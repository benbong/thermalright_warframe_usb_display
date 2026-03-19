#!/usr/bin/env python3
"""
usb_stats_display.py
Renders a 320x320 system-stats dashboard and streams it to a
ChiZhu USBDISPLAY (VID=0x87ad PID=0x70db) at a configurable frame rate.

Usage:
  sudo venv/bin/python usb_stats_display.py          # default 1 FPS
  sudo venv/bin/python usb_stats_display.py --fps 30

Supported FPS values: 1, 24, 30, 50, 60

Protocol (reverse-engineered):
  Each write = 64-byte header + RGB565 BE pixel data (320×320×2 = 204800 bytes)
  Total per frame = 204864 bytes

  Header layout (all LE unless noted):
    0x00  4B  magic        = 12 34 56 78
    0x04  4B  frame_num    LE uint32, increments each frame
    0x08  4B  width        LE uint32 = 320
    0x0C  4B  height       LE uint32 = 320
    0x10 40B  padding      all zeros
    0x38  4B  pixel_fmt    LE uint32 = 2  (RGB565)
    0x3C  4B  payload_size LE uint32 = width × height × 2
    0x40  …   pixel data   RGB565 Big-Endian, row-major

Dependencies: pyusb, numpy, Pillow, psutil
GPU stats require nvidia-smi to be on PATH.
"""

import argparse
import struct
import subprocess
import time
from datetime import datetime

import numpy as np
import psutil
import usb.core
import usb.util
from PIL import Image, ImageDraw, ImageFont

# ── USB constants ─────────────────────────────────────────────────────────────
VENDOR_ID   = 0x87ad
PRODUCT_ID  = 0x70db
INTF_NUM    = 0
OUT_EP_ADDR = 0x01

WIDTH, HEIGHT = 320, 320

HEADER_MAGIC   = b'\x12\x34\x56\x78'
PIXEL_FMT_ID   = 2          # RGB565
PAYLOAD_BYTES  = WIDTH * HEIGHT * 2   # 204800

# ── Colours ───────────────────────────────────────────────────────────────────
BG         = (0,   0,   0)
WHITE      = (255, 255, 255)
CYAN       = (0,   220, 220)
GREEN      = (0,   220, 80)
GREY       = (80,  80,  80)
BAR_GREEN  = (0,   200, 60)
BAR_YELLOW = (220, 200, 0)
BAR_RED    = (220, 40,  40)

# ── Font loader ───────────────────────────────────────────────────────────────
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()

FONT_LABEL = _load_font(22)
FONT_VALUE = _load_font(28)
FONT_TIME  = _load_font(56)
FONT_DATE  = _load_font(20)


# ── USB helpers ───────────────────────────────────────────────────────────────
def connect_display():
    dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if dev is None:
        raise SystemExit("ChiZhu USBDISPLAY not found (VID=0x87ad PID=0x70db)")

    if dev.is_kernel_driver_active(INTF_NUM):
        dev.detach_kernel_driver(INTF_NUM)

    try:
        dev.set_configuration(1)
    except usb.core.USBError as e:
        if e.errno != 16:
            raise

    usb.util.claim_interface(dev, INTF_NUM)

    cfg   = dev.get_active_configuration()
    intf  = cfg[(INTF_NUM, 0)]
    out_ep = usb.util.find_descriptor(
        intf,
        custom_match=lambda e:
            e.bEndpointAddress & usb.ENDPOINT_DIR_MASK == usb.ENDPOINT_OUT and
            e.bmAttributes & usb.ENDPOINT_TYPE_MASK == usb.ENDPOINT_TYPE_BULK
    )
    if out_ep is None:
        raise RuntimeError("Cannot find bulk-OUT endpoint")
    return dev, out_ep


def _build_header(frame_num: int) -> bytes:
    h = bytearray(64)
    h[0:4] = HEADER_MAGIC
    struct.pack_into('<I', h, 0x04, frame_num)
    struct.pack_into('<I', h, 0x08, WIDTH)
    struct.pack_into('<I', h, 0x0C, HEIGHT)
    # 0x10..0x37 → zeros (already)
    struct.pack_into('<I', h, 0x38, PIXEL_FMT_ID)
    struct.pack_into('<I', h, 0x3C, PAYLOAD_BYTES)
    return bytes(h)


def _to_rgb565_be(img: Image.Image) -> bytes:
    """Convert a PIL RGB image to RGB565 big-endian bytes."""
    arr = np.array(img.convert("RGB"), dtype=np.uint16)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    pixels = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    return pixels.astype('>u2').tobytes()   # big-endian uint16


FRAME_NUM = 3   # static, matches vendor app behaviour

def send_frame(out_ep, img: Image.Image) -> None:
    """Write one frame. The bulk write blocks until complete — no sleep needed."""
    packet = _build_header(FRAME_NUM) + _to_rgb565_be(img)
    out_ep.write(packet, timeout=5000)


# ── Stats helpers ─────────────────────────────────────────────────────────────
def _cpu_temp() -> str:
    try:
        temps = psutil.sensors_temperatures()
        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if key in temps and temps[key]:
                return f"{temps[key][0].current:.0f}°C"
    except AttributeError:
        pass
    return "N/A"


def _gpu_stats() -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) == 2:
                return f"{parts[0].strip()}%", f"{parts[1].strip()}°C"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "N/A", "N/A"


def get_stats() -> dict:
    gpu_pct, gpu_temp = _gpu_stats()
    now = datetime.now()
    return {
        "cpu_pct":  f"{psutil.cpu_percent(interval=None):.0f}%",
        "cpu_temp": _cpu_temp(),
        "gpu_pct":  gpu_pct,
        "gpu_temp": gpu_temp,
        "time_str": now.strftime("%H:%M:%S"),
        "date_str": now.strftime("%A  %d %b %Y"),
    }


# ── Rendering ─────────────────────────────────────────────────────────────────
def _bar_color(pct_str: str) -> tuple:
    try:
        v = float(pct_str.rstrip("%"))
        if v < 50:
            return BAR_GREEN
        if v < 80:
            return BAR_YELLOW
        return BAR_RED
    except ValueError:
        return GREY


def _draw_bar(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int,
              pct_str: str) -> None:
    draw.rectangle([x, y, x + w - 1, y + h - 1], outline=GREY)
    try:
        v = min(max(float(pct_str.rstrip("%")), 0), 100)
        fill_w = int((w - 2) * v / 100)
        if fill_w > 0:
            draw.rectangle([x + 1, y + 1, x + fill_w, y + h - 2],
                           fill=_bar_color(pct_str))
    except ValueError:
        pass


def _draw_stat_row(draw: ImageDraw.Draw, y: int, label: str,
                   label_color: tuple, pct: str, temp: str) -> None:
    draw.text((12, y),    label, font=FONT_LABEL, fill=label_color)
    draw.text((95, y),    pct,   font=FONT_VALUE, fill=WHITE)
    draw.text((200, y),   temp,  font=FONT_VALUE, fill=WHITE)
    _draw_bar(draw, 12, y + 34, 296, 18, pct)


def render_frame(stats: dict) -> Image.Image:
    img  = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # ── CPU row  (top)
    _draw_stat_row(draw, 18, "CPU", CYAN, stats["cpu_pct"], stats["cpu_temp"])

    draw.line([(0, 88), (WIDTH, 88)], fill=GREY, width=1)

    # ── GPU row  (middle)
    _draw_stat_row(draw, 100, "GPU", GREEN, stats["gpu_pct"], stats["gpu_temp"])

    draw.line([(0, 170), (WIDTH, 170)], fill=GREY, width=1)

    # ── Time  (large, centred)
    time_bbox = draw.textbbox((0, 0), stats["time_str"], font=FONT_TIME)
    time_w = time_bbox[2] - time_bbox[0]
    draw.text(((WIDTH - time_w) // 2, 185), stats["time_str"],
              font=FONT_TIME, fill=WHITE)

    # ── Date  (small, centred)
    date_bbox = draw.textbbox((0, 0), stats["date_str"], font=FONT_DATE)
    date_w = date_bbox[2] - date_bbox[0]
    draw.text(((WIDTH - date_w) // 2, 250), stats["date_str"],
              font=FONT_DATE, fill=GREY)

    return img


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Stream system stats to ChiZhu USBDISPLAY")
    parser.add_argument("--fps", type=int, default=1,
                        choices=[1, 24, 30, 50, 60],
                        help="Target frame rate (default: 1)")
    args = parser.parse_args()

    frame_interval = 1.0 / args.fps

    print("Connecting to ChiZhu USBDISPLAY …")
    dev, out_ep = connect_display()
    print(f"Connected. Streaming at {args.fps} FPS (Ctrl+C to stop).")

    # Prime psutil so the first reading isn't 0 %
    psutil.cpu_percent(interval=None)

    frame_count = 0
    try:
        while True:
            t0 = time.monotonic()
            try:
                stats = get_stats()
                img   = render_frame(stats)
                send_frame(out_ep, img)   # bulk write blocks ~47 ms naturally
                frame_count += 1
            except usb.core.USBError as e:
                print(f"[frame {frame_count}] USB error: {e}")
            except Exception as e:
                print(f"[frame {frame_count}] Error: {e}")
            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nStopping …")
    finally:
        usb.util.release_interface(dev, INTF_NUM)
        usb.util.dispose_resources(dev)
        print("USB interface released.")


if __name__ == "__main__":
    main()
