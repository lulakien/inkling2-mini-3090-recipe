#!/usr/bin/env python3
"""Write a 256x256 PNG, left half red, right half blue (stdlib only), for api_smoke.py --image."""
import struct, sys, zlib
w = h = 256
row = b"\x00" + (b"\xd0\x20\x20" * (w // 2)) + (b"\x20\x40\xd0" * (w // 2))
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
      chunk(b"IDAT", zlib.compress(row * h, 9)) + chunk(b"IEND", b"")
open(sys.argv[1] if len(sys.argv) > 1 else "test_image.png", "wb").write(png)
