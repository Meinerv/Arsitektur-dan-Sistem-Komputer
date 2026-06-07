#!/usr/bin/env python3
"""
generate_test_image.py
Generates a grayscale test PGM image for median filter experiments.
Usage:
  python3 generate_test_image.py [width] [height] [output.pgm]
  python3 generate_test_image.py 512 512 input.pgm
"""
import sys
import struct
import math

def generate_test_image(width=512, height=512, filename="input.pgm"):
    pixels = bytearray(width * height)
    cx, cy = width // 2, height // 2

    for y in range(height):
        for x in range(width):
            # Combination of geometric patterns
            # 1. Radial gradient
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            radial = int(255 * (1.0 - min(dist / (min(width, height) / 2), 1.0)))

            # 2. Checkerboard
            check_sz = max(width, height) // 16
            check = 200 if ((x // check_sz + y // check_sz) % 2 == 0) else 50

            # 3. Sinusoidal stripes
            freq = 8.0
            sine_x = int(127.5 + 127.5 * math.sin(2 * math.pi * freq * x / width))
            sine_y = int(127.5 + 127.5 * math.cos(2 * math.pi * freq * y / height))

            # Mix all patterns
            val = int(0.4 * radial + 0.3 * check + 0.15 * sine_x + 0.15 * sine_y)
            pixels[y * width + x] = max(0, min(255, val))

    # Write PGM P5 (binary)
    with open(filename, "wb") as f:
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode("ascii"))
        f.write(bytes(pixels))

    print(f"[generate_test_image] Created: {filename}  ({width}x{height} px)")


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    fn = sys.argv[3] if len(sys.argv) > 3 else "input.pgm"
    generate_test_image(w, h, fn)
