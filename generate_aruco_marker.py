#!/usr/bin/env python3
"""generate_aruco_marker.py

Generate ArUco markers as both high-resolution PNG and lossless SVG files.

Example
-------
Create a 600 × 600 px marker with ID 23 from the 5×5 dictionary and write
`marker_23.png` & `marker_23.svg` to the current directory:

```bash
python generate_aruco_marker.py --dict DICT_5X5_100 --id 23 --size 600 --outfile marker_23
```

The script has no external dependencies besides OpenCV (>=4.0).
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import cv2
import numpy as np

# ---------------------------------------------
# Core functionality
# ---------------------------------------------

def _parse_args() -> argparse.Namespace:  # noqa: D401
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate ArUco marker images as high-resolution PNG and vector SVG."
        )
    )
    parser.add_argument(
        "--dict",
        default="DICT_4X4_50",
        help="Predefined dictionary name, e.g. DICT_6X6_250 (see OpenCV docs)",
    )
    parser.add_argument("--id", type=int, required=True, help="Marker ID to generate")
    parser.add_argument(
        "--size",
        type=int,
        default=600,
        help="Output PNG side length in pixels (square).",
    )
    parser.add_argument(
        "--border-bits",
        type=int,
        default=1,
        help="Width of the quiet zone (in bits/modules).",
    )
    parser.add_argument(
        "--outfile",
        default="aruco_marker",
        help="Output file prefix (without extension).",
    )
    parser.add_argument(
        "--svg-mm",
        type=float,
        default=None,
        help="Override SVG physical side length (in millimetres). If omitted, one module = 1 mm.",
    )
    parser.add_argument(
        "--jpg",
        action="store_true",
        help="Also write a JPEG file (<outfile>.jpg).",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=100,
        help="JPEG quality (0–100, higher = better, larger file). Only used with --jpg.",
    )
    return parser.parse_args()


def _get_dictionary(dict_name: str) -> cv2.aruco_Dictionary:  # type: ignore
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(
            f"Dictionary '{dict_name}' not found in cv2.aruco. "
            "Check the spelling / OpenCV version."
        )
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def _draw_png(
    aruco_dict: cv2.aruco_Dictionary,  # type: ignore
    marker_id: int,
    side_pixels: int,
    border_bits: int,
) -> np.ndarray:
    """Return a high-resolution PNG image (NumPy array)."""
    return cv2.aruco.drawMarker(aruco_dict, marker_id, side_pixels, borderBits=border_bits)


def _extract_bit_matrix(
    high_res_img: np.ndarray, modules: int
) -> np.ndarray:  # shape = (modules, modules)
    """Downsample *high_res_img* to a binary \*module\* grid (0 = black, 255 = white)."""
    side_pixels = high_res_img.shape[0]
    scale = side_pixels // modules
    if scale == 0:
        raise ValueError("Image too small to determine module size.")

    bit_matrix = np.empty((modules, modules), dtype=np.uint8)
    for y in range(modules):
        for x in range(modules):
            # sample the centre of the module square
            y0 = y * scale + scale // 2
            x0 = x * scale + scale // 2
            bit_matrix[y, x] = high_res_img[y0, x0]
    return bit_matrix


def _save_svg(bit_matrix: np.ndarray, out_path: str, svg_mm: float | None = None) -> None:
    """Write *bit_matrix* (0 = black, 255 = white) into *out_path*.

    Parameters
    ----------
    bit_matrix
        Binary marker (shape = modules × modules, values 0/255).
    out_path
        Destination file path (\*.svg).
    svg_mm
        Physical side length to embed in SVG (mm). If None, defaults to
        *modules* mm so that each module maps to 1 mm.
    """
    modules = bit_matrix.shape[0]
    side_mm = svg_mm if svg_mm is not None else modules
    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{side_mm}mm" height="{side_mm}mm" viewBox="0 0 {modules} {modules}">',
        '<rect width="100%" height="100%" fill="white" />',
    ]

    for y in range(modules):
        for x in range(modules):
            if bit_matrix[y, x] == 0:  # black square
                svg_lines.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="black" />')

    svg_lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(svg_lines))


# ---------------------------------------------
# Entry point
# ---------------------------------------------

def main() -> None:  # noqa: D401
    args = _parse_args()

    aruco_dict = _get_dictionary(args.dict)
    modules = aruco_dict.markerSize + 2 * args.border_bits

    # PNG
    png_img = _draw_png(aruco_dict, args.id, args.size, args.border_bits)
    png_path = f"{args.outfile}.png"
    cv2.imwrite(png_path, png_img)

    # Optionally write JPEG
    if args.jpg:
        jpg_path = f"{args.outfile}.jpg"
        quality = max(0, min(args.jpg_quality, 100))
        cv2.imwrite(
            jpg_path,
            png_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
    else:
        jpg_path = None

    # SVG
    bit_matrix = _extract_bit_matrix(png_img, modules)
    svg_path = f"{args.outfile}.svg"
    _save_svg(bit_matrix, svg_path, svg_mm=args.svg_mm)

    msg = f"Written: {png_path} ({args.size}×{args.size}px)"
    if jpg_path:
        msg += f", {jpg_path} (JPEG)"
    msg += f", {svg_path} (vector)."
    print(msg)


if __name__ == "__main__":
    main() 