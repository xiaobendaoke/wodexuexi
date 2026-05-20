#!/usr/bin/env python3
"""Compatibility wrapper for the old Pillow figure generator.

The Pillow version used fixed pixel coordinates and macOS-only font paths,
which caused text fallback, misalignment, and English labels on the Linux
server.  Keep this wrapper so old commands still work, but delegate all real
figure generation to the Matplotlib implementation.
"""

from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).with_name("generate_readme_figures.py")


if __name__ == "__main__":
    print("Pillow generator is deprecated; running generate_readme_figures.py instead.")
    runpy.run_path(str(SCRIPT), run_name="__main__")
