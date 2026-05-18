#!/usr/bin/env python3
"""Thin wrapper so the CLI is available under scripts/ as well.

Usage examples:
  python scripts/cli_compare_fft_fd_ho3d_jdqmr.py --help
  python scripts/cli_compare_fft_fd_ho3d_jdqmr.py --N 32 --n-levels 20
"""

from pathlib import Path
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "cli_compare_fft_fd_ho3d_jdqmr.py"

if not TARGET.exists():
    raise FileNotFoundError(f"Expected CLI not found: {TARGET}")

# Ensure repo root is importable when running from scripts/
sys.path.insert(0, str(REPO_ROOT))
runpy.run_path(str(TARGET), run_name="__main__")
