"""Portable data and output paths for the I0008/A0001 scripts."""

from pathlib import Path

from loader.paths import DEFAULT_DATA_ROOT


DATA_ROOT = str(DEFAULT_DATA_ROOT)
OUTPUT_ROOT = str(Path(__file__).resolve().parent.parent)
D12_CONT = str(Path(OUTPUT_ROOT) / "d12_cont.pkl")
