"""Workspace paths shared by analysis scripts.

The sibling layout is the default. Environment variables make a clone and an
extracted telemetry collection portable without editing analysis code.
"""

import os
from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.environ.get(
    "NANOCHAT_WORKSPACE_ROOT", ANALYSIS_ROOT.parent)).expanduser().resolve()
NANOCHAT_REPO = Path(os.environ.get(
    "NANOCHAT_REPO", WORKSPACE_ROOT / "nanochat")).expanduser().resolve()
DEFAULT_DATA_ROOT = Path(os.environ.get(
    "NANOCHAT_TELEMETRY_DATA_ROOT",
    WORKSPACE_ROOT / "telemetry-data" / "sweep" / "telemetry-data",
)).expanduser().resolve()
