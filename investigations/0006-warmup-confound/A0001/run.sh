#!/usr/bin/env bash
# I0006 / A0001 - full pipeline. Regenerates every artifact in this folder.
# series.parquet and points.parquet are large regenerable caches and are not
# kept in the repository; everything else here is produced by this script.
# Run `uv sync --frozen` once; PYTHON may override the locked interpreter.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-}"
PY="${PYTHON:-$ANALYSIS_ROOT/.venv/bin/python}"

if [[ "$PY" == */* ]]; then
    if [[ ! -x "$PY" ]]; then
        echo "I0006: Python not found at $PY; run 'uv sync --frozen' in $ANALYSIS_ROOT" >&2
        exit 1
    fi
elif ! command -v "$PY" >/dev/null 2>&1; then
    echo "I0006: Python command '$PY' not found" >&2
    exit 1
fi

cd "$SCRIPT_DIR"

"$PY" -W ignore extract.py                       # -> series.parquet, runs.json
"$PY" -W ignore structure.py    > structure.txt  # schedule/grid geometry
"$PY" -W ignore analyze.py                       # -> per_family_region.csv, points.parquet
"$PY" -W ignore classify.py     > classify.txt   # -> families.csv (the deliverable)
"$PY" -W ignore headline.py     > headline.txt   # numbers quoted in result.md
"$PY" -W ignore robustness.py   > robustness.txt # sensitivity of the two choices
"$PY" -W ignore plots.py                         # -> figures/
