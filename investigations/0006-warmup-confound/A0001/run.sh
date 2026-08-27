#!/usr/bin/env bash
# I0006 / A0001 - full pipeline. Regenerates every artifact in this folder.
# series.parquet and points.parquet are large regenerable caches and are not
# kept in the repository; everything else here is produced by this script.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

$PY -W ignore extract.py                       # -> series.parquet, runs.json
$PY -W ignore structure.py    > structure.txt  # schedule/grid geometry
$PY -W ignore analyze.py                       # -> per_family_region.csv, points.parquet
$PY -W ignore classify.py     > classify.txt   # -> families.csv (the deliverable)
$PY -W ignore headline.py     > headline.txt   # numbers quoted in result.md
$PY -W ignore robustness.py   > robustness.txt # sensitivity of the two choices
$PY -W ignore plots.py                         # -> figures/
