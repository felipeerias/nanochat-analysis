#!/usr/bin/env python3
"""Report shell variables a script reads but never assigns.

The gate and run sections only execute on a GPU, so the local suite never
reaches them. A consolidation left $SHADOW read there but assigned nowhere,
and it took a pod to find out. This is a static check for that class.
"""
import re
import sys

ENV = {  # supplied by the environment or by the pod, not by the script
    "HOME", "PATH", "PWD", "BASH_SOURCE", "IFS", "RUNPOD_POD_ID",
    "RUNPOD_API_KEY", "UV_PROJECT_ENVIRONMENT", "NANOCHAT_BASE_DIR",
    "CUDA_VISIBLE_DEVICES", "TORCHDYNAMO_DISABLE", "VOLUME", "KEEP_POD",
    "SKIP_GATES", "GATES_ONLY", "CHECK_ONLY", "STOP_POD", "ALLOW_DIRTY",
    "ALLOW_EPHEMERAL", "ALLOW_STALE", "CONFIRM_VOLUME", "EXPECT_BACKEND",
    "TELEMETRY_DIR", "MIN_FREE_GB", "NANOCHAT_CHECKOUT", "N_SHARDS", "PY",
}
READ = re.compile(r'\$\{?([A-Z][A-Z0-9_]*)\b')
# NAME=..., NAME+=..., for NAME in, read -r A B C, mapfile -t NAME
ASSIGN = [re.compile(r'^\s*([A-Z][A-Z0-9_]*)\+?=', re.M),
          re.compile(r'^\s*for\s+([A-Z][A-Z0-9_]*)\s+in\b', re.M),
          re.compile(r'^\s*(?:local\s+)?read\s+(?:-\w+\s+)*([A-Z][A-Z0-9_\s]*)$', re.M),
          re.compile(r'read\s+(?:-\w+\s+)*((?:[A-Z][A-Z0-9_]*\s*)+)')]

bad = 0
for path in sys.argv[1:]:
    src = open(path, encoding="utf-8").read()
    assigned = set(ENV)
    for pat in ASSIGN:
        for m in pat.finditer(src):
            assigned.update(m.group(1).split())
    unknown = sorted({m.group(1) for m in READ.finditer(src)} - assigned)
    if unknown:
        bad += 1
        print(f"  {path}: read but never assigned: {unknown}")
    else:
        print(f"  ok    {path}")
sys.exit(1 if bad else 0)
