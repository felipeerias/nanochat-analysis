#!/usr/bin/env python3
"""Turn one manifest row into a training run, and verify what it produced.

The manifest is the a priori record of what a run is. This program resolves
one row of it and executes it directly: the argument list never leaves this
process, so there is no tab-separated hand-off to a shell that reassembles it.

The shell that calls this owns the pod - the self-stop trap, the volume and
space checks, the environment, the log, the acceptance gates. Those are
lifecycle concerns. Deciding what a run is, is not.

Git stays in the shell too. It already runs processes naturally and it needs
HEAD and the controller identity for its own gate key, so this program is told
them rather than looking them up a second time.

  run_manifest.py MANIFEST RUN_ID --head SHA [--check-only]
"""

import argparse
import json
import os
import re
import subprocess
import sys

# base_train flags a manifest may set. Every one already exists upstream; this
# adds no knob to the trainer. A flag NOT here cannot be set from a manifest at
# all - that change belongs on an experiment branch, where the commit
# identifies it and the manifest pins it. That boundary is the reason this
# table exists; the ranges are just cheap protection against paying for a pod
# to discover a typo.
RECIPE = {
    "aspect_ratio":            ("int",   8, 512),
    "window_pattern":          ("pat",   r"[SL]{1,32}", None),
    "max_seq_len":             ("int",   128, 8192),
    "warmup_steps":            ("int",   0, 100_000),
    "warmdown_ratio":          ("float", 0.0, 1.0),
    "final_lr_frac":           ("float", 0.0, 1.0),
    "embedding_lr":            ("float", 0.0, 10.0),
    "unembedding_lr":          ("float", 0.0, 10.0),
    "matrix_lr":               ("float", 0.0, 10.0),
    "scalar_lr":               ("float", 0.0, 10.0),
    "weight_decay":            ("float", 0.0, 10.0),
    "device_batch_size":       ("int",   1, 1024),
    "total_batch_size":        ("int",   1, 1 << 24),
    "num_iterations":          ("int",   1, 10**6),
    "target_param_data_ratio": ("float", 0.1, 1000.0),
}

TOP_KEYS = {"manifest_version", "telemetry_schema", "nanochat_commit",
            "defaults", "runs"}
ROW_KEYS = {"depth", "seed", "shadow", "periodic_points", "checkpoints",
            "deep_schedule", "head_dim", "recipe"}


def die(msg):
    sys.stderr.write(f"manifest: {msg}\n")
    raise SystemExit(1)


def strict_int(v, name):
    """JSON integers only: no booleans, no coercible strings."""
    if isinstance(v, bool) or not isinstance(v, int):
        die(f"{name} must be a JSON integer, got {v!r}")
    return v


def resolve(path, run_id, head):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        m = json.loads(raw)
    except ValueError as exc:
        die(f"{path} is not valid JSON: {exc}")

    unknown = sorted(set(m) - TOP_KEYS)
    if unknown:
        die(f"unknown top-level keys: {unknown}")
    if not isinstance(m.get("manifest_version"), str) or not m["manifest_version"]:
        die("manifest_version must be a nonempty string")
    if str(m.get("telemetry_schema")) != "3":
        die(f"telemetry_schema must be 3, got {m.get('telemetry_schema')!r}")
    # The manifest pins the training code and the caller enforces it here.
    # Required, so a new manifest cannot silently produce an unreproducible run.
    pin = str(m.get("nanochat_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        die("nanochat_commit must be a full 40-character commit id")
    if head and pin != head:
        die(f"manifest pins nanochat {pin} but HEAD is {head}")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", run_id):
        die(f"invalid run_id {run_id!r}")
    if run_id not in (m.get("runs") or {}):
        die(f"run_id {run_id!r} not in the manifest run table")

    row = dict(m.get("defaults") or {})
    row.update(m["runs"][run_id])
    unknown = sorted(set(row) - ROW_KEYS)
    if unknown:
        die(f"unknown row keys for {run_id}: {unknown}")

    depth = strict_int(row["depth"], "depth")
    seed = strict_int(row["seed"], "seed")
    points = strict_int(row.get("periodic_points", 25), "periodic_points")
    ckpts = strict_int(row.get("checkpoints", 0), "checkpoints")
    head_dim = strict_int(row.get("head_dim", 128), "head_dim")
    shadow = row.get("shadow", "fp32")
    schedule = row.get("deep_schedule", "pythia")

    if shadow not in ("off", "fp32"):
        die(f"shadow must be off or fp32, got {shadow!r}")
    if schedule != "pythia":
        die("official runs use deep_schedule=pythia ('every' is development"
            " only, via scripts.base_train directly)")
    if not 1 <= depth <= 64:
        die(f"depth {depth} outside [1, 64]")
    if not 0 <= seed < 10**6:
        die(f"seed {seed} outside [0, 999999]")
    if not 0 < points <= 10**4:
        die(f"periodic_points {points} outside [1, 10000]")
    if not 0 <= ckpts <= 100:
        die(f"checkpoints {ckpts} outside [0, 100]")
    if head_dim != 128:
        die(f"this sweep requires head_dim=128 (upstream default), got {head_dim}")

    # defaults' recipe overlaid by the row's, so an arm states only what differs
    recipe = dict((m.get("defaults") or {}).get("recipe") or {})
    override = m["runs"][run_id].get("recipe") or {}
    if not isinstance(override, dict):
        die("recipe must be an object")
    recipe.update(override)
    unknown = sorted(set(recipe) - set(RECIPE))
    if unknown:
        die(f"recipe keys not permitted from a manifest: {unknown}")

    flags = {}
    for key in sorted(recipe):
        kind, lo, hi = RECIPE[key]
        value = recipe[key]
        if kind == "int":
            value = strict_int(value, f"recipe.{key}")
            if not lo <= value <= hi:
                die(f"recipe.{key}={value} outside [{lo}, {hi}]")
        elif kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                die(f"recipe.{key} must be a JSON number, got {value!r}")
            # a float flag given as JSON 1 must become 1.0, or the recorded
            # user_config value will not compare against what was passed
            value = float(value)
            if not lo <= value <= hi:
                die(f"recipe.{key}={value} outside [{lo}, {hi}]")
        else:
            if not isinstance(value, str) or not re.fullmatch(lo, value):
                die(f"recipe.{key}={value!r} does not match {lo}")
        flags[key] = value

    return dict(manifest_version=m["manifest_version"], pin=pin, depth=depth,
                seed=seed, points=points, ckpts=ckpts, head_dim=head_dim,
                shadow=shadow, schedule=schedule, recipe=flags)


def train_argv(run, args):
    argv = [sys.executable, "-m", "scripts.base_train",
            f"--depth={run['depth']}", f"--seed={run['seed']}",
            f"--model-tag={args.run_id}", "--run=dummy",
            f"--telemetry-dir={args.telemetry_dir}",
            f"--telemetry-periodic-points={run['points']}",
            f"--telemetry-deep-schedule={run['schedule']}",
            f"--telemetry-shadow={run['shadow']}",
            f"--telemetry-checkpoints={run['ckpts']}",
            f"--telemetry-manifest={args.manifest}",
            f"--telemetry-manifest-run={args.run_id}"]
    if args.controller_commit:
        argv.append(f"--telemetry-controller-commit={args.controller_commit}")
    if args.controller_tree:
        argv.append(f"--telemetry-controller-tree={args.controller_tree}")
    for key, value in run["recipe"].items():
        argv.append(f"--{key.replace('_', '-')}={value}")
    return argv


def verify_argv(run, args):
    # assert the INPUTS. Whether the model matches them is base_train's own
    # arithmetic, checked by the verifier, which lives beside base_train.
    argv = [sys.executable, "runs/verify_telemetry_run.py", args.telemetry_dir,
            "--tag", args.run_id,
            "--expect", f"attention_backend={args.expect_backend}",
            "--expect", "compute_dtype=torch.bfloat16",
            "--expect", f"telemetry_config.shadow_arm={run['shadow']}",
            "--expect", f"model_config.n_layer={run['depth']}",
            "--expect", f"user_config.depth={run['depth']}",
            "--expect", f"user_config.head_dim={run['head_dim']}",
            "--expect", f"seed={run['seed']}"]
    for key, value in run["recipe"].items():
        argv += ["--expect", f"user_config.{key}={value}"]
    return argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest")
    p.add_argument("run_id")
    p.add_argument("--head", default="", help="HEAD of the training checkout")
    p.add_argument("--checkout", default=".", help="the training checkout")
    p.add_argument("--telemetry-dir", default="")
    p.add_argument("--expect-backend", default="fa3")
    p.add_argument("--controller-commit", default="")
    p.add_argument("--controller-tree", default="")
    p.add_argument("--check-only", action="store_true",
                   help="resolve and report, start nothing")
    args = p.parse_args()
    args.manifest = os.path.abspath(args.manifest)

    run = resolve(args.manifest, args.run_id, args.head)

    print(f"[check] {os.path.basename(args.manifest)} "
          f"({run['manifest_version']}) row {args.run_id}")
    print(f"[check] depth={run['depth']} seed={run['seed']} "
          f"shadow={run['shadow']} points={run['points']} "
          f"ckpt={run['ckpts']} head_dim={run['head_dim']}")
    if run["recipe"]:
        shown = " ".join(f"{k}={v}" for k, v in run["recipe"].items())
        print(f"[check] recipe: {shown}")
    else:
        print("[check] recipe: none (upstream defaults)")
    if args.head:
        print(f"[check] nanochat {args.head} matches the manifest pin")

    if args.check_only:
        print("[check] --check-only: configuration is valid, nothing started")
        return 0
    if not args.telemetry_dir:
        die("--telemetry-dir is required for a real run")

    argv = train_argv(run, args)
    print(f"[run] {' '.join(argv[2:])}", flush=True)
    result = subprocess.run(argv, cwd=args.checkout)
    if result.returncode != 0:
        sys.stderr.write(f"training exited {result.returncode}\n")
        return result.returncode

    result = subprocess.run(verify_argv(run, args), cwd=args.checkout)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
