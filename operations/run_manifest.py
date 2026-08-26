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

# The flags a manifest may set are base_train's own, read out of the checkout
# the manifest pins. Not a hand-maintained copy: a copy drifts, and the one
# that lived here was missing seven flags the controller itself already used.
# Reading them from source also means an experiment branch that adds a flag
# can use it from a manifest immediately, with no change here.
#
# The runner owns these, so a manifest must not also set them.
RUNNER_OWNED = {"run", "model_tag", "depth", "seed", "head_dim",
                "resume_from_step"}
ARG_RE = re.compile(
    r'add_argument\(\s*"--([a-z0-9-]+)"\s*(?:,\s*type=(\w+))?[^)]*?'
    r'(?:,\s*(action)="store_true")?', re.S)


def upstream_flags(checkout):
    """{flag_name: python type name} declared by base_train in this checkout."""
    path = os.path.join(checkout, "scripts", "base_train.py")
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError as exc:
        die(f"cannot read {path}: {exc}")
    out = {}
    for m in ARG_RE.finditer(src):
        name, typ, action = m.group(1).replace("-", "_"), m.group(2), m.group(3)
        out[name] = "bool" if action else (typ or "str")
    if not out:
        die(f"no arguments found in {path}; has base_train changed shape?")
    return out


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


def resolve(path, run_id, head, checkout):
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
    known = upstream_flags(checkout)
    unknown = sorted(k for k in recipe if k not in known)
    if unknown:
        die(f"not base_train flags in this checkout: {unknown}")
    clash = sorted(set(recipe) & RUNNER_OWNED)
    if clash:
        die(f"the runner sets these; a recipe must not: {clash}")

    flags = {}
    for key in sorted(recipe):
        value = recipe[key]
        kind = known[key]
        if isinstance(value, bool) and kind != "bool":
            die(f"recipe.{key} must not be a JSON boolean")
        # format per the type base_train declares, so that what we pass and
        # what argparse records string-compare: a float flag given as 1 has
        # to become 1.0 or the verification assertion fails on formatting
        if kind == "float":
            if not isinstance(value, (int, float)):
                die(f"recipe.{key} must be a number, got {value!r}")
            value = float(value)
        elif kind == "int":
            if isinstance(value, float) and value != int(value):
                die(f"recipe.{key}={value} is not an integer")
            if not isinstance(value, (int, float)):
                die(f"recipe.{key} must be an integer, got {value!r}")
            value = int(value)
        elif kind == "bool":
            if not isinstance(value, bool):
                die(f"recipe.{key} is a flag; use true or false")
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
        flag = f"--{key.replace('_', '-')}"
        # store_true flags take no value: present means true, absent false
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.append(f"{flag}={value}")
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
    p.add_argument("--print", dest="print_key", default="",
                   help="print one resolved value and exit; the shell needs "
                        "the acceptance arm for its own gate key")
    p.add_argument("--check-only", action="store_true",
                   help="resolve and report, start nothing")
    args = p.parse_args()
    args.manifest = os.path.abspath(args.manifest)

    run = resolve(args.manifest, args.run_id, args.head, args.checkout)

    if args.print_key:
        if args.print_key not in run:
            die(f"no resolved value called {args.print_key!r}")
        print(run[args.print_key])
        return 0

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
