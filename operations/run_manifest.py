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

A run row is flat: it is the arguments for that run, nothing more. There is no
separate category for some of them. Which names are valid is read from the two
parsers that declare them, so the manifest speaks base_train's own vocabulary.

  run_manifest.py MANIFEST RUN_ID --head SHA [--check-only]
"""

import argparse
import json
import os
import re
import subprocess
import sys

# A row names flags. Which flags exist is read from the checkout the manifest
# pins - base_train's own and the --telemetry-* family - so there is no list
# here to drift, and an experiment branch that adds a flag can use it at once.
PARSERS = ("scripts/base_train.py", "nanochat/telemetry.py")

# Set by the runner. A manifest naming one of these is a conflict, not a
# configuration, so it is refused rather than silently overridden.
RUNNER_OWNED = {"run", "model_tag", "telemetry_dir", "telemetry_manifest",
                "telemetry_manifest_run", "telemetry_controller_commit",
                "telemetry_controller_tree", "resume_from_step"}

ARG_RE = re.compile(
    r'add_argument\(\s*"--([a-z0-9-]+)"\s*(?:,\s*type=(\w+))?[^)]*?'
    r'(?:,\s*(action)="store_true")?', re.S)
# simple literal defaults only, so --print can report a flag a manifest leaves
# alone; anything computed is reported as unknown rather than guessed at
DEFAULT_RE = re.compile(
    r'add_argument\(\s*"--([a-z0-9-]+)".*?default=("(?:[^"]*)"|-?\d+\.?\d*)[,)]',
    re.S)

TOP_KEYS = {"manifest_version", "telemetry_schema", "nanochat_commit",
            "defaults", "runs"}


def die(msg):
    sys.stderr.write(f"manifest: {msg}\n")
    raise SystemExit(1)


def declared_flags(checkout):
    """{flag: declared type} across both parsers in this checkout."""
    out = {}
    for rel in PARSERS:
        path = os.path.join(checkout, rel)
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except OSError as exc:
            die(f"cannot read {path}: {exc}")
        for m in ARG_RE.finditer(src):
            name = m.group(1).replace("-", "_")
            out[name] = "bool" if m.group(3) else (m.group(2) or "str")
    if not out:
        die(f"no arguments declared in {PARSERS} under {checkout}")
    return out


def declared_defaults(checkout):
    """{flag: default} for flags whose default is a plain literal."""
    out = {}
    for rel in PARSERS:
        try:
            with open(os.path.join(checkout, rel), encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        for m in DEFAULT_RE.finditer(src):
            raw = m.group(2)
            out[m.group(1).replace("-", "_")] = (
                raw[1:-1] if raw.startswith('"') else raw)
    return out


def coerce(key, value, kind):
    """Format as the declaring parser will parse it back.

    A float flag written as JSON 1 has to become 1.0, or what we pass and what
    user_config records will not compare equal and verification fails on
    formatting rather than on substance.
    """
    if isinstance(value, bool) and kind != "bool":
        die(f"{key} is not a flag; it takes a {kind}")
    if kind == "float":
        if not isinstance(value, (int, float)):
            die(f"{key} must be a number, got {value!r}")
        return float(value)
    if kind == "int":
        if isinstance(value, float) and value != int(value):
            die(f"{key}={value} is not an integer")
        if not isinstance(value, (int, float)):
            die(f"{key} must be an integer, got {value!r}")
        return int(value)
    if kind == "bool":
        if not isinstance(value, bool):
            die(f"{key} is a flag; use true or false")
        return value
    if not isinstance(value, (str, int, float)):
        die(f"{key} must be a string, got {value!r}")
    return value


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
    row.update(m["runs"][run_id] or {})
    known = declared_flags(checkout)
    unknown = sorted(k for k in row if k not in known)
    if unknown:
        die(f"not flags in this checkout: {unknown}")
    clash = sorted(set(row) & RUNNER_OWNED)
    if clash:
        die(f"the runner sets these; a manifest must not: {clash}")

    flags = {k: coerce(k, row[k], known[k]) for k in sorted(row)}

    # Two narrowings of what argparse accepts, both real decisions rather than
    # validation: the shadow arm has no fp64 implementation, and the 'every'
    # deep schedule is a development shortcut that no recorded run may use.
    if flags.get("telemetry_shadow") == "fp64":
        die("telemetry_shadow=fp64 is declared but not implemented")
    if flags.get("telemetry_deep_schedule", "pythia") != "pythia":
        die("official runs use telemetry_deep_schedule=pythia")

    return dict(manifest_version=m["manifest_version"], pin=pin, flags=flags)


def train_argv(run, args):
    argv = [sys.executable, "-m", "scripts.base_train",
            f"--model-tag={args.run_id}", "--run=dummy",
            f"--telemetry-dir={args.telemetry_dir}",
            f"--telemetry-manifest={args.manifest}",
            f"--telemetry-manifest-run={args.run_id}"]
    if args.controller_commit:
        argv.append(f"--telemetry-controller-commit={args.controller_commit}")
    if args.controller_tree:
        argv.append(f"--telemetry-controller-tree={args.controller_tree}")
    for key, value in run["flags"].items():
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
            "--expect", "compute_dtype=torch.bfloat16"]
    for key, value in run["flags"].items():
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
        if args.print_key in run["flags"]:
            print(run["flags"][args.print_key])
        else:
            fallback = declared_defaults(args.checkout).get(args.print_key)
            if fallback is None:
                die(f"{args.print_key!r} is unset and has no literal default")
            print(fallback)
        return 0

    print(f"[check] {os.path.basename(args.manifest)} "
          f"({run['manifest_version']}) row {args.run_id}")
    if run["flags"]:
        for key, value in run["flags"].items():
            print(f"[check]   {key}={value}")
    else:
        print("[check]   (no arguments; upstream defaults throughout)")
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
