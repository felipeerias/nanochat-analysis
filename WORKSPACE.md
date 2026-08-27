# nanochat telemetry project — map

This project measures the training dynamics of nanochat. It adds telemetry to
pretraining, runs experiments on rented GPUs, and stores the data for later
analysis. The data comes first. The hypotheses come later.

## Two repositories

Everything lives under `~/Igalia/nanochat/`, which is **not** itself a git
repository.

| repository | holds |
|---|---|
| `nanochat/` | The fork. Branch `telemetry`. Upstream code, the instrument, its tests, its verifier, its specification. Experiment code goes on branches off `telemetry`. |
| `nanochat-analysis/` | Everything we build around it: how runs are launched, what the data showed, what to measure next. |

Not in git: `telemetry-data/` (parquet and checkpoints), `cache/` (local CPU
smoke data, `NANOCHAT_BASE_DIR`), `nanochat_agent_key.txt` (Runpod API key).

## Where things are

| path | contents |
|---|---|
| `nanochat/nanochat/telemetry.py` | The instrument. One file, 14 sections. |
| `nanochat/runs/verify_telemetry_run.py` | The verifier. Stays with the instrument: it reads private constants and rejects schemas it does not know. |
| `nanochat/tests/test_telemetry*.py` | Tests for the instrument. |
| `nanochat/docs/telemetry-spec.md` | What the instrument measures, and why. Its normative contract. |
| `nanochat/docs/history/` | Old build plans. Finished work. |
| `nanochat-analysis/DATASET.md` | **The dataset card. Start here.** Ten caveats. |
| `nanochat-analysis/loader/` | Code to read the data. |
| `nanochat-analysis/profiles/` | What each run looks like. Descriptive only. |
| `nanochat-analysis/investigations/` | One folder per question. Claims live here. |
| `nanochat-analysis/exploratory/` | Models and quick checks. Not citable. |
| `nanochat-analysis/experiments/` | Designs for data we do not have yet. |
| `nanochat-analysis/operations/` | Runner, pod setup, sweep manifests. |
| `nanochat-analysis/telemetry-v4-plan.md` | What the instrument should record next. |
| `nanochat-analysis/telemetry-plan-v2.md` | The research plan. |
| `telemetry-data/sweep/telemetry-data/` | Eight segments: seven schema-v3 runs and the schema-v1 shakedown. |

## To start work

1. Read `nanochat-analysis/DATASET.md`. It lists the runs, the schema, and the
   caveats. The caveats matter.
2. Read `nanochat-analysis/README.md`. It holds the working procedure.
3. Load the data with `nanochat-analysis/loader/telemetry_load.py`, or read the
   parquet files directly.

## Rules

- The instrument is frozen. Change it only for a strong reason.
- A manifest pins the training code with `nanochat_commit`. The runner refuses
  to start without it, or against a different HEAD. Every segment embeds its
  own copy, so the external file may change afterwards.
- A manifest row is flat: it is the arguments for that run. Valid names are
  read from the pinned checkout's own parsers, so no list exists here to
  drift. Changing the trainer itself still goes on an experiment branch,
  identified by its commit.
- Never edit `nanochat/uv.lock`. The analysis repository has its own generated
  lockfile; reproduce it with `uv sync --frozen` and update it with `uv lock`,
  never by hand.
- Never commit `nanochat_agent_key.txt`. Never copy it to the GPU volume.
- The `telemetry` branch is not for upstream. It stays a research branch.
- Commit messages use plain language and state that AI wrote the change.

## GPU runs

Runs happen on Runpod. Region AP-JP-1. GPU H100 SXM. Network volume
`nanochat_experiment` (250 GB) holds the data and the model checkpoints.
Model checkpoints are **not** in the local copy. They stay on the volume.

```bash
bash operations/telemetry_pod_setup.sh
git -C /workspace/nanochat checkout <the manifest's nanochat_commit>
bash operations/telemetry_run.sh operations/manifests/sweep-d12-d16-v1.json d12-s7
```

`GATES_ONLY=1` runs the acceptance gates and stops, which is how a controller
change is proved on real hardware without paying for a full run.

A run leaves the pod running when it finishes. Pass `STOP_POD=1` to have it
stop the pod instead; that flag makes the runner check up front that
`runpodctl` and the API key are present, so a self-stop cannot fail silently
at the end.

## Bringing run data home

On the pod, make one archive without the large checkpoint tensors, inspect its
size, and send it:

```bash
tar --exclude='*/checkpoints/*.pt' -cf /workspace/sweep-analysis.tar \
    -C /workspace telemetry-data logs
ls -lh /workspace/sweep-analysis.tar
runpodctl send /workspace/sweep-analysis.tar
```

`runpodctl send` prints the code used by `receive`. Locally, give each delivery
its own collection directory, receive the archive there, and extract it:

```bash
mkdir -p ~/Igalia/nanochat/telemetry-data/<collection-name>
cd ~/Igalia/nanochat/telemetry-data/<collection-name>
runpodctl receive <code>
tar -xf sweep-analysis.tar
```

The loader root for that collection is then
`<collection-name>/telemetry-data`; set `NANOCHAT_TELEMETRY_DATA_ROOT` to that
path when it is not the default `sweep` collection. Checkpoint inventories and
metadata are retained, but `checkpoints/*.pt` are intentionally absent, so the
full verifier's lineage-hash check is expected to fail locally. Keep the
archive until the extracted file inventory has been checked; remove it only
after the extracted copy is confirmed complete.

## Next round

- `telemetry-v4-plan.md` — what the instrument should record next, and why.
  Read it before changing the instrument.
- `experiments/` — one file per design. None are frozen. E02 and E03 can now be
  expressed as flat manifest rows; E04 and E05 still need code, so they belong on
  experiment branches.

The instrument serves many experiments, so it is justified by the questions it
makes answerable. An experiment is justified by one question and a power
calculation. Do not mix them.
