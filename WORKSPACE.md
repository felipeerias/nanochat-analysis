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
| `nanochat-analysis/DATASET.md` | **The dataset card. Start here.** Nine caveats. |
| `nanochat-analysis/loader/` | Code to read the data. |
| `nanochat-analysis/profiles/` | What each run looks like. Descriptive only. |
| `nanochat-analysis/investigations/` | One folder per question. Claims live here. |
| `nanochat-analysis/exploratory/` | Models and quick checks. Not citable. |
| `nanochat-analysis/experiments/` | Designs for data we do not have yet. |
| `nanochat-analysis/operations/` | Runner, pod setup, sweep manifests. |
| `nanochat-analysis/telemetry-v4-plan.md` | What the instrument should record next. |
| `nanochat-analysis/telemetry-plan-v2.md` | The research plan. |
| `telemetry-data/sweep/` | The parquet data. Seven schema-v3 runs. |

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
- A manifest row may carry a `recipe` of `base_train` flags. Which flags exist
  is read from the pinned checkout, not listed anywhere, so it cannot drift.
  Changing the trainer itself still goes on an experiment branch, identified
  by its commit.
- Never edit `uv.lock`. Always use `uv sync --frozen`.
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

## Next round

- `telemetry-v4-plan.md` — what the instrument should record next, and why.
  Read it before changing the instrument.
- `experiments/` — one file per design. None are frozen. E02 and E03 can now be
  expressed as manifest recipes; E04 and E05 still need code, so they belong on
  experiment branches.

The instrument serves many experiments, so it is justified by the questions it
makes answerable. An experiment is justified by one question and a power
calculation. Do not mix them.
