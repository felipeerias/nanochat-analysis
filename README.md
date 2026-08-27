# Analysis

The parquet data is in `../telemetry-data/sweep/telemetry-data/`. Read `DATASET.md` here
first. The instrument is in `../nanochat/` (branch `telemetry`); change it only
for a strong reason, and change it on a branch that a manifest then pins.

- `loader/` — shared code to read the data correctly.
- `profiles/` — what each run looks like. Descriptive. No claims.
- `investigations/` — one folder per question. Claims live here, and only here.
- `exploratory/` — models and quick checks. Single-author, not citable.
- `experiments/` — designs for data we do not have yet.
- `operations/` — the runner, the pod setup and the sweep manifests.
- `DATASET.md` — the canonical dataset card. Ten caveats. They matter.

This is a git repository and findings cite commits, so commit your work.

## Environment

The analysis repository has its own CPU environment and lockfile. From its
root, reproduce it with:

```bash
uv sync --frozen
uv run python -m unittest discover -s tests -v
```

The default workspace layout places `nanochat-analysis/`, `nanochat/`, and
`telemetry-data/` beside one another. For another layout, point the loader at
the checkout and extracted collection explicitly:

```bash
export NANOCHAT_REPO=/path/to/nanochat
export NANOCHAT_TELEMETRY_DATA_ROOT=/path/to/collection/telemetry-data
```

`NANOCHAT_WORKSPACE_ROOT` can instead move all three default locations at
once. The analysis lockfile is generated with `uv lock`; do not hand-edit it.
The separate `nanochat/uv.lock` remains frozen.

## Profiles

A profile describes one training run, or one collection of runs. It holds
fixed summaries and notes about data quality. Anyone may reuse it.

**Rule: a profile makes no comparison, ranking, or explanation.** Row counts,
verdict counts, and a loss curve are allowed. "d16 looks sharper" is not. That
is a claim, and claims go in an investigation.

Profiles are regenerated when the loader changes. That is fine. An
investigation cites the exact profile commit it used.

## Investigations

1. **Write the protocol first.** Copy `investigations/TEMPLATE-investigation.md`
   to `investigations/NNNN-slug/README.md`. State the kind, the question or
   claim, the data, the planned selection, the test or decision rule, how many
   channels you plan to test, and what an analyst may read. Commit it before
   you look at results.

2. **Run the analysis.** Work only inside your own `AXXXX/` folder. Put your
   code there. Write `result.md` from the template.

3. **Report the real numbers.** State how many channels you actually tested,
   not only the one you report. Record refuted and inconclusive results too.

4. **Do not edit a submitted result.** To correct one, add a new run with
   `supersedes: AXXXX`.

5. **The coordinator writes `conclusion.md`.** It cites every run, including
   the ones that failed.

## Independence

Analysts start from the same protocol commit. They read only what the protocol
allows. They do not read `conclusion.md` or other runs before they submit.
Each result records what it read in its `saw:` field.

Reading a profile counts as seeing its data. A hypothesis formed after reading
the profile is exploratory.

## Evidence levels

- **exploratory** — found by searching the data under study. Weak.
- **reproduced** — a second, blind analyst got the same numbers from the same
  data. This shows the computation is robust. It is not replication.
- **confirmed** — a frozen claim held on **new runs**. Only this level is strong.

## Gate

I0001 is the d12 seed-variation reference. No comparative or effect claim is
accepted until I0001 is closed. Applicable results cite
`investigations/0001-seed-variation/conclusion.md@<commit>`. Other results say
why the reference does not apply. Without it, no claim has an error bar.

I0001 measures d12 only. Its spread may not apply to d14 or d16.

## Results so far

Each folder holds a frozen protocol, two independent blind analyses, and a
conclusion. Read the conclusion first.

| | investigation | outcome |
|---|---|---|
| I0001 | seed variation (the reference) | closed; spreads span five orders of magnitude; loss 0.06%, decoherence 3.5%, curvature 25-29% |
| I0002 | bf16 vs fp32 curvature | closed; bf16 does not corrupt curvature values (0.3%, unbiased) but destroys the error bars |
| I0003 | decoherence vs depth | supported; falls about 11% at d16, confounded with width; caveated by I0006 |
| I0004 | acceptance vs depth | refuted; no degradation, d18/d20 will certify |
| I0005 | certified curvature trajectory | closed; sharpening is real and locked to the learning-rate warmdown |
| I0006 | warmup confound | closed; the alignment axis is the confound; 160 of 248 families unsafe for depth claims |
| I0007 | zero-init wake-up | closed; two tiers, everything active by update 1, depth cannot change it |
| I0008 | adaptive batching feasibility | unavailable as posed; corrected the dataset card |

Corrections these produced, now applied: the seed varies **initialization
only** (one data ordering, shared probes), and cross-depth claims are far more
limited than the seed floor alone suggests.

## Starting another analysis

The former backlog is now I0001, I0003, I0005, I0006, and I0008. Read those
conclusions before opening a new folder so an answered question is not
silently restarted. New exploratory work goes under `exploratory/`; a new
claim starts from `investigations/TEMPLATE-investigation.md` and freezes its
protocol before inspecting outcomes.
