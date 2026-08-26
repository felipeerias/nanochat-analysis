# Analysis

The parquet data is in `../telemetry-data/sweep/`. Read `DATASET.md` here
first. The instrument is in `../nanochat/` (branch `telemetry`); change it only
for a strong reason, and change it on a branch that a manifest then pins.

- `loader/` — shared code to read the data correctly.
- `profiles/` — what each run looks like. Descriptive. No claims.
- `investigations/` — one folder per question. Claims live here, and only here.
- `exploratory/` — models and quick checks. Single-author, not citable.
- `experiments/` — designs for data we do not have yet.
- `operations/` — the runner, the pod setup and the sweep manifests.
- `DATASET.md` — the canonical dataset card. Nine caveats. They matter.

This is a git repository and findings cite commits, so commit your work.

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

## Backlog

Ideas with no folder yet. Make a folder when you start one.

- Which channels have a small seed spread? Those can detect a change.
- Does Muon update decoherence change with depth?
- What does certified curvature do over training? Use `shadow_fp32` records
  with a passing per-direction verdict. Never mix the two arms.
- Is the gradient-noise scale stable across seeds and depths? See caveat 8.
- Do the absolute warmups (40 and 400 steps) distort cross-depth comparison?
