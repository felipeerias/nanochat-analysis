# Analysis

The data is in `../telemetry-data/sweep/`. Read its `DATASET.md` first.
The measuring code is in `../nanochat/` (branch `telemetry`). Do not change it.

- `loader/` — shared code to read the data correctly, plus reference values.
- `investigations/` — one folder per question or hypothesis. Work happens here.

## Procedure

1. **Write the protocol first.** Create `investigations/NNNN-slug/README.md`
   from the template. State the kind, the question or claim, the data, the
   planned selection, the test or decision rule, how many channels you plan to
   test, and what an analyst may read. Do this before you look at results.

2. **Run the analysis.** Work only inside `runs/AXXXX/`. Put your code there.
   Write `result.md` from the template.

3. **Report the real numbers.** State how many channels you actually tested,
   not only the one you report. Record refuted and inconclusive results too.

4. **Do not edit a submitted run.** To correct one, add a new run that says
   `supersedes: AXXXX`.

5. **The coordinator writes `conclusion.md`.** It cites every run, including
   the ones that failed.

## Independence

Analysts start from the same protocol. They read only what the protocol allows.
They do not read `conclusion.md` or other runs before they submit. Each result
records what it did read in its `saw:` field.

## Evidence levels

- **exploratory** — found by searching the data. Weak.
- **reproduced** — a second, blind analyst got the same numbers from the same
  data. This shows the computation is robust. It is not replication.
- **confirmed** — held on **new runs** that were not used to form the claim.
  Only this level is strong.

## Gate

I0001 is the d12 seed-variation reference. No comparative or effect claim is
accepted until I0001 is finished and cited. Without it, no claim has an error
bar. I0001 measures d12 only. Its spread may not apply to d14 or d16.

## Backlog

Ideas that have no folder yet. Make a folder when you start one.

- Which channels have a small seed spread? Those channels can detect a change.
- Does Muon update decoherence change with depth?
- What does certified curvature do over training? Use `shadow_fp32` records
  with a passing per-direction verdict. Never mix the two arms.
- Is the gradient-noise scale stable across seeds and depths? See caveat 8.
- Do the absolute warmups (40 and 400 steps) distort cross-depth comparison?
