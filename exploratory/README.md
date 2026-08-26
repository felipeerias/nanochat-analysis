# Exploratory

Single-analysis work. **Not conclusions.** Nothing here has a frozen protocol
or an independent blind check.

A note here may be referenced as a hypothesis, a model, a method or as
provenance. It may **not** be cited as empirical evidence or as a project
finding. To promote a result, freeze a protocol in `../investigations/` and run
the independent analysis. Promotion moves the *finding*, not the document: the
note stays where it is and gains a pointer to the investigation.

Two prefixes, one tier. They record what a note mostly does, not how much it is
believed:

- **T** — a model: assumptions, derivation, predictions. May also fit data.
- **X** — an empirical check or diagnostic on data we already have.

These were one folder each until the distinction stopped holding. What
separates work here from an investigation is status, not genre, and both kinds
are single-author and unreplicated. T01 and X01 fit the same 70 points.

## What a T note must state

1. The phenomenon, and the measurement that motivated it (cite the
   investigation if there is one).
2. The model: assumptions, notation, derivation. Show the working.
3. **A quantitative prediction**, ideally a scaling law or a ratio, in terms of
   quantities this project actually records.
4. What measurement would falsify it, and what precision that needs — check it
   against the noise floors in
   `../investigations/0001-seed-variation/conclusion.md`.
5. Where the model is known to break: which assumptions are false for this
   architecture, optimizer or precision.
6. What it does **not** explain.
7. Which parameters are **fitted** rather than derived. A fitted bend is not a
   threshold, and calling it one is the easiest way to overclaim.

Be explicit about the difference between a derivation and an analogy. Most
optimization theory assumes plain gradient descent on a fixed quadratic;
nanochat uses Muon with momentum, nonlinear polar orthogonalization, factored
second moments and cautious decay. A result imported from the SGD literature is
a hypothesis about this system, not a fact about it.

## Status

| id | note | status |
|---|---|---|
| T01 | warmdown curvature as a saturated schedule response | draft, 3 rounds |
| T02 | finite-step Polar Express as a rounding amplifier | draft, 2 rounds |
| X01 | is the warmdown curvature rise a power law? | refutes the pure power law |

Both T notes state which parameters are fitted rather than derived, and rank
their predictions by attribution and power rather than by the order the source
experiment happened to list its arms.

## Standing facts a model must respect

- Curvature is measured on a **single 256-token sequence**, not the 2048-token
  training loss. Those are different functions.
- Certified curvature exists along the **gradient direction only**.
- The compiled bf16 Muon update diverges from its own reference decomposition
  by roughly 3–10% per matrix; the optimizer does not apply the update its math
  specifies.
- `eta * lambda_max` is not a valid stability statistic for Muon.
- The seed floor is initialization-only: one data ordering exists in all data
  collected so far.

## Known open conflict

T01 ranks E02's arm D as the cleanest falsifier of the learning-rate-specific
model, because arms A, B and C move the Muon momentum schedule and the learning
rate together. `../experiments/E02-warmdown-sharpening.md` still calls A, B and
C its primary contrast, and does not cite T01 at all. They answer different
decision nodes, and the wording in both should be reconciled.
