# Open questions

Write the question and the test before you look. Add new questions freely.
Status: `open`, `in progress`, `answered (Fxxxx)`, `dropped`.

---

**Q0001 — How much does a d12 result vary between seeds?**
Test: for each metric family, compute the spread across the five d12 runs
(seeds 7-11) at matched normalized progress. Rank families by relative spread.
Confirmatory. Status: open. This must become F0001. See the gate in README.md.

**Q0002 — Which telemetry channels can detect a change?**
Test: from Q0001, list the channels whose seed spread is small. These channels
can show the effect of a future change to the training recipe. Channels with
large spread cannot. Exploratory. Status: open.

**Q0003 — Does the Muon update decoherence change with scale?**
Test: compare `muon/replay_update_relerr` across d12, d14, d16 at matched
progress. Compare the difference to the d12 seed spread. Note: the compiled
optimizer diverges from the reference decomposition by design; this asks
whether the size of that divergence moves with scale. Exploratory. Status: open.

**Q0004 — What does certified curvature do over training?**
Test: use only shadow_fp32 records whose per-direction verdict passed. Plot
gHg, eta*, and dhd against normalized progress for the five d12 runs.
Exploratory. Status: open. Warning: the native bf16 arm is uncertified
everywhere. Do not mix the arms.

**Q0005 — Is the gradient-noise scale stable across seeds and depths?**
Test: compare `noise/b_noise` and `noise/s2` across seeds, then across depths.
Read caveat 8 in `DATASET.md` first: noise is measured on a device batch, not
the logical batch. Exploratory. Status: open.

**Q0006 — Do the absolute warmups distort cross-depth comparison?**
Test: the LR warmup ends at step 40 and the Muon momentum ramp at step 400 in
every run. As a fraction of training this is about 16% at d12 and 7% at d16.
Check whether metrics differ across depths mainly inside that window.
Exploratory. Status: open.
