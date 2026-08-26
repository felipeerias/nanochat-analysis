# nanochat-analysis

Workspace map: `WORKSPACE.md`. Working procedure: `README.md`.
Dataset card: `DATASET.md` — read its caveats before using the data.

- Claims go in `investigations/`, and only there. They need a frozen protocol
  and two independent blind analyses.
- `exploratory/` is single-author and unreplicated. Referenceable as a
  hypothesis or a method, never citable as evidence.
- `experiments/` are designs for data we do not have. None are frozen.
- `operations/` launches runs against the nanochat commit its manifest pins.

The instrument is in `../nanochat/`. Do not change it from here.
Commit messages use plain language and state that AI wrote the change.
