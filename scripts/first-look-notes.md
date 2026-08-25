# First end-to-end analysis - d12-iter (freeze gate)

- rows: {'continuous': 194066, 'periodic': 85686, 'sparse': 1239, 'offline': 12}, undefined: {'continuous': 0, 'periodic': 738, 'sparse': 29, 'offline': 0}
- native verdicts: {'passed': 0, 'inconclusive': 0, 'failed': 3} - every deep checkpoint is UNCERTIFIED at bf16; curvature figures are labeled accordingly. Certified numbers await the shadow arm.
- train loss 10.3976 -> 2.7726; final val-probe loss 2.9020
- Muon decoherence (median/max per checkpoint): step 1: 0.0000/0.1459, step 1001: 0.0513/0.2744, step 2001: 0.0314/0.0712
- telemetry overhead 103.4s; top sections: noise=25.6s, update_effectiveness=18.0s, probes=15.4s, periodic_scan=11.9s, calibration=11.4s

## Data-shape frictions

- none: every quantity above was computable from parquet + provenance alone with the documented joins.
