# Multi-Seed Retest Summary (`--seed 0,1,2`)

This folder contains a grouped retest matrix for the most promising experiments after the explicit seed-plumbing fix.

- Experiments: `control`, `target_fold`, `bf16`, `fp16`, `trainint20`
- Seeds: `0`, `1`, `2`
- Raw files: `artifacts/benchmarks/multiseed_retest_seeds012/<experiment>/seed<seed>/{stdout,stderr}.txt`
- Runner script: `run_multiseed_reward_retests.sh`
- Visualization assets: `plot_evolution.py`, `evolution_comparison.png`, `evolution_comparison.html`, `evolution_comparison.raw.csv`, `evolution_comparison.agg.csv`

## Aggregate vs control

| Experiment | avg `rate_mean` | vs control | avg `returns_mean` | delta | avg `returns_last` | delta | snapshot MAE vs control | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| control | 224876.19 | `0.00%` | 22.9109 | `0.0000` | 39.7161 | `0.0000` | 0.0000 | reference |
| target_fold | 226721.95 | `+0.82%` | 22.4703 | `-0.4406` | 40.7057 | `+0.9896` | 0.7727 | closest reward match |
| bf16 | 329178.63 | `+46.38%` | 22.4967 | `-0.4142` | 41.0651 | `+1.3490` | 3.4004 | faster but reward distribution shifted/variable |
| fp16 | 326796.50 | `+45.32%` | 23.1201 | `+0.2092` | 39.3776 | `-0.3385` | 2.0672 | strong speedup, moderate reward-shape drift |
| trainint20 | 224882.19 | `+0.00%` | 21.9835 | `-0.9273` | 38.9167 | `-0.7995` | 1.5370 | no throughput reason to keep |

Notes:

- `snapshot MAE vs control` is the mean absolute difference between aligned `returns` snapshots for the same seed. Lower is more distribution-similar.
- `target_fold` is the closest reward-distribution match to control in this 3-seed pass.
- `fp16` is the strongest large-throughput candidate from this pass, but its reward trajectory still differs more from control than `target_fold`.
- `bf16` kept the large speedup but showed the widest reward variability across seeds.

## Per-seed details

| Experiment | Seed | `rate_mean` | `returns_mean` | `returns_last` | `returns_max` |
|---|---:|---:|---:|---:|---:|
| control | 0 | 224801.41 | 22.4925 | 36.9688 | 40.0469 |
| control | 1 | 224912.49 | 24.0720 | 45.7656 | 46.1484 |
| control | 2 | 224914.65 | 22.1680 | 36.4141 | 40.9297 |
| target_fold | 0 | 226808.25 | 22.4984 | 36.3750 | 40.0469 |
| target_fold | 1 | 226848.35 | 24.0975 | 45.7656 | 45.8906 |
| target_fold | 2 | 226509.26 | 20.8149 | 39.9766 | 43.0859 |
| bf16 | 0 | 329338.36 | 24.1232 | 46.5781 | 50.8047 |
| bf16 | 1 | 329618.10 | 25.1637 | 41.1328 | 43.2031 |
| bf16 | 2 | 328579.44 | 18.2031 | 35.4844 | 35.4844 |
| fp16 | 0 | 327710.90 | 21.5033 | 36.0938 | 39.6875 |
| fp16 | 1 | 326143.39 | 24.2528 | 43.6250 | 47.8828 |
| fp16 | 2 | 326535.22 | 23.6042 | 38.4141 | 42.5000 |
| trainint20 | 0 | 224808.60 | 22.5261 | 36.9688 | 40.0469 |
| trainint20 | 1 | 224887.17 | 22.5887 | 39.8047 | 40.5625 |
| trainint20 | 2 | 224950.79 | 20.8359 | 39.9766 | 43.0859 |
