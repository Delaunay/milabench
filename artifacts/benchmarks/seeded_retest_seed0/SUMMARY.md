# Seeded Retest Summary (`--seed 0`)

This folder contains a fresh direct-GPU retest pass after the explicit seed-plumbing change in `benchmarks/purejaxrl/dqn.py`.

- Raw files live in one subfolder per experiment: `control/`, `target_fold/`, `bf16/`, `fp16/`, `numenvs192/`, `numenvs256/`, `trainint20/`
- Runner script: `run_seeded_reward_retests.sh`
- Comparison basis: each candidate is compared to the `control/` run in this same folder

| Experiment | `rate_mean` | vs control | `returns_last` | vs control | `returns_mean` | vs control | `returns_max` | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| control | 225191.48 | `0.00%` | 33.4766 | `0.0000` | 20.0046 | `0.0000` | 36.3516 | reference |
| target_fold | 226775.15 | `+0.70%` | 33.4766 | `0.0000` | 19.9887 | `-0.0160` | 36.3516 | reward-neutral at seed 0 |
| bf16 | 328764.14 | `+45.99%` | 46.5781 | `+13.1016` | 24.1274 | `+4.1228` | 50.8047 | strong seed-0 win |
| fp16 | 326620.32 | `+45.04%` | 39.6016 | `+6.1250` | 24.5344 | `+4.5298` | 46.3828 | strong seed-0 win |
| numenvs192 | 322527.36 | `+43.22%` | 25.5156 | `-7.9609` | 15.4694 | `-4.5352` | 26.0104 | reward regression |
| numenvs256 | 429367.84 | `+90.67%` | 24.0117 | `-9.4648` | 14.8974 | `-5.1072` | 26.8047 | reward regression |
| trainint20 | 224915.84 | `-0.12%` | 36.9688 | `+3.4922` | 22.5041 | `+2.4995` | 40.0469 | reward better, speed flat |

Interpretation:

- These are single-run, single-seed results after explicit seed stabilization.
- They supersede the older "mixed seed plumbing" reward comparisons for this exact seed, but they are not yet a full multi-seed promotion gate.
- The biggest reversals versus the earlier reward conclusions are `target_fold`, `bf16`, and `fp16`, which all look materially better under this seeded retest than they did before.
