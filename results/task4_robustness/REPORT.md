# Task 4 — robustness and release decision

Repeated the provisional `lstm8_p1` configuration from scratch with seeds 7, 42 and 123. Architecture, features, loss weighting, data, optimizer and stopping rules remain fixed. Seed 42 reproduces the Task 3 checkpoint tensors and every validation score exactly.

| Seed | AP | Precision | Recall | FPR | TP / FP / FN / TN | False bursts/day | Incidents | Feasible? |
|---|---:|---:|---:|---:|---|---:|---:|---|
| 7 | 0.0279 | 3.22% | 40.02% | 15.26% | 361 / 10847 / 541 / 60233 | 1.070 | 26/47 | False |
| 42 | 0.0235 | 2.98% | 41.91% | 17.29% | 378 / 12289 / 524 / 58791 | 0.489 | 26/47 | False |
| 123 | 0.0194 | 1.82% | 40.24% | 27.59% | 363 / 19609 / 539 / 51471 | 1.267 | 28/47 | False |

**Release decision: `no_release_candidate`.** 0/3 seeds have any feasible policy, across 135 policies. The targets are unchanged: precision ≥30%, recall ≥40%, FPR ≤1%, false bursts/day ≤0.1 and observable incident recall ≥60%.

Every seed uses the same policy-selection rule. If no feasible policy exists, the table shows the best diagnostic precision while retaining at least 40% recall; these diagnostic thresholds are not deployment recommendations. Per-node policy fallbacks are fitted on pooled normal validation histories only.

## Seed variation

| Measure | Mean | Sample standard deviation |
|---|---:|---:|
| average_precision | 0.02362 | 0.00425 |
| precision | 0.02674 | 0.00751 |
| recall | 0.40724 | 0.01030 |
| false_positive_rate | 0.20045 | 0.06610 |
| false_bursts_per_node_day | 0.94201 | 0.40416 |
| event_recall_observable | 0.56738 | 0.02457 |

Three seeds describe variability, not a precise confidence interval for future performance. The configuration was selected using seed 42 in Task 3, so its validation performance was already inspected and may be optimistic. Each seed selects its own validation checkpoint and policy. All three results are reported; no test-score seed selection occurred.

## Uncertainty and coverage

The 95% percentile intervals in `node_bootstrap_intervals.json` resample whole development servers, 2,000 times, preserving correlations within each server. They condition on the already-selected validation policy and do not capture selection uncertainty, shared rack-wide events or future drift. There are only nine development nodes. All comparisons retain the same 71,982 endpoints and 902 critical targets, with 47/47 observable raw incidents. Per-node results, incident records and delays are saved.

## Freeze and independent evaluation

The release rule was declared before these repeats: all three seeds need feasible policies; seed 42 is the primary release candidate, and its global fallback must also pass the target gate on validation. A failing gate yields no release candidate. `research_snapshot.json` freezes the research model/preprocessing/policy hashes without certifying operational performance.

No reserved server measurements, labels or predictions were accessed. The old exploratory period was not scored. Independent evaluation is deferred when there is no qualified release, preserving the final recordings for a revised, validation-qualified pipeline. The dashboard default is unchanged.

## What to investigate next

The repeated model and the supervised tree comparison have not established a usable detector. Investigate training-only source feature availability and label alignment, rather than promising that more seeds or higher class weights will fix the false alarms. See `TRAINING_FEATURE_AUDIT.md` for measured availability of additional thermal/electrical inputs. Their predictive value is unproven. Do not create artificial cooling labels, discard difficult servers or rebalance evaluation recordings.

## Reproduce

Run `.venv\Scripts\python.exe scripts\run_task4.py`, then `scripts\audit_task4_features.py` and `scripts\verify_task4.py`. The frozen plan records all seeds, input fingerprints, policy grid, release rule and bootstrap method. All three seed folders include weights, preprocessing, examples, histories, scores and raw decisions.
