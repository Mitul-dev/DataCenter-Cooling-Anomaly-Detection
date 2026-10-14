# Task 2: threshold and persistence tradeoffs

Evaluated **298 policies** using **71,982 identical validation endpoints**, including **902 critical readings** and **47 original incidents**. No heldout server was scored and no model was retrained.

## Decision

**No evaluated policy meets all frozen targets.** Tightening alarms alone is insufficient among the tested configurations. There is no feasible selected policy, and the dashboard default has not been changed to hide this outcome. Proceed to the supervised LSTM and supervised baseline in Task 3.

Targets: precision >=30%, recall >=40%, FPR <=1%, false bursts <=0.1 per scored node-day, and observable-incident recall >=60%. No target was relaxed. An all-normal predictor cannot pass the recall constraints.

## Representative operating points

| Policy | Model / rule | Precision | Recall | FPR | False bursts/day | Incidents caught |
|---|---|---:|---:|---:|---:|---:|
| Current fixed policy (p0170) | cooling12_8 / immediate | 3.72% | 19.96% | 6.55% | 1.022 | 28 / 47 |
| Highest precision retaining >=40% recall (p0286) | isolation_forest / immediate | 3.52% | 40.24% | 14.00% | 2.267 | 30 / 47 |
| Highest recall within false-alarm budgets (p0210) | robust24 / immediate | 10.61% | 7.54% | 0.81% | 0.032 | 10 / 47 |

These policies use different models and/or scores. The table is a validation tradeoff illustration, not a claim that a threshold change improved one trained model or that test performance improved. The current fixed-policy result is on this shared cohort, not the larger native historical cohort reported in Task 1.

## Experiment design

The saved base8, cooling12_8 and robust24 autoencoders each contribute whole-window and latest-reading error scores. Isolation Forest contributes its anomaly score. All use the frozen 24-reading eligibility context, shared calendar validation period and known endpoint targets. Width-8 models still consume eight readings; the longer context sets common eligibility only. Existing training cohorts and early-stopping histories differ, so primary controlled comparisons still require retraining under protocol v1.

Normal-history percentiles 95, 99, 99.5, 99.9 and 99.95 are calibrated globally and per server using only fully known normal validation histories. Global precision-recall candidates maximize empirical precision at recall floors 40%, 60% and 80%, plus F0.5. These candidates use validation labels and are not independent performance estimates. Strict-threshold code handles score ties consistently with precision-recall curve definitions.

Every threshold is evaluated with immediate alarms, two consecutive high scores and at least two highs in three complete readings. All rules are causal and reset at server changes or gaps between eligible scored timestamps. Two-of-three can remain positive when the current score is low; its delay and false-positive costs are included in the reading decisions. Historical fixed policies provide references; Isolation Forest requires three dense consecutive highs to retain its original 30-minute confirmation span.

Precision-recall curves and `global_threshold_precision_limits.csv` describe global immediate thresholds only. Temporal and per-node policies are assessed in `policy_comparison.csv`. No policy-grid result proves a theoretical ceiling on every possible detector. The Pareto table compares precision, recall and false bursts; feasible selection additionally requires the frozen FPR and incident-recall targets.

Per-node percentile specifications also store the pooled normal-validation threshold as the fallback for an unseen server. No reserved server data is used to calibrate that fallback. Historical fixed policies remain references rather than deployable unseen-node candidates.

## Notification cooldown

- p0170: 766 raw false bursts and 28 raw detected incidents; a 60-minute rising-edge cooldown emits 715 notifications, of which 687 are false by their notification timestamps, and notifies inside 28 incidents. Raw precision remains 3.72% and raw recall remains 19.96%.
- p0210: 24 raw false bursts and 10 raw detected incidents; a 60-minute rising-edge cooldown emits 34 notifications, of which 24 are false by their notification timestamps, and notifies inside 10 incidents. Raw precision remains 10.61% and raw recall remains 7.54%.

Cooldown changes message delivery, not the reading predictions. A suppressed notification can hide an incident. It is therefore never used to satisfy the raw false-positive or false-burst targets. Notification truth is evaluated at the notification timestamp; a burst spanning an incident can have a false notification at its earlier start.

## Verification and limits

Model/scaler and cohort fingerprints are recorded in the predeclared experiment plan. Every score is finite and belongs to the same validation endpoints. Representative policies were cross-checked against the original evaluator for confusion counts, bursts, incidents and delay. The reserved nodes and their label values remain untouched. Validation selection can overfit, especially at high percentiles with very few exceedances; the frozen final independent evaluation is still required. All results here are diagnostic and should be labelled as such in the final report.

Files: `policy_comparison.csv`, `pareto_frontier.csv`, `global_threshold_precision_limits.csv`, `notification_demonstration.csv`, policy specifications, decision CSVs and three plotted tradeoff figures.
