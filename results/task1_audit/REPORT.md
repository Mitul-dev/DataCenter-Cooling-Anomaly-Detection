# Task 1: data, failure and evaluation audit

This audit evaluates the existing selected model on validation recordings only. No new model was trained, no threshold was changed, and no prediction or label-value inspection was performed on the seven reserved final nodes. Historical test results remain exploratory.

## Problem statement

Predict whether the current 15-minute endpoint has a Nagios CRITICAL server-health label from preceding and current thermal, fan and power telemetry. This is current-state detection. Labels do not establish cooling-specific failures, future-fault prediction or a physical diagnosis. [M100 data descriptor](https://www.nature.com/articles/s41597-023-02174-3).

## Findings from the validation audit

At the unchanged policy, the 30-minute validation evaluation has precision 4.59%, recall 22.94%, FPR 5.05%, and 533 false bursts. Scoring every 15 minutes has precision 4.47%, recall 22.24%, FPR 5.04%, and 700 false bursts. Different sampling grids change the cohort and alarm opportunities; this is not a model improvement. Saved 30-minute validation scores and confusion counts were independently reproduced.

Of 3,699 false-positive validation endpoints at 15-minute cadence, 41 still contain a preceding critical reading in their input history, 0 have missing original measurements somewhere in the 22-sensor history, and 0 have filled measurements. These groups can overlap. They identify associations to investigate, not proven causes of the model errors. The quality counts include all 22 original sensors; the selected model uses 12.

The servers contributing the most false-positive endpoints are: 13 (433), 11 (424), 17 (421). Full per-node metrics remain available; pooled metrics must not conceal server differences.

None of these validation false positives had missing/imputed input measurements. Most also lacked a critical reading in their recent history. This audit therefore does not support missing-data repair or residual critical history as the main explanation of false alarms. The percentile threshold admits roughly 5% of normal validation endpoints, and normal-pattern changes deserve investigation. These are observations and hypotheses, not a proven physical fault diagnosis.

The largest training-to-validation normal-reading median shifts, measured in within-server training standard deviations, are: gpu_mem_temp on 11 (-1.10), gpu_core_temp on 13 (-1.03), p1_vdd_temp on 17 (-1.02), dimm_temp on 13 (-0.99), gpu_mem_temp on 13 (-0.97). These are descriptive shifts, not validated fault indicators. Raw sensor quality failures can affect these estimates.

## Cases for review

- highest_score_false_alarm: server 11, 2022-03-28T11:00:00+00:00; endpoint outcome FP; highest latest-reading reconstruction-error contributor gpu_core_temp.
- normal_endpoint_with_critical_history: server 13, 2022-03-30T15:45:00+00:00; endpoint outcome FP; highest latest-reading reconstruction-error contributor p1_vdd_temp.
- detected_incident: server 7, 2022-01-25T08:30:00+00:00; endpoint outcome TP; highest latest-reading reconstruction-error contributor total_power.
- missed_observable_incident: server 7, 2022-03-16T14:15:00+00:00; endpoint outcome FN; highest latest-reading reconstruction-error contributor fan0_0.

The corresponding PNGs show the score, fixed threshold, critical endpoints and original sensor readings. The top sensor explains latest-reading squared error only; the whole-window decision also includes earlier readings and does not prove the fault cause.

## Sensor-quality findings

- train, label 0: 10 implausible negative sensor cells in 5 rows. Original values remain preserved; invalid flags are retained for future ablation.
- train, label 2: 16 implausible negative sensor cells in 8 rows. Original values remain preserved; invalid flags are retained for future ablation.
- exploratory, label 0: 4 implausible negative sensor cells in 2 rows. Original values remain preserved; invalid flags are retained for future ablation.
- exploratory, label 2: 100 implausible negative sensor cells in 50 rows. Original values remain preserved; invalid flags are retained for future ablation.

Unknown labels remain unknown. Genuine fan zeros, high readings and rare events are preserved. Quality-indicator candidates are saved in `artifacts/quality_indicators.parquet`; their usefulness is not yet established. No future label or Nagios state is added as a feature.

## Frozen evaluation protocol

The autoencoder reference, robust-scaled autoencoder, Isolation Forest and selected model were also scored at 15-minute cadence with their fixed historical thresholds. `historical_reference_15min.csv` records them. Persistence was converted to preserve its minimum elapsed confirmation span (two high samples 30 minutes apart become three high samples 15 minutes apart). This additionally checks intermediate readings. These native cohorts differ by input width; the table is a diagnostic, not a fair ranking. No threshold or model was tuned.

Historical row splits differed by server: server 16 training continued into dates assigned to validation on other servers. New models must use calendar-aligned boundaries: training before 22 January 2022 UTC; validation from 22 January to before 16 April 2022 UTC; exploratory recordings from 16 April onward. These new boundaries do not make previously inspected recordings blind. Existing models were trained under the older row splits, so they are diagnostic references and must be retrained for a fair primary comparison.

The primary matched cohort requires a consecutive, finite 24-reading context across all 22 base sensors, with no node/split/gap crossings and a known endpoint target. Width-8 and width-24 candidates use exactly these endpoints. Unknown historical labels do not disqualify otherwise usable measurements or become input features; normal-only training/calibration requires all historical labels normal. Training stride is four and validation/evaluation stride is one. Limited causal filling resets at every node, timestamp gap and calendar boundary.

| Split | Eligible windows | Critical endpoints | Observable / raw incidents |
|---|---:|---:|---:|
| train | 111,070 | 1,445 | 415 / 1023 |
| validation | 71,982 | 902 | 47 / 47 |
| exploratory | 139,777 | 895 | 34 / 34 |

Training coverage uses its stride-four sampled cohort. All-incident denominators preserve unobservable incidents. Exposure equals scored bins multiplied by the explicit 15-minute cadence, not wall-clock server uptime. Notification cooldown must remain separate from raw reading decisions.

Final independent nodes are 2, 3, 4, 5, 6, 18 and 19 over [16 April, 29 September 2022) UTC. Only schema and timestamps were inventoried now; their sensor values, labels and predictions are reserved for final evaluation. Unknown/finite coverage will be reported after applying the frozen eligibility rule. This separately evaluates unseen-node and later-period generalization, not just performance on familiar servers.

Select policies on validation only. Targets are precision >=30%, recall >=40%, FPR <=1%, false bursts <=0.1 per scored node-day, and observable-incident recall >=60%. Among feasible policies maximize precision, then F0.5, then minimize false bursts, then maximize recall. If none is feasible, report no feasible policy. These goals are not claimed achievements or professor acceptance rules.

## Next task

Task 2 is threshold/persistence tradeoff analysis on validation data, followed by label-aligned supervised baselines if the existing scores cannot meet the targets. The manifest, cohort CSVs and SHA-256 checksums in `results/protocol_v1` fix the scope before new experiments.
