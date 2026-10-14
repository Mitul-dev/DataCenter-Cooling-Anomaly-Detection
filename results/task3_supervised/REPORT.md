# Task 3 — supervised training and fair validation comparison

Trained 9 models, including six predefined LSTM classifiers, on calendar training data. All candidates use the same 71,982 eligible validation endpoints (902 critical; prevalence 1.25%). Reserved servers and the exploratory period were not scored.

## Results

| Model | AP | Precision | Recall | FPR | False bursts/day | Incidents | Feasible? |
|---|---:|---:|---:|---:|---:|---:|---|
| lstm8_p1 | 0.0235 | 2.98% | 41.91% | 17.29% | 0.489 | 26/47 | False |
| lstm8_p3 | 0.0213 | 2.64% | 40.02% | 18.75% | 1.120 | 30/47 | False |
| lstm8_p10 | 0.0151 | 1.45% | 51.22% | 44.28% | 1.222 | 31/47 | False |
| lstm24_p3 | 0.0164 | 1.31% | 98.78% | 94.76% | 0.221 | 46/47 | False |
| lstm8_corr_p3 | 0.0174 | 1.68% | 61.64% | 45.85% | 1.718 | 39/47 | False |
| lstm8_quality_p3 | 0.0146 | 1.43% | 40.02% | 35.13% | 0.951 | 26/47 | False |
| gradient_boosting8 | 0.0152 | 1.25% | 100.00% | 99.92% | 0.115 | 47/47 | False |
| autoencoder8_refit | 0.0217 | 1.92% | 40.02% | 25.91% | 2.439 | 37/47 | False |
| isolation_forest8_refit | 0.0215 | 2.48% | 40.02% | 19.97% | 3.508 | 36/47 | False |

**0 neural policies and 0 total policies meet all five frozen targets.** 441 policies were evaluated. The provisional DL candidate is `lstm8_p1`.

Each row shows the best feasible policy if one exists; otherwise it shows a diagnostic policy maximizing precision with recall at least 40%. A diagnostic policy is not a recommended release policy. AP is the ranking measure for the score mode used in that row. All policies, exact thresholds and raw decisions are saved alongside this report.

## Training and safeguards

- Training has 111,070 known endpoint targets, including 1,445 critical readings. Unknown historical labels are permitted as context but never supplied as features. Validation retains its natural class prevalence.
- Scaling uses finite normal training rows only. Features are clipped at 10 training standard deviations. Correlation selection uses normal training statistics only; the quality ablation appends fractions of missing, invalid and causally filled sensors.
- Neural training uses Adam, batch 512, up to 25 epochs, patience five, gradient clipping one and seed 42. The supervised checkpoint maximizes validation AP. Loss weights 1, 3 and 10 affect training only.
- The tree classifier sees latest values, means, standard deviations and slopes from the same eight readings. Its 50/100/200 iteration checkpoint is chosen on the explicit chronological validation cohort; random internal validation is disabled.
- The autoencoder and Isolation Forest are freshly trained on 104,230 fully known normal histories from the frozen training cohort. Autoencoder stopping uses the 68,978 normal validation histories. They are comparable references under the shared split; they replace neither the historical artifacts nor their reported results.
- Policies use the frozen target gate: precision ≥30%, recall ≥40%, FPR ≤1%, false bursts ≤0.1/scored node-day, observable incident recall ≥60%. Persistence resets at gaps and nodes. Per-node thresholds retain a validation-only global fallback for unseen servers.
- Scores are uncalibrated. Nagios labels concern general current server health, not confirmed cooling faults or future fault forecasting.

## Limits and next step

These are development results from one seed, with configuration/checkpoint/policy selection on the same validation period. They are optimistic for deployment. Task 4 must repeat the finalist with seeds 7, 42 and 123, freeze the full pipeline using validation only, and then evaluate the independent servers once. The dashboard default and existing report/PPT remain unchanged until a verified release is available.

## Reproduce

Run `.venv\Scripts\python.exe scripts\train_task3.py` to reproduce or resume the frozen experiment. Run `scripts\verify_task3.py` to reproduce policies and standalone inference. `experiment_plan.json` records input/source hashes, fixed configurations and stopping rules. Individual folders contain weights, preprocessing, histories, validation scores and example CSV windows.
