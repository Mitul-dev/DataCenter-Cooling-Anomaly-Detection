# Validation-only sensor-feature comparison

Professor feedback prompted an audit of additional channels and a controlled reduction experiment. Six new LSTMs were trained; three existing LSTM seed runs are the matched reference. This is development evidence, not a new independent test or a replacement for the submitted model.

## Input definitions

- engineered32: thirteen source measurement/control inputs, six derived consistency features and thirteen missing flags.
- raw13: thirteen raw inputs only; no derived features or missing flags.
- raw5: five training-ranked raw variables, with no derived features or extra flags.

Selected raw5: outdoor_temp, cooling_valve, supply_temp, mixed_temp, outdoor_damper.

The ranking uses whole training days held out internally within winter/spring. It never selects from summer validation or autumn test outcomes. Sensor-group permutation is conditional on the screening tree and correlations, and does not establish physical causality or an optimal LSTM subset.

## Matched experiment

All sets use the same 61,187 training and 33,432 summer validation endpoints, 30-minute history, hidden size 32, 16-unit head, dropout 0.2, Adam 0.001, batch 512, balanced BCE training weight, gradient clip one, maximum 30 epochs and AP patience six. Seeds are 7, 42 and 123. Preprocessing fits normal active training observations only. Each model receives the same thirty-policy validation search and target gate.

| Inputs | Parameters | AP mean +/- sample SD | Mean precision | Mean recall | Mean FPR | Feasible seeds |
|---|---:|---:|---:|---:|---:|---:|
| engineered32 | 8,993 | 0.9820 +/- 0.0027 | 100.00% | 51.73% | 0.00% | 3/3 |
| raw13 | 6,561 | 0.8933 +/- 0.0236 | 89.42% | 43.06% | 31.02% | 0/3 |
| raw5 | 5,537 | 0.8947 +/- 0.0183 | 90.22% | 40.18% | 26.31% | 0/3 |

None of the six reduced-input runs meets the original validation gate. Their reported policies are diagnostic fallbacks satisfying the recall constraint where available, not qualified releases. The current engineered model is retained. Removing derived relationships while reducing raw inputs did not improve this budgeted comparison.

| Model | AP | Precision | Recall | FPR | Feasible |
|---|---:|---:|---:|---:|---|
| engineered32_seed_7 | 0.9836 | 100.00% | 64.42% | 0.00% | True |
| engineered32_seed_42 | 0.9834 | 100.00% | 48.66% | 0.00% | True |
| engineered32_seed_123 | 0.9789 | 100.00% | 42.10% | 0.00% | True |
| raw13_seed_7 | 0.9182 | 92.86% | 40.04% | 18.47% | False |
| raw13_seed_42 | 0.8713 | 86.42% | 41.85% | 39.47% | False |
| raw13_seed_123 | 0.8904 | 88.99% | 47.29% | 35.11% | False |
| raw5_seed_7 | 0.8745 | 87.21% | 40.49% | 35.62% | False |
| raw5_seed_42 | 0.8995 | 92.18% | 40.04% | 20.37% | False |
| raw5_seed_123 | 0.9102 | 91.28% | 40.01% | 22.95% | False |

## Interpretation and next step

Five raw sensors and five engineered model features are different hypotheses. This study tests the former. A future wider-data experiment should also compare five selected engineered features and disclose the raw measurements each requires. These results do not establish that every five-feature approach must fail.

MZVAV-1 already provides all current measurement/control channels; unused status flags are constant during eligible operation. Additional genuine simulated airflow, fan-power or position channels require a richer aligned source. SENSOR_EXPANSION_PLAN.md identifies an official 30-variable annual simulation source and an overlap-aware evaluation design. No additional readings or sensor types have been fabricated, downloaded or added here.

All 270 policies and six complete fresh validation score arrays reproduce; six fresh subset CSV windows reproduce; 42 unit tests pass. No new autumn predictions were made. The source, original trained artifacts, submission report and dashboard remain unchanged.

Reproduce with `python scripts/audit_hvac_sensors.py`, `python scripts/train_hvac_feature_study.py`, `python scripts/verify_hvac_feature_study.py` and `python scripts/report_hvac_feature_study.py`. Completed models are reused only after integrity checks.