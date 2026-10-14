# Cooling equipment sensor-fault detection with an LSTM

Primary study: author-labelled outdoor-air temperature sensor-bias detection in an operating simulated air-handling unit. M100 remains supporting research; its performance is not the source of the following benchmark accuracy.

Independent autumn test: accuracy 92.82%, precision 99.74%, recall 91.86%, F1 95.64%, AP 0.9962, FPR 1.44%. The FPR misses the original 1% goal. No test-based threshold or seed tuning occurred.

| Model | Accuracy | Precision | Recall | FPR | AP |
|---|---:|---:|---:|---:|---:|
| lstm_seed_7 | 96.66% | 99.73% | 96.36% | 1.55% | 0.9988 |
| lstm_seed_42 | 92.82% | 99.74% | 91.86% | 1.44% | 0.9962 |
| lstm_seed_123 | 85.55% | 99.71% | 83.39% | 1.47% | 0.9973 |
| boosting | 76.04% | 100.00% | 72.05% | 0.00% | 0.9914 |

## Data, reference outcome and separation

Source: LBNL AFDD collection, DOI 10.25984/1824861, member MZVAV-1.csv. The 189 days contain 272,160 one-minute records with original 0/1 condition ground truth. Inventory Table 1-2 documents normal scenarios and outdoor-air sensor biases of +/-1, +/-2 and +/-4 Celsius. Input temperatures are Fahrenheit. This is EnergyPlus-Dymola simulation of one AHU, not datacenter failure measurements.

| Partition | Seasons | Raw / scored days | Endpoints | Fault endpoints |
|---|---|---:|---:|---:|
| train | winter + spring | 91 / 77 | 61,187 | 52,446 |
| validation | summer | 49 / 42 | 33,432 | 28,656 |
| test | autumn | 49 / 42 | 33,432 | 28,656 |

Each input contains 30 consecutive occupied, fan-on minutes on one day. Whole seasons/scenario blocks remain in one partition. Nights/nonoperating days are outside the model task; all raw fault days stay in the coverage denominator. Native labels match the inventory for every source date. Dates, seasons, day IDs, bias intensity and labels are not model inputs.

## Inputs, cleaning and learning

Thirteen temperature, fan/damper/valve command and duct-pressure inputs yield six consistency differences and 13 missing indicators: 32 channels. The command-weighted mixing residual is a heuristic, not measured mixing fraction. Negative outdoor Fahrenheit temperatures remain valid. Forward filling uses at most five prior minutes and resets at days/gaps. Remaining missing inputs use normal active training medians. Scaling uses normal active training observations only, clipped at ten standard deviations.

The LSTM has hidden size 32 and a 16-unit ReLU classification head with dropout 0.2. Adam uses learning rate 0.001 and batch 512, maximum 30 epochs, patience six on validation AP and gradient clipping one. BCE positive weight equals the training normal/fault ratio, about 1/6. It balances the training loss without changing evaluation prevalence. The tree receives latest/mean/std/slope over the same history, with fixed 15-leaf boosting and 50/100/200 validation checkpoints.

Seeds 7, 42 and 123 all pass the validation policy gate; validation AP mean 0.9820, sample SD 0.0027. Seed 42 remains the predeclared primary although seed 7 performs better on test. All model/checkpoint/policy artifacts were hashed before the one-time autumn evaluation.

## Alarm policy and measured tradeoff

Normal-validation percentiles and PR thresholds compete under the original precision/recall/FPR/false-burst/fault-day gate. The frozen primary requires two high scores in three consecutive scored minutes at logit threshold 6.967112; state resets at days/gaps. Scores are uncalibrated. A single uploaded window cannot establish persistence.

Confusion counts: TP 26,324, FP 69, FN 2,332, TN 4,707. The 69 false readings form one burst: 0.0431 bursts per scored device-day. False discovery ratio is 0.26%, using a different denominator than FPR. Detected days: 35/36 observable and 35/42 total. Median delay from midnight imposition is 392 minutes, including overnight downtime/warmup; median delay from first eligible score is 2.0 minutes.

The tree has zero test false positives but lower reading recall (72.05%). The LSTM has greater sensitivity with FPR 1.44%, above the 1% goal. This tradeoff remains unchanged after evaluation.

## Limits and reproducibility

Experimental fault prevalence is 85.71%. An all-fault classifier would score 85.71% accuracy and 100% FPR, so accuracy alone is misleading. Healthy validation/test each contain one scenario week; adjacent minutes/days are correlated. These results do not estimate deployment prevalence or independent-unit reliability. Only one simulated unit and fault family are tested. No early warning or general fault diagnosis is established.

The original M100 study tested feature removal, classifiers, persistence and additional inputs but missed the false-alarm targets. Its published preparation selects batch-client scheduler state rather than a cooling-specific fault. The preserved negative results motivate this documented, user-approved dataset/scope change.

Source archive/inventory: data/hvac_benchmark. Protocols, histories, model/preprocessing, scores, decisions, case/severity tables and verification: results/hvac_v1. Reproduce with prepare_hvac.py, train_hvac.py, verify_hvac.py --validation-only, evaluate_hvac.py, verify_hvac.py, build_hvac_evidence.py. The evaluation script returns existing final results on rerun; a new experiment requires a new version and evaluation design.

[Official dataset](https://data.openei.org/submissions/910). Source inventory pages 3-7 define the system, units and scenarios.
