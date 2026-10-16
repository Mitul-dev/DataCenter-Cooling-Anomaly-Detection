# Fault detection across four sensors

This extension identifies four individual temperature/pressure sensor biases in a simulated dual-duct AHU. The previous outdoor-air detector remains a separate benchmark. The four new targets do not include outdoor-air temperature.

## Reserved test result

Primary model: all_inputs, seed 42, selected on validation. Five-class accuracy 92.87%; macro F1 89.36%; any-fault detection recall 93.05%.
Healthy FPR is 2.10%: 580 false readings / 27,640 healthy readings. The 1% goal is missed. There are 52 false bursts across 40 active healthy calendar days. Thresholds were not retuned after this test.

| Faulty sensor | Precision | Correct-sensor recall | F1 |
|---|---:|---:|---:|
| Cold-deck temperature sensor | 99.43% | 91.03% | 95.05% |
| Hot-deck temperature sensor | 99.13% | 87.52% | 92.96% |
| Cold-deck pressure sensor | 99.21% | 95.68% | 97.41% |
| Hot-deck pressure sensor | 99.50% | 95.88% | 97.66% |

## Source and data cleaning

Official source: https://faultdetection.lbl.gov/dataset/simulated-dd-ahu-dataset/. Dataset collection DOI 10.25984/1881324. Methodology and fault definitions: https://doi.org/10.1038/s41597-023-02197-w, Table 4.
The downloaded archive contains 56 CSVs. We selected one healthy scenario and sixteen native sensor-bias scenarios: four severity filenames for each of four sensor locations. No labels were synthesized. Each selected CSV contains 525,600 one-minute rows in 2018, representing parallel versions of one simulated unit, rather than seventeen independent units.
The source contains 114 measurement/control points. Our declared 26 AHU inputs comprise 21 measurements and five commands. Dates, scenario names, native filename severity and the class label are excluded from the input. Temperatures retain Fahrenheit. The pressure filenames use +/-2inwg and +/-4inwg while the paper states +/-0.2 and +/-0.4 inH2O. We retain their original identifiers and make no independently verified pressure-severity magnitude claim.
Operating windows require SYS_CTL=1, both supply fan speeds positive, and thirty contiguous one-minute readings within one source day. Missing/nonfinite values receive at most five causal forward-filled readings within a day/gap; remaining gaps use medians fitted on healthy active training rows only. Standardization uses the same training-only rows and clips to +/-10 standard deviations. No nonfinite values appeared in the audited training columns. Inactive/shutdown windows remain unscored; scenario-specific raw and scored day counts appear in results/multisensor_v1/cohort_counts.csv.
The initially proposed single-duct source failed an integrity audit: all four outdoor-bias tables were identical and the documented sa_bias filenames were absent. The audit is retained in results/sda_source_audit. Its mislabeled/duplicate cases were not used in these models.

## Calendar separation and development history

All scenarios use the same calendar-week partition: ISO-week modulo six equals zero for test, five for validation, and all other values for training. Matching calendar/weather periods never cross a partition. Histories never cross a day, operating gap or scenario. This tests held-out periods of the same simulated unit and known bias families, not new physical equipment or unseen fault severities.
Eligible endpoints: 2,096,259 training, 463,842 validation, 463,058 test. To control overlapping-window training cost, optimization samples one endpoint per hour (36,404 windows). Checkpoint selection samples validation every fifteen minutes (30,877). Alarm-policy validation and test score every eligible minute.
Version 1 trained two input families for at most 20 epochs and failed the false-alarm/recall validation gate. Version 2 explicitly logged a validation refinement: up to 80 epochs, eight-epoch patience, doubled healthy loss weight and checkpoint selection using macro AP across all five classes. Four unique LSTMs were trained in total. All use the fixed primary seed 42, a 32-unit LSTM, 16-unit ReLU head, dropout 0.2, weighted cross entropy, Adam 0.001, batch size 512 and gradient clipping 1. There is no repeated-seed robustness claim for this extension.
Version 2 compared 126 alarm policies for each of the four model candidates, including reused version-1 weights: 504 policies. Shared healthy percentiles or a larger budget for one fault class determine per-class thresholds. Immediate, two-consecutive and two-of-three rules reset at day/scenario/gap boundaries. Validation macro F1 selects among candidates meeting healthy FPR <=1% and each-sensor recall >=40%. Model and thresholds were frozen before the single final test. Both study versions remain visible; their total search is disclosed.
The selected primary has 8,293 parameters, runs 50 epochs and restores epoch 42. Its validation FPR is 0.90%; its reserved test FPR is 2.10%. The selected alarm rule is immediate.

## Five input comparison

A training-only tree selector ranks grouped latest/mean/std/slope histories. It fits earlier internal training weeks and permutes whole daily trajectories in held-out training weeks, with three fixed repeats. No validation or test labels enter ranking. The five selected inputs are hot supply fan speed, hot supply fan power, cold supply fan speed, cooling valve command and cold supply fan power. These comprise four measurements and one command, rather than five fault-target sensors.
| Candidate | Inputs | Test accuracy | Macro F1 | Healthy FPR |
|---|---:|---:|---:|---:|
| v1_all_inputs | 26 | 77.84% | 75.22% | 2.25% |
| v1_top5_inputs | 5 | 63.22% | 54.88% | 32.69% |
| all_inputs | 26 | 92.87% | 89.36% | 2.10% |
| top5_inputs | 5 | 64.27% | 56.42% | 37.99% |

Neither raw-five model met validation targets. The reduction removed evidence useful for hot-deck temperature identification. This conditional permutation ranking is exploratory; correlated features share information. It does not establish that every possible five-feature or engineered-feature model will fail. PCA is not implemented in this study.

## Limits and demonstration

Fault prevalence is 94.03% in the experimental test sample. A constant majority class would score 23.88% on this five-class task. Five-class accuracy, per-sensor recall and healthy false alarms answer different questions. Minute observations overlap and are correlated, so large row counts do not imply independent equipment evidence.
The source imposes one known sensor bias per scenario. Concurrent faults, other equipment faults, missing-data robustness, field deployment, unseen severity generalization and future-failure prediction are unproven. The already-inspected reserved weeks cannot be reused for a fresh improvement claim. Improving the 2.10% false-alarm result requires a new development/evaluation design and additional independent data.
Run Start Dashboard.bat and select Four-sensor extension. Replay a healthy day and one case for each sensor. The app displays the failed 1% target. Uploaded thirty-row CSVs use saved transformations and model weights; softmax scores are uncalibrated. The old outdoor-air study remains selectable. Retraining requires the separately downloaded source archive; the saved demonstration runs offline.
