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


Input channels: 26; measured targets: four individually biased sensors; primary parameters: 8,293.

Research prototype. The reserved-test 1% healthy FPR target is unmet. Single simulated unit, fixed seed, no simultaneous fault coverage. See REPORT.md and verification.json for full methods and evidence.
