# More sensor data and fewer model inputs

Prepared 16 October 2026 in response to professor feedback. This is a research extension; the submitted model and its independent autumn result remain unchanged.

## What the present source can support

MZVAV-1 has thirteen measurement/control inputs, plus timestamp, labels, occupancy and fan status. The latter status signals are constant during eligible operation. All thirteen measurement/control inputs are already available to the current model, which derives six consistency features and thirteen missing flags for 32 channels. This file does not contain additional airflow, humidity or power measurements. Some of its existing inputs are also constant in active training operation.

Adding columns from another AHU by row number/date would not create additional measurements of this equipment. Duplicating rows would not create new independent evidence either.

## Completed feature comparison

The separate `results/hvac_feature_v2` study compares existing engineered32, raw13 and a training-selected raw5 across seeds 7, 42 and 123. The comparison keeps the source cohorts, 30-minute histories, LSTM hidden size 32, head/dropout, loss weighting, optimizer, batch size, maximum 30 epochs, six-epoch patience and validation alarm-policy search matched. No new autumn predictions were made.

Sensor selection uses only original winter/spring training days. A tree is fitted to earlier eligible days within each documented scenario, then sensor-group permutation importance is measured on the last two eligible training days per scenario. Entire donor-day trajectories preserve each scored minute pattern. Ranking is conditional on this selector and correlated inputs; the five candidates are not asserted to be the globally optimal LSTM subset.

The selected five raw inputs are outdoor temperature, cooling-coil valve command, supply-air temperature, mixed-air temperature and outdoor-damper command. This is five raw sensor/control variables, not five features selected from all the engineered channels. These two reduction questions must be distinguished in the next experiment.

| Input set | Model channels | Parameters | Validation AP mean | Mean FPR at selected policies | Feasible seeds |
|---|---:|---:|---:|---:|---:|
| Existing engineered32 | 32 | 8,993 | 0.9820 | 0.00% | 3/3 |
| Raw13 | 13 | 6,561 | 0.8933 | 31.02% | 0/3 |
| Selected raw5 | 5 | 5,537 | 0.8947 | 26.31% | 0/3 |

These are summer validation results, not replacement autumn test results. The failed reduced policies are diagnostic policies with the original recall constraint; they are not qualified releases. Feature removal reduced parameters but did not improve the tested false-alarm/recall tradeoff. In particular, the reductions removed the useful engineered sensor relationships as well as redundant inputs. A further study should evaluate selected engineered features too, instead of claiming that all five-feature models must perform poorly.

## Suitable richer simulation source

The official LBNL simulated single-duct AHU collection lists 30 recorded data points, one-minute sampling and 365 days. Its PDF inventory lists twenty faulty cases and one healthy case, across five fault types; the website's brief CSV count differs and must be checked against the actual archive. Outdoor-temperature biases of +/-2 and +/-4 Celsius are explicitly documented, so the present fault family can be retained. The inventory includes outdoor/return/supply airflow, fan power and actual actuator positions, beyond the channels in MZVAV-1. The listed archive is about 593 MB and expands to approximately 2.69 GB.

Official source: https://faultdetection.lbl.gov/dataset/simulated-sd-ahu/

Source inventory: https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_SDAHU/LBNL_FDD_Data_Sets_SDAHU.pdf

No rich-source archive has been downloaded, merged, relabelled or trained as part of the completed feature comparison. Another option is a separate expanded synthetic simulator, but its assumptions and generated ground truth would need independent documentation. The published simulation source is the preferred research direction when it contains suitable native fault cases.

## Next experiment design

1. Audit the richer source's sensor schema, units and native fault definitions. Decide whether to retain the outdoor-air bias task or explicitly extend to multiple fault families. Do not silently combine distinct labels.
2. Check overlap with MZVAV-1 using common-channel/timestamp fingerprints. Both sources model similar equipment/weather, so a new file is not automatically independent new evidence.
3. Reserve fresh calendar/weather periods before model fitting. Keep the same calendar period in one partition across all parallel healthy/fault scenario files. Exclude previously inspected old-source test observations from any claimed new untouched evaluation.
4. Use training data only to fit cleaning/scaling and rank features, with whole-day internal selection folds. Remove constants and obvious duplicates. Compare all suitable recorded measurements, five selected raw measurements, and five selected engineered features as separate hypotheses. Report how many physical measurements each engineered feature needs.
5. Keep histories, labels, endpoints, model budget and three seeds matched. Tune thresholds/persistence on validation. Compare AP, recall, FPR, false bursts, fault coverage, parameter count and measured inference time. Do not select only by accuracy on a fault-heavy sample.
6. Freeze the complete pipeline, evaluate once on the newly reserved cohort, and promote a new model only if its measured tradeoff justifies it. Then update the app, notebook, report, presentation and package with the new source and measured scope.

More data means additional scenarios/weather periods or richer genuine simulated measurements. More columns and fewer deployed model inputs can coexist: collect the wider source first, then determine which inputs the model needs through the controlled ablation.
