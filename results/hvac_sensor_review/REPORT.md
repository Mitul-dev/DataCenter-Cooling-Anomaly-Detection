# Sensor expansion and top-five review

This audit uses only the original winter/spring training seasons. The existing trained model and final report are unchanged.

The source already supplies all thirteen measurement/control inputs used by the project. The remaining fan-status/occupancy columns are constant during eligible operation; timestamps and labels are excluded. Additional humidity, airflow or power channels are not present in MZVAV-1. Channels from a different unit cannot be appended as if they were simultaneous observations of this AHU.

## Provisional top five

outdoor_temp, cooling_valve, supply_temp, mixed_temp, outdoor_damper

Ranking uses whole-day partitions within the original winter/spring training set: the last two eligible days of each documented scenario are used only for screening, and the remaining training days fit the selector. Each sensor is permuted together with its latest/mean/std/slope summaries. Five fixed repeats preserve the scored minute-of-day pattern within donor days. Balanced log loss prevents the fault majority from hiding healthy-class errors. Correlations can mask or distribute importance, so this ranking needs a retrained LSTM ablation.

The first diagnostic selector fit winter and screened spring. It had AP 0.8539 and poor balanced loss, so its ranking was not treated as reliable. Its diagnostic and candidate ranking are preserved separately. Neither screen reads the original summer validation or autumn test outcomes.

## Controlled comparison

Compare three predeclared sets: existing engineered32, raw13 and selected raw5. Keep endpoints, histories, architecture family, training budget, seeds and validation-only policy selection matched. Report recall/FPR/false bursts, parameter count and inference cost. Five sensors do not guarantee higher accuracy.

The original autumn data have already been inspected. A comparison there must be labelled retrospective. An independent improvement claim requires new reserved scenario runs, weather periods or equipment units. Any expanded simulator must be a separate, documented synthetic study with faults injected by scenario and labels kept out of model inputs.

This screening preceded the controlled LSTM comparison in results/hvac_feature_v2/REPORT.md. That follow-on study trained the raw13/raw5 subsets and found no qualified reduction; it did not replace the submitted model.