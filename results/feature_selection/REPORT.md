# Data cleaning and feature selection experiment

All four models use width 8, seed 42, the same model architecture and training settings, and exactly the same training, validation and test endpoints. Feature removal reduces the input/output parameter count. Configuration selection uses validation F1; test results remain exploratory because this recording has already been examined.

## Cleaning

- nonfinite values become missing
- negative temperatures, fan speeds and powers become missing under an indoor-rack plausibility rule
- zero fan speeds/power and high measurements remain unchanged
- forward fill at most four consecutive 15-minute readings; reset at gaps, node and split boundaries
- no backfill, interpolation, removal of rare high readings or invented labels
- unknown labels remain recorded but are excluded from labelled evaluation windows

Original readings remain in the source archive and original caches. Cleaned Parquet exports retain every row, labels, split, counts of invalid/imputed/remaining missing sensors, and the sensor values after limited causal filling. Duplicates fail validation for manual reconciliation; none exist in the current dataset. Remaining missing readings are not invented.

Measured data quality: {"rows_preserved": 751363, "duplicate_node_timestamp_keys": 0, "original_missing_cells": 1304207, "implausible_negative_cells": 130, "causally_filled_cells": 75640, "remaining_missing_cells": 1228697}.

## Feature selection

Normal-training data removed constants (standard deviation <=1e-8), inputs with over 20% missing measurements, and redundant inputs with absolute Pearson correlation >=0.95. The deterministic greedy order follows the original feature list. This produced 16 inputs: ambient, fan0_0, fan1_0, fan1_1, p0_vdd_temp, p1_vdd_temp, dimm_temp, p0_core_temp, p1_core_temp, gpu_core_temp, gpu_mem_temp, p0_power, p1_power, total_power, ps0_input_power, ps1_input_power.

The cooling candidate is a predefined 12-feature subset: ambient, three distinct fan channels, seven internal temperatures and total power. Correlation is pooled across nodes and may obscure node-specific relationships; selection is not proof that a removed sensor is unnecessary for a physical diagnosis.

## Measured results

| Experiment | Features | Validation F1 | Test precision | Test recall | Test F1 | Test AP | False bursts/day |
|---|---:|---:|---:|---:|---:|---:|---:|
| feature_reference8 | 22 | 0.0750 | 0.0211 | 0.7227 | 0.0411 | 0.1031 | 0.692 |
| clean22_8 | 22 | 0.0744 | 0.0208 | 0.6914 | 0.0405 | 0.1018 | 0.672 |
| correlation8 | 16 | 0.0748 | 0.0314 | 0.6855 | 0.0601 | 0.1231 | 0.849 |
| cooling12_8 | 12 | 0.0764 | 0.0383 | 0.6797 | 0.0725 | 0.0789 | 0.955 |

Matched-cohort validation winner: **cooling12_8**. Dashboard default: **cooling12_8**. The default changed because validation performance improved; test results did not select it.

The validation F1 difference between the reference (0.0750) and the winner (0.0764) is small and comes from one seed. This is not evidence of a reliable general improvement. Compared with the reference on identical test readings, the winner changes falsely flagged readings from 17,137 to 8,745, false-alarm bursts from 1,056 to 1,459, incidents caught from 17 to 15, and average precision from 0.1031 to 0.0789. Fewer false-positive readings can form more separate bursts when alarms are less continuous. Performance therefore involves a tradeoff rather than a universal accuracy improvement.

Cohort counts: {"train": {"original_eligible": 105150, "cleaned_eligible": 105150, "common": 105150, "common_critical_endpoints": 0}, "validation": {"original_eligible": 37074, "cleaned_eligible": 37074, "common": 37074, "common_critical_endpoints": 388}, "test": {"original_eligible": 73300, "cleaned_eligible": 73300, "common": 73300, "common_critical_endpoints": 512}}.

## Limits

- removing correlated sensors can hide individual fan faults
- implausible temperature readings can themselves signal a sensor-health anomaly
- previous test results were inspected; new experiments are exploratory
- one seed and one rack subset; no guaranteed performance improvement

The existing final PDF and PowerPoint describe the earlier experiments; this file is a supplementary experiment report. Their previous results have not been silently replaced.
