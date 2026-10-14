# Source label and sensor audit

Checked the dataset authors’ documentation and preparation code on 14 October 2026, before the recovery experiment.

The published aggregation code summarizes IPMI measurements over 15-minute bins. Its second preparation step requests Nagios `state` for `description='batchs::client::state'`, then joins that outcome to the sensor timestamps. The plugin documentation describes this check as the scheduler’s view of a compute node’s readiness to run jobs. It also documents service-state encoding 0=OK, 1=WARNING, 2=CRITICAL and 3=UNKNOWN. This is a more specific outcome than the project’s earlier general server-health description.

The local aggregated files contain a numeric `value` outcome without alert descriptions, operator-drain details or state-type tags. The association with the batch-client check is inferred from the authors’ published preparation pipeline; the archive alone cannot independently verify each original alert or explain why it occurred. Keep the existing value-0/value-2 task and unknown-label exclusions. A critical scheduler-state outcome is not a confirmed cooling fault, and contemporaneous sensor aggregates do not establish advance prediction.

The IPMI documentation identifies memory/I/O electrical measurements as power in watts, power-supply input/output voltage in volts, output current in amperes, and PCIe/GPU/ambient measurements as temperatures in degrees Celsius. Standard deviations retain the corresponding units; spreads and maxima are derived from these measurements. The four `gv100card*` fields have unspecified meaning/units in the documentation and zero training variation in the previous audit, so the recovery set excludes them.

The additional-input cleaning rule is a declared indoor-rack plausibility assumption: nonfinite and negative absolute measurements/variations become missing. This is not a manufacturer-specified operating range. Real zeros and high measurements remain. Incomplete multi-device groups become missing rather than an artificially low maximum; training-only fallback plus indicators preserves endpoint coverage.

Sources:

- [Dataset record and aggregation description](https://zenodo.org/records/7541722)
- [Authors’ first aggregation step](https://gitlab.com/ecs-lab/exadata/-/blob/main/parquet_dataset/node_aggregated_data/step1.py)
- [Authors’ label query and join](https://gitlab.com/ecs-lab/exadata/-/blob/main/parquet_dataset/node_aggregated_data/step2.py)
- [Nagios plugin definitions](https://gitlab.com/ecs-lab/exadata/-/blob/main/documentation/plugins/nagios.md)
- [IPMI metric definitions and units](https://gitlab.com/ecs-lab/exadata/-/blob/main/documentation/plugins/ipmi.md)

This source review refines the task description; it does not rewrite labels, consume independent test data or explain every observed classifier error.
