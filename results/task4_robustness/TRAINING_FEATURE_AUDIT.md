# Training-only additional-feature audit

Inspected only the nine development-node archive members and their rows before 2022-01-22T00:00:00+00:00. No labels are read by this audit. The development schemas share 243 columns including metadata and target; 48 candidate source measurements are inventoried.

| Candidate group | Measurements | Min finite fraction across nodes |
|---|---:|---:|
| additional per-component measurements | 9 | 88.77% |
| extrema hidden by averaged/pooled features | 20 | 88.77% |
| power-supply voltage/current | 6 | 88.96% |
| variation within aggregation intervals | 13 | 88.77% |

The archive offers additional per-component electrical readings, power-supply voltage/current, within-interval variability, and individual sensor extrema. Averaging multiple sensors can obscure a faulty device, but that is a hypothesis to test, not evidence of improved model accuracy. No CPU-utilization, memory-usage or network-health fields were found in these schemas. The broad Nagios target may describe faults that this thermal/electrical data cannot explain.

## Concrete bounded follow-up

1. Check original label construction and, if accessible, alert-specific fault types against the physical sensor task. Do not substitute sensor-derived labels for the reference outcome.
2. Before training another configuration, declare one extended input set from this training-only inventory: power-supply voltage/current, memory/I/O electrical inputs and compact aggregates of sensor maxima and within-interval variability. Confirm units and sensor validity rules for unfamiliar names.
3. Fit feature preprocessing on the fixed training period. Retain the existing endpoints for a primary matched-cohort comparison; do not let extra missing sensors silently exclude difficult validation cases. Compare the extended supervised LSTM with an extended tree baseline under the same training and alarm constraints.
4. Require useful later-period validation ranking and a feasible operating point before consuming the independent test recordings. If the richer telemetry still fails, the defensible options are a clearly reported negative study or a separately agreed project/dataset change.

This audit does not rank features using validation labels, train an extra model, alter the frozen targets or access reserved servers. Availability and variance do not prove predictive value. The complete source names and per-node training counts are saved in the adjacent JSON/CSV files.
