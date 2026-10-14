# Task 3 generalization diagnostic

Training AP is measured on the fitting data and is optimistic; it does not select an additional model. Validation is later in time, on the frozen natural-prevalence cohort. The training endpoints are sampled hourly, whereas validation endpoints are scored every 15 minutes. This comparison describes the gap without estimating its cause.

| Model | Training AP | Validation AP |
|---|---:|---:|
| lstm8_p1 | 0.0239 | 0.0235 |
| lstm8_p3 | 0.4061 | 0.0213 |
| lstm8_p10 | 0.2401 | 0.0151 |
| lstm24_p3 | 0.2365 | 0.0164 |
| lstm8_corr_p3 | 0.2698 | 0.0174 |
| lstm8_quality_p3 | 0.1898 | 0.0146 |
| gradient_boosting8 | 0.7124 | 0.0152 |

Per-node endpoint target counts and prevalence are in `label_distribution.csv`. These counts can reveal label concentration but cannot prove that a sensor change caused the weak validation result. Nagios health faults may not be identifiable from the selected thermal/fan/power measurements. Additional telemetry and fault-specific labels are possibilities to investigate, not established remedies.

No independent holdout measurements, labels or predictions were accessed. This diagnostic adds no configurations, thresholds or selection criteria.
