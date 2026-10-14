# Task 5 — bounded additional-input recovery

Trained one extended LSTM and one extended tree classifier. The extension adds 22 measured/aggregated inputs to the original 22, plus 22 unavailability flags and one median-fallback fraction: 67 model inputs. The same 111,070 training and 71,982 validation endpoints are retained, with 1,445 and 902 critical targets respectively. No extra missing measurements exclude an endpoint.

| Model | AP | Precision | Recall | FPR | False bursts/day | Incidents | Feasible? |
|---|---:|---:|---:|---:|---:|---:|---|
| lstm8_p1 | 0.0235 | 2.98% | 41.91% | 17.29% | 0.489 | 26/47 | False |
| gradient_boosting8 | 0.0152 | 1.25% | 100.00% | 99.92% | 0.115 | 47/47 | False |
| extended_lstm8_p1 | 0.0247 | 1.93% | 40.02% | 25.87% | 1.963 | 32/47 | False |
| extended_boosting8 | 0.0147 | 1.25% | 100.00% | 99.86% | 0.129 | 47/47 | False |

**Outcome: `no_qualified_recovery_candidate`.** 0 recovery policies meet all five original targets. There are 90 recovery policies and 180 total matched-baseline/recovery policies.

A row without a feasible policy shows the best diagnostic precision at recall ≥40%; it is not a release operating point. A recovery candidate additionally needs AP improvement over its original same-type baseline. The seed remains 42; architecture, optimizer, class weighting and history width were not searched.

## Added information and safeguards

Added inputs cover power-supply voltage/current, CPU memory/I/O power, PCIe temperature, ambient/power variability, maximum fan variability, GPU temperature maxima/spreads and ambient/power maxima. GPU/fan aggregate features require all specified devices. The full source definitions are frozen in the plan. See `LABEL_AND_SENSOR_AUDIT.md` for verified source meanings and the more precise batch-client label scope.

Forward filling uses only prior readings, at most four, and resets at splits, nodes and gaps. Remaining extra missing values use medians of normal training rows only. Unavailability flags preserve the original missing state, and the fallback fraction identifies median substitution. Extra scaling uses the same normal training rows as the original base scaler; validation supplies no preprocessing statistics. The original 22-sensor scaler is reused exactly. Values beyond ten training standard deviations are clipped; real zeros and high original measurements are preserved in exported CSVs.

The LSTM uses hidden size 32, dropout 0.2, positive loss weight one, Adam 0.001, batch 512, maximum 25 epochs and patience five on validation AP. The tree uses the same latest/mean/std/slope construction, learning rate 0.05, 15 leaves, minimum leaf size 30 and L2 one; explicit validation selects 50/100/200 iterations. Input dimension and parameter count grow; this is an input-set comparison, not proof of any individual sensor’s causal contribution.

Normal percentile/PR thresholds and persistence follow the original grid and feasibility gate. Node-bootstrap intervals condition on selected policies and cannot establish future performance. Configuration/checkpoint/policy selection uses the already-inspected validation period, so it remains development evidence.

## Decision and next work

Reserved servers and the later exploratory period were not scored. A successful single-seed recovery would require repeated seeds and a frozen pipeline before independent evaluation. A failed recovery does not justify an accuracy claim or changing the reference labels. It supports reporting that the tested physical telemetry does not adequately predict the selected batch-client outcome under these splits. Further model search or a replacement dataset must be a separately bounded and documented next decision.

The dashboard, earlier PDF/PPT and original artifacts remain unchanged. This report supplements them; submission documents still need to include the later experiments and corrected label scope.

## Reproduce

Run `.venv\Scripts\python.exe scripts\run_task5.py` then `scripts\verify_task5.py`. `scripts\predict_task5.py` scores an exported example window without the archive. Individual model folders contain trusted weights/preprocessing, histories, scores, examples, decisions and incident details. Original source readings and labels are never overwritten.
