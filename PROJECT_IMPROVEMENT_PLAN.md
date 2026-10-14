# One-week deep learning project improvement plan

Prepared 14 October 2026. Submission is next week; the exact submission date is not specified. The professor has no required model type. This is a plan, not a record of completed experiments or promised results.

Progress: Tasks 1Ã¢â‚¬â€œ5 were completed on 14 October 2026. Historical policies, six supervised configurations, three-seed repeats and the bounded additional-input recovery all failed the unchanged target gate. Task 5's extended LSTM has diagnostic precision 1.93%, recall 40.02% and FPR 25.87%, despite a slight AP increase. Thirty-five tests, full validation inference and reproduction of all 180 original/recovery policies passed. A source review found that the authors' preparation code selects the batch-client scheduler-state check; it does not provide verified cooling-fault labels. No operational release is frozen and independent M100 evaluation remains deferred. See `results/task5_recovery_v1/REPORT.md`, `LABEL_AND_SENSOR_AUDIT.md` and `NEXT_OPTIONS.md`. This was the status before the approved cooling-equipment scope change documented below.

## Current primary study

The user approved the labelled cooling-equipment dataset option on 14 October 2026. The current submission is the `hvac_v1` simulated-AHU sensor-bias benchmark, with a frozen autumn evaluation, updated dashboard, report, presentation and notebook. Its 1.44% test FPR remains above the 1% goal. The M100 status and work plan below are preserved historical records; see the completed scope-change section and README.md for current deliverables.

## Objective and current evidence

Build a defensible deep learning project that detects current server-health anomalies with substantially fewer false alarms. Keep the existing autoencoder as an experimental reference and develop a supervised LSTM classifier as the main new candidate.

Current selected experiment: `cooling12_8`. On 73,300 scored test endpoints:

| Measure | Current result |
|---|---:|
| True positives / false positives | 348 / 8,745 |
| False negatives / true negatives | 164 / 64,043 |
| Precision | 3.83% |
| Recall | 67.97% |
| F1 | 7.25% |
| False positive rate, FP / (FP + TN) | 12.01% |
| False discovery ratio, FP / (TP + FP) | 96.17% |
| Average precision | 0.0789 |
| False alarm bursts per scored node-day | 0.955 |
| Detected original incidents | 15 / 34; 22 observable |

These measures describe different denominators. Reducing the number of notifications does not itself improve the underlying reading-level predictions. Only 512 scored endpoints are critical, and node 10 contributes 397 of them. Results must include per-node metrics and incident coverage.

The existing policy maximizes validation endpoint F1. Its winning threshold is the normal-validation 95th percentile. This operating point does not satisfy the user's priority of limiting false positives. Feature removal raised test F1 but worsened average precision, total incident detection and false-burst burden. The validation F1 difference from the original width-8 model is only 0.0750 to 0.0764, from one seed.

## Proposed development targets

Use the following as initial engineering goals, not professor acceptance criteria or guaranteed results:

- Reading-level precision at least 30%, while reading-level recall remains at least 40%.
- False positive rate at most 1% of scored known-normal readings.
- At most 0.1 false alarm bursts per scored node-day, while detecting at least 60% of observable incidents.
- Report all-incident recall, including unobservable incidents, alongside observable recall.
- Maintain natural class prevalence in validation and evaluation; report exact counts and uncertainty.

Before testing models, fix one primary policy-selection rule: maximize validation precision among policies satisfying the recall and false-alarm constraints. Use F0.5 and alarm delay as secondary comparisons. If no policy satisfies the constraints, show the feasible tradeoff frontier and record that the target was missed; do not silently relax constraints or select an all-normal predictor.

## Day 1: audit the task and freeze the evaluation protocol

1. Review representative validation false alarms, detected incidents and missed incidents. Check label timestamps, sensor outages, node-specific shifts, fan zeros, power changes and implausible temperature readings. Keep examples that explain both model failure and data limitations.
2. Use a precise task statement: current general server-health anomaly detection from thermal, fan and power telemetry. Nagios labels do not identify cooling-specific faults. Do not claim early warning, fault-cause diagnosis or energy savings from contemporaneous classification.
3. Preserve original measurements and labels. Keep missing/invalid-sensor indicators as candidate features; an invalid sensor can itself be a health fault. Avoid treating every extreme value as noise. Retain genuine zero fan speeds and high temperatures/power readings.
4. Split raw readings before constructing sequences. Use causal processing and chronological development folds, with no sequences crossing node boundaries, split boundaries or time gaps. Fit all scaling and feature selection on training data only.
5. Compare candidates on exactly the same valid window endpoints. Keep unknown labels excluded from supervised loss and labelled metrics, rather than mapping them to normal. Explicitly report coverage lost to missing inputs and unknown labels.
6. Score every 15 minutes where a valid input history exists. Make alarm cadence configurable; the current evaluator assumes 30-minute scored intervals. Recalculate persistence, burst boundaries and scored exposure using the actual cadence. Rescore the existing baselines at the same cadence; retain historical 30-minute results separately.
7. Treat the already-inspected test recordings as an exploratory benchmark. Reserve the previously unused nodes 2, 3, 4, 5, 6, 18 and 19 for a separate final check over a fixed later time period. Check feature and label availability before predictions; freeze the exact cohort and date cutoff in a manifest. This is also an unseen-node generalization test, and must be labelled as such. Do not use those final-period labels to fit features, select models or tune thresholds. If class or incident coverage is too small, disclose it rather than choosing a cohort after seeing model performance.

Deliverables: task statement, false-alarm case audit, evaluation/holdout manifest, configurable-cadence evaluation tests and data-quality report.

## Day 2: establish the attainable false-alarm tradeoff

1. Reuse saved autoencoder and Isolation Forest models before expensive retraining. Produce validation precision-recall and threshold-versus-false-alarm curves on the matched cohort.
2. Compare a small predefined threshold grid, including normal-validation percentiles 99, 99.5, 99.9 and 99.95, plus score thresholds derived from validation precision-recall curves. Compare global and training/validation-calibrated per-node policies. Use a global fallback for previously unseen nodes; never fit a threshold on final holdout labels.
3. Evaluate immediate decisions, two consecutive high scores and two-of-three high scores. Reset state at gaps and node changes and measure the delay cost on short incidents.
4. Keep raw reading predictions separate from displayed notifications. Hysteresis or cooldown may reduce alert fragmentation, but must not be presented as a reduction in reading-level false positives. Report every policy's missed incidents and delay.
5. Select operating points using the fixed validation constraints rather than unconstrained F1. Threshold adjustment changes the operating point, not the underlying score-ranking quality.

Decision gate: if an existing detector achieves useful precision at acceptable recall, retain it as a credible baseline. If the validation ranking is weak, proceed to supervised learning rather than spending the week searching ever more thresholds.

Deliverables: validation tradeoff curves, comparison of feasible alarm policies and a documented selection rule.

## Days 3Ã¢â‚¬â€œ4: train a label-aligned deep learning model

There are 6,953 CRITICAL raw training readings in the existing chronological split. These are correlated readings, not 6,953 independent incidents. The current autoencoder excludes them from training; a supervised model can learn which patterns distinguish the labelled states.

Main candidate: an LSTM encoder with a small classification head. Start with an 8-reading input history, hidden size 32 and a binary logit for the current endpoint label. Train on both normal and critical training windows using `BCEWithLogitsLoss`. Apply modest loss weighting only on training data and tune it on validation; do not assume the full inverse-frequency weight will reduce false alarms. Higher positive weights can increase recall at the expense of precision.

Bound the experiment budget:

- Start with positive loss weights 1, 3 and 10, with the same architecture and features.
- Try a 24-reading history or hidden size 64 only if the first validation experiments justify it.
- Limit architecture/weight exploration to at most six predefined neural configurations before freezing a finalist.
- Use chronological validation for early stopping and configuration selection. Set maximum epochs and patience in the manifest, and report which criterion stopped training.

Use four principal comparison models:

| Model | Purpose |
|---|---|
| Existing LSTM autoencoder | Original normal-only deep learning reference |
| Isolation Forest | Unsupervised machine learning reference |
| Gradient-boosted classifier | Strong supervised non-DL baseline |
| Supervised LSTM classifier | Main label-aligned deep learning candidate |

Give the supervised tree baseline the same sensor history through causal trailing statistics, and the same labelled endpoints. Avoid random internal validation when early-stopping a time-series classifier. Compare full telemetry plus data-quality flags with a reduced feature set in a controlled ablation. Do not remove a sensor merely because it correlates with another during normal operation: their divergence can signal a fault.

PCA is optional. Run at most one small PCA ablation after the supervised model and alarm calibration work. It is not the primary strategy for addressing the present false positives.

Deliverables: saved classifiers and preprocessing, reproducible training scripts, histories, matched-cohort model table and feature/quality-flag ablation.

## Day 5: verify robustness and freeze the model

1. Run the finalist with seeds 7, 42 and 123. Apply the same validation selection rule to each; report mean, variation and individual counts. Do not cherry-pick the seed with the best test score.
2. Choose a release model using a predeclared validation-only rule. Freeze model, feature list, preprocessing, threshold and alarm state machine before the final heldout evaluation.
3. Evaluate the frozen pipeline on the independent heldout nodes/time period once. Report this separately from the old exploratory recordings, because unseen-node performance answers an additional generalization question.
4. Include confusion matrices, precision-recall curves, AP, F0.5/F1, FPR, false bursts per scored node-day, detected/total/observable incidents and delays. Include per-node results to expose node 10's dominance in the old test cohort.
5. Estimate uncertainty using nodes/time blocks or incidents rather than treating overlapping windows as independent observations. Where there are too few independent incidents, report that limitation instead of a precise-looking confidence interval.
6. Verify fresh inference against saved scores, timestamp/gap behaviour and missing-input handling. Verify that the packaged model uses exactly the preprocessing evaluated in the report.

Decision gate: if neither the DL model nor the supervised baseline meets the targets, inspect whether the selected telemetry can predict the broad Nagios labels at all. Document the ceiling and investigate relevant additional training features. If a project-scope/dataset change is needed, propose a concrete labelled cooling-equipment benchmark before replacing the project. Do not manufacture cooling labels from temperature thresholds, delete difficult servers or rebalance the test set to improve the headline numbers.

## Day 6: rebuild the academic submission and demo

- Update the report and PPT together; the existing final PDF and slides predate feature selection.
- Explain problem scope, data source, label meaning, chronological splits, cleaning, architecture, training, class imbalance, baselines and policy selection.
- Present the main comparison and ablations, including failures and the precision-recall/false-alarm tradeoffs.
- Include one successful detection, one false alarm and one missed incident with original readings and model outputs.
- Document dataset limitations, sensor-quality effects, incident coverage, previously inspected recordings and unseen-node evaluation.
- Update the dashboard with the frozen classifier, its actual trained feature list and threshold, reproducible CSV inference and honest alarm explanations. Do not present an uncalibrated score as a fault probability.
- Add a model card, exact environment, test commands and source-data instructions. Keep the trained demo runnable without the large source archive.

Deliverables: updated report, approximately 12Ã¢â‚¬â€œ15 clear presentation slides, reproducible notebook, model card and dashboard.

## Day 7 or remaining time: submission buffer

Test a fresh GitHub checkout in a clean environment, verify that the selected model and dashboard load, build the final ZIP and rehearse a short demonstration. Upload only verified changes and give the professor the repository link, report and PPT. Keep this buffer rather than introducing a new architecture immediately before submission.

## Completed scope change and submission, 14 October 2026

The user selected a labelled cooling-equipment dataset after M100 feature recovery failed. The primary experiment is `hvac_v1`: native outdoor-air sensor-bias labels in one simulated AHU from the LBNL AFDD collection. Whole seasons and source days remain separated. Three LSTM seeds and gradient boosting use the same 30-minute histories; seed 42 was declared primary before autumn evaluation.

The frozen test result is 92.82% accuracy, 99.74% precision, 91.86% recall and 1.44% FPR. **The original 1% FPR goal is still not met.** No post-test tuning occurred. This is a course prototype, not a validated deployment across real equipment. M100 negative results and unscored reserved servers remain intact.

The updated dashboard, 11-page report, 14-slide presentation, executed notebook, model card and compact bundle are the current submission. See README.md and results/hvac_v1. The plan above remains a record of the M100 investigation.

## Sources informing this plan

- [M100 data descriptor and Nagios label scope](https://www.nature.com/articles/s41597-023-02174-3)
- [Decision-threshold tuning and validation separation](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [Time-series and grouped evaluation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [PyTorch binary classification loss and positive-weight tradeoff](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.BCEWithLogitsLoss.html)

The engineering targets and one-week work allocation are proposed for this project; the sources do not guarantee that these targets are achievable or that a professor will accept a particular result.
