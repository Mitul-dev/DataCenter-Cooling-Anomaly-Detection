# Data center telemetry anomaly detection

Deep Learning project, Section B. Team: Rishabh Srivastava (2430010149), Mitul Seth (2430010099), Madhav Mittal (2430010096).

The project compares a normal-only LSTM autoencoder and Isolation Forest with supervised LSTM and gradient-boosted classifiers using thermal, fan and power telemetry. A source review found that the authors' published dataset preparation selects Nagios `batchs::client::state`, which concerns scheduler readiness. The earlier general server-health description is broader than that documented check; original alert metadata is not present in the local aggregate files. The evaluated task does not establish cooling-specific diagnosis or prediction of future faults. The controlled comparison, three-seed check and added-input recovery still miss the false-alarm targets; see `results/task5_recovery_v1/REPORT.md` and `LABEL_AND_SENSOR_AUDIT.md`.

## Submission files

- `submission/DCCAD_Final_Report.pdf`: earlier report; the planned submission update must incorporate Tasks 1–3.
- `submission/DCCAD_Final_Presentation_Reviewed.pptx`: earlier presentation; the planned submission update must incorporate Tasks 1–3.
- `M100_submission.ipynb`: clean notebook for reproducing and examining the experiment workflow.
- `results/selected_model.json`: identifies the LSTM selected using validation data.
- `results/<experiment>/`: trained model, scaler, history, scores, window metadata, incident records and replay data.
- `results/feature_selection/REPORT.md`: supplementary cleaning and feature-selection report, including matched-cohort results and tradeoffs.
- `dashboard/app.py`: updated demo with recorded replay, sensor explanations and fresh CSV inference.
- `M100_explore.ipynb` and `dashboard/legacy_app.py`: original work retained for traceability.

## Run the demo

The GitHub checkout includes the trained models, replay data, reports and slides, so the demo runs without the original dataset archive. Download `0.tar` from the dataset link below only when retraining. The 225 MiB submission ZIP and the source archive are excluded from Git; you can recreate the ZIP with `python scripts/package_submission.py`.

Python 3.11 is the tested environment. From this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

If the local environment is already prepared, run `run_dashboard.ps1` or use the last command. The app resolves data paths relative to its own source, so it also runs when launched from another directory.

The replay controls choose an experiment, server and alarm rule. The default experiment comes from `results/selected_model.json`; its alarm policy was chosen on labelled validation data. Manually changing the replay rule is exploratory and does not alter the reported results.

The fresh-inference tab runs the saved model on a CSV. Download its actual 15-minute example window, then upload it to reproduce a score. A single window cannot establish a two-reading persistence streak. If a timestamp column is supplied, the app checks exact 15-minute cadence. Enhanced feature windows must use the definitions below.

## Data and preprocessing

Source: [M100 aggregated anomaly-detection dataset](https://zenodo.org/records/7541722), DOI 10.5281/zenodo.7541722, and its [data descriptor](https://www.nature.com/articles/s41597-023-02174-3).

The local `0.tar` contains sixteen node files. Experiments retain the fixed nine-node subset 7, 8, 9, 10, 11, 13, 14, 16 and 17. The tar is read directly; files are not extracted through untrusted archive paths. The submission ZIP omits the large archive. Download `0.tar` from the dataset record and place it in the project root to reproduce training. It is already present in this local project.

Historical experiments use a chronological 70/10/20 row split per node. Short gaps are filled forward at most four rows, with filling reset at timestamp discontinuities. Scalers fit finite normal training rows only. Windows never cross split boundaries or node boundaries. Those normal-only training windows contain no CRITICAL readings. Task 3 instead uses the frozen shared calendar boundaries, known endpoint targets and 15-minute validation cadence; its supervised models learn from both normal and critical training endpoints. See the protocol and Task 3 sections below.

Standard experiments clip scaled inputs to [-5, 5]. Robust24 uses median/IQR scaling with no clipping. Original recorded readings are kept separately for dashboard plots, so inverse scaling does not incorrectly present clipped values as raw measurements.

## Task 5: bounded additional-input recovery

One extended LSTM and one tree classifier add 22 electrical/thermal inputs to the original 22, with 23 missing/fallback indicators. Both preserve all 111,070 training and 71,982 validation endpoints. The original base scaler is reused; added-input scaling and median fallback use normal training rows only. Four-reading causal filling resets at gaps, nodes and splits. Indicators retain sensor unavailability rather than silently removing readings.

**No recovery policy passes all five targets.** At the diagnostic operating point the extended LSTM gives precision 1.93%, recall 40.02%, FPR 25.87% and 1.963 false bursts/scored node-day. Its AP increases slightly from 0.02349 to 0.02465, but this does not improve the required operating point. The extended tree AP is 0.01470. All 180 matched original/recovery policy results and the two full validation score arrays reproduced; 35 unit tests passed. Reserved servers remain unscored and no replacement model was released.

```powershell
.\.venv\Scripts\python.exe scripts\run_task5.py
.\.venv\Scripts\python.exe scripts\verify_task5.py
.\.venv\Scripts\python.exe scripts\predict_task5.py --model extended_lstm8_p1 --csv results\task5_recovery_v1\extended_lstm8_p1\example_window.csv
```

`results/task5_recovery_v1` contains the frozen recovery plan, weights/preprocessing, source-label audit, exact feature definitions, histories, shared scores, decisions, incident/per-node results, uncertainty intervals and inspected plots. CSV inference needs 24 timestamped readings of all 44 measured/derived inputs; extra missing values use saved training medians and do not remove the endpoint. Scores remain uncalibrated. `NEXT_OPTIONS.md` describes a concrete cooling-equipment benchmark alternative and the option of completing M100 as an honest negative comparative study. A dataset/scope change is pending the user's choice; current artifacts and submission files are preserved.

## Task 4: robustness check and release gate

The provisional `lstm8_p1` was trained from scratch with seeds 7, 42 and 123 under the same frozen configuration and cohorts. Seed 42 reproduces the Task 3 checkpoint tensors and validation scores exactly. All three seeds fail the validation targets: their diagnostic policies average precision 2.67%, recall 40.72% and FPR 20.05%. None of 135 policies is feasible. **No model qualifies for release**, and independent test servers remain untouched.

```powershell
.\.venv\Scripts\python.exe scripts\run_task4.py
.\.venv\Scripts\python.exe scripts\audit_task4_features.py
.\.venv\Scripts\python.exe scripts\verify_task4.py
```

`results/task4_robustness/REPORT.md` contains individual counts, seed variation, per-node/incident results, inspected plots and the release decision. Uncertainty intervals resample entire development servers; they condition on selected validation policies and do not account for model selection or future drift. `research_snapshot.json` preserves model/preprocessing/policy fingerprints without certifying release performance. The predeclared release rule requires all three seeds to pass and uses seed 42 as the primary candidate; its global fallback must also pass. It cannot cherry-pick a seed or relax the original targets.

`TRAINING_FEATURE_AUDIT.md` inventories 48 additional thermal/electrical measurements from development training rows only, including power-supply voltage/current, component measurements, interval variation and sensor extrema. Their usefulness is unproven. The next work is a bounded investigation of richer inputs and label alignment before independent evaluation; details are in `RECOVERY_NEXT_STEP.md`. The dashboard and earlier PDF/PPT have not been switched to a failed classifier. Thirty-one unit tests and reproduction of all 135 policies, three fresh-inference examples, exact seed-42 scores, per-node metrics and node-bootstrap intervals passed.

## Task 3: supervised classifiers and controlled comparison

`results/task3_supervised/REPORT.md` records nine freshly trained models: six predefined LSTM classifier configurations, a gradient-boosted classifier, and retrained autoencoder/Isolation Forest references. The neural experiments compare loss weights 1/3/10, eight versus 24 readings, 22 versus 16 training-selected sensors, and 22 sensors plus three data-quality indicators. Scaling and feature selection use calendar training data only. All validation comparisons use the same 71,982 endpoints, with 902 critical targets; the reserved servers and later exploratory period are not scored.

**None of 441 tested policies meets all five targets.** The provisional DL candidate, `lstm8_p1`, has validation AP 0.02349. Its diagnostic policy gives precision 2.98%, recall 41.91%, FPR 17.29%, 367 false bursts and 26/47 detected incidents. It is not a release recommendation. Removing correlated sensors and adding quality indicators did not improve AP in this single-seed comparison. `GENERALIZATION.md` shows that the weight-three LSTM has training AP 0.4061 versus validation AP 0.0213; the tree baseline similarly drops from 0.7124 to 0.0152. Training AP is optimistic and does not establish the cause of the gap.

```powershell
.\.venv\Scripts\python.exe scripts\train_task3.py
.\.venv\Scripts\python.exe scripts\diagnose_task3.py
.\.venv\Scripts\python.exe scripts\verify_task3.py
.\.venv\Scripts\python.exe scripts\predict_task3.py --model lstm8_p1 --csv results\task3_supervised\lstm8_p1\example_window.csv
```

Training needs `0.tar` or the original development cache and the frozen protocol files. It resumes verified completed model artifacts. The verifier checks the original archive/logical data, training-only preprocessing, checkpoint selection, matched score arrays, all policies and standalone CSV inference for all nine models. The last command runs without the archive and returns an uncalibrated endpoint score; persistent alarm decisions require a chronological score stream. CSV scoring requires 24 consecutive timestamped 15-minute readings of all 22 original sensors, from one server. Causal filling is limited to four readings, with no history available before the uploaded window. All 24 readings must be resolved even when the model consumes only the final eight.

Weights, preprocessing, histories, scores, policies, decisions, per-node metrics, feature-selection decisions and three inspected plots are in `results/task3_supervised`. `experiment_plan.json` preserves the pre-training experiment and source hashes. `implementation_repairs.json` documents a plotting-only repair; the training configurations, saved weights, scores and policy search were unchanged. The dashboard still uses the earlier selected autoencoder. Repeated seeds, release selection, independent evaluation and report/PPT/dashboard integration remain later tasks.

## Task 2: threshold and persistence analysis

`results/task2_thresholds/REPORT.md` contains the validation-only comparison of 298 alarm policies on the same 71,982 endpoints. Four historical models contribute seven score variants. The search compares global/per-server normal percentiles, precision-recall-derived global thresholds, immediate decisions, two consecutive highs and two-of-three highs. It uses the frozen precision, recall, FPR, false-burst and incident-recall targets without relaxing them. The result is **no feasible policy** among the tested configurations; the current dashboard default remains unchanged. This diagnostic uses historical weights and does not replace the future controlled, calendar-aligned retraining comparison.

```powershell
.\.venv\Scripts\python.exe scripts\analyze_thresholds.py
.\.venv\Scripts\python.exe scripts\verify_task2.py
```

The first command loads the existing models and scores only the frozen validation cohort. It never trains a model or scores the reserved final nodes. The second reproduces all policy metrics from saved validation scores and checks the model/cohort fingerprints. The report, full policy table, Pareto frontier, score curves and alarm-budget detail plot are in `results/task2_thresholds`. Notification cooldown is evaluated separately and cannot improve the reported raw reading predictions.

## Task 1 audit and the new evaluation protocol

`results/task1_audit/REPORT.md` contains the validation-only false-alarm audit, sensor-quality findings and representative cases. It also documents fixed-policy rescoring at 15-minute cadence. These historical models were trained under the old row-based splits, so the rescoring is diagnostic and does not claim a model improvement or a fair ranking across different retained cohorts.

`results/protocol_v1/manifest.json` freezes the next experiments: shared calendar cutoffs, 15-minute scoring, a common 24-reading context for width-8/24 comparisons, training-only preprocessing and precision/false-alarm constraints. The seven reserved nodes are 2, 3, 4, 5, 6, 18 and 19. Only their schemas and timestamps have been inventoried; their sensor values, label values and predictions are reserved for final independent evaluation. The primary cohort uses a known endpoint target; unknown historical labels are not model inputs and do not exclude otherwise usable histories. Normal-only training/calibration additionally requires all history labels normal.

```powershell
.\.venv\Scripts\python.exe scripts\audit_task1.py
.\.venv\Scripts\python.exe scripts\verify_task1.py
```

This command needs the original `0.tar` archive. It reproduces the audit and checks the frozen manifest and logical input fingerprint. The generated NPZ cache checksum is informational because ZIP-container timestamps can differ across machines; the raw archive, logical data and cohort checksums fix the inputs. Changing an immutable protocol field requires a new version and a documented rationale. `artifacts/quality_indicators.parquet` preserves per-sensor missing, invalid and imputed flags as candidates for later experiments. No classifier training or threshold optimization occurs in Task 1.

## Cleaning and feature-selection workflow

The reproducible workflow is `scripts/clean_and_select.py`. It preserves the source archive and original experiments and creates `artifacts/cleaned_telemetry.parquet` (all 22 sensor inputs after cleaning) and `artifacts/selected_telemetry.parquet` (the correlation-selected inputs). Both exports preserve all rows, unknown labels and split assignments. Quality counts identify invalid readings, filled values and remaining missing values. These exports contain telemetry at 15-minute resolution, not preformed model windows.

Nonfinite values become missing. Negative fan speeds, power and temperatures become missing under a stated indoor-rack plausibility rule; this is a project assumption rather than a manufacturer-specified sensor range. Genuine zeros and high measurements remain. Filling uses only preceding readings, at most four, resetting at gaps and node/split boundaries. Long outages remain missing. The original source readings remain available for audit. Invalid sensor readings can themselves signal health anomalies, so cleaning is evaluated rather than assumed beneficial.

Feature selection uses only normal training rows. It removes inputs with over 20% missing readings, constants (standard deviation <=1e-8), and redundant inputs with absolute Pearson correlation >=0.95. The greedy order follows the original feature list. `results/feature_selection/feature_decisions.csv` records every decision and the retained counterpart for correlated inputs. Neither validation nor test statistics influence these decisions.

Four width-8 models compare the original 22 inputs, cleaned 22 inputs, correlation-selected inputs, and a predefined 12-input cooling subset. They use identical training, validation and test endpoints, seed, training settings and architecture; parameter counts change with input dimensionality. Validation endpoint F1 selects the configuration. Test results remain exploratory because the recording was already inspected during development. The workflow retains the current dashboard default unless the new validation score improves it.

```powershell
.\.venv\Scripts\python.exe scripts\clean_and_select.py --prepare-only
.\.venv\Scripts\python.exe scripts\clean_and_select.py
.\.venv\Scripts\python.exe scripts\build_results.py
.\.venv\Scripts\python.exe scripts\verify_artifacts.py
```

The first command creates cleaned datasets, selection decisions and the common cohort without training. The second runs the full four-model comparison (up to 30 epochs each). Detailed measured results and limitations are in `results/feature_selection/REPORT.md` and `comparison.csv`. The final PDF and PPT describe the earlier experiments; the feature-selection report supplements them.

## Original and enhanced feature definitions

The 22 base features match the original reduced telemetry: ambient temperature; eight fan speeds; fan disk power; two CPU voltage-regulator temperatures; mean DIMM, CPU and GPU temperatures; processor and total power; and two power-supply input powers.

Enhanced24 adds eleven features:

- `cpu_temp_max`: maximum available CPU core `_temp_max` measurement across both processors.
- `gpu_temp_max`: maximum available GPU core `_temp_max` measurement.
- `fan_min` and `fan_spread`: minimum fan `_avg` reading and maximum-minus-minimum across eight fans.
- `cpu_above_ambient`: maximum of the two mean processor core temperatures minus ambient.
- `gpu_above_ambient`: mean GPU core temperature minus ambient.
- `cpu_temp_change`, `fan_change`, `power_change`: difference from the preceding 15-minute reading for p0 core temperature, fan0_0 and total power. Differences reset at time gaps.
- `ps0_output_voltage`, `ps1_output_voltage`: the original power-supply output voltage averages.

## Reproduce the experiments

```powershell
python scripts\evaluate_saved.py
python scripts\train_baseline.py
python scripts\train_experiments.py --epochs 30 --configs base24 base8 base48 enhanced24 robust24
python scripts\build_results.py
```

The default neural training settings are hidden size 64, latent size 16, Adam 0.001, batch size 512, seed 42, up to 30 epochs and patience five on normal validation MSE. The scripts use GPU if the installed PyTorch runtime exposes one; otherwise they use CPU with four threads. The delivered runs use CPU. Floating-point implementations and package versions can change numerical results.

Each model evaluates whole-window versus latest-reading error, validation normal percentiles 95/99/99.5, global versus per-node thresholds, and one versus two consecutive high scores. Endpoint F1 on labelled validation selects settings; ties use raw incident recall then a lower false-burst rate. Model selection also uses validation. Training loss and test metrics do not select the model.

The original TensorFlow scores are separately audited and explicitly labelled as an original reference. The new PyTorch runs are independently trained; their weights are not converted TensorFlow checkpoints.

## Evaluation definitions

`y_window` is positive if any reading in the input window is CRITICAL. `y_end` uses only the label at the last timestamp. These targets have different prevalence and are reported separately.

Raw incidents are consecutive original CRITICAL 15-minute bins on one node. Non-CRITICAL labels and missing timestamps split incidents. An alarm detects an incident only if its timestamp falls inside an original incident bin. Delay starts at the incident's first bin. Matching after an incident has ended does not count.

Both total and observable incident recall are reported. Observable incidents contain at least one scored endpoint. Unobservable incidents stay in the total denominator. Historical runs use 30-minute scored intervals; Tasks 1–3 explicitly use 15 minutes. Alarm bursts and persistence reset at a node change or any gap relative to that experiment's cadence. A scored node-day is scored-bin count multiplied by cadence/1440. This excludes missing exposure and is not a wall-clock uptime estimate. False alarms are relative to Nagios labels and may include unlabelled unusual behaviour.

AP is `average_precision_score`, not trapezoidal PR-AUC. Endpoint and per-node results, raw incident records, false bursts per scored node-day and median delay are included. Width and feature changes can change coverage. The existing recordings were inspected during earlier development, so conclusions remain exploratory despite validation-only new setting selection.

## Check the project

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
python scripts\verify_artifacts.py
```

The tests check gap-aware persistence, incident timing and coverage denominators. The artifact verifier checks saved-model inference against recorded scores, rejects irregular input timestamps, checks finite metrics and inspects result files. Only load the trusted model/scaler files supplied with this project; joblib files use Python serialisation.

## Scope and next work

The detector is a course prototype. Its labels cover general server health, its cohort is a single rack subset and runs use one seed. Several alarm policies still have low precision. Prospective evaluation, alert-specific cooling labels, more nodes, repeated seeds and operational false-alarm review are needed before deployment. The report and presentation include failures and competing baseline strengths.
