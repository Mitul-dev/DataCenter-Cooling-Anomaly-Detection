# Data center telemetry anomaly detection

Deep Learning project, Section B. Team: Rishabh Srivastava (2430010149), Mitul Seth (2430010099), Madhav Mittal (2430010096).

The project trains an LSTM autoencoder on normal thermal, fan and power telemetry and compares it with Isolation Forest. The reference labels are general Nagios server-health alerts. The evaluated task does not establish cooling-specific diagnosis or prediction of future faults.

## Submission files

- `submission/DCCAD_Final_Report.pdf`: final report with verified experiments, definitions, limitations and references.
- `submission/DCCAD_Final_Presentation_Reviewed.pptx`: final presentation with editable evidence tables and charts.
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

Each node uses a chronological 70/10/20 row split. Short gaps are filled forward at most four rows, with filling reset at timestamp discontinuities. Scalers fit finite normal training rows only. Windows never cross split boundaries or node boundaries. Known normal/CRITICAL labels and finite, consecutive input readings are required. Training windows contain no CRITICAL readings. Normal training is label-assisted; labelled validation selects the final alarm policy.

Standard experiments clip scaled inputs to [-5, 5]. Robust24 uses median/IQR scaling with no clipping. Original recorded readings are kept separately for dashboard plots, so inverse scaling does not incorrectly present clipped values as raw measurements.

## Data cleaning and feature selection

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

Both total and observable incident recall are reported. Observable incidents contain at least one scored endpoint. Unobservable incidents stay in the total denominator. Alarm bursts and persistence reset whenever consecutive scored timestamps are not 30 minutes apart. A scored node-day is scored-bin count multiplied by 30/1440. This excludes missing exposure and is not a wall-clock uptime estimate. False alarms are relative to Nagios labels and may include unlabelled unusual behaviour.

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
