# Cooling Equipment Sensor-Fault Detection with an LSTM

Deep Learning project, Section B. Team: Rishabh Srivastava (2430010149), Mitul Seth (2430010099), Madhav Mittal (2430010096).

This course project now identifies faults at **four sensor locations** in a simulated dual-duct air-handling unit: cold-deck and hot-deck air temperature, and cold-deck and hot-deck duct pressure. Healthy operation is the fifth model class. Each native scenario contains one sensor bias. The earlier outdoor-air detector and M100 investigation remain separate supporting studies.

## Current four-sensor extension

The LBNL dual-duct source provides 114 recorded points. This study uses 26 AHU inputs: **21 measurement channels and five control commands**. Seventeen selected source files cover a full year. Matching calendar weeks stay together across all scenarios. A fixed-seed LSTM uses 30-minute histories; training-only feature selection also tests five raw inputs.

| Reserved test metric | Result |
|---|---:|
| Five-class accuracy | 92.87% |
| Macro F1 | 89.36% |
| Any-fault detection recall | 93.05% |
| Correct-sensor recall range | 87.52% to 95.88% |
| Healthy false-positive rate | **2.10%** |
| Healthy false readings | 580 / 27,640 |

**The 1% healthy FPR goal remains unmet.** The selected full-input model passed validation at 0.90% FPR, then missed the target on reserved test weeks. Five selected inputs performed worse and failed validation. The test is already inspected and was not used to retune thresholds. Broader fault coverage is established only within this simulation; these are different targets/data from the earlier outdoor-air benchmark, so its metrics are not a matched comparison.

- [Updated PBL report with retained cover and certificate](submission/HVAC_Multisensor_PBL_Report.pdf)
- [Editable updated report](submission/HVAC_Multisensor_PBL_Report.docx)
- [Updated 12-slide presentation](submission/HVAC_Multisensor_Presentation.pptx)
- [Executed multi-sensor notebook](HVAC_multisensor_submission.ipynb)
- [Complete methods and results](results/multisensor_v2/REPORT.md)
- [Model card](results/multisensor_v2/MODEL_CARD.md)
- [Verification](results/multisensor_v2/verification.json)

The dashboard opens the four-sensor extension by default. Its sidebar retains the earlier outdoor-air benchmark. Saved models, examples and replay run offline. The **1.7 GB raw dual-duct source archive and local caches are excluded from Git and the submission ZIP**; retraining downloads the public source separately:

```powershell
python scripts/download_sda.py --dual
python scripts/prepare_multisensor.py
python scripts/train_multisensor.py
python scripts/refine_multisensor.py
python scripts/evaluate_multisensor.py --study multisensor_v2
python scripts/verify_multisensor.py
```

The logged development compares four unique LSTMs, fixed seed 42, two input sets and 504 final validation policies. Verification reproduces eight score arrays and twenty fresh CSV examples; all 46 unit tests pass. Only one seed is evaluated for this extension. Inactive/shutdown periods, simultaneous faults, new physical units and unseen severities are outside the measured scope. See [source attribution](data/dda_benchmark/SOURCE.md).

## Earlier outdoor-air benchmark results

Seed 42 was declared primary before evaluation. Its 30-minute history and two-of-three alarm rule were frozen using summer validation before the reserved autumn evaluation.

| Autumn metric | Result |
|---|---:|
| Accuracy | 92.82% |
| Alarm precision | 99.74% |
| Fault-reading recall | 91.86% |
| F1 | 95.64% |
| False-positive rate | 1.44% |
| TP / FP / FN / TN | 26,324 / 69 / 2,332 / 4,707 |
| Observable fault days detected | 35 / 36 |
| All raw fault days detected | 35 / 42 |

**The 1.44% test FPR exceeds the original 1% goal.** The 69 false readings form one burst. False discovery ratio is 0.26%, with a different denominator than FPR. No threshold was adjusted after evaluation. Seed 7 has higher test accuracy but does not replace the predeclared primary. The tree has zero false positives and lower recall (72.05%); all models are reported.

This is one simulated AHU and one fault family. Experimental fault prevalence is 85.71%, so an all-fault prediction would already score 85.71% accuracy with 100% FPR. The results do not establish deployment reliability, future-fault prediction or general cooling-failure diagnosis. Healthy validation/test each cover one scenario week; adjacent minutes and days are correlated.

## Submission

- [PBL report with university cover and certificate](submission/HVAC_PBL_Project_Report.pdf)
- [Editable PBL report](submission/HVAC_PBL_Project_Report.docx)
- [Earlier technical report](submission/HVAC_DL_Project_Report.pdf)
- [14-slide editable PowerPoint](submission/HVAC_DL_Project_Presentation.pptx)
- [Executed notebook with figures](HVAC_submission.ipynb)
- [Model card](results/hvac_v1/MODEL_CARD.md)
- [Results and methodology](results/hvac_v1/REPORT.md)
- Local bundle: `submission/HVAC_Project_Submission.zip`, recreated with `python scripts/package_hvac.py`. Generated submission ZIPs are excluded from Git.

The ZIP includes the updated multi-sensor submission, trained models, replay/evidence, source code and the earlier submission. Its original small outdoor-air source archive is included. The larger dual-duct source downloads separately. Supporting M100 reports are included; complete historical experiments remain in the GitHub repository.

## Run the dashboard

Python 3.11.9 on Windows with CPU inference was tested. From this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

If dependencies are already installed, double-click `Start Dashboard.bat`. If port 8501 is occupied:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.port 8502
```

Open the address Streamlit prints. The four-sensor dashboard replays reserved calendar weeks and scores an uploaded example with exactly 30 consecutive operating one-minute readings. Temperatures use Fahrenheit. Labels and imposed bias never enter the model. Its selected immediate policy uses uncalibrated class scores and per-sensor thresholds. The separately selectable earlier study retains its two-of-three policy.

## Data and separation

Source: [LBNL AFDD evaluation collection](https://data.openei.org/submissions/910), DOI [10.25984/1824861](https://doi.org/10.25984/1824861). Unchanged archive and inventory are included in `data/hvac_benchmark`; attribution and selection are in [SOURCE.md](data/hvac_benchmark/SOURCE.md).

`MZVAV-1.csv` contains 272,160 one-minute readings across 189 source days, with native condition labels and outdoor-air sensor biases of +/-1, +/-2 and +/-4 Celsius. All daily labels match the published inventory. Measurement temperatures are Fahrenheit; source local timestamps have unspecified timezone.

| Partition | Seasons | Raw / scored days | Scored endpoints |
|---|---|---:|---:|
| Training | Winter + spring | 91 / 77 | 61,187 |
| Validation | Summer | 49 / 42 | 33,432 |
| Final test | Autumn | 49 / 42 | 33,432 |

Whole source days and scenario blocks stay in one partition. A window requires 30 consecutive occupied, fan-on minutes and resets at days or gaps. Nights and nonoperating days are unscored; raw fault days remain in coverage denominators. This operating-mode task was frozen before fitting.

Thirteen raw temperature/control/pressure inputs, six consistency features and thirteen missing indicators provide 32 channels. Negative outdoor Fahrenheit readings remain valid. Filling uses at most five previous minutes within the same day/contiguous segment. Remaining missing values and scaling use normal active training data only. Mixing consistency uses damper command as a proxy, not measured airflow fraction.

## Reproduce and verify

Using the installed environment (`python` below means its interpreter):

```powershell
python scripts/prepare_hvac.py
python scripts/train_hvac.py
python scripts/verify_hvac.py --validation-only
python scripts/evaluate_hvac.py
python scripts/verify_hvac.py
python scripts/build_hvac_evidence.py
python scripts/run_tests.py
python scripts/package_hvac.py
```

Saved complete runs and final evaluation are reused after integrity checks. Evaluation does not retune on the already-inspected autumn data. New experiments require a new version and newly reserved evaluation data.

Verification reproduces four complete validation/test score arrays, all 120 validation policies, training-only preprocessing, source-label alignment and four fresh CSV examples. The 39 unit tests and dashboard replay checks passed. [Runtime](results/hvac_v1/environment.json), [protocol](results/hvac_v1/experiment_plan.json) and [verification](results/hvac_v1/verification.json) document the evidence.

View the executed notebook on GitHub. To rerun interactively, optionally install JupyterLab into this environment (`python -m pip install jupyterlab`) and launch `python -m jupyterlab`. Source data and models are included.

## Professor feedback on more data and five inputs

The separate [sensor-expansion plan](SENSOR_EXPANSION_PLAN.md) documents the 16 October review. A [three-seed validation comparison](results/hvac_feature_v2/REPORT.md) tested the existing 32 channels against thirteen raw inputs and five training-selected raw inputs. Neither reduced raw-input set met the validation gate; the submitted model remains the same. This is a five-raw-input ablation, not a claim about every possible five-engineered-feature model. No new autumn predictions were made.

The original file has no additional varying measurement channels to add. The initially proposed annual single-duct source failed its integrity audit. The completed multi-sensor extension therefore uses the separate dual-duct source described above. Both raw-five studies are retained as evidence that fewer inputs do not automatically improve accuracy or false positives.

## Supporting M100 investigation

[Historical documentation](docs/M100_RESEARCH.md) describes the original project. Tasks 1-5 preserve split audits, threshold experiments, supervised comparisons, seed repetition, feature recovery and a source-label audit under `results/task*_*/`. Seven reserved M100 servers remain unscored. Original reports/notebooks retain historical names; use the HVAC files above for the current submission. The archived dashboard is `dashboard/m100_app.py` and requires historical artifacts from the full repository.
