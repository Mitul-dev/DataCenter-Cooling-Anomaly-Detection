# Model card: operating AHU sensor-bias detector

**Intended use:** a deep-learning course prototype for current outdoor-air temperature sensor-bias detection in an occupied, running simulated air-handling unit. This is the primary study following the user's approved change from the M100 dataset. It is not a general cooling-failure diagnostic or early-warning deployment.

**Source:** LBNL AFDD collection, DOI 10.25984/1824861, `MZVAV-1.csv`. The archive/inventory remain unchanged in `data/hvac_benchmark`. All native daily condition labels match the published scenario dates. Measurements are Fahrenheit; imposed bias intensity is Celsius. Source timestamps are local/unspecified.

**Inputs:** 30 consecutive one-minute readings, wholly within one occupied/fan-on source day. Thirteen raw temperature/control/pressure inputs, six consistency features and thirteen original-missing flags give 32 channels. Labels, calendar, day identifiers, scenario and imposed intensity are excluded. The mixing residual uses damper command as a proxy, not measured flow fraction.

**Preprocessing:** nonfinite inputs become missing; negative outdoor temperatures remain. Forward-fill at most five minutes with resets at day/gaps. Remaining missing inputs use normal active training medians. Standard scaling uses those normal training observations only; clip at ten standard deviations. Severe outage/missing patterns have not been independently validated by this simulation benchmark.

**Architecture/training:** LSTM hidden size 32, 16-unit ReLU head, dropout 0.2 and one logit (8,993 trainable parameters). Adam 0.001, batch 512, gradient clipping one, at most 30 epochs, validation AP patience six. Positive BCE weight equals the training normal/fault ratio (about 1/6). Seeds 7/42/123 are reported; seed 42 was predeclared as primary.

**Separation:** winter/spring training (91 raw days, 61,187 eligible endpoints); summer validation (49 raw days, 33,432 endpoints); autumn test (49 raw days, 33,432 endpoints). Scenario blocks and days never cross partitions. Nights/nonoperating days are outside the scoring task; their fault-day coverage remains disclosed.

**Frozen policy:** logit >6.967112, at least two highs in three consecutive scored minutes. Reset at days, units and gaps. One uploaded window produces only one endpoint score and cannot establish persistence. No probability calibration is claimed.

**Independent autumn results:** accuracy 92.82%, precision 99.74%, recall 91.86%, F1 95.64%, AP 0.9962, FPR 1.44%, false discovery ratio 0.26%. TP=26,324; FP=69; FN=2,332; TN=4,707. False readings form one burst. Detected fault days: 35/36 observable, 35/42 total. Median delay: 392 minutes from midnight imposition; two minutes from first eligible endpoint. Test FPR exceeds the original 1% goal. No post-test tuning occurred.

**Baseline/tradeoff:** the same-history tree has 72.05% recall and zero test false positives. The primary LSTM trades a small false-alarm cost for greater sensitivity. The highest test-scoring neural seed does not replace primary seed 42.

**Limits:** one simulated unit and fault family, experimental prevalence 85.71%, one healthy scenario week in each validation/test partition, and correlated adjacent days/readings. An all-fault predictor would have 85.71% accuracy and 100% FPR. This does not estimate real deployment prevalence or reliability on another physical unit. No binomial interval over overlapping minutes is presented as an independent reliability estimate.

**Verification:** all four full validation/test score arrays, all 120 validation policies, native-label/scenario alignment, training-only transforms and fresh CSV examples reproduce. Thirty-nine unit tests pass. The model/checkpoint/policy hashes preceded the one-time autumn evaluation. Existing final results are returned on rerun; a revised investigation needs a new version/evaluation design.

**Demo and documents:** `dashboard/app.py`, `HVAC_submission.ipynb`, `submission/HVAC_DL_Project_Report.pdf` and `submission/HVAC_DL_Project_Presentation.pptx`. The original M100 artifacts and source archive remain intact as supporting research.
