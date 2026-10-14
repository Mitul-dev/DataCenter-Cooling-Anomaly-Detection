# HVAC benchmark preflight and frozen protocol

The source archive contains six collections. Selection was based on label/independent-day availability and documented provenance, before model fitting. RTU has no native normal labels; SZCAV has only one healthy day; the smaller collections have fewer independent runs. The largest MZVAV-1 file is selected and its native labels are verified against the inventory for every date. Labels are not corrected or synthesized.

The task is sensor-bias detection in an operating simulated AHU, not confirmed datacenter cooling failure. Occupied/fan-on filtering is declared before training and applied identically to all partitions. Nights and nonoperating days are reported as unscored; all raw condition days remain in the fault-day coverage denominator.

| Partition | Seasons | Raw days | Scored days | Endpoints | Fault endpoints |
|---|---|---:|---:|---:|---:|
| train | winter + spring | 91 | 77 | 61187 | 52446 |
| validation | summer | 49 | 42 | 33432 | 28656 |
| test | autumn | 49 | 42 | 33432 | 28656 |

Scenarios and whole seasons do not cross partitions. No shuffled overlapping-window evaluation is allowed. The final season is reserved until preprocessing, checkpoint and policy selection are frozen. Source times are local/unspecified, not asserted as UTC. Temperature inputs are Fahrenheit; imposed sensor-bias levels in the inventory are Celsius.

The physical consistency features use measurements/control commands only. They do not determine labels. Model training uses the author-provided ground truth. Scenario/calendar fields are metadata only and never model inputs.

The dataset is strongly fault-heavy because it was designed for equipment-fault experiments. Accuracy alone is misleading: an all-fault classifier would have high accuracy but 100% FPR. Endpoint precision, recall, FPR, AP, false bursts and observed/total fault-day coverage are required. No operational prevalence claim is made.

[Official data record](https://data.openei.org/submissions/910). Fault/unit/schedule definitions are in the accompanying source inventory, pages 3–7.
