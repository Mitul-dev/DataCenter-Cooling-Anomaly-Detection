# Public AFDD source data

Downloaded on 14 October 2026 after the user chose the cooling-equipment benchmark direction.

- Collection: *Data Sets for Evaluation of Building Fault Detection and Diagnostics Algorithms*, Lin and Mitchell, Lawrence Berkeley National Laboratory (2019).
- DOI: [10.25984/1824861](https://doi.org/10.25984/1824861)
- Official record and license information: [OEDI submission 910](https://data.openei.org/submissions/910)
- `afdd_source.zip`: original CSV collection, 7,010,416 bytes.
- `source_inventory.pdf`: original inventory, 1,222,383 bytes; identical inventory is also bundled inside the source ZIP.

The project does not extract arbitrary ZIP paths. It reads the named `MZVAV-1.csv` member. Neither source archive nor inventory is modified. Their checksums and the verified scenario dates are in `results/hvac_v1/experiment_plan.json`.

The primary file is a simulation of outdoor-air temperature sensor bias in a multi-zone variable-air-volume AHU. Native 0/1 condition labels are retained and verified against inventory Table 1-2. This is a cooling-equipment sensor-fault benchmark, not a datacenter field-failure dataset.
