# LBNL dual-duct AHU simulation source

Official dataset: https://faultdetection.lbl.gov/dataset/simulated-dd-ahu-dataset/

Collection DOI: https://doi.org/10.25984/1881324

Reference: Granderson et al., A labeled dataset for building HVAC systems operating in faulted and fault-free states, Scientific Data 10, 342 (2023), https://doi.org/10.1038/s41597-023-02197-w. Table 4 documents temperature and pressure sensor biases at cold/hot supply decks.

The unmodified public archive was downloaded on 16 October 2026. Its URL, size and SHA-256 are in download.json. SHA-256: `bac73c863ca4ecad52846dbdd41d5daf9d844619e928f01b9eb18c1546d45c50`. Compressed size: 1,775,170,872 bytes. It contains 56 CSVs; 17 native healthy/sensor-bias scenarios are used, each with 525,600 rows in 2018.

The archive's 114 recorded points include measurements, controls and setpoints. This study declares 26 AHU inputs; the complete alias list and native scenario mappings are in results/multisensor_v1/experiment_plan.json and scenarios.json. Source filenames provide the author-defined class targets, never model features. The pressure filename tokens differ from paper Table 4: native +/-2inwg and +/-4inwg versus documented +/-0.2 and +/-0.4 inH2O. Native identifiers are retained; independently verified numerical pressure severity is not claimed.

Raw archive and cached parquet files are excluded from Git and submission ZIPs due to size. Run `python scripts/download_sda.py --dual` then `python scripts/prepare_multisensor.py` to reproduce local caches. The saved demonstration needs only bundled weights, preprocessing, input examples and replay.

The proposed single-duct source was rejected after its duplicate/missing-filename audit. That audit remains in results/sda_source_audit. Those faulty files are not used in the four-sensor models. The original outdoor-air benchmark retains its independent older source and scope.
