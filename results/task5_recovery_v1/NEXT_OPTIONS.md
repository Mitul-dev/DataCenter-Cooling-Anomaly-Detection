# Submission choices after recovery v1

The recovery experiment is complete and failed the original target gate. The additional-input LSTM's diagnostic policy gives precision 1.93%, recall 40.02% and FPR 25.87%, versus 2.98%, 41.91% and 17.29% for the original matched seed-42 LSTM. No recovery policy is feasible. These are validation development results, not independent-test accuracy.

## Recommended: a cooling-equipment benchmark

Keep this repository's existing experiments as a documented investigation, and build a separately versioned primary experiment on a dataset with equipment-fault scenarios defined by its authors. A concrete candidate is the [Lawrence Berkeley National Laboratory building fault-detection evaluation collection](https://data.openei.org/submissions/910), DOI 10.25984/1824861. Its official record lists an approximately 6.69 MB CSV archive plus an approximately 1.17 MB inventory, covering air-handling and rooftop equipment from laboratory/simulation sources.

This would change the principal scope to **cooling-equipment fault detection**, not demonstrate datacenter server cooling performance. Confirm archive access, documented case labels, sensor coverage and sufficient independent runs before committing to the switch. Equipment/case/day groups must be split before window creation; no random overlapping-window split. Scenario labels must come from the inventory, not be inferred from sensor thresholds. Report experimental prevalence and simulation/laboratory provenance explicitly.

If that preflight succeeds, use one compact supervised LSTM and a tree baseline, fit preprocessing on training groups only, choose alarm settings on separate validation groups, freeze the pipeline and evaluate untouched groups once. Reuse the project's evaluation discipline and demo structure. Good performance is a hypothesis to test, not a promise.

## Keep M100 and finish an honest comparative study

Retain the current dataset and refine the title/task to current batch-client state classification from physical telemetry. Rebuild the report, slides and demo around the measured failures, feature ablations, three-seed variation, source-label audit and limitations. This can be a defensible research submission, but it cannot claim that the false-positive requirement has been solved, and it may not meet the professor's expectations described by the user.

Further M100 architectures, PCA variants, extra loss weights or threshold searches would extend the agreed experiment budget. They should not be added automatically to avoid reporting this negative result. Independent M100 test servers remain reserved for any later qualified pipeline.

The choice changes the submission's dataset and claims, so it requires the user's direction. No replacement dataset has been downloaded or substituted, and the old artifacts remain intact.
