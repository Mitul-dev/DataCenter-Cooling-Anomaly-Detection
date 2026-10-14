# Recovery step after the failed release gate

Task 4 completed its robustness experiment and release decision, but no seed has a feasible policy. The independent evaluation step is therefore deferred. The final servers and dates, natural class prevalence and original target gate remain unchanged.

The next stage should declare a new, bounded recovery experiment before fitting it. This is a documented follow-up after the original six-configuration search failed, rather than silently extending that search or changing its results.

1. Check the original reference-label construction and whether fault categories align with physical cooling/electrical telemetry. Keep Nagios labels as the outcome; do not manufacture cooling-failure labels from temperatures. Record unavailable alert-specific metadata as a limitation.
2. Declare one compact extended sensor set using the training-only inventory: power-supply voltage/current, memory/I/O power, within-interval sensor variability and extrema. Confirm source meanings and validity assumptions. Do not choose features using final-server measurements or labels.
3. Fit one extended LSTM and one extended tree baseline with frozen training and validation boundaries. Retain the same endpoint cohort for the primary comparison. Use training-only handling of any additional missing inputs and expose unavailable-sensor indicators; do not silently discard hard validation readings.
4. Keep the original class distribution and alarm constraints. If ranking and the operating point improve, repeat the revised finalist, freeze its full pipeline and evaluate the reserved servers once. If the recovery experiment still fails, report that the general health labels are not adequately predictable from the tested telemetry. Any dataset or project-scope replacement is a separate decision.

Availability of extra sensors does not guarantee improvement. The three-seed results rule out claiming the present classifier is ready for submission as an accurate detector. They support a reproducible negative result and a focused investigation into the input/label mismatch.
