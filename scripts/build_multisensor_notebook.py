"""Execute a concise notebook against the saved, frozen multi-sensor evidence."""
import base64
import contextlib
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

ROOT=Path(__file__).resolve().parents[1]
cells=[];namespace={};execution=0


def markdown(text):
    cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True)))


def code(source,figure=False):
    global execution
    execution+=1;stream=io.StringIO()
    with contextlib.redirect_stdout(stream):exec(compile(source,'HVAC_multisensor_submission.ipynb','exec'),namespace)
    output=stream.getvalue();outputs=[]
    if output:outputs.append(dict(output_type='stream',name='stdout',text=output.splitlines(keepends=True)))
    if figure:
        image=io.BytesIO();namespace['fig'].savefig(image,format='png',dpi=125,bbox_inches='tight')
        outputs.append(dict(output_type='display_data',metadata={},data={'image/png':base64.b64encode(image.getvalue()).decode(),'text/plain':['Executed evidence figure']}))
        namespace['plt'].close(namespace['fig'])
    cells.append(dict(cell_type='code',execution_count=execution,metadata={},source=source.splitlines(keepends=True),outputs=outputs))


def main():
    markdown('# Fault detection across four sensors\n\nDSE 3170 PBL 3, Section B. Rishabh Srivastava, Mitul Seth and Madhav Mittal.\n\nThis extension identifies cold-deck/hot-deck temperature and pressure sensor biases in one simulated dual-duct AHU. The fifth outcome is healthy operation. The earlier outdoor-air benchmark remains separate. Each new native scenario contains one fault; simultaneous faults are untested.\n\nSource: [LBNL dual-duct AHU](https://faultdetection.lbl.gov/dataset/simulated-dd-ahu-dataset/). Collection DOI 10.25984/1881324. Fault definitions: Scientific Data 10, 342 (2023), DOI 10.1038/s41597-023-02197-w, Table 4.\n\nThe reserved test is already inspected. This notebook reads saved results and performs fresh inference, without retuning thresholds.')
    code('from pathlib import Path\nimport json, sys\nimport numpy as np\nimport pandas as pd\nimport torch\nROOT = Path.cwd()\nsys.path.insert(0, str(ROOT / "src"))\nR = ROOT / "results/multisensor_v2"\ntorch.set_num_threads(4)\nsummary = json.loads((R / "test_summary.json").read_text())\nevidence = json.loads((R / "evidence_summary.json").read_text())\nfrozen = json.loads((R / "frozen_model.json").read_text())\nprint("Primary:", summary["primary_family"], "seed", summary["seed"])\nprint("Classes:", *summary["classes"], sep="\\n")')
    markdown('## Source audit and calendar separation\n\nWe selected one healthy file and sixteen native sensor-bias files from a 56-CSV archive. Each file represents the same modeled unit under a different condition. Matching ISO calendar weeks stay in the same partition across every scenario: modulo six equal to zero tests, five validates, and all other values train. Thirty one-minute readings must remain operating, contiguous and in one source day. Temperatures use Fahrenheit. Pressure filename severity tokens differ from the paper, so verified numerical pressure severities are not claimed.\n\nAn earlier single-duct source was rejected because outdoor-bias tables were identical and the documented supply-bias filenames were missing. The audit remains in the repository.')
    code('counts = pd.read_csv(ROOT / "results/multisensor_v1/cohort_counts.csv")\nprint(counts.groupby("partition").endpoints.sum().to_string())\nprint("Hourly training windows:", evidence["sampled_train_windows"])\nprint("Quarter-hour checkpoint validation windows:", evidence["checkpoint_validation_windows"])\nprint("All-minute policy validation and test endpoints are retained.")')
    markdown('## Input measurements and five-input comparison\n\nThe wider model uses 21 measurements and five commands. Cleaning/scaling fit only healthy active training rows, causal filling resets at day/gap boundaries, and labels/filenames never enter the model. A training-only tree selector uses whole-day grouped permutation on internal held-out training weeks. The five candidates comprise four measurements and one command. Rankings remain conditional on the selector and correlated inputs.')
    code('ranking = pd.read_csv(ROOT / "results/multisensor_v1/training_feature_ranking.csv")\nprint(ranking.head(5).to_string(index=False))\nfrom dccad.multisensor import SensorClassifier\ncheckpoint = torch.load(R / summary["primary_family"] / "model.pt", map_location="cpu", weights_only=True)\nmodel = SensorClassifier(len(checkpoint["features"]), len(checkpoint["classes"]))\nmodel.load_state_dict(checkpoint["state_dict"])\nmodel.eval()\nprint(model)\nprint("Parameters:", sum(p.numel() for p in model.parameters()))')
    markdown('## Reserved test results\n\nVersion 1 tested two input sets for up to 20 epochs. Before test inference, Version 2 recorded a longer training budget and increased healthy-class loss weight. Four unique LSTMs and one fixed seed were used. Final selection compared 126 validation policies per candidate, or 504 policies, then froze the primary model and thresholds. The full-input Version 2 model won on validation.\n\nHealthy FPR is FP divided by all healthy minutes. Correct-sensor recall requires the matching faulty sensor label. Any-fault recall only requires raising some fault alarm.')
    code('m = summary["primary_test"]\nprint(pd.DataFrame(m["per_class"])[["sensor", "support", "precision", "recall", "f1"]].to_string(index=False))\nprint("Accuracy:", "{:.2%}".format(m["accuracy"]))\nprint("Macro F1:", "{:.2%}".format(m["macro_f1"]))\nprint("Healthy FPR:", "{:.2%}".format(m["healthy_fpr"]), "from", m["healthy_false_readings"], "/", m["healthy_readings"])\nassert not summary["test_used_for_selection"]\nprint("The 1% FPR target is unmet:", not summary["test_targets_met"])')
    code('import matplotlib.pyplot as plt\nmatrix = np.asarray(m["confusion_matrix"])\nfig, ax = plt.subplots(figsize=(7, 5))\nax.imshow(matrix / matrix.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)\nfor i in range(5):\n    for j in range(5):\n        ax.text(j, i, f"{matrix[i,j]:,}", ha="center", va="center", color="white" if i==j else "black", fontsize=9)\nnames = ["Healthy", "Cold temp", "Hot temp", "Cold pressure", "Hot pressure"]\nax.set_xticks(range(5), names, rotation=25, ha="right")\nax.set_yticks(range(5), names)\nax.set(xlabel="Prediction", ylabel="Native scenario condition", title="Reserved test confusion counts")\nfig.tight_layout()',figure=True)
    code('comparison = pd.read_csv(R / "model_comparison.csv")\nprint(comparison[["model", "inputs", "accuracy", "macro_f1", "healthy_fpr", "selected_before_test"]].to_string(index=False))\nprint("Majority five-class baseline:", "{:.2%}".format(evidence["five_class_majority_baseline_accuracy"]))\nprint("Experimental fault prevalence:", "{:.2%}".format(evidence["test_fault_prevalence"]))')
    markdown('## Fresh CSV inference\n\nThe saved primary uses an immediate alarm policy. Each faulty sensor has its own validation threshold. If multiple sensors pass, the larger class score wins. Softmax scores remain uncalibrated. Reference labels do not affect the scorer.')
    code('from dccad.multisensor import score_window\nfolder = R / summary["primary_family"]\nwindow = pd.read_csv(folder / "example_class_2.csv")\nscores = score_window(folder, window)\nexpected = json.loads((folder / "fresh_window_checks.json").read_text())[2]["scores"]\nnp.testing.assert_allclose(list(scores.values()), list(expected.values()), rtol=2e-5, atol=2e-6)\nwindow["fault_label"] = "incorrect metadata"\nassert score_window(folder, window) == scores\nprint(pd.Series(scores, name="Uncalibrated class score").to_string())\nprint("Frozen thresholds:", frozen["policy"]["thresholds"])')
    markdown('## Reproducibility and limitations\n\nThe source archive downloads separately for retraining. Saved models, replay data and examples run offline. Start Dashboard.bat opens the four-sensor extension; the sidebar preserves the earlier outdoor-air experiment.\n\nThe 2.10% test healthy FPR exceeds the original 1% goal. There are 52 false bursts across 40 active healthy calendar days. Five selected raw inputs perform worse. One simulated unit, overlapping minutes, known bias severities and one fixed seed limit generalization. No field validation, simultaneous-fault coverage or future-failure prediction is established. Do not retune on the already-inspected test and call it a fresh result.')
    code('checks = json.loads((R / "verification.json").read_text())\nassert checks["status"] == "passed"\nprint("Complete score arrays reproduced:", checks["complete_score_arrays"])\nprint("Validation policies reproduced:", checks["validation_policies_reproduced"])\nprint("Fresh CSV checks:", checks["fresh_csv_and_label_exclusion_checks"])\nprint("Next evaluation requires new independent reserved data.")')
    nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
            language_info=dict(name='python',version='3.11'),execution_note='All code cells executed sequentially against saved project evidence.'),nbformat=4,nbformat_minor=4)
    path=ROOT/'HVAC_multisensor_submission.ipynb';path.write_text(json.dumps(nb,indent=1),encoding='utf-8')
    print('Created executed notebook with',execution,'code cells.')


if __name__=='__main__':main()
