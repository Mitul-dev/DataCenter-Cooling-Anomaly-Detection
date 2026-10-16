"""Build figures and written evidence directly from the fixed multi-sensor results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/multisensor_v2'
SHORT=['Healthy','Cold temp','Hot temp','Cold pressure','Hot pressure']


def main():
    summary=json.loads((OUT/'test_summary.json').read_text());metric=summary['primary_test']
    frozen=json.loads((OUT/'frozen_model.json').read_text());table=[]
    for family,metrics in summary['all_families'].items():
        folder=OUT/family;model=json.loads((folder/'summary.json').read_text())
        policy=json.loads((folder/'validation_policy.json').read_text())
        table.append(dict(model=family,inputs=model['channels'],parameters=model['parameters'],epochs_run=model['epochs_run'],
                          best_epoch=model['best_epoch'],validation_macro_f1=policy['validation_metrics']['macro_f1'],
                          validation_healthy_fpr=policy['validation_metrics']['healthy_fpr'],
                          validation_feasible=policy['validation_metrics']['feasible'],
                          accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],healthy_fpr=metrics['healthy_fpr'],
                          fault_detection_recall=metrics['fault_detection_recall'],
                          selected_before_test=family==summary['primary_family']))
    pd.DataFrame(table).to_csv(OUT/'model_comparison.csv',index=False)
    per_class=pd.DataFrame(metric['per_class']);per_class.to_csv(OUT/'test_by_sensor.csv',index=False)
    matrix=np.array(metric['confusion_matrix']);normal=matrix/matrix.sum(axis=1,keepdims=True)
    fig,ax=plt.subplots(figsize=(8.2,5.6))
    im=ax.imshow(normal,vmin=0,vmax=1,cmap='Blues')
    for i in range(5):
        for j in range(5):
            ax.text(j,i,f'{matrix[i,j]:,}\n{normal[i,j]:.1%}',ha='center',va='center',fontsize=9,color='white' if normal[i,j]>.55 else '#123550')
    ax.set_xticks(range(5),SHORT,rotation=25,ha='right');ax.set_yticks(range(5),SHORT)
    ax.set(xlabel='Predicted condition',ylabel='Native scenario condition',title='Reserved calendar weeks: four sensor fault types')
    fig.colorbar(im,ax=ax,label='Fraction within source condition');fig.tight_layout()
    fig.savefig(OUT/'test_confusion_matrix.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.8));x=np.arange(4)
    ax.bar(x-.18,per_class.recall.iloc[1:],.36,label='Correct-sensor recall',color='#137B9A')
    ax.bar(x+.18,per_class.precision.iloc[1:],.36,label='Precision',color='#123550')
    ax.set_xticks(x,SHORT[1:]);ax.set(ylim=(0,1.08),ylabel='Fraction',title='Primary LSTM: sensor identification on reserved weeks')
    ax.legend(loc='lower right');fig.tight_layout();fig.savefig(OUT/'test_sensor_performance.png',dpi=170);plt.close(fig)
    counts=pd.read_csv(ROOT/'results/multisensor_v1/cohort_counts.csv')
    days=pd.read_csv(OUT/summary['primary_family']/'test_per_day.csv')
    primary=json.loads((OUT/summary['primary_family']/'summary.json').read_text())
    prevalence=1-matrix[0].sum()/matrix.sum();majority=matrix.sum(axis=1).max()/matrix.sum()
    overview=dict(test_fault_prevalence=float(prevalence),five_class_majority_baseline_accuracy=float(majority),
                  eligible_train_endpoints=int(counts[counts.partition=='train'].endpoints.sum()),
                  eligible_validation_endpoints=int(counts[counts.partition=='validation'].endpoints.sum()),
                  eligible_test_endpoints=int(counts[counts.partition=='test'].endpoints.sum()),
                  sampled_train_windows=primary['train_windows'],checkpoint_validation_windows=primary['checkpoint_validation_windows'],
                  healthy_active_test_calendar_days=int(days[days.y==0].date.nunique()),
                  raw_test_days_per_scenario=int(counts[counts.partition=='test'].raw_days.max()),
                  measurement_inputs=21,command_inputs=5,raw_candidate_inputs=26,
                  trainable_parameters=primary['parameters'],best_epoch=primary['best_epoch'],epochs_run=primary['epochs_run'])
    (OUT/'evidence_summary.json').write_text(json.dumps(overview,indent=2))
    pct=lambda x:f'{x:.2%}'
    lines=['# Fault detection across four sensors','',
           'This extension identifies four individual temperature/pressure sensor biases in a simulated dual-duct AHU. The previous outdoor-air detector remains a separate benchmark. The four new targets do not include outdoor-air temperature.','',
           '## Reserved test result','',
           f"Primary model: {summary['primary_family']}, seed 42, selected on validation. Five-class accuracy {pct(metric['accuracy'])}; macro F1 {pct(metric['macro_f1'])}; any-fault detection recall {pct(metric['fault_detection_recall'])}.",
           f"Healthy FPR is {pct(metric['healthy_fpr'])}: {metric['healthy_false_readings']:,} false readings / {metric['healthy_readings']:,} healthy readings. The 1% goal is missed. There are {metric['healthy_false_bursts']} false bursts across {overview['healthy_active_test_calendar_days']} active healthy calendar days. Thresholds were not retuned after this test.",'',
           '| Faulty sensor | Precision | Correct-sensor recall | F1 |','|---|---:|---:|---:|']
    for r in metric['per_class'][1:]:lines.append(f"| {r['sensor']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} |")
    lines += ['', '## Source and data cleaning','',
              'Official source: https://faultdetection.lbl.gov/dataset/simulated-dd-ahu-dataset/. Dataset collection DOI 10.25984/1881324. Methodology and fault definitions: https://doi.org/10.1038/s41597-023-02197-w, Table 4.',
              'The downloaded archive contains 56 CSVs. We selected one healthy scenario and sixteen native sensor-bias scenarios: four severity filenames for each of four sensor locations. No labels were synthesized. Each selected CSV contains 525,600 one-minute rows in 2018, representing parallel versions of one simulated unit, rather than seventeen independent units.',
              'The source contains 114 measurement/control points. Our declared 26 AHU inputs comprise 21 measurements and five commands. Dates, scenario names, native filename severity and the class label are excluded from the input. Temperatures retain Fahrenheit. The pressure filenames use +/-2inwg and +/-4inwg while the paper states +/-0.2 and +/-0.4 inH2O. We retain their original identifiers and make no independently verified pressure-severity magnitude claim.',
              'Operating windows require SYS_CTL=1, both supply fan speeds positive, and thirty contiguous one-minute readings within one source day. Missing/nonfinite values receive at most five causal forward-filled readings within a day/gap; remaining gaps use medians fitted on healthy active training rows only. Standardization uses the same training-only rows and clips to +/-10 standard deviations. No nonfinite values appeared in the audited training columns. Inactive/shutdown windows remain unscored; scenario-specific raw and scored day counts appear in results/multisensor_v1/cohort_counts.csv.',
              'The initially proposed single-duct source failed an integrity audit: all four outdoor-bias tables were identical and the documented sa_bias filenames were absent. The audit is retained in results/sda_source_audit. Its mislabeled/duplicate cases were not used in these models.','',
              '## Calendar separation and development history','',
              'All scenarios use the same calendar-week partition: ISO-week modulo six equals zero for test, five for validation, and all other values for training. Matching calendar/weather periods never cross a partition. Histories never cross a day, operating gap or scenario. This tests held-out periods of the same simulated unit and known bias families, not new physical equipment or unseen fault severities.',
              f"Eligible endpoints: {overview['eligible_train_endpoints']:,} training, {overview['eligible_validation_endpoints']:,} validation, {overview['eligible_test_endpoints']:,} test. To control overlapping-window training cost, optimization samples one endpoint per hour ({overview['sampled_train_windows']:,} windows). Checkpoint selection samples validation every fifteen minutes ({overview['checkpoint_validation_windows']:,}). Alarm-policy validation and test score every eligible minute.",
              'Version 1 trained two input families for at most 20 epochs and failed the false-alarm/recall validation gate. Version 2 explicitly logged a validation refinement: up to 80 epochs, eight-epoch patience, doubled healthy loss weight and checkpoint selection using macro AP across all five classes. Four unique LSTMs were trained in total. All use the fixed primary seed 42, a 32-unit LSTM, 16-unit ReLU head, dropout 0.2, weighted cross entropy, Adam 0.001, batch size 512 and gradient clipping 1. There is no repeated-seed robustness claim for this extension.',
              'Version 2 compared 126 alarm policies for each of the four model candidates, including reused version-1 weights: 504 policies. Shared healthy percentiles or a larger budget for one fault class determine per-class thresholds. Immediate, two-consecutive and two-of-three rules reset at day/scenario/gap boundaries. Validation macro F1 selects among candidates meeting healthy FPR <=1% and each-sensor recall >=40%. Model and thresholds were frozen before the single final test. Both study versions remain visible; their total search is disclosed.',
              f"The selected primary has {primary['parameters']:,} parameters, runs {primary['epochs_run']} epochs and restores epoch {primary['best_epoch']}. Its validation FPR is {pct(frozen['policy']['validation_metrics']['healthy_fpr'])}; its reserved test FPR is {pct(metric['healthy_fpr'])}. The selected alarm rule is {frozen['policy']['rule']}.",'',
              '## Five input comparison','',
              'A training-only tree selector ranks grouped latest/mean/std/slope histories. It fits earlier internal training weeks and permutes whole daily trajectories in held-out training weeks, with three fixed repeats. No validation or test labels enter ranking. The five selected inputs are hot supply fan speed, hot supply fan power, cold supply fan speed, cooling valve command and cold supply fan power. These comprise four measurements and one command, rather than five fault-target sensors.',
              '| Candidate | Inputs | Test accuracy | Macro F1 | Healthy FPR |','|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['model']} | {r['inputs']} | {pct(r['accuracy'])} | {pct(r['macro_f1'])} | {pct(r['healthy_fpr'])} |")
    lines += ['', 'Neither raw-five model met validation targets. The reduction removed evidence useful for hot-deck temperature identification. This conditional permutation ranking is exploratory; correlated features share information. It does not establish that every possible five-feature or engineered-feature model will fail. PCA is not implemented in this study.','',
              '## Limits and demonstration','',
              f"Fault prevalence is {pct(prevalence)} in the experimental test sample. A constant majority class would score {pct(majority)} on this five-class task. Five-class accuracy, per-sensor recall and healthy false alarms answer different questions. Minute observations overlap and are correlated, so large row counts do not imply independent equipment evidence.",
              'The source imposes one known sensor bias per scenario. Concurrent faults, other equipment faults, missing-data robustness, field deployment, unseen severity generalization and future-failure prediction are unproven. The already-inspected reserved weeks cannot be reused for a fresh improvement claim. Improving the 2.10% false-alarm result requires a new development/evaluation design and additional independent data.',
              'Run Start Dashboard.bat and select Four-sensor extension. Replay a healthy day and one case for each sensor. The app displays the failed 1% target. Uploaded thirty-row CSVs use saved transformations and model weights; softmax scores are uncalibrated. The old outdoor-air study remains selectable. Retraining requires the separately downloaded source archive; the saved demonstration runs offline.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'MODEL_CARD.md').write_text('\n'.join(lines[:16])+f'\n\nInput channels: 26; measured targets: four individually biased sensors; primary parameters: {primary["parameters"]:,}.\n\nResearch prototype. The reserved-test 1% healthy FPR target is unmet. Single simulated unit, fixed seed, no simultaneous fault coverage. See REPORT.md and verification.json for full methods and evidence.\n',encoding='utf-8')
    print(json.dumps(overview,indent=2))


if __name__=='__main__':main()
