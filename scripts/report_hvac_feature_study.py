"""Describe the measured validation-only feature ablation without replacing the submission."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/hvac_feature_v2'


def main():
    verification=json.loads((OUT/'verification.json').read_text());assert verification['passed']
    groups=pd.read_csv(OUT/'validation_feature_comparison.csv')
    models=pd.read_csv(OUT/'validation_model_comparison.csv')
    summary=json.loads((OUT/'summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(groups.family,groups.ap_mean,yerr=groups.ap_sd,capsize=5,color=['#126b86','#7894a0','#b65f45'])
    axes[0].set(ylabel='Validation AP',ylim=(0,1.02),title='Mean and sample SD across three seeds')
    axes[1].bar(groups.family,100*groups.fpr_mean,color=['#126b86','#7894a0','#b65f45'])
    axes[1].axhline(1,color='black',ls='--',label='1% target')
    axes[1].set(ylabel='Mean validation FPR (%)',title='Selected policies; reduced sets unqualified');axes[1].legend()
    fig.suptitle('Same cohort and model budget; no new autumn scoring');fig.tight_layout()
    fig.savefig(OUT/'feature_comparison.png',dpi=160);plt.close(fig)
    lines=['# Validation-only sensor-feature comparison','',
        'Professor feedback prompted an audit of additional channels and a controlled reduction experiment. Six new LSTMs were trained; three existing LSTM seed runs are the matched reference. This is development evidence, not a new independent test or a replacement for the submitted model.','',
        '## Input definitions','',
        '- engineered32: thirteen source measurement/control inputs, six derived consistency features and thirteen missing flags.',
        '- raw13: thirteen raw inputs only; no derived features or missing flags.',
        '- raw5: five training-ranked raw variables, with no derived features or extra flags.',
        '', 'Selected raw5: '+', '.join(summary['top_five_sensors'])+'.', '',
        'The ranking uses whole training days held out internally within winter/spring. It never selects from summer validation or autumn test outcomes. Sensor-group permutation is conditional on the screening tree and correlations, and does not establish physical causality or an optimal LSTM subset.',
        '', '## Matched experiment','',
        'All sets use the same 61,187 training and 33,432 summer validation endpoints, 30-minute history, hidden size 32, 16-unit head, dropout 0.2, Adam 0.001, batch 512, balanced BCE training weight, gradient clip one, maximum 30 epochs and AP patience six. Seeds are 7, 42 and 123. Preprocessing fits normal active training observations only. Each model receives the same thirty-policy validation search and target gate.','',
        '| Inputs | Parameters | AP mean +/- sample SD | Mean precision | Mean recall | Mean FPR | Feasible seeds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in groups.to_dict('records'):
        lines.append(f"| {r['family']} | {r['parameters']:,} | {r['ap_mean']:.4f} +/- {r['ap_sd']:.4f} | {r['precision_mean']:.2%} | {r['recall_mean']:.2%} | {r['fpr_mean']:.2%} | {r['feasible_seeds']}/{r['seeds']} |")
    lines += ['', 'None of the six reduced-input runs meets the original validation gate. Their reported policies are diagnostic fallbacks satisfying the recall constraint where available, not qualified releases. The current engineered model is retained. Removing derived relationships while reducing raw inputs did not improve this budgeted comparison.',
        '', '| Model | AP | Precision | Recall | FPR | Feasible |', '|---|---:|---:|---:|---:|---|']
    for r in models.to_dict('records'):
        lines.append(f"| {r['model']} | {r['average_precision']:.4f} | {r['precision']:.2%} | {r['recall']:.2%} | {r['false_positive_rate']:.2%} | {r['feasible']} |")
    lines += ['', '## Interpretation and next step','',
        'Five raw sensors and five engineered model features are different hypotheses. This study tests the former. A future wider-data experiment should also compare five selected engineered features and disclose the raw measurements each requires. These results do not establish that every five-feature approach must fail.',
        '', 'MZVAV-1 already provides all current measurement/control channels; unused status flags are constant during eligible operation. Additional genuine simulated airflow, fan-power or position channels require a richer aligned source. SENSOR_EXPANSION_PLAN.md identifies an official 30-variable annual simulation source and an overlap-aware evaluation design. No additional readings or sensor types have been fabricated, downloaded or added here.',
        '', 'All 270 policies and six complete fresh validation score arrays reproduce; six fresh subset CSV windows reproduce; 42 unit tests pass. No new autumn predictions were made. The source, original trained artifacts, submission report and dashboard remain unchanged.',
        '', 'Reproduce with `python scripts/audit_hvac_sensors.py`, `python scripts/train_hvac_feature_study.py`, `python scripts/verify_hvac_feature_study.py` and `python scripts/report_hvac_feature_study.py`. Completed models are reused only after integrity checks.']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('Built feature-comparison chart and report.')


if __name__=='__main__':main()
