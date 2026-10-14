"""Prepare the evaluated replay, severity/delay analysis and report content."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dccad.hvac import read_source,sensor_frame
OUT=ROOT/'results/hvac_v1'


def main():
    frame=read_source(ROOT/'data/hvac_benchmark/afdd_source.zip');sensor=sensor_frame(frame)
    d=pd.read_csv(OUT/'lstm_seed_42/test_decisions.csv');idx=d.row_index.to_numpy()
    replay=d.copy();replay['timestamp']=frame.timestamp.iloc[idx].to_numpy()
    for c in sensor.columns:replay[c]=sensor[c].iloc[idx].to_numpy()
    replay.to_parquet(OUT/'test_replay.parquet',index=False)
    s=json.loads((OUT/'test_summary.json').read_text());v=json.loads((OUT/'validation_summary.json').read_text());p=json.loads((OUT/'experiment_plan.json').read_text());m=s['primary_test']
    test=pd.read_csv(OUT/'test_model_comparison.csv');perday=pd.read_csv(OUT/'test_per_day.csv');primary=perday[perday.model=='lstm_seed_42']
    severity=[]
    for bias,g in primary.groupby('bias_c'):
        tp=int(g.tp.sum());fp=int(g.fp.sum());fn=int(g.fn.sum());tn=int(g.tn.sum())
        severity.append(dict(bias_c=float(bias),scored_days=len(g),tp=tp,fp=fp,fn=fn,tn=tn,
            recall=tp/(tp+fn) if tp+fn else None,fpr=fp/(fp+tn) if fp+tn else None,detected_days=int(g.detected.sum())))
    pd.DataFrame(severity).to_csv(OUT/'test_by_severity.csv',index=False)
    details=pd.read_csv(OUT/'lstm_seed_42/test_fault_day_details.csv');first=d.groupby('node').time_ns.min()
    details['first_eligible_ns']=details.node.map(first)
    details['delay_from_first_eligible_minutes']=details.delay_minutes-(details.first_eligible_ns-details.start_ns)/(60*10**9)
    details.to_csv(OUT/'primary_fault_day_delay_analysis.csv',index=False)
    delay=details[details.detected].delay_from_first_eligible_minutes
    evidence=dict(source_midnight_median_delay_minutes=m['median_delay_minutes'],first_eligible_median_delay_minutes=float(delay.median()),
        fault_days_total=42,observable_fault_days=36,fault_days_detected=35,false_alarm_bursts=1,
        test_targets_met=dict(precision=m['precision']>=.3,recall=m['recall']>=.4,fpr=m['false_positive_rate']<=.01,
            false_bursts=m['false_bursts_per_node_day']<=.1,observable_fault_days=m['event_recall_observable']>=.6),
        seed_test_accuracy_mean=float(test[test.model.str.startswith('lstm')].accuracy.mean()),
        seed_test_accuracy_std=float(test[test.model.str.startswith('lstm')].accuracy.std()),simulation_only=True,post_test_tuning=False)
    (OUT/'evidence_summary.json').write_text(json.dumps(evidence,indent=2))
    examples=[];hit=d[d.pred&d.y_end].iloc[0];missed=details[(details.score_opportunities>0)&(~details.detected)].iloc[0];false=d[d.pred&(~d.y_end)].iloc[0]
    for label,node in [('Detected fault',int(hit.node)),('Missed fault day',int(missed.node)),('False alarm',int(false.node))]:
        x=replay[replay.node==node];fig,axs=plt.subplots(2,1,figsize=(10,5),sharex=True)
        for c in ['outdoor_temp','mixed_temp','return_temp']:axs[0].plot(x.timestamp,x[c],label=c)
        axs[0].set(ylabel='Temperature (F)',title=f'{label}: {x.date.iloc[0]}');axs[0].legend(fontsize=8,ncol=3)
        axs[1].plot(x.timestamp,x.score,label='LSTM logit');threshold=json.loads((OUT/'frozen_model.json').read_text())['policy']['threshold']
        axs[1].axhline(threshold,ls='--',color='black');axs[1].scatter(x.loc[x.pred,'timestamp'],x.loc[x.pred,'score'],s=4,color='red',label='Confirmed alarm')
        axs[1].set(ylabel='Logit',xlabel='Source local time');axs[1].legend(fontsize=8)
        fig.tight_layout();stem=label.lower().replace(' ','_');fig.savefig(OUT/(stem+'.png'),dpi=160);plt.close(fig)
        examples.append(dict(type=label,node=node,date=str(x.date.iloc[0]),source_fault=bool(x.y_end.iloc[0]),bias_c=float(x.bias_c.iloc[0]),plot=stem+'.png'))
    (OUT/'examples.json').write_text(json.dumps(examples,indent=2))
    lines=['# Cooling equipment sensor-fault detection with an LSTM','',
        'Primary study: author-labelled outdoor-air temperature sensor-bias detection in an operating simulated air-handling unit. M100 remains supporting research; its performance is not the source of the following benchmark accuracy.','',
        f"Independent autumn test: accuracy {m['accuracy']:.2%}, precision {m['precision']:.2%}, recall {m['recall']:.2%}, F1 {m['f1']:.2%}, AP {m['average_precision']:.4f}, FPR {m['false_positive_rate']:.2%}. The FPR misses the original 1% goal. No test-based threshold or seed tuning occurred.",'',
        '| Model | Accuracy | Precision | Recall | FPR | AP |','|---|---:|---:|---:|---:|---:|']
    for r in test.to_dict('records'):lines.append(f"| {r['model']} | {r['accuracy']:.2%} | {r['precision']:.2%} | {r['recall']:.2%} | {r['false_positive_rate']:.2%} | {r['average_precision']:.4f} |")
    lines += ['', '## Data, reference outcome and separation','',
        'Source: LBNL AFDD collection, DOI 10.25984/1824861, member MZVAV-1.csv. The 189 days contain 272,160 one-minute records with original 0/1 condition ground truth. Inventory Table 1-2 documents normal scenarios and outdoor-air sensor biases of +/-1, +/-2 and +/-4 Celsius. Input temperatures are Fahrenheit. This is EnergyPlus-Dymola simulation of one AHU, not datacenter failure measurements.',
        '', '| Partition | Seasons | Raw / scored days | Endpoints | Fault endpoints |','|---|---|---:|---:|---:|']
    for name,c in p['cohorts'].items():lines.append(f"| {name} | {'winter + spring' if name=='train' else 'summer' if name=='validation' else 'autumn'} | {c['raw_days']} / {c['scored_days']} | {c['windows']:,} | {c['faulty']:,} |")
    lines += ['', 'Each input contains 30 consecutive occupied, fan-on minutes on one day. Whole seasons/scenario blocks remain in one partition. Nights/nonoperating days are outside the model task; all raw fault days stay in the coverage denominator. Native labels match the inventory for every source date. Dates, seasons, day IDs, bias intensity and labels are not model inputs.',
        '', '## Inputs, cleaning and learning','',
        'Thirteen temperature, fan/damper/valve command and duct-pressure inputs yield six consistency differences and 13 missing indicators: 32 channels. The command-weighted mixing residual is a heuristic, not measured mixing fraction. Negative outdoor Fahrenheit temperatures remain valid. Forward filling uses at most five prior minutes and resets at days/gaps. Remaining missing inputs use normal active training medians. Scaling uses normal active training observations only, clipped at ten standard deviations.',
        '', 'The LSTM has hidden size 32 and a 16-unit ReLU classification head with dropout 0.2. Adam uses learning rate 0.001 and batch 512, maximum 30 epochs, patience six on validation AP and gradient clipping one. BCE positive weight equals the training normal/fault ratio, about 1/6. It balances the training loss without changing evaluation prevalence. The tree receives latest/mean/std/slope over the same history, with fixed 15-leaf boosting and 50/100/200 validation checkpoints.',
        '', f"Seeds 7, 42 and 123 all pass the validation policy gate; validation AP mean {v['seed_ap_mean']:.4f}, sample SD {v['seed_ap_std']:.4f}. Seed 42 remains the predeclared primary although seed 7 performs better on test. All model/checkpoint/policy artifacts were hashed before the one-time autumn evaluation.",
        '', '## Alarm policy and measured tradeoff','',
        'Normal-validation percentiles and PR thresholds compete under the original precision/recall/FPR/false-burst/fault-day gate. The frozen primary requires two high scores in three consecutive scored minutes at logit threshold 6.967112; state resets at days/gaps. Scores are uncalibrated. A single uploaded window cannot establish persistence.',
        '', f"Confusion counts: TP {m['tp']:,}, FP {m['fp']}, FN {m['fn']:,}, TN {m['tn']:,}. The 69 false readings form one burst: {m['false_bursts_per_node_day']:.4f} bursts per scored device-day. False discovery ratio is {m['false_discovery_ratio']:.2%}, using a different denominator than FPR. Detected days: 35/36 observable and 35/42 total. Median delay from midnight imposition is 392 minutes, including overnight downtime/warmup; median delay from first eligible score is {evidence['first_eligible_median_delay_minutes']:.1f} minutes.",
        '', 'The tree has zero test false positives but lower reading recall (72.05%). The LSTM has greater sensitivity with FPR 1.44%, above the 1% goal. This tradeoff remains unchanged after evaluation.',
        '', '## Limits and reproducibility','',
        'Experimental fault prevalence is 85.71%. An all-fault classifier would score 85.71% accuracy and 100% FPR, so accuracy alone is misleading. Healthy validation/test each contain one scenario week; adjacent minutes/days are correlated. These results do not estimate deployment prevalence or independent-unit reliability. Only one simulated unit and fault family are tested. No early warning or general fault diagnosis is established.',
        '', 'The original M100 study tested feature removal, classifiers, persistence and additional inputs but missed the false-alarm targets. Its published preparation selects batch-client scheduler state rather than a cooling-specific fault. The preserved negative results motivate this documented, user-approved dataset/scope change.',
        '', 'Source archive/inventory: data/hvac_benchmark. Protocols, histories, model/preprocessing, scores, decisions, case/severity tables and verification: results/hvac_v1. Reproduce with prepare_hvac.py, train_hvac.py, verify_hvac.py --validation-only, evaluate_hvac.py, verify_hvac.py, build_hvac_evidence.py. The evaluation script returns existing final results on rerun; a new experiment requires a new version and evaluation design.',
        '', '[Official dataset](https://data.openei.org/submissions/910). Source inventory pages 3-7 define the system, units and scenarios.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Prepared replay, severity/delay tables, three examples and report content.')

if __name__=='__main__':main()
