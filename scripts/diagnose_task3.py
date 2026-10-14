"""Describe train/validation generalization without additional model selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from dccad.data import load_reduced
from dccad.protocol import protocol_clean
from dccad.supervised import LSTMClassifier,features_from_clean,sequence_inputs,predict_logits,trailing_statistics


def main():
    torch.set_num_threads(4);out=ROOT/'results/task3_supervised'
    plan=json.loads((out/'experiment_plan.json').read_text())
    train=pd.read_csv(ROOT/plan['cohorts']['train']['path']);val=pd.read_csv(ROOT/plan['cohorts']['validation']['path'])
    data=load_reduced(ROOT/'0.tar',ROOT/'artifacts/reduced_base.npz');_,filled,_=protocol_clean(data)
    records=[];nodes=[]
    for name,meta in [('training_sampled',train),('validation_dense',val)]:
        for node,group in meta.groupby('node'):
            nodes.append(dict(split=name,node=int(node),endpoints=len(group),critical=int(group.y_end.sum()),prevalence=float(group.y_end.mean())))
    pd.DataFrame(nodes).to_csv(out/'label_distribution.csv',index=False)
    for cfg in plan['configs']:
        if cfg['kind'] not in ['lstm','boosting']:continue
        folder=out/cfg['name'];summary=json.loads((folder/'summary.json').read_text())
        pre=joblib.load(folder/'preprocessing.joblib')
        values=features_from_clean(data['raw'],filled,pre['scaler'],pre['features'],pre['clip'])
        x=sequence_inputs(values,train.row_index,pre['width'])
        if cfg['kind']=='lstm':
            cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            model=LSTMClassifier(len(cp['features']),cp['hidden'],cp['dropout']);model.load_state_dict(cp['state_dict'])
            score=predict_logits(model,x);mode='logit'
        else:
            model=joblib.load(folder/'model.joblib')
            with threadpool_limits(limits=4):score=model.predict_proba(trailing_statistics(x))[:,1]
            mode='score'
        vap=summary['score_metrics'][mode]['average_precision']
        records.append(dict(model=cfg['name'],training_ap=float(average_precision_score(train.y_end,score)),validation_ap=vap,
            training_prevalence=float(train.y_end.mean()),validation_prevalence=float(val.y_end.mean())))
    pd.DataFrame(records).to_csv(out/'generalization_diagnostic.csv',index=False)
    text=['# Task 3 generalization diagnostic','',
        'Training AP is measured on the fitting data and is optimistic; it does not select an additional model. Validation is later in time, on the frozen natural-prevalence cohort. The training endpoints are sampled hourly, whereas validation endpoints are scored every 15 minutes. This comparison describes the gap without estimating its cause.','',
        '| Model | Training AP | Validation AP |','|---|---:|---:|']
    for r in records:text.append(f"| {r['model']} | {r['training_ap']:.4f} | {r['validation_ap']:.4f} |")
    text += ['', 'Per-node endpoint target counts and prevalence are in `label_distribution.csv`. These counts can reveal label concentration but cannot prove that a sensor change caused the weak validation result. Nagios health faults may not be identifiable from the selected thermal/fan/power measurements. Additional telemetry and fault-specific labels are possibilities to investigate, not established remedies.','',
        'No independent holdout measurements, labels or predictions were accessed. This diagnostic adds no configurations, thresholds or selection criteria.']
    (out/'GENERALIZATION.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(pd.DataFrame(records).to_string(index=False))

if __name__=='__main__':main()
