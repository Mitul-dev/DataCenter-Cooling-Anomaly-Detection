"""One-shot autumn comparison after validation/checkpoint/policy freezing."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dccad.hvac import read_source,sensor_frame,causal_clean,transform,WIDTH
from dccad.supervised import LSTMClassifier,predict_logits,sequence_inputs,trailing_statistics
from dccad.alarm_policies import PolicyEvaluator,apply_rule
from dccad.evaluation import incident_metrics

OUT=ROOT/'results/hvac_v1';DATA=ROOT/'data/hvac_benchmark'
NAMES=['lstm_seed_7','lstm_seed_42','lstm_seed_123','boosting']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')


def main():
    torch.set_num_threads(4);plan=json.loads((OUT/'experiment_plan.json').read_text());frozen=json.loads((OUT/'frozen_model.json').read_text())
    assert frozen['protocol_sha256']==sha(OUT/'experiment_plan.json')
    assert frozen['model_sha256']==sha(OUT/'lstm_seed_42/model.pt') and frozen['preprocessing_sha256']==sha(OUT/'lstm_seed_42/preprocessing.joblib')
    evaluation=dict(primary_model='lstm_seed_42',models=NAMES,protocol_sha256=sha(OUT/'experiment_plan.json'),
        freeze_sha256=sha(OUT/'frozen_model.json'),test_cohort_sha256=plan['cohorts']['test']['sha256'],
        model_artifacts={name:{f:sha(OUT/name/f) for f in ['model.joblib' if name=='boosting' else 'model.pt','preprocessing.joblib','validation_policy.json']} for name in NAMES},
        scope='one-time independent-season research comparison; all seeds reported; no post-test tuning',
        qualification=frozen['qualified'],cadence_minutes=1)
    p=OUT/'evaluation_freeze.json'
    if p.exists() and json.loads(p.read_text())!=evaluation:raise RuntimeError('Frozen evaluation inputs changed; do not retune on autumn results')
    if (OUT/'test_summary.json').exists():
        existing=json.loads((OUT/'test_summary.json').read_text());assert existing['evaluation_freeze_sha256']==sha(p)
        print('Autumn already evaluated. Returning saved results without new predictions.');print(json.dumps(existing,indent=2));return
    write(p,evaluation)
    cohort=ROOT/plan['cohorts']['test']['path'];assert sha(cohort)==evaluation['test_cohort_sha256']
    meta=pd.read_csv(cohort);episodes=pd.read_csv(OUT/'test_fault_days.csv')[['node','start_ns','end_ns','readings']]
    frame=read_source(DATA/'afdd_source.zip');raw=sensor_frame(frame).to_numpy(dtype=np.float32)
    clean,filled=causal_clean(raw,frame.date.to_numpy(),frame.timestamp.dt.as_unit('ns').astype('int64').to_numpy())
    records=[];day_records=[];curves={};engine=PolicyEvaluator(meta,episodes,1)
    for name in NAMES:
        folder=OUT/name;summary=json.loads((folder/'summary.json').read_text());pre=joblib.load(folder/'preprocessing.joblib')
        values=transform(clean,filled,pre);x=sequence_inputs(values,meta.row_index,WIDTH)
        if summary['kind']=='lstm':
            cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model=LSTMClassifier(len(cp['features']),32,.2)
            model.load_state_dict(cp['state_dict']);score=predict_logits(model,x).astype(np.float64)
        else:
            model=joblib.load(folder/'model.joblib')
            with threadpool_limits(limits=4):score=model.predict_proba(trailing_statistics(x))[:,1]
        policy=json.loads((folder/'validation_policy.json').read_text())
        pred=apply_rule(meta,score,policy['threshold'],policy['rule'],1)
        engine.register_scores(name,score);metrics=engine.evaluate(pred,name)
        metrics['accuracy']=(metrics['tp']+metrics['tn'])/len(meta)
        record=dict(model=name,**metrics);records.append(record)
        pd.DataFrame(dict(node=meta.node,date=meta.date,time_ns=meta.time_ns,row_index=meta.row_index,y_end=meta.y_end,score=score,pred=pred,bias_c=meta.bias_c,scenario=meta.scenario)).to_csv(folder/'test_decisions.csv',index=False)
        np.savez_compressed(folder/'test_scores.npz',score=score,node=meta.node,time_ns=meta.time_ns,row_index=meta.row_index,y_end=meta.y_end)
        _,details=incident_metrics(meta,pred,episodes,1);details.to_csv(folder/'test_fault_day_details.csv',index=False)
        write(folder/'test_metrics.json',metrics);curves[name]=precision_recall_curve(meta.y_end,score)
        for date,group in meta.groupby('date'):
            keep=meta.date==date;m=meta[keep].reset_index(drop=True);e=PolicyEvaluator(m,episodes[episodes.node.isin(m.node)],1)
            e.register_scores('score',score[keep]);day_records.append(dict(model=name,date=date,bias_c=float(group.bias_c.iloc[0]),**e.evaluate(pred[keep],'score')))
    pd.DataFrame(records).to_csv(OUT/'test_model_comparison.csv',index=False);pd.DataFrame(day_records).to_csv(OUT/'test_per_day.csv',index=False)
    primary=next(r for r in records if r['model']=='lstm_seed_42')
    summary=dict(primary_model='lstm_seed_42',evaluation_freeze_sha256=sha(p),test_windows=len(meta),test_faults=int(meta.y_end.sum()),
        experimental_fault_prevalence=float(meta.y_end.mean()),validation_qualified=frozen['qualified'],primary_test=primary,
        all_fault_baseline_accuracy=float(meta.y_end.mean()),all_fault_baseline_fpr=1.,all_normal_baseline_recall=0.,
        test_scored=True,post_test_tuning=False,source='MZVAV-1, simulated outdoor-air sensor bias, autumn holdout',
        caveats=['One simulated AHU and one fault family','Only one healthy scenario week per validation/test season','Fault-heavy experimental prevalence; not deployment prevalence','Sensor bias remains labelled during nights; nonoperating periods excluded from model scoring and disclosed'])
    write(OUT/'test_summary.json',summary);figures(records,curves)
    print(pd.DataFrame(records)[['model','precision','recall','false_positive_rate','accuracy','detected','false_bursts']].to_string(index=False))
    print(json.dumps(summary,indent=2))


def figures(records,curves):
    fig,ax=plt.subplots(figsize=(9,5))
    for name,(p,r,_) in curves.items():ax.step(r,p,where='post',label=name)
    ax.set(xlabel='Recall',ylabel='Precision',title='Frozen model/policy comparison: autumn test',ylim=(0,1.02),xlim=(0,1));ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'test_precision_recall.png',dpi=170);plt.close(fig)
    primary=next(r for r in records if r['model']=='lstm_seed_42');matrix=np.array([[primary['tn'],primary['fp']],[primary['fn'],primary['tp']]])
    fig,ax=plt.subplots(figsize=(5,4));ax.imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    ax.set_xticks([0,1],['Healthy','Fault']);ax.set_yticks([0,1],['Healthy','Fault']);ax.set(xlabel='Predicted',ylabel='Author label',title='Primary LSTM: independent autumn season')
    fig.tight_layout();fig.savefig(OUT/'test_confusion_matrix.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
