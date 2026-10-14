"""Fixed AHU benchmark models; validation only, no autumn test scores."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dccad.hvac import (read_source,sensor_frame,operation_mask,causal_clean,fit_preprocessor,transform,
    day_episodes,MODEL_FEATURES,RAW_FEATURES,WIDTH)
from dccad.supervised import LSTMClassifier,predict_logits,sequence_inputs,trailing_statistics
from dccad.alarm_policies import PolicyEvaluator,apply_rule
from dccad.protocol import constrained_policy_key
from dccad.robustness import choose_seed_policy
from dccad.evaluation import metrics,incident_metrics
from analyze_thresholds import pr_candidates

OUT=ROOT/'results/hvac_v1';DATA=ROOT/'data/hvac_benchmark'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')


def train_seed(s,xt,xv,yt,yv,weight,pre,frame,val):
    folder=OUT/f'lstm_seed_{s}';folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():
        r=json.loads((folder/'summary.json').read_text())
        for f,h in r['checksums'].items():assert sha(folder/f)==h
        return r
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=LSTMClassifier(len(MODEL_FEATURES),32,.2);opt=torch.optim.Adam(model.parameters(),lr=.001)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(weight)))
    loader=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(yt.astype(np.float32))),batch_size=512,
        shuffle=True,generator=torch.Generator().manual_seed(s))
    best=-np.inf;wait=0;history=[]
    for epoch in range(1,31):
        start=time.time();model.train();total=0.
        for x,y in loader:
            opt.zero_grad();loss=loss_fn(model(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(x)
        score=predict_logits(model,xv);ap=float(average_precision_score(yv,score))
        history.append(dict(epoch=epoch,train_loss=total/len(xt),validation_ap=ap,seconds=time.time()-start))
        if ap>best+1e-8:best=ap;wait=0;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:wait+=1
        print('AHU seed',s,'epoch',epoch,'AP',round(ap,6),flush=True)
        if wait>=6:break
    model.load_state_dict(state);score=predict_logits(model,xv)
    torch.save(dict(state_dict=state,features=MODEL_FEATURES,width=WIDTH,hidden=32,dropout=.2),folder/'model.pt')
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    r=dict(name=f'lstm_seed_{s}',kind='lstm',seed=s,features=MODEL_FEATURES,width=WIDTH,score_mode='logit',
        best_epoch=best_epoch,epochs_run=len(history),stop_reason='patience' if wait>=6 else 'maximum_epochs',
        pos_weight=float(weight),average_precision=best)
    return save_model(folder,r,score,pre,frame,val)


def save_model(folder,r,score,pre,frame,val):
    joblib.dump(pre,folder/'preprocessing.joblib')
    np.savez_compressed(folder/'validation_scores.npz',node=val.node,time_ns=val.time_ns,row_index=val.row_index,y_end=val.y_end,score=score)
    end=int(val.iloc[0].row_index);example=sensor_frame(frame.iloc[end-WIDTH+1:end+1]).reset_index(drop=True)
    example.insert(0,'timestamp',frame.timestamp.iloc[end-WIDTH+1:end+1].astype(str).to_numpy())
    example['occupied']=1;example['supply_fan_on']=1;example.to_csv(folder/'example_window.csv',index=False)
    write(folder/'example_metadata.json',dict(validation_position=0,expected_score=float(score[0]),row_index=end))
    r.update(validation_windows=len(val),validation_faults=int(val.y_end.sum()),
        checksums={f:sha(folder/f) for f in ['model.pt' if r['kind']=='lstm' else 'model.joblib','preprocessing.joblib','validation_scores.npz','history.csv']},test_scored=False)
    write(folder/'summary.json',r);print('Completed AHU model',r['name'],'AP',r['average_precision'],flush=True);return r


def tree(xt,xv,yt,yv,weight,pre,frame,val):
    folder=OUT/'boosting';folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():
        r=json.loads((folder/'summary.json').read_text())
        for f,h in r['checksums'].items():assert sha(folder/f)==h
        return r
    model=HistGradientBoostingClassifier(learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.,
        max_iter=200,early_stopping=False,random_state=42)
    tx=trailing_statistics(xt);vx=trailing_statistics(xv);weights=np.where(yt,weight,1.)
    history=[];best=-np.inf
    with threadpool_limits(limits=4):
        model.fit(tx,yt,sample_weight=weights)
        for step,p in enumerate(model.staged_predict_proba(vx),1):
            if step in [50,100,200]:
                ap=float(average_precision_score(yv,p[:,1]));history.append(dict(iterations=step,validation_ap=ap))
                if ap>best:best=ap;best_step=step;best_score=p[:,1].copy()
        model.set_params(max_iter=best_step);model.fit(tx,yt,sample_weight=weights);score=model.predict_proba(vx)[:,1]
    np.testing.assert_allclose(score,best_score,rtol=1e-10,atol=1e-12)
    joblib.dump(model,folder/'model.joblib');pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    return save_model(folder,dict(name='boosting',kind='boosting',seed=42,features=MODEL_FEATURES,width=WIDTH,
        score_mode='score',average_precision=best,best_iterations=best_step),score,pre,frame,val)


def evaluate(meta,episodes,models):
    engine=PolicyEvaluator(meta,episodes,1);records=[];specs={};selected=[];curves={};normal=~meta.y_end.to_numpy(bool)
    for r in models:
        name=r['name'];score=np.load(OUT/name/'validation_scores.npz')['score'].astype(np.float64)
        engine.register_scores(name,score);candidates,_,curves[name]=pr_candidates(meta.y_end,score)
        candidates+=[(f'normal_percentile_{q}',float(np.percentile(score[normal],q))) for q in [95,99,99.5,99.9,99.95]]
        candidates+=[('default',0. if r['kind']=='lstm' else .5)];rows=[]
        for method,t in candidates:
            for rule in ['immediate','consecutive_2','two_of_three']:
                pred=apply_rule(meta,score,t,rule,1)
                row=dict(policy_id=f'p{len(records)+1:04d}',model=name,kind=r['kind'],method=method,rule=rule,**engine.evaluate(pred,name))
                row['feasible']=constrained_policy_key(row) is not None;records.append(row);rows.append(row)
                specs[row['policy_id']]=dict(threshold=t,rule=rule,cadence_minutes=1,reset='day/unit/gap')
        chosen,status=choose_seed_policy(rows);chosen=dict(chosen,status=status);selected.append(chosen)
        spec=specs[chosen['policy_id']];pred=apply_rule(meta,score,spec['threshold'],spec['rule'],1)
        write(OUT/name/'validation_policy.json',dict(metrics=chosen,status=status,**spec))
        pd.DataFrame(dict(node=meta.node,date=meta.date,time_ns=meta.time_ns,row_index=meta.row_index,y_end=meta.y_end,score=score,pred=pred,scenario=meta.scenario,bias_c=meta.bias_c)).to_csv(OUT/name/'validation_decisions.csv',index=False)
        standard=metrics(meta.y_end,score,pred);events,details=incident_metrics(meta,pred,episodes,1)
        for k,v in standard.items():np.testing.assert_allclose(v,chosen[k])
        for k in ['detected','false_bursts','scored_node_days']:np.testing.assert_allclose(events[k],chosen[k])
        details.to_csv(OUT/name/'validation_fault_day_details.csv',index=False)
    pd.DataFrame(records).to_csv(OUT/'validation_policy_comparison.csv',index=False);write(OUT/'validation_policy_specs.json',specs)
    pd.DataFrame(selected).to_csv(OUT/'validation_model_comparison.csv',index=False)
    neural=[r for r in selected if r['kind']=='lstm'];primary=next(r for r in neural if r['model']=='lstm_seed_42')
    # Research checkpoint is fixed to primary seed 42; qualification is a separate gate.
    freeze=dict(name='lstm_seed_42',kind='lstm',seed=42,model_sha256=sha(OUT/'lstm_seed_42/model.pt'),
        preprocessing_sha256=sha(OUT/'lstm_seed_42/preprocessing.joblib'),features=MODEL_FEATURES,width=WIDTH,
        policy=specs[primary['policy_id']],policy_id=primary['policy_id'],validation_metrics=primary,
        qualified=all(r['feasible'] for r in neural),qualification_rule='all three neural seeds feasible; primary seed 42',
        interpretation='Research benchmark checkpoint; qualification does not establish real deployment performance',
        protocol_sha256=sha(OUT/'experiment_plan.json'),test_scored=False)
    write(OUT/'frozen_model.json',freeze)
    summary=dict(status='validation_qualified' if freeze['qualified'] else 'validation_targets_missed',
        models=4,neural_seeds=[7,42,123],feasible_neural_seeds=sum(r['feasible'] for r in neural),
        feasible_policies=sum(r['feasible'] for r in records),policies=len(records),primary_model='lstm_seed_42',
        test_scored=False,validation_windows=len(meta),validation_faults=int(meta.y_end.sum()),
        raw_fault_days=len(episodes),observable_fault_days=primary['observable_episodes'],
        seed_ap_mean=float(np.mean([r['average_precision'] for r in neural])),seed_ap_std=float(np.std([r['average_precision'] for r in neural],ddof=1)))
    write(OUT/'validation_summary.json',summary)
    fig,ax=plt.subplots(figsize=(9,5))
    for name,(p,r,_) in curves.items():ax.step(r,p,where='post',label=name)
    ax.axhline(meta.y_end.mean(),color='gray',ls=':',label='all-fault ranking baseline')
    ax.set(xlabel='Recall',ylabel='Precision',title='AHU summer validation; whole-season separation',xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'validation_precision_recall.png',dpi=170);plt.close(fig)
    print(pd.DataFrame(selected)[['model','average_precision','precision','recall','false_positive_rate','false_bursts_per_node_day','event_recall_observable','feasible']].to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    plan=json.loads((OUT/'experiment_plan.json').read_text());assert sha(DATA/'afdd_source.zip')==plan['archive_sha256']
    frame=read_source(DATA/'afdd_source.zip')
    raw=sensor_frame(frame).to_numpy(dtype=np.float32);times=frame.timestamp.dt.as_unit('ns').astype('int64').to_numpy()
    clean,filled=causal_clean(raw,frame.date.to_numpy(),times)
    fit=frame.season.isin(['winter','spring']).to_numpy()&(~frame['Fault Detection Ground Truth'].to_numpy(bool))&operation_mask(frame)
    pre=fit_preprocessor(raw,clean,filled,fit);joblib.dump(pre,OUT/'preprocessing.joblib')
    write(OUT/'preprocessing_audit.json',dict(fit_rows=int(fit.sum()),fit_days=frame.loc[fit,'date'].nunique(),
        fit_seasons=frame.loc[fit,'season'].unique().tolist(),source_and_derived_features=pre['features'],
        temperatures_below_zero_retained=True,test_statistics_used=False,
        implementation_sha256={f:sha(ROOT/f) for f in ['src/dccad/hvac.py','scripts/train_hvac.py','src/dccad/supervised.py']}))
    values=transform(clean,filled,pre);meta={}
    for name in ['train','validation']:
        p=ROOT/plan['cohorts'][name]['path'];assert sha(p)==plan['cohorts'][name]['sha256'];meta[name]=pd.read_csv(p)
    train=meta['train'];val=meta['validation'];xt=sequence_inputs(values,train.row_index,WIDTH);xv=sequence_inputs(values,val.row_index,WIDTH)
    yt=train.y_end.to_numpy(bool);yv=val.y_end.to_numpy(bool);weight=float((~yt).sum()/yt.sum())
    models=[train_seed(s,xt,xv,yt,yv,weight,pre,frame,val) for s in [7,42,123]]
    models.append(tree(xt,xv,yt,yv,weight,pre,frame,val))
    ep=pd.read_csv(OUT/'validation_fault_days.csv')[['node','start_ns','end_ns','readings']]
    evaluate(val,ep,models)

if __name__=='__main__':main()
