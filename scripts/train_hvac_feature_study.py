"""Matched sensor ablation on original training/validation; no new autumn scoring."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from dccad.hvac import read_source,sensor_frame,causal_clean,operation_mask,RAW_FEATURES,MODEL_FEATURES,WIDTH
from dccad.hvac_feature_study import fit_subset,transform_subset,score_window
from dccad.supervised import LSTMClassifier,sequence_inputs,predict_logits
from dccad.alarm_policies import PolicyEvaluator,apply_rule
from dccad.protocol import constrained_policy_key
from dccad.robustness import choose_seed_policy
from analyze_thresholds import pr_candidates
OUT=ROOT/'results/hvac_feature_v2';OLD=ROOT/'results/hvac_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False))


def train_one(family,seed,features,pre,xt,xv,yt,yv,frame,val,plan_hash):
    name=f'{family}_seed_{seed}';folder=OUT/name;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():
        r=json.loads((folder/'summary.json').read_text());assert r['plan_sha256']==plan_hash
        for file,h in r['checksums'].items():assert sha(folder/file)==h
        return r
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=LSTMClassifier(len(features),32,.2);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    weight=float((~yt).sum()/yt.sum());criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    loader=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(yt.astype(np.float32))),batch_size=512,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=-np.inf;wait=0;history=[]
    for epoch in range(1,31):
        start=time.perf_counter();model.train();total=0.
        for x,y in loader:
            optimizer.zero_grad();loss=criterion(model(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=loss.item()*len(x)
        pred=predict_logits(model,xv);ap=float(average_precision_score(yv,pred))
        history.append(dict(epoch=epoch,train_loss=total/len(xt),validation_ap=ap,seconds=time.perf_counter()-start))
        if ap>best+1e-8:best=ap;wait=0;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:wait+=1
        print(name,'epoch',epoch,'AP',round(ap,6),flush=True)
        if wait>=6:break
    model.load_state_dict(state);pred=predict_logits(model,xv)
    torch.save(dict(state_dict=state,features=features,width=WIDTH,hidden=32,dropout=.2),folder/'model.pt')
    write(folder/'preprocessing.json',pre);pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    np.savez_compressed(folder/'validation_scores.npz',score=pred,node=val.node,time_ns=val.time_ns,row_index=val.row_index,y_end=val.y_end)
    end=int(val.row_index.iloc[0]);example=sensor_frame(frame.iloc[end-WIDTH+1:end+1])[features].reset_index(drop=True)
    example.insert(0,'timestamp',frame.timestamp.iloc[end-WIDTH+1:end+1].astype(str).to_numpy());example['occupied']=1;example['supply_fan_on']=1
    example.to_csv(folder/'example_window.csv',index=False)
    fresh=score_window(folder,example);np.testing.assert_allclose(fresh,pred[0],rtol=2e-5,atol=2e-5)
    result=dict(name=name,family=family,seed=seed,features=features,channels=len(features),parameters=sum(p.numel() for p in model.parameters()),
        best_epoch=best_epoch,epochs_run=len(history),validation_ap=best,stop_reason='patience' if wait>=6 else 'maximum_epochs',
        fresh_csv_score=fresh,expected_csv_score=float(pred[0]),plan_sha256=plan_hash,test_scored=False,
        checksums={f:sha(folder/f) for f in ['model.pt','preprocessing.json','history.csv','validation_scores.npz','example_window.csv']})
    write(folder/'summary.json',result);return result


def compare(val,episodes,models):
    engine=PolicyEvaluator(val,episodes,1);all_policies=[];selected=[];specs={}
    for r in models:
        name=r['name'];folder=ROOT/r['artifact_path'];score=np.load(folder/'validation_scores.npz')['score'].astype(np.float64)
        engine.register_scores(name,score);candidates,_,_=pr_candidates(val.y_end,score)
        candidates += [(f'normal_percentile_{q}',float(np.percentile(score[~val.y_end.to_numpy(bool)],q))) for q in [95,99,99.5,99.9,99.95]]
        candidates += [('default',0.)];rows=[]
        for method,threshold in candidates:
            for rule in ['immediate','consecutive_2','two_of_three']:
                prediction=apply_rule(val,score,threshold,rule,1)
                row=dict(policy_id=f'p{len(all_policies)+1:04d}',model=name,method=method,rule=rule,**engine.evaluate(prediction,name))
                row['feasible']=constrained_policy_key(row) is not None;all_policies.append(row);rows.append(row)
                specs[row['policy_id']]=dict(threshold=threshold,rule=rule,cadence_minutes=1)
        chosen,status=choose_seed_policy(rows)
        result=dict(chosen,status=status,family=r['family'],seed=r['seed'],channels=r['channels'],parameters=r['parameters'])
        selected.append(result)
        target=OUT/name;target.mkdir(exist_ok=True)
        write(target/'validation_policy.json',dict(metrics=result,**specs[chosen['policy_id']]))
    pd.DataFrame(all_policies).to_csv(OUT/'validation_policy_comparison.csv',index=False);write(OUT/'validation_policy_specs.json',specs)
    table=pd.DataFrame(selected);table.to_csv(OUT/'validation_model_comparison.csv',index=False)
    aggregate=[]
    for family,g in table.groupby('family',sort=False):
        aggregate.append(dict(family=family,channels=int(g.channels.iloc[0]),parameters=int(g.parameters.iloc[0]),seeds=len(g),feasible_seeds=int(g.feasible.sum()),
            ap_mean=float(g.average_precision.mean()),ap_sd=float(g.average_precision.std()),precision_mean=float(g.precision.mean()),recall_mean=float(g.recall.mean()),
            fpr_mean=float(g.false_positive_rate.mean()),false_bursts_mean=float(g.false_bursts.mean())))
    pd.DataFrame(aggregate).to_csv(OUT/'validation_feature_comparison.csv',index=False)
    print(table[['model','average_precision','precision','recall','false_positive_rate','feasible']].to_string(index=False),flush=True)
    write(OUT/'summary.json',dict(stage='validation_only_feature_ablation',models=9,new_models_trained=6,validation_policies=len(all_policies),
        top_five_sensors=models[-1]['features'],families=aggregate,new_test_predictions=False,independent_improvement_established=False,
        original_submission_replaced=False,additional_sensor_types_added=False,more_sensor_observations_added=False))


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    review=ROOT/'results/hvac_sensor_review/review.json';selected=json.loads(review.read_text())['candidate_top_five']
    plan=dict(study='hvac_feature_v2',purpose='Same-data sensor-subset ablation; summer validation only',
        feature_sets=dict(engineered32=MODEL_FEATURES,raw13=RAW_FEATURES,raw5=selected),seeds=[7,42,123],primary_seed=42,
        hidden=32,dropout=.2,learning_rate=.001,batch=512,max_epochs=30,patience=6,selection='maximum validation AP; same fixed 30-policy alarm search and original targets',
        audit_sha256=sha(review),source_sha256=sha(ROOT/'data/hvac_benchmark/afdd_source.zip'),
        cohort_sha256={n:sha(OLD/f'{n}_cohort.csv') for n in ['train','validation']},
        implementation_sha256={p:sha(ROOT/p) for p in ['scripts/train_hvac_feature_study.py','src/dccad/hvac_feature_study.py']},
        old_test_already_inspected=True,new_test_scoring_authorized=False,additional_architecture_search=False,
        interpretation='Development comparison only; new reserved scenario data required for an independent improvement claim')
    plan_path=OUT/'experiment_plan.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())==plan
    else:write(plan_path,plan)
    frame=read_source(ROOT/'data/hvac_benchmark/afdd_source.zip')
    frame=frame[frame.season.isin(['winter','spring','summer'])].reset_index(drop=True)
    raw=sensor_frame(frame).to_numpy(dtype=np.float32);clean,filled=causal_clean(raw,frame.date.to_numpy(),frame.timestamp.dt.as_unit('ns').astype('int64').to_numpy())
    fit=frame.season.isin(['winter','spring']).to_numpy()&operation_mask(frame)&(~frame['Fault Detection Ground Truth'].to_numpy(bool))
    train=pd.read_csv(OLD/'train_cohort.csv');val=pd.read_csv(OLD/'validation_cohort.csv')
    yt=train.y_end.to_numpy(bool);yv=val.y_end.to_numpy(bool);models=[]
    for seed in [7,42,123]:
        models.append(dict(name=f'engineered32_seed_{seed}',family='engineered32',seed=seed,channels=32,parameters=8993,artifact_path=f'results/hvac_v1/lstm_seed_{seed}'))
    for family,features in [('raw13',RAW_FEATURES),('raw5',selected)]:
        indices=[RAW_FEATURES.index(c) for c in features];c=clean[:,indices];f=filled[:,indices]
        pre=fit_subset(c,f,fit,features);values=transform_subset(c,f,pre)
        xt=sequence_inputs(values,train.row_index,WIDTH);xv=sequence_inputs(values,val.row_index,WIDTH)
        for seed in [7,42,123]:
            r=train_one(family,seed,features,pre,xt,xv,yt,yv,frame,val,sha(plan_path));r['artifact_path']=str((OUT/r['name']).relative_to(ROOT)).replace('\\','/');models.append(r)
    write(OUT/'models.json',models)
    episodes=pd.read_csv(OLD/'validation_fault_days.csv')[['node','start_ns','end_ns','readings']]
    compare(val,episodes,models)


if __name__=='__main__':main()
