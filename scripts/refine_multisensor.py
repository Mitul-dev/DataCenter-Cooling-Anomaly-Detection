"""Logged validation refinement; the reserved calendar test remains unopened."""
import copy
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import average_precision_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import train_multisensor as initial
from prepare_multisensor import DATA,CLASSES,FEATURES,sha,write
from dccad.multisensor import SensorClassifier,predict,fit_preprocessing,classification_metrics
from dccad.alarm_policies import apply_rule
OUT=ROOT/'results/multisensor_v2'
OLD=ROOT/'results/multisensor_v1'


def train(family,pre,shared):
    folder=OUT/family;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():
        r=json.loads((folder/'summary.json').read_text())
        for file,digest in r['checksums'].items():assert sha(folder/file)==digest
        return
    xt,yt,xv,yv,_=shared
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=SensorClassifier(len(pre['features']),len(CLASSES));opt=torch.optim.Adam(model.parameters(),lr=.001)
    count=np.bincount(yt,minlength=5);weights=len(yt)/(5*count);weights[0]*=2
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    loader=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(yt)),batch_size=512,shuffle=True,
                      generator=torch.Generator().manual_seed(42))
    best=-np.inf;wait=0;history=[]
    for epoch in range(1,81):
        start=time.perf_counter();model.train();total=0.
        for x,y in loader:
            opt.zero_grad();loss=loss_fn(model(x),y);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(x)
        p=predict(model,xv)
        ap=float(np.mean([average_precision_score(yv==k,p[:,k]) for k in range(5)]))
        history.append(dict(epoch=epoch,train_loss=total/len(xt),validation_macro_ap_all_classes=ap,seconds=time.perf_counter()-start))
        if ap>best+1e-8:
            best=ap;wait=0;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:wait+=1
        print(family,'epoch',epoch,'five-class macro AP',round(ap,6),flush=True)
        if wait>=8:break
    torch.save(dict(state_dict=state,features=pre['features'],classes=CLASSES,hidden=32,width=30),folder/'model.pt')
    write(folder/'preprocessing.json',pre);pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    write(folder/'summary.json',dict(family=family,seed=42,features=pre['features'],channels=len(pre['features']),classes=CLASSES,
          parameters=sum(p.numel() for p in model.parameters()),train_windows=len(xt),checkpoint_validation_windows=len(xv),
          best_epoch=best_epoch,epochs_run=len(history),validation_macro_ap_all_classes=best,
          class_counts=count.tolist(),class_weights=weights.tolist(),test_predictions_made=False,
          checksums={f:sha(folder/f) for f in ['model.pt','preprocessing.json','history.csv']}))


def fast_metrics(y,pred):
    c=np.bincount(y*5+pred,minlength=25).reshape(5,5)
    recalls=np.diag(c)/c.sum(axis=1)
    f1=2*np.diag(c)/np.maximum(c.sum(axis=0)+c.sum(axis=1),1)
    return float(f1.mean()),float(1-recalls[0]),recalls


def tune(folder,meta,p):
    y=meta.y.to_numpy(int);normal=y==0;rows=[];specs=[]
    quantiles=[90.,95.,97.5,99.,99.25,99.5,99.75,99.9,99.95,100.]
    thresholds={q:np.percentile(p[normal,1:],q,axis=0) for q in quantiles}
    configs=[(f'equal_p{q}',[q]*4) for q in quantiles]
    for special in range(4):
        for q in [99.,99.25,99.5,99.75]:
            for other in [99.9,99.95]:
                vector=[other]*4;vector[special]=q
                configs.append((f'budget_sensor{special+1}_p{q}_others_p{other}',vector))
    for rule in ['immediate','consecutive_2','two_of_three']:
        # Cache each univariate persistence result; policy combinations reuse it.
        hits={(k,q):apply_rule(meta,p[:,k+1],thresholds[q][k],rule,1) for k in range(4) for q in quantiles}
        for method,vector in configs:
            active=np.column_stack([hits[k,q] for k,q in enumerate(vector)])
            pred=np.where(active.any(axis=1),np.where(active,p[:,1:],-1).argmax(axis=1)+1,0)
            f1,fpr,recall=fast_metrics(y,pred)
            feasible=fpr<=.01 and np.all(recall[1:]>=.4)
            rows.append(dict(method=method,rule=rule,macro_f1=f1,healthy_fpr=fpr,recall=recall.tolist(),feasible=bool(feasible)))
            specs.append(dict(method=method,rule=rule,thresholds=[float(thresholds[q][k]) for k,q in enumerate(vector)],quantiles=vector))
    best=max(range(len(rows)),key=lambda i:(rows[i]['feasible'],rows[i]['macro_f1'],-rows[i]['healthy_fpr']))
    from dccad.multisensor import decisions
    spec=specs[best];pred=decisions(meta,p,spec['thresholds'],spec['rule'])
    metrics=classification_metrics(y,pred,p,CLASSES,meta)
    selected=dict(**spec,validation_metrics=dict(**metrics,feasible=rows[best]['feasible']),classes=CLASSES)
    write(folder/'validation_policy.json',selected)
    write(folder/'validation_policy_search.json',dict(policies=rows,specifications=specs))
    np.savez_compressed(folder/'validation_scores.npz',probability=p,y=y,node=meta.node,time_ns=meta.time_ns)
    print('Refined validation',folder.name,'FPR',round(metrics['healthy_fpr'],5),'recalls',rows[best]['recall'],'feasible',rows[best]['feasible'],flush=True)
    return selected


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    inherited=json.loads((OLD/'experiment_plan.json').read_text())
    scenarios=json.loads((OLD/'scenarios.json').read_text())
    plan=copy.deepcopy(inherited)
    plan.update(study='multisensor_v2',validation_refinement='V1 failed 1% healthy FPR plus 40% each-sensor recall; test remains unscored',
                previous_validation_study_sha256=sha(OLD/'frozen_model.json'),
                previous_test_predictions_made=False,maximum_epochs=80,patience=8,
                criterion='maximum five-class macro AP including healthy; healthy class training weight doubled',
                feature_selection='Reuse frozen training-only V1 top five; no reranking',
                threshold_selection='126 policies per family: equal healthy quantile or one sensor with a larger false-alarm budget, three causal persistence rules',
                candidate_families=['v1_all_inputs','v1_top5_inputs','all_inputs','top5_inputs'],
                reported_total_new_models=4,seeds=[42],
                implementation={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/dccad/multisensor.py',ROOT/'scripts/train_multisensor.py']})
    if (OUT/'experiment_plan.json').exists():assert json.loads((OUT/'experiment_plan.json').read_text())==plan
    else:write(OUT/'experiment_plan.json',plan)
    write(OUT/'scenarios.json',scenarios)
    selected=json.loads((OLD/'feature_selection.json').read_text())['selected']
    pre=json.loads((OLD/'all_inputs/preprocessing.json').read_text());shared=initial.arrays(pre,scenarios)
    train('all_inputs',pre,shared)
    indices=[FEATURES.index(c) for c in selected];small={**pre,'features':selected}
    for key in ['median','mean','scale']:small[key]=[pre[key][i] for i in indices]
    sub=(shared[0][:,:,indices].copy(),shared[1],shared[2][:,:,indices].copy(),shared[3],shared[4])
    train('top5_inputs',small,sub)
    import shutil
    for name in ['all_inputs','top5_inputs']:
        folder=OUT/f'v1_{name}';folder.mkdir(exist_ok=True)
        for file in ['model.pt','preprocessing.json','summary.json','history.csv']:
            shutil.copyfile(OLD/name/file,folder/file)
    policies={}
    for family in plan['candidate_families']:
        folder=OUT/family
        if family.startswith('v1_'):
            saved=np.load(OLD/family[3:]/'validation_scores.npz');p=saved['probability']
            parts=[]
            for r in scenarios:
                meta=pd.read_parquet(DATA/f"{r['scenario']}_cohort.parquet")
                parts.append(meta[meta.partition=='validation'])
            meta=pd.concat(parts,ignore_index=True)
            np.testing.assert_array_equal(saved['node'],meta.node)
        else:meta,p=initial.full_scores(folder,scenarios,'validation')
        policies[family]=tune(folder,meta,p)
    winner=max(policies,key=lambda f:(policies[f]['validation_metrics']['feasible'],policies[f]['validation_metrics']['macro_f1'],-policies[f]['validation_metrics']['healthy_fpr']))
    frozen=dict(family=winner,classes=CLASSES,policy=policies[winner],seed=42,
                plan_sha256=sha(OUT/'experiment_plan.json'),
                families={f:{name:sha(OUT/f/name) for name in ['model.pt','preprocessing.json','validation_policy.json']} for f in policies},
                qualified=bool(policies[winner]['validation_metrics']['feasible']),test_predictions_made=False)
    write(OUT/'frozen_model.json',frozen)
    print('V2 frozen primary',winner,'qualified',frozen['qualified'],flush=True)


if __name__=='__main__':main()
