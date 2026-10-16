"""Fixed multiclass LSTM comparison using training and validation only."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, log_loss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from dccad.multisensor import SensorClassifier, fit_preprocessing, transform, predict, decisions, classification_metrics
from dccad.supervised import sequence_inputs, trailing_statistics
from prepare_multisensor import OUT, DATA, FEATURES, CLASSES, sha, write


def processed(frame, pre):
    return transform(frame[pre['features']].to_numpy(np.float32), frame.timestamp.dt.strftime('%Y-%m-%d').to_numpy(),
                     frame.timestamp.dt.as_unit('ns').astype('int64').to_numpy(), pre)


def arrays(pre, scenarios):
    train_x=[]; train_y=[]; val_x=[]; val_y=[]; train_meta=[]
    for record in scenarios:
        name=record['scenario'];frame=pd.read_parquet(DATA/f'{name}.parquet')
        meta=pd.read_parquet(DATA/f'{name}_cohort.parquet')
        values=processed(frame, pre)
        minutes=meta.timestamp.dt.hour*60+meta.timestamp.dt.minute
        for split, stride in [('train',60),('validation',15)]:
            g=meta[(meta.partition==split)&(minutes%stride==stride-1)]
            x=sequence_inputs(values,g.row_index,30)
            if split=='train':
                train_x.append(x);train_y.append(g.y.to_numpy(np.int64));train_meta.append(g)
            else:
                val_x.append(x);val_y.append(g.y.to_numpy(np.int64))
    return np.concatenate(train_x),np.concatenate(train_y),np.concatenate(val_x),np.concatenate(val_y),pd.concat(train_meta,ignore_index=True)


def rank_inputs(xt,yt,meta):
    table_path=OUT/'training_feature_ranking.csv'
    if table_path.exists():
        return pd.read_csv(table_path).feature.head(5).tolist()
    tx=trailing_statistics(xt)
    held=(meta.timestamp.dt.isocalendar().week.to_numpy(int)%6)==4
    selector=ExtraTreesClassifier(n_estimators=100,max_depth=12,min_samples_leaf=10,
                                  class_weight='balanced',random_state=42,n_jobs=4)
    selector.fit(tx[~held],yt[~held])
    vx=tx[held]; y=yt[held]; hm=meta.loc[held].reset_index(drop=True)
    weights=1/np.bincount(y,minlength=len(CLASSES))[y]
    base=log_loss(y,selector.predict_proba(vx),sample_weight=weights,labels=np.arange(len(CLASSES)))
    groups={}
    for _,g in hm.groupby(['scenario','date'],sort=False):
        idx=g.index.to_numpy();pattern=tuple((g.timestamp.dt.hour*60+g.timestamp.dt.minute).tolist())
        groups.setdefault(pattern,[]).append(idx)
    rows=[];rng=np.random.default_rng(2026)
    # Shuffle each complete daily trajectory jointly across its four history statistics.
    maps=[]
    for repeat in range(3):
        donor=np.arange(len(hm))
        for block in groups.values():
            for recipient,source in zip(block,rng.permutation(len(block))):
                donor[recipient]=block[source]
        maps.append(donor)
    for j,feature in enumerate(FEATURES):
        columns=[j+len(FEATURES)*s for s in range(4)];losses=[]
        for donor in maps:
            perm=vx.copy();perm[:,columns]=vx[donor][:,columns]
            losses.append(log_loss(y,selector.predict_proba(perm),sample_weight=weights,labels=np.arange(len(CLASSES)))-base)
        rows.append(dict(feature=feature,importance_mean=float(np.mean(losses)),importance_sd=float(np.std(losses,ddof=1))))
    ranking=pd.DataFrame(rows).sort_values(['importance_mean','feature'],ascending=[False,True])
    ranking.to_csv(table_path,index=False)
    write(OUT/'feature_selection.json',dict(method='Whole-day grouped permutation on held-out training weeks',
                                           repeats=3,seed=2026,selector_seed=42,selector_balanced_log_loss=float(base),
                                           candidate_features=len(FEATURES),selected=ranking.feature.head(5).tolist(),
                                           validation_or_test_used=False,
                                           limitation='Conditional selector importance; correlated inputs can share importance; not globally optimal'))
    return ranking.feature.head(5).tolist()


def train(family,pre,scenarios,shared=None):
    folder=OUT/family;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():
        result=json.loads((folder/'summary.json').read_text())
        for name,digest in result['checksums'].items():assert sha(folder/name)==digest
        return
    xt,yt,xv,yv,_=shared if shared is not None else arrays(pre,scenarios)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=SensorClassifier(len(pre['features']),len(CLASSES));optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    counts=np.bincount(yt,minlength=len(CLASSES));weights=len(yt)/(len(CLASSES)*counts)
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    loader=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(yt)),batch_size=512,shuffle=True,
                      generator=torch.Generator().manual_seed(42))
    best=-np.inf;patience=0;history=[]
    for epoch in range(1,21):
        start=time.perf_counter();model.train();total=0.
        for x,y in loader:
            optimizer.zero_grad();loss=criterion(model(x),y);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=loss.item()*len(x)
        p=predict(model,xv)
        ap=float(np.mean([average_precision_score(yv==k,p[:,k]) for k in range(1,len(CLASSES))]))
        history.append(dict(epoch=epoch,train_loss=total/len(xt),validation_fault_macro_ap=ap,seconds=time.perf_counter()-start))
        if ap>best+1e-8:
            best=ap;patience=0;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:patience+=1
        print(family,'epoch',epoch,'macro AP',round(ap,5),'seconds',round(history[-1]['seconds'],1),flush=True)
        if patience>=4:break
    torch.save(dict(state_dict=state,features=pre['features'],classes=CLASSES,hidden=32,width=30),folder/'model.pt')
    write(folder/'preprocessing.json',pre);pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    write(folder/'summary.json',dict(family=family,seed=42,features=pre['features'],channels=len(pre['features']),
                                    classes=CLASSES,parameters=sum(p.numel() for p in model.parameters()),
                                    train_windows=len(xt),checkpoint_validation_windows=len(xv),
                                    best_epoch=best_epoch,epochs_run=len(history),validation_fault_macro_ap=best,
                                    class_counts=counts.tolist(),class_weights=weights.tolist(),
                                    test_predictions_made=False,
                                    checksums={f:sha(folder/f) for f in ['model.pt','preprocessing.json','history.csv']}))


def full_scores(folder,scenarios,split):
    pre=json.loads((folder/'preprocessing.json').read_text())
    cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    model=SensorClassifier(len(pre['features']),len(CLASSES));model.load_state_dict(cp['state_dict'])
    parts=[];scores=[]
    for record in scenarios:
        name=record['scenario'];meta=pd.read_parquet(DATA/f'{name}_cohort.parquet')
        g=meta[meta.partition==split].reset_index(drop=True)
        frame=pd.read_parquet(DATA/f'{name}.parquet');values=processed(frame,pre)
        # Only the requested split's windows enter model inference.
        p=predict(model,sequence_inputs(values,g.row_index,30))
        parts.append(g);scores.append(p)
    return pd.concat(parts,ignore_index=True),np.concatenate(scores)


def tune(folder,meta,p):
    healthy=meta.y.to_numpy()==0;rows=[];specs=[]
    for method in ['p90','p95','p97.5','p99','p99.5','p99.9','p100','fixed0.5','fixed0.7','fixed0.9']:
        threshold=np.percentile(p[healthy,1:],float(method[1:]),axis=0).tolist() if method.startswith('p') else [float(method[5:])]*4
        for rule in ['immediate','consecutive_2','two_of_three']:
            pred=decisions(meta,p,threshold,rule)
            metrics=classification_metrics(meta.y,pred,p,CLASSES,meta)
            feasible=metrics['healthy_fpr']<=.01 and all(r['recall']>=.4 for r in metrics['per_class'][1:])
            rows.append(dict(method=method,rule=rule,feasible=feasible,**metrics))
            specs.append(dict(thresholds=threshold,rule=rule,method=method))
    # Validation macro F1 selects both policy and feature family; primary seed stays fixed.
    winner=max(range(len(rows)),key=lambda i:(rows[i]['feasible'],rows[i]['macro_f1'],-rows[i]['healthy_fpr']))
    selected=dict(**specs[winner],validation_metrics=rows[winner],classes=CLASSES)
    write(folder/'validation_policy.json',selected);write(folder/'validation_policy_search.json',rows)
    np.savez_compressed(folder/'validation_scores.npz',probability=p,y=meta.y,node=meta.node,time_ns=meta.time_ns)
    return selected


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    plan=json.loads((OUT/'experiment_plan.json').read_text());assert plan['primary_seed']==42
    scenarios=json.loads((OUT/'scenarios.json').read_text())
    for r in scenarios:
        assert sha(DATA/f"{r['scenario']}.parquet")==r['cache_sha256']
        assert sha(DATA/f"{r['scenario']}_cohort.parquet")==r['cohort_sha256']
    normal=next(r for r in scenarios if r['class_id']==0)
    frame=pd.read_parquet(DATA/f"{normal['scenario']}.parquet")
    fit=frame.operating & (frame.partition=='train')
    pre=fit_preprocessing(frame[FEATURES].to_numpy(np.float32),fit.to_numpy(),FEATURES)
    shared=arrays(pre,scenarios);selected=rank_inputs(shared[0],shared[1],shared[4])
    write(OUT/'training_implementation.json',dict(plan_sha256=sha(OUT/'experiment_plan.json'),
          files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/dccad/multisensor.py',ROOT/'scripts/prepare_multisensor.py']},
          selected_five=selected,normal_training_fit_rows=int(fit.sum()),test_predictions_made=False))
    train('all_inputs',pre,scenarios,shared)
    indices=[FEATURES.index(c) for c in selected]
    small={**pre,'features':selected}
    for key in ['median','mean','scale']:small[key]=[pre[key][i] for i in indices]
    small_shared=(shared[0][:,:,indices].copy(),shared[1],shared[2][:,:,indices].copy(),shared[3],shared[4])
    train('top5_inputs',small,scenarios,small_shared)
    policies={}
    for family in ['all_inputs','top5_inputs']:
        folder=OUT/family;meta,p=full_scores(folder,scenarios,'validation')
        policies[family]=tune(folder,meta,p)
        print('Validation',family,policies[family]['validation_metrics']['macro_f1'],policies[family]['validation_metrics']['healthy_fpr'],flush=True)
    winner=max(policies,key=lambda f:(policies[f]['validation_metrics']['feasible'],policies[f]['validation_metrics']['macro_f1'],-policies[f]['validation_metrics']['healthy_fpr']))
    frozen=dict(family=winner,classes=CLASSES,policy=policies[winner],seed=42,
                plan_sha256=sha(OUT/'experiment_plan.json'),
                families={f:{name:sha(OUT/f/name) for name in ['model.pt','preprocessing.json','validation_policy.json']} for f in policies},
                qualified=bool(policies[winner]['validation_metrics']['feasible']),test_predictions_made=False)
    if (OUT/'frozen_model.json').exists():assert json.loads((OUT/'frozen_model.json').read_text())==frozen
    else:write(OUT/'frozen_model.json',frozen)
    print('Frozen selected family:',winner,'qualified:',frozen['qualified'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
