"""Bounded, reproducible Task 3 training and validation; never scores final servers."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import gc,hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier,IsolationForest
from sklearn.metrics import average_precision_score,precision_recall_curve
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dccad.data import load_reduced,BASE_FEATURES
from dccad.protocol import protocol_clean,constrained_policy_key
from dccad.preprocessing import select_features
from dccad.supervised import LSTMClassifier,QUALITY_NAMES,features_from_clean,sequence_inputs,predict_logits,trailing_statistics
from dccad.model import LSTMAutoencoder,predict_errors
from dccad.alarm_policies import PolicyEvaluator,apply_rule
from dccad.evaluation import calibrate,metrics,incident_metrics
from analyze_thresholds import pr_candidates,QUANTILES,RULES

OUT=ROOT/'results'/'task3_supervised'
CONFIGS=[
    dict(name='lstm8_p1',kind='lstm',width=8,pos_weight=1,feature_set='full'),
    dict(name='lstm8_p3',kind='lstm',width=8,pos_weight=3,feature_set='full'),
    dict(name='lstm8_p10',kind='lstm',width=8,pos_weight=10,feature_set='full'),
    dict(name='lstm24_p3',kind='lstm',width=24,pos_weight=3,feature_set='full'),
    dict(name='lstm8_corr_p3',kind='lstm',width=8,pos_weight=3,feature_set='correlation'),
    dict(name='lstm8_quality_p3',kind='lstm',width=8,pos_weight=3,feature_set='quality'),
    dict(name='gradient_boosting8',kind='boosting',width=8,feature_set='full'),
    dict(name='autoencoder8_refit',kind='autoencoder',width=8,feature_set='full'),
    dict(name='isolation_forest8_refit',kind='isolation_forest',width=8,feature_set='full')]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)

def freeze():
    p=ROOT/'results'/'protocol_v1'/'manifest.json';manifest=json.loads(p.read_text())
    assert sha(p)==(p.parent/'manifest.sha256').read_text().strip()
    for split in ['train','validation']:
        c=manifest['cohort_files'][split];assert sha(ROOT/c['path'])==c['sha256']
    plan=dict(task=3,seed=42,configs=CONFIGS,protocol_sha256=sha(p),
        cohorts={s:manifest['cohort_files'][s] for s in ['train','validation']},
        scaler='StandardScaler on finite normal calendar-training rows, all 22 sensors',clip=10.,
        neural=dict(hidden=32,dropout=.2,learning_rate=.001,batch_size=512,max_epochs=25,patience=5,
                    gradient_clip=1.,stopping='highest labelled calendar-validation average precision'),
        autoencoder=dict(hidden=64,latent=16,max_epochs=25,patience=5,stopping='lowest normal-history calendar-validation MSE'),
        boosting=dict(learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.,
            max_iter=200,checkpoints=[50,100,200],early_stopping=False,class_weight=None,
            inputs='latest, mean, population std and OLS slope over same 8-reading sensor history'),
        isolation_forest=dict(n_estimators=200,max_samples=256,contamination='auto'),
        loss='BCEWithLogitsLoss; training weights only; no oversampling',eligibility_width=24,
        targets=manifest['policy_targets'],policy_search=dict(quantiles=QUANTILES,rules=RULES,
            scopes=['global','per_node'],pr_recall_floors=[.4,.6,.8],include_best_f0_5=True),
        reserved_nodes_accessed=False,exploratory_period_scored=False,
        classifier_selection='feasible validation policy key, otherwise highest validation AP as a provisional candidate only',
        inference='scores are uncalibrated; labels, timestamps and node IDs are not model inputs',
        source_checksums={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'src/dccad/supervised.py',ROOT/'src/dccad/protocol.py']})
    path=OUT/'experiment_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        original=json.loads(path.read_text());repairs=json.loads((OUT/'implementation_repairs.json').read_text())
        # Preserve the pre-training manifest while allowing a documented plotting-only fix.
        expected=json.loads(json.dumps(original));repair=repairs[0]
        assert repair['training_or_policy_changes'] is False
        assert expected['source_checksums'][repair['source']]==repair['pre_training_sha256']
        expected['source_checksums'][repair['source']]=repair['repaired_sha256']
        if expected!=plan:raise RuntimeError('Frozen Task 3 plan changed; create a new experiment version')
        return manifest,original
    js(path,plan);return manifest,plan

def train_neural(cfg,xt,xv,yt,yv,normal_v):
    seed();ae=cfg['kind']=='autoencoder'
    model=LSTMAutoencoder(xt.shape[-1],cfg['width'],64,16) if ae else LSTMClassifier(xt.shape[-1])
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    criterion=torch.nn.MSELoss() if ae else torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(cfg['pos_weight'])))
    loader=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(yt.astype(np.float32))),
        batch_size=512,shuffle=True,generator=torch.Generator().manual_seed(42))
    best=-np.inf;wait=0;history=[];best_state=None
    for epoch in range(1,26):
        start=time.time();model.train();total=0.
        for x,y in loader:
            optimizer.zero_grad();output=model(x);loss=criterion(output,x if ae else y)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=loss.item()*len(x)
        if ae:
            score,_,_=predict_errors(model,xv[normal_v]);value=-float(score.mean());ap=None
        else:
            score=predict_logits(model,xv);ap=float(average_precision_score(yv,score));value=ap
        row=dict(epoch=epoch,train_loss=total/len(xt),validation_ap=ap,
            normal_validation_mse=-value if ae else None,seconds=time.time()-start)
        history.append(row)
        if value>best+1e-8:
            best=value;wait=0;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:wait+=1
        print(cfg['name'],json.dumps(row),flush=True)
        if wait>=5:break
    model.load_state_dict(best_state)
    return model,history,dict(best_epoch=best_epoch,epochs_run=len(history),stop_reason='patience' if wait>=5 else 'maximum_epochs')

def train_one(cfg,data,filled,scaler,reduced,train,val):
    folder=OUT/cfg['name'];folder.mkdir(exist_ok=True)
    complete=folder/'summary.json'
    if complete.exists():
        r=json.loads(complete.read_text())
        for name,digest in r['artifact_checksums'].items():assert sha(folder/name)==digest
        print('Reusing completed model:',cfg['name'],flush=True);return r
    seed();names=reduced if cfg['feature_set']=='correlation' else BASE_FEATURES.copy()
    if cfg['feature_set']=='quality':names+=QUALITY_NAMES
    pre=dict(scaler=scaler,features=names,clip=10.,eligibility_width=24,width=cfg['width'],cadence_minutes=15)
    joblib.dump(pre,folder/'preprocessing.joblib');js(folder/'preprocessing.json',{k:v for k,v in pre.items() if k!='scaler'})
    values=features_from_clean(data['raw'],filled,scaler,names)
    normal_t=(train.history_critical_count==0)&(train.history_unknown_count==0)
    normal_v=((val.history_critical_count==0)&(val.history_unknown_count==0)).to_numpy()
    training_meta=train[normal_t] if cfg['kind'] in ['autoencoder','isolation_forest'] else train
    xt=sequence_inputs(values,training_meta.row_index,cfg['width']);xv=sequence_inputs(values,val.row_index,cfg['width'])
    yt=training_meta.y_end.to_numpy(dtype=np.float32);yv=val.y_end.to_numpy(dtype=bool)
    start=time.time();kind=cfg['kind'];history=[]
    if kind in ['lstm','autoencoder']:
        model,history,stopping=train_neural(cfg,xt,xv,yt,yv,normal_v)
        cp=dict(kind=kind,state_dict=model.state_dict(),features=names,width=cfg['width'],hidden=64 if kind=='autoencoder' else 32,latent=16,dropout=.2)
        torch.save(cp,folder/'model.pt')
        if kind=='lstm':variants={'logit':predict_logits(model,xv)}
        else:
            whole,latest,_=predict_errors(model,xv);variants={'whole_window':whole,'latest_reading':latest}
    elif kind=='boosting':
        model=HistGradientBoostingClassifier(learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,
            l2_regularization=1.,max_iter=200,early_stopping=False,random_state=42)
        tx=trailing_statistics(xt);vx=trailing_statistics(xv)
        with threadpool_limits(limits=4):
            model.fit(tx,yt);best_ap=-1
            for step,pred in enumerate(model.staged_predict_proba(vx),1):
                if step in [50,100,200]:
                    ap=float(average_precision_score(yv,pred[:,1]));history.append(dict(iterations=step,validation_ap=ap))
                    if ap>best_ap:best_ap=ap;best_step=step;best_score=pred[:,1].copy()
            # Refit the selected predeclared iteration count, using exactly the same training data.
            model.set_params(max_iter=best_step);model.fit(tx,yt)
            score=model.predict_proba(vx)[:,1];np.testing.assert_allclose(score,best_score,rtol=1e-10,atol=1e-12)
        variants={'score':score};joblib.dump(model,folder/'model.joblib')
        stopping=dict(best_iterations=best_step,iterations_evaluated=[50,100,200],stop_reason='predeclared_validation_checkpoints')
    else:
        model=IsolationForest(n_estimators=200,max_samples=256,contamination='auto',random_state=42,n_jobs=2)
        model.fit(xt.reshape(len(xt),-1));variants={'anomaly_score':-model.score_samples(xv.reshape(len(xv),-1))}
        joblib.dump(model,folder/'model.joblib');stopping=dict(stop_reason='fixed_200_trees')
    np.savez_compressed(folder/'validation_scores.npz',row_index=val.row_index,node=val.node,time_ns=val.time_ns,y_end=val.y_end,**variants)
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    # A fully observed original window makes the standalone CSV demonstration exact.
    chosen=next(i for i,end in enumerate(val.row_index) if np.isfinite(data['raw'][end-23:end+1]).all() and (data['raw'][end-23:end+1]>=0).all())
    end=int(val.iloc[chosen].row_index)
    example=pd.DataFrame(data['raw'][end-23:end+1],columns=BASE_FEATURES)
    example.insert(0,'timestamp',pd.to_datetime(data['time_ns'][end-23:end+1],utc=True))
    example.to_csv(folder/'example_window.csv',index=False)
    js(folder/'example_metadata.json',dict(validation_position=chosen,row_index=end,node=int(val.iloc[chosen].node),
        expected_scores={k:float(s[chosen]) for k,s in variants.items()}))
    artifacts=['preprocessing.joblib','preprocessing.json','validation_scores.npz','model.pt' if kind in ['lstm','autoencoder'] else 'model.joblib']
    result=dict(**cfg,features=names,seed=42,training_windows=len(xt),training_positives=int(yt.sum()),
        validation_windows=len(xv),validation_positives=int(yv.sum()),seconds=time.time()-start,**stopping,
        score_metrics={k:dict(average_precision=float(average_precision_score(yv,s))) for k,s in variants.items()},
        artifact_checksums={p:sha(folder/p) for p in artifacts},reserved_nodes_scored=False)
    js(complete,result);del model,xt,xv,values;gc.collect()
    print('Completed',cfg['name'],result['score_metrics'],flush=True);return result

def evaluate(manifest,plan,val,summaries):
    episodes=pd.read_csv(ROOT/'results/protocol_v1/validation_incident_coverage.csv')[['node','start_ns','end_ns','readings']]
    engine=PolicyEvaluator(val,episodes,15);normal=(val.history_critical_count==0)&(val.history_unknown_count==0)
    records=[];specs={};scores={};curves={};limits=[]
    for r in summaries:
        z=np.load(OUT/r['name']/'validation_scores.npz')
        for mode in r['score_metrics']:
            score=z[mode].astype(np.float64);key=r['name']+'__'+mode;scores[key]=score;engine.register_scores(key,score)
            candidates,ceiling,curve=pr_candidates(val.y_end,score);curves[key]=curve
            for floor,v in ceiling.items():limits.append(dict(model=r['name'],mode=mode,recall_floor=float(floor),**v))
            thresholds=[]
            for q in QUANTILES:
                for per_node in [False,True]:thresholds.append(('normal_percentile','per_node' if per_node else 'global',q,calibrate(val[normal],score[normal],q,per_node)))
            thresholds += [(method,'global',None,t) for method,t in candidates]
            if r['kind']=='lstm':thresholds.append(('default_zero_logit','global',None,0.))
            elif r['kind']=='boosting':thresholds.append(('default_score_0.5','global',None,.5))
            for method,scope,q,threshold in thresholds:
                for rule in RULES:
                    pred=apply_rule(val,score,threshold,rule)
                    row=dict(policy_id=f'p{len(records)+1:04d}',model=r['name'],kind=r['kind'],score_mode=mode,
                        threshold_method=method,scope=scope,quantile=q,rule=rule,**engine.evaluate(pred,key))
                    row['feasible']=constrained_policy_key(row) is not None;records.append(row)
                    spec=dict(score_key=key,threshold=threshold,rule=rule)
                    if isinstance(threshold,dict):spec['unseen_node_global_threshold']=float(np.percentile(score[normal],q))
                    specs[row['policy_id']]=spec
    pd.DataFrame(records).to_csv(OUT/'policy_comparison.csv',index=False);js(OUT/'policy_specs.json',specs)
    pd.DataFrame(limits).to_csv(OUT/'global_threshold_precision_limits.csv',index=False)
    comparisons=[];pernode=[]
    for r in summaries:
        rows=[x for x in records if x['model']==r['name']];feasible=[x for x in rows if x['feasible']]
        representative=max(feasible,key=constrained_policy_key) if feasible else max([x for x in rows if x['recall']>=.4],key=lambda x:(x['precision'],x['event_recall_observable'],-x['false_bursts_per_node_day']))
        rep=dict(representative,features=len(r['features']),training_windows=r['training_windows'],status='feasible' if feasible else 'diagnostic_at_recall_40_percent')
        comparisons.append(rep);js(OUT/r['name']/'validation_policy.json',dict(status=rep['status'],metrics=representative,**specs[representative['policy_id']]))
        score=scores[specs[rep['policy_id']]['score_key']];threshold=specs[rep['policy_id']]['threshold'];pred=apply_rule(val,score,threshold,rep['rule'])
        pd.DataFrame(dict(node=val.node,time_ns=val.time_ns,row_index=val.row_index,y_end=val.y_end,score=score,pred=pred)).to_csv(OUT/r['name']/'validation_decisions.csv',index=False)
        # Reference evaluator cross-checks the efficient policy implementation.
        standard=metrics(val.y_end,score,pred);events,_=incident_metrics(val,pred,episodes,cadence_minutes=15)
        for k in ['tp','fp','fn','tn','precision','recall','f0_5','false_positive_rate']:np.testing.assert_allclose(standard[k],rep[k])
        for k in ['false_bursts','detected','event_recall_observable']:np.testing.assert_allclose(events[k],rep[k])
        for node in val.node.unique():
            keep=val.node==node;m=val[keep].reset_index(drop=True);ep=episodes[episodes.node==node]
            e=PolicyEvaluator(m,ep,15);e.register_scores('score',score[keep]);pernode.append(dict(model=r['name'],node=int(node),**e.evaluate(pred[keep],'score')))
    pd.DataFrame(comparisons).to_csv(OUT/'model_comparison.csv',index=False);pd.DataFrame(pernode).to_csv(OUT/'per_node_validation.csv',index=False)
    neural=[x for x in records if x['kind']=='lstm' and x['feasible']]
    if neural:
        selected=max(neural,key=constrained_policy_key);candidate=selected['model'];status='feasible_validation_candidate'
    else:
        selected=None;candidate=max([r for r in summaries if r['kind']=='lstm'],key=lambda r:r['score_metrics']['logit']['average_precision'])['name'];status='no_feasible_neural_policy'
    feasible=[x for x in records if x['feasible']]
    summary=dict(task=3,status=status,provisional_dl_candidate=candidate,selected_neural_policy=selected,
        feasible_neural_policies=len(neural),feasible_policies=len(feasible),policies_evaluated=len(records),
        models_trained=len(summaries),validation_rows=len(val),validation_positives=int(val.y_end.sum()),
        validation_prevalence=float(val.y_end.mean()),independent_holdout_scored=False,
        exploratory_period_scored=False,dashboard_default_changed=False,
        protocol_sha256=plan['protocol_sha256'],next_task=4,
        caveat='Single-seed validation development results; release and final accuracy are not established.')
    js(OUT/'summary.json',summary);plot(curves,records,comparisons)
    report(summary,comparisons,summaries,plan)
    print(json.dumps(summary,indent=2),flush=True)

def plot(curves,records,comparisons):
    fig,ax=plt.subplots(figsize=(10,6))
    for name in ['lstm8_p1','lstm8_p3','lstm8_p10','lstm24_p3','lstm8_corr_p3','lstm8_quality_p3','gradient_boosting8','autoencoder8_refit','isolation_forest8_refit']:
        key=next(k for k in curves if k.startswith(name+'__'));p,r,_=curves[key];ax.step(r,p,where='post',label=name,alpha=.8)
    ax.axhline(.30,color='black',ls='--',lw=1);ax.axvline(.40,color='black',ls=':',lw=1)
    ax.set(xlabel='Endpoint recall',ylabel='Endpoint precision',title='Task 3: shared chronological validation (single seed)',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'precision_recall.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for name in set(r['model'] for r in records):
        rows=[r for r in records if r['model']==name];ax.scatter([r['false_positive_rate'] for r in rows],[r['recall'] for r in rows],s=12,alpha=.6,label=name)
    ax.axvline(.01,color='black',ls='--');ax.axhline(.4,color='black',ls=':');ax.set_xscale('symlog',linthresh=.001)
    ax.set(xlabel='False positive rate',ylabel='Recall',title='Reading-level policy tradeoffs on validation',xlim=(0,1.03))
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'false_positive_tradeoff.png',dpi=170);plt.close(fig)
    neural=[x for x in comparisons if x['kind']=='lstm'];fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,k,label in zip(axs,['average_precision','precision'],['Average precision','Precision at diagnostic/feasible operating point']):
        ax.bar([r['model'] for r in neural],[r[k] for r in neural]);ax.tick_params(axis='x',rotation=35);ax.set_ylabel(label)
        ax.set_ylim(0,max(r[k] for r in neural)*1.25)
    fig.suptitle('Controlled supervised LSTM ablations; seed 42');fig.tight_layout();fig.savefig(OUT/'ablations.png',dpi=170);plt.close(fig)

def report(s,rows,models,plan):
    text=['# Task 3 — supervised training and fair validation comparison','',
        f"Trained {len(models)} models, including six predefined LSTM classifiers, on calendar training data. All candidates use the same {s['validation_rows']:,} eligible validation endpoints ({s['validation_positives']} critical; prevalence {s['validation_prevalence']:.2%}). Reserved servers and the exploratory period were not scored.",'',
        '## Results','',
        '| Model | AP | Precision | Recall | FPR | False bursts/day | Incidents | Feasible? |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:text.append(f"| {r['model']} | {r['average_precision']:.4f} | {r['precision']:.2%} | {r['recall']:.2%} | {r['false_positive_rate']:.2%} | {r['false_bursts_per_node_day']:.3f} | {r['detected']}/{r['observable_episodes']} | {r['feasible']} |")
    text += ['',f"**{s['feasible_neural_policies']} neural policies and {s['feasible_policies']} total policies meet all five frozen targets.** {s['policies_evaluated']} policies were evaluated. The provisional DL candidate is `{s['provisional_dl_candidate']}`.",'',
        'Each row shows the best feasible policy if one exists; otherwise it shows a diagnostic policy maximizing precision with recall at least 40%. A diagnostic policy is not a recommended release policy. AP is the ranking measure for the score mode used in that row. All policies, exact thresholds and raw decisions are saved alongside this report.',
        '', '## Training and safeguards','',
        '- Training has 111,070 known endpoint targets, including 1,445 critical readings. Unknown historical labels are permitted as context but never supplied as features. Validation retains its natural class prevalence.',
        '- Scaling uses finite normal training rows only. Features are clipped at 10 training standard deviations. Correlation selection uses normal training statistics only; the quality ablation appends fractions of missing, invalid and causally filled sensors.',
        '- Neural training uses Adam, batch 512, up to 25 epochs, patience five, gradient clipping one and seed 42. The supervised checkpoint maximizes validation AP. Loss weights 1, 3 and 10 affect training only.',
        '- The tree classifier sees latest values, means, standard deviations and slopes from the same eight readings. Its 50/100/200 iteration checkpoint is chosen on the explicit chronological validation cohort; random internal validation is disabled.',
        '- The autoencoder and Isolation Forest are freshly trained on 104,230 fully known normal histories from the frozen training cohort. Autoencoder stopping uses the 68,978 normal validation histories. They are comparable references under the shared split; they replace neither the historical artifacts nor their reported results.',
        '- Policies use the frozen target gate: precision ≥30%, recall ≥40%, FPR ≤1%, false bursts ≤0.1/scored node-day, observable incident recall ≥60%. Persistence resets at gaps and nodes. Per-node thresholds retain a validation-only global fallback for unseen servers.',
        '- Scores are uncalibrated. Nagios labels concern general current server health, not confirmed cooling faults or future fault forecasting.',
        '', '## Limits and next step','',
        'These are development results from one seed, with configuration/checkpoint/policy selection on the same validation period. They are optimistic for deployment. Task 4 must repeat the finalist with seeds 7, 42 and 123, freeze the full pipeline using validation only, and then evaluate the independent servers once. The dashboard default and existing report/PPT remain unchanged until a verified release is available.',
        '', '## Reproduce','',
        'Run `.venv\\Scripts\\python.exe scripts\\train_task3.py` to reproduce or resume the frozen experiment. Run `scripts\\verify_task3.py` to reproduce policies and standalone inference. `experiment_plan.json` records input/source hashes, fixed configurations and stopping rules. Individual folders contain weights, preprocessing, histories, validation scores and example CSV windows.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest,plan=freeze()
    data=load_reduced(ROOT/'0.tar',ROOT/'artifacts/reduced_base.npz')
    assert list(data['features'])==BASE_FEATURES
    assert not set(data['node']).intersection(manifest['independent_holdout_nodes'])
    sanitized,filled,split=protocol_clean(data)
    train=pd.read_csv(ROOT/manifest['cohort_files']['train']['path']);val=pd.read_csv(ROOT/manifest['cohort_files']['validation']['path'])
    for meta,which in [(train,0),(val,1)]:
        idx=meta.row_index.to_numpy();np.testing.assert_array_equal(meta.y_end,data['label'][idx]==2)
        assert (split[idx]==which).all()
    fit=(split==0)&(data['label']==0)&np.isfinite(filled).all(axis=1)
    scaler=StandardScaler().fit(filled[fit]);joblib.dump(scaler,OUT/'training_scaler.joblib')
    reduced,selection,corr=select_features(data,sanitized,filled,split)
    selection.to_csv(OUT/'feature_selection.csv',index=False);corr.to_csv(OUT/'training_correlation.csv')
    js(OUT/'preprocessing_audit.json',dict(scaler_fit_rows=int(fit.sum()),scaler_fit_latest_ns=int(data['time_ns'][fit].max()),
        correlation_selected_features=reduced,train_rows=len(train),validation_rows=len(val),reserved_nodes_read=False))
    summaries=[train_one(c,data,filled,scaler,reduced,train,val) for c in CONFIGS]
    evaluate(manifest,plan,val,summaries)

if __name__=='__main__':main()
