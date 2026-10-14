"""Repeat the fixed Task 3 LSTM and gate release without touching final servers."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib,json,random,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import average_precision_score,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dccad.data import load_reduced,BASE_FEATURES
from dccad.protocol import protocol_clean,constrained_policy_key
from dccad.supervised import LSTMClassifier,features_from_clean,sequence_inputs,predict_logits
from dccad.alarm_policies import PolicyEvaluator,apply_rule
from dccad.evaluation import calibrate,metrics,incident_metrics
from dccad.robustness import choose_seed_policy,release_gate,node_bootstrap
from analyze_thresholds import pr_candidates,QUANTILES,RULES
from audit_task1 import data_digest,digest

OUT=ROOT/'results/task4_robustness'
SOURCE=ROOT/'results/task3_supervised/lstm8_p1'
SEEDS=[7,42,123]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def freeze():
    manifest_path=ROOT/'results/protocol_v1/manifest.json'
    manifest=json.loads(manifest_path.read_text());assert sha(manifest_path)==(manifest_path.parent/'manifest.sha256').read_text().strip()
    previous=json.loads((ROOT/'results/task3_supervised/summary.json').read_text())
    assert previous['provisional_dl_candidate']=='lstm8_p1'
    t3=json.loads((ROOT/'results/task3_supervised/experiment_plan.json').read_text())
    plan=dict(task=4,configuration='lstm8_p1',seeds=SEEDS,primary_seed=42,
        feature_names=BASE_FEATURES,width=8,eligibility_width=24,clip=10.,neural=t3['neural'],pos_weight=1,
        protocol_sha256=sha(manifest_path),cohorts=t3['cohorts'],targets=manifest['policy_targets'],
        policy_search=t3['policy_search'],include_default_logit_zero=True,
        inputs_sha256={p:sha(SOURCE/p) for p in ['model.pt','preprocessing.joblib','validation_scores.npz']},
        task3_summary_sha256=sha(ROOT/'results/task3_supervised/summary.json'),
        policy_selection='same feasible-policy key as Task 3; otherwise max precision with recall >=40% for diagnosis only',
        release_rule='all three seeds must have a feasible validation policy; use seed 42, never cherry-pick a seed; its global fallback must also pass the same validation targets',
        uncertainty=dict(method='whole-node percentile bootstrap on each fixed selected validation policy',replicates=2000,seed=20261014,
            limitation='Nine development nodes from one rack; conditional on selected policies. Selection uncertainty and future drift are not captured.'),
        heldout_access='no reserved node measurements or labels; no holdout scoring unless a release passes the validation gate in a later task',
        training_configuration_search=False)
    p=OUT/'experiment_plan.json'
    if p.exists() and json.loads(p.read_text())!=plan:raise RuntimeError('Task 4 frozen inputs changed; use a new experiment version')
    write(p,plan)
    return manifest,plan


def run_seed(seed,xtrain,xval,ytrain,yval,plan):
    folder=OUT/f'seed_{seed}';folder.mkdir(exist_ok=True)
    p=folder/'summary.json'
    if p.exists():
        summary=json.loads(p.read_text())
        for name,hash_value in summary['artifact_checksums'].items():assert sha(folder/name)==hash_value
        return summary
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=LSTMClassifier(len(BASE_FEATURES),32,.2)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.))
    loader=DataLoader(TensorDataset(torch.from_numpy(xtrain),torch.from_numpy(ytrain.astype(np.float32))),
        batch_size=512,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=-np.inf;wait=0;history=[];state=None
    for epoch in range(1,26):
        start=time.time();model.train();total=0.
        for x,y in loader:
            optimizer.zero_grad();loss=criterion(model(x),y);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=loss.item()*len(x)
        score=predict_logits(model,xval);ap=float(average_precision_score(yval,score))
        history.append(dict(epoch=epoch,train_loss=total/len(xtrain),validation_ap=ap,seconds=time.time()-start))
        if ap>best+1e-8:
            best=ap;wait=0;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:wait+=1
        print('Seed',seed,'epoch',epoch,'validation AP',round(ap,6),flush=True)
        if wait>=5:break
    model.load_state_dict(state);score=predict_logits(model,xval)
    torch.save(dict(kind='lstm',state_dict=state,features=BASE_FEATURES,width=8,hidden=32,dropout=.2),folder/'model.pt')
    shutil.copy2(SOURCE/'preprocessing.joblib',folder/'preprocessing.joblib')
    shutil.copy2(SOURCE/'preprocessing.json',folder/'preprocessing.json')
    shutil.copy2(SOURCE/'example_window.csv',folder/'example_window.csv')
    example=json.loads((SOURCE/'example_metadata.json').read_text())
    example['expected_scores']={'logit':float(score[example['validation_position']])};write(folder/'example_metadata.json',example)
    meta=pd.read_csv(ROOT/plan['cohorts']['validation']['path'])
    np.savez_compressed(folder/'validation_scores.npz',logit=score,node=meta.node,time_ns=meta.time_ns,row_index=meta.row_index,y_end=meta.y_end)
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    repeated=False
    if seed==42:
        original=torch.load(SOURCE/'model.pt',map_location='cpu',weights_only=True)
        for k,v in state.items():torch.testing.assert_close(v,original['state_dict'][k],rtol=0,atol=0)
        with np.load(SOURCE/'validation_scores.npz') as z:np.testing.assert_array_equal(score,z['logit'])
        repeated=True
    summary=dict(kind='lstm',width=8,features=BASE_FEATURES,seed=seed,configuration='lstm8_p1',
        best_epoch=best_epoch,epochs_run=len(history),stop_reason='patience' if wait>=5 else 'maximum_epochs',
        training_windows=len(xtrain),training_positives=int(ytrain.sum()),validation_windows=len(xval),
        validation_positives=int(yval.sum()),average_precision=best,seed42_exact_reproduction=repeated,
        reserved_nodes_scored=False,artifact_checksums={name:sha(folder/name) for name in ['model.pt','preprocessing.joblib','validation_scores.npz','history.csv']})
    write(p,summary);return summary


def evaluate(meta,episodes,plan,summaries):
    engine=PolicyEvaluator(meta,episodes,15);normal=(meta.history_critical_count==0)&(meta.history_unknown_count==0)
    records=[];specs={};selected=[];nodes=[];intervals={};score_map={};curves={}
    for summary in summaries:
        seed=summary['seed'];folder=OUT/f'seed_{seed}'
        score=np.load(folder/'validation_scores.npz')['logit'].astype(np.float64)
        key=str(seed);score_map[key]=score;engine.register_scores(key,score)
        candidates,_,curves[key]=pr_candidates(meta.y_end,score);thresholds=[]
        for q in QUANTILES:
            for per_node in [False,True]:thresholds.append(('normal_percentile','per_node' if per_node else 'global',q,calibrate(meta[normal],score[normal],q,per_node)))
        thresholds += [(method,'global',None,t) for method,t in candidates]+[('default_zero_logit','global',None,0.)]
        seed_rows=[]
        for method,scope,q,threshold in thresholds:
            for rule in RULES:
                pred=apply_rule(meta,score,threshold,rule)
                row=dict(seed=seed,policy_id=f's{seed}_p{len(seed_rows)+1:03d}',threshold_method=method,
                    scope=scope,quantile=q,rule=rule,**engine.evaluate(pred,key))
                row['feasible']=constrained_policy_key(row) is not None
                records.append(row);seed_rows.append(row)
                spec=dict(threshold=threshold,rule=rule)
                if isinstance(threshold,dict):spec['unseen_node_global_threshold']=float(np.percentile(score[normal],q))
                specs[row['policy_id']]=spec
        chosen,status=choose_seed_policy(seed_rows);spec=specs[chosen['policy_id']]
        pred=apply_rule(meta,score,spec['threshold'],spec['rule'])
        fallback=spec.get('unseen_node_global_threshold',spec['threshold'])
        fb=engine.evaluate(apply_rule(meta,score,fallback,spec['rule']),key)
        selected.append(dict(chosen,status=status,global_fallback_validation=fb))
        write(folder/'validation_policy.json',dict(status=status,metrics=chosen,**spec,global_fallback_validation=fb))
        pd.DataFrame(dict(node=meta.node,time_ns=meta.time_ns,row_index=meta.row_index,y_end=meta.y_end,score=score,pred=pred)).to_csv(folder/'validation_decisions.csv',index=False)
        standard=metrics(meta.y_end,score,pred);events,details=incident_metrics(meta,pred,episodes,cadence_minutes=15)
        details.to_csv(folder/'incident_details.csv',index=False)
        for k,v in standard.items():np.testing.assert_allclose(chosen[k],v)
        for k in ['false_bursts','detected','scored_node_days','event_recall_observable']:np.testing.assert_allclose(chosen[k],events[k])
        pernode=[]
        for node in meta.node.unique():
            keep=meta.node==node;m=meta[keep].reset_index(drop=True);e=PolicyEvaluator(m,episodes[episodes.node==node],15)
            e.register_scores(key,score[keep]);r=dict(seed=seed,node=int(node),**e.evaluate(pred[keep],key));pernode.append(r);nodes.append(r)
        intervals[key]=node_bootstrap(pernode,**{'replicates':plan['uncertainty']['replicates'],'seed':plan['uncertainty']['seed']})
    pd.DataFrame(records).to_csv(OUT/'policy_comparison.csv',index=False);write(OUT/'policy_specs.json',specs)
    pd.DataFrame([{k:v for k,v in r.items() if k!='global_fallback_validation'} for r in selected]).to_csv(OUT/'seed_comparison.csv',index=False)
    pd.DataFrame(nodes).to_csv(OUT/'per_node_validation.csv',index=False);write(OUT/'node_bootstrap_intervals.json',intervals)
    variation={}
    for k in ['average_precision','precision','recall','false_positive_rate','false_bursts_per_node_day','event_recall_observable']:
        v=np.array([r[k] for r in selected]);variation[k]=dict(mean=float(v.mean()),sample_std=float(v.std(ddof=1)),minimum=float(v.min()),maximum=float(v.max()))
    release=release_gate(selected)
    decision=dict(status='qualified_validation_candidate' if release else 'no_release_candidate',selected_seed=42 if release else None,
        selected_policy=release['policy_id'] if release else None,release_gate=plan['release_rule'],
        seed_policy_statuses={str(r['seed']):r['status'] for r in selected},
        reason='All seeds and the primary global fallback pass the frozen targets' if release else 'At least one seed fails the frozen validation targets or the primary fallback fails',
        independent_holdout_scored=False,dashboard_default_changed=False)
    write(OUT/'release_decision.json',decision)
    # Freeze research snapshots even if no model qualifies for operational release.
    write(OUT/'research_snapshot.json',dict(purpose='immutable reference for robustness results; not a qualified operational release',
        snapshots={str(r['seed']):dict(artifacts=r['artifact_checksums'],configuration=r['configuration'],features=r['features']) for r in summaries},
        policy_specs_sha256=sha(OUT/'policy_specs.json'),protocol_sha256=plan['protocol_sha256']))
    if release:
        seed_folder=OUT/'seed_42';write(OUT/'frozen_candidate.json',dict(seed=42,model_sha256=sha(seed_folder/'model.pt'),
            preprocessing_sha256=sha(seed_folder/'preprocessing.joblib'),features=BASE_FEATURES,width=8,eligibility_width=24,
            policy=specs[release['policy_id']],validation_metrics={k:v for k,v in release.items() if k!='global_fallback_validation'},protocol_sha256=plan['protocol_sha256']))
    result=dict(task=4,seeds=SEEDS,configuration='lstm8_p1',validation_windows=len(meta),
        policies_evaluated=len(records),feasible_policies=sum(r['feasible'] for r in records),
        feasible_seeds=sum(r['status']=='feasible' for r in selected),variation=variation,release_status=decision['status'],
        seed42_exact_reproduction=next(r for r in summaries if r['seed']==42)['seed42_exact_reproduction'],
        independent_holdout_scored=False,exploratory_period_scored=False,dashboard_default_changed=False,
        next_step='Investigate richer training features and label alignment before independent evaluation' if not release else 'Evaluate the frozen candidate once on independent holdout')
    write(OUT/'summary.json',result);plots(selected,curves,summaries,intervals);report(result,selected,decision,plan)
    print(json.dumps(result,indent=2),flush=True)


def plots(selected,curves,summaries,intervals):
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for r in summaries:
        history=pd.read_csv(OUT/f"seed_{r['seed']}"/'history.csv')
        axs[0].plot(history.epoch,history.validation_ap,marker='.',label=f"Seed {r['seed']}")
        p,rec,_=curves[str(r['seed'])];axs[1].step(rec,p,where='post',label=f"Seed {r['seed']}")
    axs[0].set(xlabel='Epoch',ylabel='Validation average precision',title='Fixed architecture; different training seeds')
    axs[1].set(xlabel='Recall',ylabel='Precision',ylim=(0,.32),xlim=(0,1),title='Validation PR curves (precision shown to 32%)')
    axs[1].axhline(.30,ls='--',color='black');axs[1].axvline(.4,ls=':',color='black')
    for ax in axs:ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'seed_stability.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,k,label,target in zip(axs,['precision','false_positive_rate','event_recall_observable'],['Precision','False positive rate','Observable incident recall'],[.3,.01,.6]):
        for i,r in enumerate(selected):
            v=r[k];ci=intervals[str(r['seed'])][k]
            ax.plot([i,i],[ci['lower_95'],ci['upper_95']],color='tab:blue');ax.scatter(i,v,color='tab:blue')
        ax.axhline(target,ls='--',color='black');ax.set_xticks(range(3),[str(r['seed']) for r in selected]);ax.set(xlabel='Training seed',ylabel=label,ylim=(0,max(target*1.2,max(intervals[str(r['seed'])][k]['upper_95'] for r in selected)*1.1)))
    fig.suptitle('Diagnostic validation policies; 95% whole-node bootstrap intervals')
    fig.tight_layout();fig.savefig(OUT/'node_uncertainty.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4));r=next(x for x in selected if x['seed']==42)
    matrix=np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
    ax.imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    ax.set_xticks([0,1],['Normal','Critical']);ax.set_yticks([0,1],['Normal','Critical']);ax.set(xlabel='Predicted',ylabel='Actual',title='Seed 42 diagnostic policy: validation')
    fig.tight_layout();fig.savefig(OUT/'confusion_matrix_seed42.png',dpi=170);plt.close(fig)


def report(result,selected,decision,plan):
    lines=['# Task 4 — robustness and release decision','',
        'Repeated the provisional `lstm8_p1` configuration from scratch with seeds 7, 42 and 123. Architecture, features, loss weighting, data, optimizer and stopping rules remain fixed. Seed 42 reproduces the Task 3 checkpoint tensors and every validation score exactly.','',
        '| Seed | AP | Precision | Recall | FPR | TP / FP / FN / TN | False bursts/day | Incidents | Feasible? |','|---|---:|---:|---:|---:|---|---:|---:|---|']
    for r in selected:lines.append(f"| {r['seed']} | {r['average_precision']:.4f} | {r['precision']:.2%} | {r['recall']:.2%} | {r['false_positive_rate']:.2%} | {r['tp']} / {r['fp']} / {r['fn']} / {r['tn']} | {r['false_bursts_per_node_day']:.3f} | {r['detected']}/{r['observable_episodes']} | {r['status']=='feasible'} |")
    lines += ['',f"**Release decision: `{decision['status']}`.** {result['feasible_seeds']}/3 seeds have any feasible policy, across {result['policies_evaluated']} policies. The targets are unchanged: precision ≥30%, recall ≥40%, FPR ≤1%, false bursts/day ≤0.1 and observable incident recall ≥60%.",'',
        'Every seed uses the same policy-selection rule. If no feasible policy exists, the table shows the best diagnostic precision while retaining at least 40% recall; these diagnostic thresholds are not deployment recommendations. Per-node policy fallbacks are fitted on pooled normal validation histories only.','',
        '## Seed variation','', '| Measure | Mean | Sample standard deviation |','|---|---:|---:|']
    for k,v in result['variation'].items():lines.append(f"| {k} | {v['mean']:.5f} | {v['sample_std']:.5f} |")
    lines += ['', 'Three seeds describe variability, not a precise confidence interval for future performance. The configuration was selected using seed 42 in Task 3, so its validation performance was already inspected and may be optimistic. Each seed selects its own validation checkpoint and policy. All three results are reported; no test-score seed selection occurred.','',
        '## Uncertainty and coverage','',
        'The 95% percentile intervals in `node_bootstrap_intervals.json` resample whole development servers, 2,000 times, preserving correlations within each server. They condition on the already-selected validation policy and do not capture selection uncertainty, shared rack-wide events or future drift. There are only nine development nodes. All comparisons retain the same 71,982 endpoints and 902 critical targets, with 47/47 observable raw incidents. Per-node results, incident records and delays are saved.','',
        '## Freeze and independent evaluation','',
        'The release rule was declared before these repeats: all three seeds need feasible policies; seed 42 is the primary release candidate, and its global fallback must also pass the target gate on validation. A failing gate yields no release candidate. `research_snapshot.json` freezes the research model/preprocessing/policy hashes without certifying operational performance.',
        '', 'No reserved server measurements, labels or predictions were accessed. The old exploratory period was not scored. Independent evaluation is deferred when there is no qualified release, preserving the final recordings for a revised, validation-qualified pipeline. The dashboard default is unchanged.','',
        '## What to investigate next','',
        'The repeated model and the supervised tree comparison have not established a usable detector. Investigate training-only source feature availability and label alignment, rather than promising that more seeds or higher class weights will fix the false alarms. See `TRAINING_FEATURE_AUDIT.md` for measured availability of additional thermal/electrical inputs. Their predictive value is unproven. Do not create artificial cooling labels, discard difficult servers or rebalance evaluation recordings.','',
        '## Reproduce','',
        'Run `.venv\\Scripts\\python.exe scripts\\run_task4.py`, then `scripts\\audit_task4_features.py` and `scripts\\verify_task4.py`. The frozen plan records all seeds, input fingerprints, policy grid, release rule and bootstrap method. All three seed folders include weights, preprocessing, examples, histories, scores and raw decisions.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest,plan=freeze()
    data=load_reduced(ROOT/'0.tar',ROOT/'artifacts/reduced_base.npz')
    assert data_digest(data)==(ROOT/'results/protocol_v1/base_data.sha256').read_text().strip()
    assert digest(ROOT/'0.tar')==manifest['source_archive_sha256']
    assert not set(data['node']).intersection(manifest['independent_holdout_nodes'])
    _,filled,split=protocol_clean(data)
    pre=joblib.load(SOURCE/'preprocessing.joblib');values=features_from_clean(data['raw'],filled,pre['scaler'],BASE_FEATURES,10.)
    cohorts={}
    for key in ['train','validation']:
        p=ROOT/plan['cohorts'][key]['path'];assert sha(p)==plan['cohorts'][key]['sha256'];cohorts[key]=pd.read_csv(p)
    train=cohorts['train'];val=cohorts['validation']
    for m,which in [(train,0),(val,1)]:
        idx=m.row_index.to_numpy();assert (split[idx]==which).all();np.testing.assert_array_equal(data['label'][idx]==2,m.y_end)
    xt=sequence_inputs(values,train.row_index,8);xv=sequence_inputs(values,val.row_index,8)
    summaries=[run_seed(s,xt,xv,train.y_end.to_numpy(),val.y_end.to_numpy(),plan) for s in SEEDS]
    episodes=pd.read_csv(ROOT/'results/protocol_v1/validation_incident_coverage.csv')[['node','start_ns','end_ns','readings']]
    evaluate(val,episodes,plan,summaries)

if __name__=='__main__':main()
