"""Task 5 recovery v1: exactly one extended LSTM and one extended tree classifier."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,precision_recall_curve
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dccad.data import load_reduced,BASE_FEATURES
from dccad.protocol import protocol_clean,constrained_policy_key
from dccad.recovery import EXTRA_FEATURES,FEATURES,SOURCE_COLUMNS,GROUPS,load_extra,fit_extension,preprocess_recovery_csv
from dccad.supervised import sequence_inputs,predict_logits,trailing_statistics,LSTMClassifier
from dccad.alarm_policies import PolicyEvaluator,apply_rule
from dccad.evaluation import calibrate,metrics,incident_metrics
from dccad.robustness import choose_seed_policy,node_bootstrap
from train_task3 import train_neural,seed
from analyze_thresholds import pr_candidates,QUANTILES,RULES
from audit_task1 import data_digest,digest

OUT=ROOT/'results/task5_recovery_v1'
NAMES=['extended_lstm8_p1','extended_boosting8']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')


def freeze():
    p=ROOT/'results/protocol_v1/manifest.json';manifest=json.loads(p.read_text())
    assert sha(p)==(p.parent/'manifest.sha256').read_text().strip()
    original=json.loads((ROOT/'results/task3_supervised/experiment_plan.json').read_text())
    plan=dict(task=5,experiment='recovery_v1',seed=42,models=NAMES,
        search_budget='one fixed additional-input set; one LSTM and one tree architecture; no extra weight/width search',
        protocol_sha256=sha(p),cohorts=original['cohorts'],targets=manifest['policy_targets'],
        width=8,eligibility_width=24,base_features=BASE_FEATURES,extra_features=EXTRA_FEATURES,model_features=FEATURES,
        source_columns=SOURCE_COLUMNS,aggregates=GROUPS,
        negative_rule='Negative absolute indoor-rack measurements, spreads and standard deviations become missing; genuine zeros/high measurements retained',
        aggregation='All specified devices must be observed; incomplete groups become missing',
        preprocessing='Original base scaler reused; extra causal ffill <=4 resets at node/gap/split; remaining extra missing values use normal-training medians; extra scaler on same normal-training rows as base; clip +/-10; raw unavailability indicators and median-fallback fraction appended',
        neural=original['neural'],pos_weight=1,boosting=original['boosting'],policy_search=original['policy_search'],
        include_default_threshold=True,comparison_models=['lstm8_p1','gradient_boosting8'],
        baseline_artifacts={name:{f:sha(ROOT/'results/task3_supervised'/name/f) for f in ['validation_scores.npz','preprocessing.joblib','summary.json']} for name in ['lstm8_p1','gradient_boosting8']},
        architecture_selection='no configuration tuning; single-seed recovery candidate requires feasible policy and AP improvement over its matched original baseline; repeated seeds and freeze required before independent evaluation',
        independent_holdout_scored=False,exploratory_period_scored=False,
        label='Existing value==2 versus value==0 outcome preserved; companion preparation code selects batchs::client::state; alert-specific raw metadata absent',
        implementation_sha256={f:sha(ROOT/f) for f in ['src/dccad/recovery.py','scripts/train_task3.py','src/dccad/supervised.py','src/dccad/alarm_policies.py']})
    path=OUT/'experiment_plan.json'
    # JSON normalizes tuple aggregate definitions; compare using the serialized representation.
    plan=json.loads(json.dumps(plan))
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Recovery inputs/configuration changed; use a new experiment version')
    write(path,plan);return manifest,plan


def train_one(name,values,data,extras,train,val,pre,plan):
    folder=OUT/name;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():
        r=json.loads((folder/'summary.json').read_text())
        for f,h in r['artifact_checksums'].items():assert sha(folder/f)==h
        print('Reusing completed recovery model:',name,flush=True);return r
    seed();kind='lstm' if name==NAMES[0] else 'boosting'
    xt=sequence_inputs(values,train.row_index,8);xv=sequence_inputs(values,val.row_index,8)
    yt=train.y_end.to_numpy(dtype=np.float32);yv=val.y_end.to_numpy(dtype=bool)
    history=[];start=time.time()
    if kind=='lstm':
        model,history,stopping=train_neural(dict(name=name,kind='lstm',width=8,pos_weight=1),xt,xv,yt,yv,
            ((val.history_critical_count==0)&(val.history_unknown_count==0)).to_numpy())
        score=predict_logits(model,xv)
        torch.save(dict(state_dict=model.state_dict(),features=FEATURES,width=8,hidden=32,dropout=.2,kind=kind),folder/'model.pt')
        mode='logit'
    else:
        model=HistGradientBoostingClassifier(learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,
            l2_regularization=1.,max_iter=200,early_stopping=False,random_state=42)
        tx=trailing_statistics(xt);vx=trailing_statistics(xv)
        with threadpool_limits(limits=4):
            model.fit(tx,yt);best=-np.inf
            for step,pred in enumerate(model.staged_predict_proba(vx),1):
                if step in [50,100,200]:
                    ap=float(average_precision_score(yv,pred[:,1]));history.append(dict(iterations=step,validation_ap=ap))
                    if ap>best:best=ap;best_step=step;best_score=pred[:,1].copy()
            model.set_params(max_iter=best_step);model.fit(tx,yt);score=model.predict_proba(vx)[:,1]
            np.testing.assert_allclose(score,best_score,rtol=1e-10,atol=1e-12)
        stopping=dict(best_iterations=best_step,stop_reason='predeclared_50_100_200_validation_checkpoints')
        joblib.dump(model,folder/'model.joblib');mode='score'
    joblib.dump(pre,folder/'preprocessing.joblib')
    np.savez_compressed(folder/'validation_scores.npz',node=val.node,time_ns=val.time_ns,row_index=val.row_index,y_end=val.y_end,**{mode:score})
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    # Export a complete original window so standalone preprocessing is exactly comparable.
    i=next(i for i,end in enumerate(val.row_index) if np.isfinite(data['raw'][end-23:end+1]).all()
        and (data['raw'][end-23:end+1]>=0).all() and np.isfinite(extras[end-23:end+1]).all())
    end=int(val.iloc[i].row_index)
    example=pd.DataFrame(np.column_stack([data['raw'][end-23:end+1],extras[end-23:end+1]]),columns=BASE_FEATURES+EXTRA_FEATURES)
    example.insert(0,'timestamp',pd.to_datetime(data['time_ns'][end-23:end+1],utc=True));example.to_csv(folder/'example_window.csv',index=False)
    write(folder/'example_metadata.json',dict(validation_position=i,row_index=end,node=int(val.iloc[i].node),expected_score=float(score[i]),score_mode=mode))
    summary=dict(name=name,kind=kind,width=8,features=FEATURES,seed=42,score_mode=mode,average_precision=float(average_precision_score(yv,score)),
        training_windows=len(xt),training_positives=int(yt.sum()),validation_windows=len(xv),validation_positives=int(yv.sum()),
        seconds=time.time()-start,**stopping,artifact_checksums={f:sha(folder/f) for f in ['model.pt' if kind=='lstm' else 'model.joblib','preprocessing.joblib','validation_scores.npz','history.csv']},
        reserved_nodes_scored=False)
    write(folder/'summary.json',summary);print('Recovery model completed:',name,'AP',summary['average_precision'],flush=True)
    return summary


def evaluate(meta,episodes,plan,summaries):
    engine=PolicyEvaluator(meta,episodes,15);normal=(meta.history_critical_count==0)&(meta.history_unknown_count==0)
    records=[];specs={};comparisons=[];curves={};nodes=[];intervals={}
    models=[dict(name='lstm8_p1',kind='lstm',score_mode='logit'),dict(name='gradient_boosting8',kind='boosting',score_mode='score')]+summaries
    for r in models:
        name=r['name'];folder=OUT/name if name in NAMES else ROOT/'results/task3_supervised'/name
        z=np.load(folder/'validation_scores.npz')
        for field in ['node','row_index','time_ns','y_end']:np.testing.assert_array_equal(z[field],meta[field])
        score=z[r['score_mode']].astype(np.float64);engine.register_scores(name,score)
        candidates,_,curves[name]=pr_candidates(meta.y_end,score);thresholds=[]
        for q in QUANTILES:
            for per_node in [False,True]:thresholds.append(('normal_percentile','per_node' if per_node else 'global',q,calibrate(meta[normal],score[normal],q,per_node)))
        thresholds += [(method,'global',None,t) for method,t in candidates]+[('default','global',None,0. if r['kind']=='lstm' else .5)]
        rows=[]
        for method,scope,q,t in thresholds:
            for rule in RULES:
                pred=apply_rule(meta,score,t,rule)
                row=dict(policy_id=f'p{len(records)+1:04d}',model=name,kind=r['kind'],score_mode=r['score_mode'],
                    threshold_method=method,scope=scope,quantile=q,rule=rule,**engine.evaluate(pred,name))
                row['feasible']=constrained_policy_key(row) is not None;records.append(row);rows.append(row)
                spec=dict(model=name,score_mode=r['score_mode'],threshold=t,rule=rule)
                if isinstance(t,dict):spec['unseen_node_global_threshold']=float(np.percentile(score[normal],q))
                specs[row['policy_id']]=spec
        chosen,status=choose_seed_policy(rows);chosen=dict(chosen,status=status);comparisons.append(chosen)
        spec=specs[chosen['policy_id']];pred=apply_rule(meta,score,spec['threshold'],spec['rule'])
        if name in NAMES:
            write(folder/'validation_policy.json',dict(status=status,metrics=chosen,**spec))
            pd.DataFrame(dict(node=meta.node,time_ns=meta.time_ns,row_index=meta.row_index,y_end=meta.y_end,score=score,pred=pred)).to_csv(folder/'validation_decisions.csv',index=False)
        standard=metrics(meta.y_end,score,pred);events,details=incident_metrics(meta,pred,episodes,15)
        for k,v in standard.items():np.testing.assert_allclose(v,chosen[k])
        for k in ['detected','false_bursts','scored_node_days']:np.testing.assert_allclose(events[k],chosen[k])
        if name in NAMES:details.to_csv(folder/'incident_details.csv',index=False)
        pernode=[]
        for node in meta.node.unique():
            keep=meta.node==node;m=meta[keep].reset_index(drop=True);e=PolicyEvaluator(m,episodes[episodes.node==node],15)
            e.register_scores('score',score[keep]);nr=dict(model=name,node=int(node),**e.evaluate(pred[keep],'score'));pernode.append(nr);nodes.append(nr)
        intervals[name]=node_bootstrap(pernode)
    pd.DataFrame(records).to_csv(OUT/'policy_comparison.csv',index=False);write(OUT/'policy_specs.json',specs)
    pd.DataFrame(comparisons).to_csv(OUT/'model_comparison.csv',index=False);pd.DataFrame(nodes).to_csv(OUT/'per_node_validation.csv',index=False)
    write(OUT/'node_bootstrap_intervals.json',intervals)
    original={r['kind']:r for r in comparisons if r['model'] not in NAMES}
    candidates=[r for r in comparisons if r['model'] in NAMES and r['feasible'] and r['average_precision']>original[r['kind']]['average_precision']]
    selected=max(candidates,key=constrained_policy_key) if candidates else None
    summary=dict(task=5,experiment='recovery_v1',models_trained=2,validation_windows=len(meta),validation_positives=int(meta.y_end.sum()),
        measurement_features=44,indicator_features=23,total_model_features=len(FEATURES),additional_missing_inputs_drop_endpoints=False,
        policies_evaluated=len(records),recovery_policies_evaluated=sum(r['model'] in NAMES for r in records),
        feasible_recovery_policies=sum(r['model'] in NAMES and r['feasible'] for r in records),
        status='candidate_requires_seed_robustness' if selected else 'no_qualified_recovery_candidate',
        recovery_candidate=selected['model'] if selected else None,selected_policy=selected,
        independent_holdout_scored=False,exploratory_period_scored=False,dashboard_default_changed=False,
        next_step='Repeat qualified revised candidate before freeze and independent evaluation' if selected else 'Report the failed recovery honestly; discuss task/dataset scope before further model search')
    write(OUT/'summary.json',summary);plots(curves,comparisons,summaries);report(summary,comparisons)
    print(json.dumps(summary,indent=2),flush=True)


def plots(curves,comparison,summaries):
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for name,(p,r,_) in curves.items():axs[0].step(r,p,where='post',label=name)
    axs[0].axhline(.3,color='black',ls='--');axs[0].axvline(.4,color='black',ls=':')
    axs[0].set(xlim=(0,1),ylim=(0,.32),xlabel='Recall',ylabel='Precision',title='Matched validation PR curves (shown to 32%)');axs[0].legend(fontsize=8)
    names=[r['model'] for r in comparison];aps=[r['average_precision'] for r in comparison]
    axs[1].bar(names,aps);axs[1].tick_params(axis='x',rotation=25)
    axs[1].set(ylabel='Average precision',ylim=(0,max(aps)*1.3),title='Original versus added-input models; seed 42')
    fig.tight_layout();fig.savefig(OUT/'recovery_comparison.png',dpi=170);plt.close(fig)
    r=next(x for x in comparison if x['model']==NAMES[0]);matrix=np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
    fig,ax=plt.subplots(figsize=(5,4));ax.imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    ax.set_xticks([0,1],['Normal','Critical']);ax.set_yticks([0,1],['Normal','Critical']);ax.set(xlabel='Predicted',ylabel='Actual',title='Extended LSTM diagnostic policy: validation')
    fig.tight_layout();fig.savefig(OUT/'extended_lstm_confusion.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4));history=pd.read_csv(OUT/NAMES[0]/'history.csv')
    axs[0].plot(history.epoch,history.train_loss,marker='.');axs[0].set(xlabel='Epoch',ylabel='Training BCE loss')
    axs[1].plot(history.epoch,history.validation_ap,marker='.');axs[1].set(xlabel='Epoch',ylabel='Validation AP')
    fig.suptitle('Extended LSTM training; fixed configuration');fig.tight_layout();fig.savefig(OUT/'training_history.png',dpi=170);plt.close(fig)


def report(s,rows):
    lines=['# Task 5 — bounded additional-input recovery','',
        'Trained one extended LSTM and one extended tree classifier. The extension adds 22 measured/aggregated inputs to the original 22, plus 22 unavailability flags and one median-fallback fraction: 67 model inputs. The same 111,070 training and 71,982 validation endpoints are retained, with 1,445 and 902 critical targets respectively. No extra missing measurements exclude an endpoint.','',
        '| Model | AP | Precision | Recall | FPR | False bursts/day | Incidents | Feasible? |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['model']} | {r['average_precision']:.4f} | {r['precision']:.2%} | {r['recall']:.2%} | {r['false_positive_rate']:.2%} | {r['false_bursts_per_node_day']:.3f} | {r['detected']}/{r['observable_episodes']} | {r['feasible']} |")
    lines += ['',f"**Outcome: `{s['status']}`.** {s['feasible_recovery_policies']} recovery policies meet all five original targets. There are {s['recovery_policies_evaluated']} recovery policies and {s['policies_evaluated']} total matched-baseline/recovery policies.",'',
        'A row without a feasible policy shows the best diagnostic precision at recall ≥40%; it is not a release operating point. A recovery candidate additionally needs AP improvement over its original same-type baseline. The seed remains 42; architecture, optimizer, class weighting and history width were not searched.','',
        '## Added information and safeguards','',
        'Added inputs cover power-supply voltage/current, CPU memory/I/O power, PCIe temperature, ambient/power variability, maximum fan variability, GPU temperature maxima/spreads and ambient/power maxima. GPU/fan aggregate features require all specified devices. The full source definitions are frozen in the plan. See `LABEL_AND_SENSOR_AUDIT.md` for verified source meanings and the more precise batch-client label scope.',
        '', 'Forward filling uses only prior readings, at most four, and resets at splits, nodes and gaps. Remaining extra missing values use medians of normal training rows only. Unavailability flags preserve the original missing state, and the fallback fraction identifies median substitution. Extra scaling uses the same normal training rows as the original base scaler; validation supplies no preprocessing statistics. The original 22-sensor scaler is reused exactly. Values beyond ten training standard deviations are clipped; real zeros and high original measurements are preserved in exported CSVs.',
        '', 'The LSTM uses hidden size 32, dropout 0.2, positive loss weight one, Adam 0.001, batch 512, maximum 25 epochs and patience five on validation AP. The tree uses the same latest/mean/std/slope construction, learning rate 0.05, 15 leaves, minimum leaf size 30 and L2 one; explicit validation selects 50/100/200 iterations. Input dimension and parameter count grow; this is an input-set comparison, not proof of any individual sensor’s causal contribution.',
        '', 'Normal percentile/PR thresholds and persistence follow the original grid and feasibility gate. Node-bootstrap intervals condition on selected policies and cannot establish future performance. Configuration/checkpoint/policy selection uses the already-inspected validation period, so it remains development evidence.',
        '', '## Decision and next work','',
        'Reserved servers and the later exploratory period were not scored. A successful single-seed recovery would require repeated seeds and a frozen pipeline before independent evaluation. A failed recovery does not justify an accuracy claim or changing the reference labels. It supports reporting that the tested physical telemetry does not adequately predict the selected batch-client outcome under these splits. Further model search or a replacement dataset must be a separately bounded and documented next decision.',
        '', 'The dashboard, earlier PDF/PPT and original artifacts remain unchanged. This report supplements them; submission documents still need to include the later experiments and corrected label scope.',
        '', '## Reproduce','',
        'Run `.venv\\Scripts\\python.exe scripts\\run_task5.py` then `scripts\\verify_task5.py`. `scripts\\predict_task5.py` scores an exported example window without the archive. Individual model folders contain trusted weights/preprocessing, histories, scores, examples, decisions and incident details. Original source readings and labels are never overwritten.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest,plan=freeze()
    data=load_reduced(ROOT/'0.tar',ROOT/'artifacts/reduced_base.npz')
    assert data_digest(data)==(ROOT/'results/protocol_v1/base_data.sha256').read_text().strip()
    assert digest(ROOT/'0.tar')==manifest['source_archive_sha256']
    assert not set(data['node']).intersection(manifest['independent_holdout_nodes'])
    _,filled,split=protocol_clean(data);extras=load_extra(ROOT/'0.tar',data)
    base_pre=joblib.load(ROOT/'results/task3_supervised/lstm8_p1/preprocessing.joblib')
    values,pre,audit=fit_extension(data,filled,extras,split,base_pre['scaler'])
    write(OUT/'preprocessing_audit.json',audit);joblib.dump(pre,OUT/'preprocessing.joblib')
    meta={}
    for key in ['train','validation']:
        p=ROOT/plan['cohorts'][key]['path'];assert sha(p)==plan['cohorts'][key]['sha256'];meta[key]=pd.read_csv(p)
    summaries=[train_one(name,values,data,extras,meta['train'],meta['validation'],pre,plan) for name in NAMES]
    episodes=pd.read_csv(ROOT/'results/protocol_v1/validation_incident_coverage.csv')[['node','start_ns','end_ns','readings']]
    evaluate(meta['validation'],episodes,plan,summaries)

if __name__=='__main__':main()
