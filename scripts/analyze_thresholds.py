"""Task 2: validation-only policy search on frozen, shared endpoints."""
import gc
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from dccad.data import load_reduced
from dccad.model import LSTMAutoencoder,predict_errors
from dccad.protocol import protocol_clean,constrained_policy_key
from dccad.evaluation import calibrate,metrics,incident_metrics
from dccad.alarm_policies import apply_rule,notifications,PolicyEvaluator

OUT=ROOT/'results'/'task2_thresholds'
NAMES=['base8','cooling12_8','robust24','isolation_forest']
QUANTILES=[95.,99.,99.5,99.9,99.95]
RULES=['immediate','consecutive_2','two_of_three']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def experiment_plan():
    manifest_path=ROOT/'results'/'protocol_v1'/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert sha(manifest_path)==(manifest_path.parent/'manifest.sha256').read_text().strip()
    cohort=ROOT/manifest['cohort_files']['validation']['path']
    assert sha(cohort)==manifest['cohort_files']['validation']['sha256']
    plan={'task':2,'scope':'exploratory policy tuning of historical models on a shared frozen validation cohort',
        'protocol_sha256':sha(manifest_path),'validation_cohort_sha256':sha(cohort),
        'models':NAMES,'score_modes':'whole_window and latest_reading for each LSTM; anomaly score for Isolation Forest',
        'quantiles':QUANTILES,'threshold_scopes':['global','per_node'],
        'quantile_fit':'fully known normal validation histories only',
        'pr_candidates':['highest empirical precision at recall >=0.4','highest empirical precision at recall >=0.6',
                         'highest empirical precision at recall >=0.8','highest empirical F0.5'],
        'pr_fit':'validation only; global score threshold; exact ties follow >= semantics',
        'alarm_rules':RULES,'cadence_minutes':15,'two_of_three_definition':'at least two highs in three complete consecutive eligible readings; current reading need not be high',
        'reset':'any node change or gap between eligible scored timestamps',
        'notification_demonstration':'60-minute cooldown on rising edges, separate from raw decisions; reset at gaps/nodes',
        'policy_targets':manifest['policy_targets'],'selection':manifest['policy_selection'],
        'heldout_nodes_accessed':False,'new_model_training':False,
        'training_caveat':'Historical training sets and early-stopping periods differ. This is a diagnostic on identical validation endpoints, not the primary retrained-model comparison.',
        'model_checksums':{name:{p.name:sha(p) for p in [ROOT/'results'/name/('model.joblib' if name=='isolation_forest' else 'model.pt'),ROOT/'results'/name/'scaler.joblib']} for name in NAMES}}
    path=OUT/'experiment_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise RuntimeError('Task 2 inputs or search plan changed; use a new version instead of silently replacing the experiment')
    write_json(path,plan)
    return manifest,plan


def scores_for(name,data,filled,meta):
    folder=ROOT/'results'/name;r=json.loads((folder/'summary.json').read_text())
    if name=='isolation_forest':features=data['features'].tolist()
    else:
        cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        features=cp['features']
    indices=[data['features'].tolist().index(c) for c in features]
    scaler=joblib.load(folder/'scaler.joblib')
    values=scaler.transform(filled[:,indices])
    clip=r.get('clip',5.)
    if clip is not None:values=np.clip(values,-clip,clip)
    rows=meta.row_index.to_numpy()[:,None]-np.arange(r['width']-1,-1,-1)[None,:]
    x=np.ascontiguousarray(values[rows],dtype=np.float32)
    assert np.isfinite(x).all()
    if name=='isolation_forest':
        model=joblib.load(folder/'model.joblib');variants={'anomaly_score':-model.score_samples(x.reshape(len(x),-1))}
    else:
        model=LSTMAutoencoder(len(features),cp['width'],cp.get('hidden',64),cp.get('latent',16))
        model.load_state_dict(cp['state_dict'])
        mean,end,_=predict_errors(model,x)
        variants={'whole_window':mean,'latest_reading':end}
    np.savez_compressed(OUT/(name+'_validation_scores.npz'),row_index=meta.row_index.to_numpy(),
                       node=meta.node.to_numpy(),time_ns=meta.time_ns.to_numpy(),y_end=meta.y_end.to_numpy(),**variants)
    del model,x,values;gc.collect()
    return r,{k:np.asarray(v,dtype=np.float64) for k,v in variants.items()}


def pr_candidates(y,score):
    precision,recall,threshold=precision_recall_curve(y,score)
    p=precision[:-1];r=recall[:-1]
    candidates=[];limits={}
    for floor in [.4,.6,.8]:
        indices=np.flatnonzero(r>=floor)
        i=max(indices,key=lambda j:(p[j],r[j],threshold[j]))
        value=float(np.nextafter(np.float64(threshold[i]),-np.inf))
        candidates.append((f'pr_precision_recall_{floor:g}',value))
        limits[str(floor)]={'precision':float(p[i]),'recall':float(r[i]),'threshold':value}
    denominator=.25*p+r
    f=np.divide(1.25*p*r,denominator,out=np.zeros_like(p),where=denominator>0)
    i=int(np.argmax(f));candidates.append(('pr_f0_5',float(np.nextafter(np.float64(threshold[i]),-np.inf))))
    return candidates,limits,(precision,recall,threshold)


def evaluate_search(meta,episodes,manifest,data,filled):
    engine=PolicyEvaluator(meta,episodes,cadence_minutes=15)
    normal=(meta.history_critical_count.to_numpy()==0)&(meta.history_unknown_count.to_numpy()==0)
    assert int(normal.sum())==manifest['cohort_files']['validation']['normal_only_calibration_windows']
    records=[];specs={};curves={};all_scores={};curve_limits=[];baseline_ids=[]
    def add(model,mode,method,scope,rule,threshold,quantile=None,baseline=False):
        score_key=model+'__'+mode;score=all_scores[score_key]
        pred=apply_rule(meta,score,threshold,rule,cadence_minutes=15)
        record={'policy_id':f'p{len(records)+1:04d}','model':model,'score_mode':mode,
            'threshold_method':method,'scope':scope,'rule':rule,'quantile':quantile,'baseline':baseline,
            **engine.evaluate(pred,score_key)}
        record['feasible']=constrained_policy_key(record) is not None
        records.append(record);specs[record['policy_id']]={'score_key':score_key,'threshold':threshold,'rule':rule}
        if scope=='per_node' and method=='normal_percentile':
            specs[record['policy_id']]['unseen_node_global_threshold']=float(np.percentile(score[normal],quantile))
        return record
    for name in NAMES:
        summary,variants=scores_for(name,data,filled,meta)
        for mode,score in variants.items():
            key=name+'__'+mode;all_scores[key]=score;engine.register_scores(key,score)
            candidates,limits,curve=pr_candidates(meta.y_end.to_numpy(),score)
            curves[key]=curve
            for floor,value in limits.items():curve_limits.append({'model':name,'score_mode':mode,'recall_floor':float(floor),**value})
            for q in QUANTILES:
                for per_node in [False,True]:
                    threshold=calibrate(meta[normal],score[normal],q,per_node)
                    for rule in RULES:add(name,mode,'normal_percentile','per_node' if per_node else 'global',rule,threshold,q)
            for method,threshold in candidates:
                for rule in RULES:add(name,mode,method,'global',rule,threshold)
        policy=summary['selected_policy']
        mode='anomaly_score' if name=='isolation_forest' else policy['score_mode']
        threshold=policy['threshold']
        if isinstance(threshold,dict):threshold={int(k):v for k,v in threshold.items()}
        score_key=name+'__'+mode
        # Historical IF confirmation span requires three dense readings, evaluated explicitly.
        if policy['persistence']==2:
            pred=apply_rule(meta,all_scores[score_key],threshold,'consecutive_2')
            high=apply_rule(meta,all_scores[score_key],threshold,'immediate')
            previous=np.r_[False,high[:-1]]
            before=np.r_[False,False,high[:-2]][:len(high)]
            complete=~engine.reset&~np.r_[True,engine.reset[:-1]]
            pred=high&previous&before&complete
            record={'policy_id':f'p{len(records)+1:04d}','model':name,'score_mode':mode,
                'threshold_method':'historical_fixed','scope':'per_node' if isinstance(threshold,dict) else 'global',
                'rule':'consecutive_3','quantile':None,'baseline':True,**engine.evaluate(pred,score_key)}
            record['feasible']=constrained_policy_key(record) is not None;records.append(record)
            specs[record['policy_id']]={'score_key':score_key,'threshold':threshold,'rule':'consecutive_3'}
        else:
            record=add(name,mode,'historical_fixed','per_node' if isinstance(threshold,dict) else 'global','immediate',threshold,baseline=True)
        baseline_ids.append(record['policy_id'])
        print('Evaluated model policies:',name,'total policies:',len(records),flush=True)
    pd.DataFrame(records).to_csv(OUT/'policy_comparison.csv',index=False)
    pd.DataFrame(curve_limits).to_csv(OUT/'global_threshold_precision_limits.csv',index=False)
    return engine,records,specs,all_scores,curves,baseline_ids


def pareto(records):
    points=[]
    for r in records:
        dominated=any(s['precision']>=r['precision'] and s['recall']>=r['recall']
            and s['false_bursts_per_node_day']<=r['false_bursts_per_node_day']
            and (s['precision']>r['precision'] or s['recall']>r['recall'] or s['false_bursts_per_node_day']<r['false_bursts_per_node_day']) for s in records)
        if not dominated:points.append(r)
    return points


def raw_predictions(meta,scores,spec):
    if spec['rule']!='consecutive_3':return apply_rule(meta,scores,spec['threshold'],spec['rule'])
    from dccad.evaluation import consecutive_alarm
    return consecutive_alarm(meta,scores,spec['threshold'],3,cadence_minutes=15)


def finish(engine,records,specs,scores,curves,baseline_ids,plan):
    feasible=[r for r in records if r['feasible']]
    selected=max(feasible,key=constrained_policy_key) if feasible else None
    retaining=[r for r in records if r['recall']>=.4]
    best_retaining=max(retaining,key=lambda r:(r['precision'],r['f0_5'],-r['false_bursts_per_node_day'])) if retaining else None
    quiet=[r for r in records if r['tp']>0 and r['false_positive_rate']<=.01 and r['false_bursts_per_node_day']<=.1]
    best_quiet=max(quiet,key=lambda r:(r['recall'],r['precision'],r['event_recall_observable'])) if quiet else None
    frontier=pareto(records);pd.DataFrame(frontier).to_csv(OUT/'pareto_frontier.csv',index=False)
    # Example notifications preserve raw predictions and expose lost incident notifications.
    baseline=next(r for r in records if r['baseline'] and r['model']=='cooling12_8')
    illustrations=[]
    for record in [baseline,best_quiet]:
        if record is None:continue
        spec=specs[record['policy_id']];score=scores[spec['score_key']]
        pred=raw_predictions(engine.meta,score,spec);notice=notifications(engine.meta,pred,60)
        delivery=engine.evaluate(notice,spec['score_key'])
        illustrations.append({'policy_id':record['policy_id'],'raw_precision':record['precision'],'raw_recall':record['recall'],
            'raw_false_bursts':record['false_bursts'],'raw_detected_incidents':record['detected'],
            'displayed_notifications':int(notice.sum()),'false_notifications':delivery['fp'],
            'notified_incidents':delivery['detected'],'notified_incident_recall_all':delivery['event_recall_all'],
            'cooldown_minutes':60,'raw_decisions_unchanged':True})
    pd.DataFrame(illustrations).to_csv(OUT/'notification_demonstration.csv',index=False)
    # Independently cross-check immediate, temporal and node-specific policies.
    checked=[]
    chosen=[baseline,best_retaining,best_quiet,selected]
    for rule in ['consecutive_2','two_of_three']:
        for scope in ['global','per_node']:
            chosen.append(next(r for r in records if r['model']=='cooling12_8' and r['rule']==rule
                and r['scope']==scope and r['threshold_method']=='normal_percentile' and r['quantile']==99.))
    for record in chosen:
        if record is None or record['policy_id'] in checked:continue
        spec=specs[record['policy_id']];score=scores[spec['score_key']]
        pred=raw_predictions(engine.meta,score,spec)
        reference={**metrics(engine.meta.y_end,score,pred),**incident_metrics(engine.meta,pred,engine.episodes,cadence_minutes=15)[0]}
        for key,value in reference.items():
            if key=='cadence_minutes':continue
            if value is None:assert record[key] is None
            else:np.testing.assert_allclose(record[key],value,rtol=1e-12,atol=1e-12,err_msg=key)
        checked.append(record['policy_id'])
    output={'status':'feasible_policy_found' if selected else 'no_feasible_policy',
        'validation_rows':len(engine.meta),'critical_endpoints':int(engine.y.sum()),'raw_incidents':len(engine.episodes),
        'policies_evaluated':len(records),'feasible_policies':len(feasible),'selected_policy':selected,
        'best_precision_while_retaining_40pct_recall':best_retaining,
        'best_recall_within_false_alarm_budgets':best_quiet,'current_model_fixed_policy_on_shared_cohort':baseline,
        'pareto_policies':len(frontier),'targets':plan['policy_targets'],
        'selection_rule':'All frozen targets must pass; no fallback policy is labelled feasible',
        'models_retrained':False,'holdout_accessed':False,'dashboard_default_changed':False,
        'interpretation':'Exploratory validation tuning of existing models; candidate training and early stopping histories differ.',
        'reference_evaluator_crosschecks':checked,'next_task':3}
    misleading=[r for r in records if r['recall']>=.4 and r['false_bursts_per_node_day']<=.1]
    output['low_burst_high_recall_policies']={'count':len(misleading),
        'minimum_false_positive_rate':min(r['false_positive_rate'] for r in misleading) if misleading else None,
        'maximum_precision':max(r['precision'] for r in misleading) if misleading else None,
        'interpretation':'Long continuous alarms can have few distinct false bursts while falsely flagging most normal readings; all frozen targets remain necessary.'}
    write_json(OUT/'summary.json',output)
    write_json(OUT/'policy_specs.json',specs)
    write_json(OUT/'verification.json',{'passed':True,'same_validation_rows_for_all_models':True,
        'finite_scores':True,'frozen_targets_used':True,'reference_metrics_matched':checked,
        'reserved_nodes_scored':False,'scores_read_only_from_saved_models':True,
        'policies_evaluated':len(records),'feasible_policies':len(feasible)})
    # Save representative reading decisions for review; not a deployable final policy.
    for record,label in [(baseline,'current'),(best_retaining,'recall_constrained'),(best_quiet,'low_burden')]:
        if record is None:continue
        spec=specs[record['policy_id']];frame=engine.meta.copy();frame['score']=scores[spec['score_key']]
        frame['alarm']=raw_predictions(engine.meta,frame.score.to_numpy(),spec)
        frame.to_csv(OUT/(label+'_decisions.csv'),index=False)
    plot_results(records,curves,engine.y,baseline,best_retaining,best_quiet)
    write_report(output,illustrations)
    print(json.dumps({k:output[k] for k in ['status','policies_evaluated','feasible_policies']},indent=2),flush=True)


def plot_results(records,curves,y,baseline,best_retaining,best_quiet):
    fig,ax=plt.subplots(figsize=(9,5.5))
    for key,(p,r,t) in curves.items():ax.plot(r,p,label=key.replace('__',' / '),alpha=.8)
    ax.axhline(float(np.mean(y)),ls=':',color='gray',label='Critical prevalence')
    ax.axhline(.30,ls='--',color='#bc3636',label='Precision target');ax.axvline(.40,ls='--',color='#c67b22',label='Recall target')
    ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='Global single-threshold validation curves')
    ax.legend(fontsize=8,loc='upper right');ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(OUT/'precision_recall.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.7));colors={'base8':'#19638a','cooling12_8':'#c67b22','robust24':'#528044','isolation_forest':'#8962a1'}
    for name in NAMES:
        group=[r for r in records if r['model']==name and not r['baseline']]
        axes[0].scatter([r['recall'] for r in group],[r['precision'] for r in group],s=14,alpha=.5,color=colors[name],label=name)
        axes[1].scatter([r['false_bursts_per_node_day'] for r in group],[r['recall'] for r in group],s=14,alpha=.5,color=colors[name],label=name)
    axes[0].axhline(.3,ls='--',color='gray');axes[0].axvline(.4,ls='--',color='gray')
    axes[0].set(xlabel='Recall',ylabel='Precision',title='Threshold and temporal-rule tradeoff',xlim=(0,1),ylim=(0,1))
    axes[1].axvline(.1,ls='--',color='gray');axes[1].axhline(.4,ls='--',color='gray')
    axes[1].set(xlabel='False bursts per scored node-day',ylabel='Recall',title='False-alarm budget versus recall',ylim=(0,1))
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(OUT/'policy_tradeoffs.png',dpi=160);plt.close(fig)
    # Detail near the operational false-alarm budget, colour by reading-level FPR.
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    group=[r for r in records if r['false_bursts_per_node_day']<=.3]
    points=ax.scatter([r['false_bursts_per_node_day'] for r in group],[r['recall'] for r in group],
        c=[r['false_positive_rate'] for r in group],cmap='viridis',vmin=0,vmax=1,s=28,alpha=.8)
    ax.axvline(.1,ls='--',color='#bc3636',label='False-burst budget')
    ax.axhline(.4,ls='--',color='#c67b22',label='Minimum recall')
    ax.set(xlim=(-.005,.3),ylim=(0,1),xlabel='False bursts per scored node-day',ylabel='Recall',
        title='Low-burst policies can still have high reading-level false positives')
    fig.colorbar(points,ax=ax,label='Reading-level false-positive rate')
    ax.legend(fontsize=8,loc='center right');ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(OUT/'false_alarm_budget_detail.png',dpi=160);plt.close(fig)


def write_report(summary,illustrations):
    text='# Task 2: threshold and persistence tradeoffs\n\n'
    text+=f"Evaluated **{summary['policies_evaluated']} policies** using **{summary['validation_rows']:,} identical validation endpoints**, including **{summary['critical_endpoints']} critical readings** and **{summary['raw_incidents']} original incidents**. No heldout server was scored and no model was retrained.\n\n"
    text+='## Decision\n\n'
    text+=('A candidate meets all frozen validation targets; it remains exploratory until independent evaluation.\n\n' if summary['selected_policy'] else '**No evaluated policy meets all frozen targets.** Tightening alarms alone is insufficient among the tested configurations. There is no feasible selected policy, and the dashboard default has not been changed to hide this outcome. Proceed to the supervised LSTM and supervised baseline in Task 3.\n\n')
    text+='Targets: precision >=30%, recall >=40%, FPR <=1%, false bursts <=0.1 per scored node-day, and observable-incident recall >=60%. No target was relaxed. An all-normal predictor cannot pass the recall constraints.\n\n'
    text+='## Representative operating points\n\n'
    text+='| Policy | Model / rule | Precision | Recall | FPR | False bursts/day | Incidents caught |\n|---|---|---:|---:|---:|---:|---:|\n'
    for label,r in [('Current fixed policy',summary['current_model_fixed_policy_on_shared_cohort']),
                    ('Highest precision retaining >=40% recall',summary['best_precision_while_retaining_40pct_recall']),
                    ('Highest recall within false-alarm budgets',summary['best_recall_within_false_alarm_budgets'])]:
        if r is None:
            text+=f'| {label} | No candidate | — | — | — | — | — |\n';continue
        text+=f"| {label} ({r['policy_id']}) | {r['model']} / {r['rule']} | {r['precision']:.2%} | {r['recall']:.2%} | {r['false_positive_rate']:.2%} | {r['false_bursts_per_node_day']:.3f} | {r['detected']} / {r['episodes']} |\n"
    text+='\nThese policies use different models and/or scores. The table is a validation tradeoff illustration, not a claim that a threshold change improved one trained model or that test performance improved. The current fixed-policy result is on this shared cohort, not the larger native historical cohort reported in Task 1.\n\n'
    text+='## Experiment design\n\n'
    text+='The saved base8, cooling12_8 and robust24 autoencoders each contribute whole-window and latest-reading error scores. Isolation Forest contributes its anomaly score. All use the frozen 24-reading eligibility context, shared calendar validation period and known endpoint targets. Width-8 models still consume eight readings; the longer context sets common eligibility only. Existing training cohorts and early-stopping histories differ, so primary controlled comparisons still require retraining under protocol v1.\n\n'
    text+='Normal-history percentiles 95, 99, 99.5, 99.9 and 99.95 are calibrated globally and per server using only fully known normal validation histories. Global precision-recall candidates maximize empirical precision at recall floors 40%, 60% and 80%, plus F0.5. These candidates use validation labels and are not independent performance estimates. Strict-threshold code handles score ties consistently with precision-recall curve definitions.\n\n'
    text+='Every threshold is evaluated with immediate alarms, two consecutive high scores and at least two highs in three complete readings. All rules are causal and reset at server changes or gaps between eligible scored timestamps. Two-of-three can remain positive when the current score is low; its delay and false-positive costs are included in the reading decisions. Historical fixed policies provide references; Isolation Forest requires three dense consecutive highs to retain its original 30-minute confirmation span.\n\n'
    text+='Precision-recall curves and `global_threshold_precision_limits.csv` describe global immediate thresholds only. Temporal and per-node policies are assessed in `policy_comparison.csv`. No policy-grid result proves a theoretical ceiling on every possible detector. The Pareto table compares precision, recall and false bursts; feasible selection additionally requires the frozen FPR and incident-recall targets.\n\n'
    warning=summary['low_burst_high_recall_policies']
    if warning['count']:
        text+=f"There are {warning['count']} policies with recall >=40% and false bursts <=0.1/day, but their lowest reading-level FPR is {warning['minimum_false_positive_rate']:.2%} and their highest precision is {warning['maximum_precision']:.2%}. Almost-continuous alarms can merge into very few bursts while flagging most normal readings. The separate FPR and precision constraints correctly reject this misleading result. The detail plot colours these policies by reading-level FPR.\n\n"
    text+='Per-node percentile specifications also store the pooled normal-validation threshold as the fallback for an unseen server. No reserved server data is used to calibrate that fallback. Historical fixed policies remain references rather than deployable unseen-node candidates.\n\n'
    text+='## Notification cooldown\n\n'
    for row in illustrations:
        text+=f"- {row['policy_id']}: {row['raw_false_bursts']} raw false bursts and {row['raw_detected_incidents']} raw detected incidents; a 60-minute rising-edge cooldown emits {row['displayed_notifications']} notifications, of which {row['false_notifications']} are false by their notification timestamps, and notifies inside {row['notified_incidents']} incidents. Raw precision remains {row['raw_precision']:.2%} and raw recall remains {row['raw_recall']:.2%}.\n"
    text+='\nCooldown changes message delivery, not the reading predictions. A suppressed notification can hide an incident. It is therefore never used to satisfy the raw false-positive or false-burst targets. Notification truth is evaluated at the notification timestamp; a burst spanning an incident can have a false notification at its earlier start.\n\n'
    text+='## Verification and limits\n\n'
    text+='Model/scaler and cohort fingerprints are recorded in the predeclared experiment plan. Every score is finite and belongs to the same validation endpoints. Representative policies were cross-checked against the original evaluator for confusion counts, bursts, incidents and delay. The reserved nodes and their label values remain untouched. Validation selection can overfit, especially at high percentiles with very few exceedances; the frozen final independent evaluation is still required. All results here are diagnostic and should be labelled as such in the final report.\n\n'
    text+='Files: `policy_comparison.csv`, `pareto_frontier.csv`, `global_threshold_precision_limits.csv`, `notification_demonstration.csv`, policy specifications, decision CSVs and three plotted tradeoff figures.\n'
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    manifest,plan=experiment_plan()
    meta=pd.read_csv(ROOT/manifest['cohort_files']['validation']['path'])
    episodes=pd.read_csv(ROOT/'results'/'protocol_v1'/'validation_incident_coverage.csv')[['node','start_ns','end_ns','readings']]
    assert not set(meta.node).intersection(manifest['independent_holdout_nodes'])
    data=load_reduced(ROOT/'0.tar',ROOT/'artifacts'/'reduced_base.npz')
    _,filled,split=protocol_clean(data)
    assert (split[meta.row_index.to_numpy()]==1).all()
    np.testing.assert_array_equal(data['label'][meta.row_index.to_numpy()]==2,meta.y_end)
    engine,records,specs,scores,curves,baseline_ids=evaluate_search(meta,episodes,manifest,data,filled)
    finish(engine,records,specs,scores,curves,baseline_ids,plan)


if __name__=='__main__':main()
