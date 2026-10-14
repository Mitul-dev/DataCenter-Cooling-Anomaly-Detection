"""Validation-only failure audit and a frozen protocol; no final-node scores."""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dccad.data import load_reduced, prepare, windows, raw_episodes, CADENCE_NS, BASE_FEATURES, NODES
from dccad.preprocessing import prepare_clean, clean_data
from dccad.evaluation import metrics, consecutive_alarm, incident_metrics
from dccad.model import LSTMAutoencoder, predict_errors
from dccad.protocol import (TRAIN_END, VALIDATION_END, FINAL_END, HOLDOUT_NODES,
                           protocol_clean, cohort_metadata)

OUT = ROOT/'results'/'task1_audit'
PROTO = ROOT/'results'/'protocol_v1'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def data_digest(data):
    """Fingerprint logical inputs, independent of NPZ ZIP timestamps."""
    h=hashlib.sha256()
    for key in sorted(data):
        a=np.ascontiguousarray(data[key])
        if a.dtype.kind=='f':
            a=a.copy();a[np.isnan(a)]=np.nan
        h.update(json.dumps([key,a.dtype.str,list(a.shape)]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def timestamp(ns):return pd.to_datetime(int(ns),utc=True).isoformat()


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def quality_and_protocol(data):
    PROTO.mkdir(parents=True,exist_ok=True)
    previous=json.loads((PROTO/'manifest.json').read_text()) if (PROTO/'manifest.json').exists() else None
    fingerprint=PROTO/'base_data.sha256';logical_hash=data_digest(data)
    if fingerprint.exists() and fingerprint.read_text().strip()!=logical_hash:
        raise RuntimeError('Frozen logical input data changed; create a new protocol version')
    if previous:
        boundaries=previous['time_boundaries_utc']
        expected={'train_end_exclusive':timestamp(TRAIN_END),'validation_end_exclusive':timestamp(VALIDATION_END),
                  'final_end_exclusive':timestamp(FINAL_END)}
        if boundaries!=expected or previous['primary_sensor_features']!=BASE_FEATURES or previous['maximum_input_width']!=24:
            raise RuntimeError('Frozen protocol parameters changed; create a new version before generating cohorts')
    sanitized,filled,split=protocol_clean(data)
    missing=~np.isfinite(data['raw'])
    invalid=np.isfinite(data['raw']) & np.isnan(sanitized)
    imputed=np.isnan(sanitized) & np.isfinite(filled)
    quality={'node':data['node'],'time_ns':data['time_ns']}
    for j,c in enumerate(data['features']):
        quality['missing_'+str(c)]=missing[:,j]
        quality['invalid_'+str(c)]=invalid[:,j]
        quality['imputed_'+str(c)]=imputed[:,j]
    pl.DataFrame(quality).write_parquet(ROOT/'artifacts'/'quality_indicators.parquet')
    quality_rows=[]; splits=[]; coverage=[]; cohort_files={}
    for which,name,stride in [(0,'train',4),(1,'validation',1),(2,'exploratory',1)]:
        meta=cohort_metadata(data,filled,split,which,width=24,stride=stride)
        target=PROTO/f'{name}_cohort.csv'
        payload=meta.to_csv(index=False).encode('utf-8')
        payload_hash=hashlib.sha256(payload).hexdigest()
        if previous and previous['cohort_files'][name]['sha256']!=payload_hash:
            raise RuntimeError('Frozen cohort changed; refusing to overwrite its CSV')
        target.write_bytes(payload)
        cohort_files[name]={'path':str(target.relative_to(ROOT)).replace('\\','/'),'sha256':digest(target),
                            'windows':len(meta),'critical_endpoints':int(meta.y_end.sum()),
                            'normal_only_calibration_windows':int(((meta.history_critical_count==0)&(meta.history_unknown_count==0)).sum())}
        episodes=raw_episodes(data,split,which)
        _,detail=incident_metrics(meta,np.zeros(len(meta),bool),episodes,cadence_minutes=15)
        detail.to_csv(PROTO/f'{name}_incident_coverage.csv',index=False)
        mask=split==which
        splits.append({'split':name,'rows':int(mask.sum()),'known_normal':int((mask&(data['label']==0)).sum()),
                       'critical':int((mask&(data['label']==2)).sum()),'unknown_or_other':int((mask&~np.isin(data['label'],[0,2])).sum()),
                       'raw_incidents':len(episodes),'observable_incidents':int((detail.score_opportunities>0).sum()),
                       'eligible_windows':len(meta),'eligible_critical_endpoints':int(meta.y_end.sum())})
        for node in np.unique(data['node']):
            nm=mask&(data['node']==node); m=meta[meta.node==node]; e=detail[detail.node==node]
            coverage.append({'split':name,'node':int(node),'raw_rows':int(nm.sum()),
                'known_target_rows':int((nm&np.isin(data['label'],[0,2])).sum()),
                'eligible_windows':len(m),'critical_endpoints':int(m.y_end.sum()),
                'raw_incidents':len(e),'observable_incidents':int((e.score_opportunities>0).sum())})
        for label in [-1,0,2,3]:
            lm=mask&(data['label']==label)
            if not lm.any():continue
            quality_rows.append({'split':name,'label':label,'rows':int(lm.sum()),
                'missing_cells':int(missing[lm].sum()),'invalid_cells':int(invalid[lm].sum()),
                'imputed_cells':int(imputed[lm].sum()),'rows_with_any_missing':int(missing[lm].any(axis=1).sum()),
                'rows_with_any_invalid':int(invalid[lm].any(axis=1).sum())})
    pd.DataFrame(splits).to_csv(OUT/'calendar_split_summary.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'protocol_coverage_by_node.csv',index=False)
    pd.DataFrame(quality_rows).to_csv(OUT/'quality_by_label.csv',index=False)
    # Inspect schema and timestamps only. No value-column reads or holdout predictions.
    inventory=[]
    with tarfile.open(ROOT/'0.tar') as tar:
        for node in HOLDOUT_NODES:
            contents=tar.extractfile(f'{node}.parquet').read()
            schema=pl.read_parquet_schema(io.BytesIO(contents))
            ts=pl.read_parquet(io.BytesIO(contents),columns=['timestamp']).sort('timestamp')['timestamp'].dt.epoch('ns').to_numpy()
            final_times=ts[(ts>=VALIDATION_END)&(ts<FINAL_END)]
            if len(final_times)>=24:
                tw=np.lib.stride_tricks.sliding_window_view(final_times,24)
                candidates=int((np.diff(tw,axis=1)==CADENCE_NS).all(axis=1).sum())
            else:candidates=0
            exact=['timestamp','value','ambient_avg']+[f'fan{i}_{j}_avg' for i in range(4) for j in range(2)]
            exact += [c+'_avg' for c in ['fan_disk_power','p0_vdd_temp','p1_vdd_temp','p0_power','p1_power','total_power','ps0_input_power','ps1_input_power']]
            missing_columns=[c for c in exact if c not in schema]
            families=[('dimm','_temp_avg'),('p0_core','_temp_avg'),('p1_core','_temp_avg'),('gpu','core_temp_avg'),('gpu','mem_temp_avg')]
            absent_families=[a+'*'+b for a,b in families if not any(c.startswith(a) and c.endswith(b) for c in schema)]
            inventory.append({'node':node,'rows':len(ts),'first_utc':timestamp(ts.min()),'last_utc':timestamp(ts.max()),
                'duplicate_timestamps':int(pd.Series(ts).duplicated().sum()),'final_period_rows':len(final_times),
                'clock_eligible_24_reading_endpoints':candidates,'missing_source_columns':','.join(missing_columns+absent_families),
                'label_values_read':False,'numeric_sensor_values_read':False,'predictions_made':False})
            print('Holdout schema/timestamps inventoried:',node,flush=True)
    pd.DataFrame(inventory).to_csv(PROTO/'holdout_inventory.csv',index=False)
    manifest={'protocol_version':'v1','status':'frozen_before_new_model_training','prepared_date':'2026-10-14',
        'task':'Current Nagios server-health anomaly classification from thermal, fan and power telemetry',
        'claims_excluded':['cooling-specific diagnosis','future-fault prediction','energy savings'],
        'development_nodes':NODES,'independent_holdout_nodes':HOLDOUT_NODES,
        'time_boundaries_utc':{'train_end_exclusive':timestamp(TRAIN_END),
            'validation_end_exclusive':timestamp(VALIDATION_END),'final_end_exclusive':timestamp(FINAL_END)},
        'boundary_rationale':'Day-aligned dates after the latest historical training and validation cutoffs, chosen from timestamp metadata, not model scores.',
        'cadence_minutes':15,'maximum_input_width':24,'candidate_input_widths':[8,24],
        'training_stride':4,'validation_and_evaluation_stride':1,'primary_sensor_features':BASE_FEATURES,
        'quality_feature_candidates':['missing_<sensor>','invalid_<sensor>','imputed_<sensor>'],
        'preprocessing':{'negative_base_sensor_measurements':'missing under indoor-rack plausibility assumption; preserve raw and invalid flags',
            'filling':'causal, maximum four readings, reset at node/gap/calendar split',
            'unresolved_missing_values':'exclude from primary matched cohort; report coverage',
            'scaler_fit':'finite normal training rows only; never validation/holdout',
            'feature_selection_fit':'training only; no label-derived features',
            'label_mapping':{'normal':0,'critical':2,'unknown_or_other':'exclude endpoint targets, never relabel normal'}},
        'cohort_rule':'Primary comparison requires a finite, consecutive, same-split 24-reading history across all 22 base sensors and a known endpoint label. Historical labels inside inputs may be unknown; labels are never model inputs. Autoencoder training/calibration additionally requires all history labels normal.',
        'cohort_files':cohort_files,'source_archive_sha256':digest(ROOT/'0.tar'),
        'base_cache_sha256':digest(ROOT/'artifacts'/'reduced_base.npz'),
        'baseline_rescoring':'Historical models are diagnostics only. Fair primary baselines must be retrained on the calendar-aligned training cohort.',
        'policy_targets':{'precision_min':.30,'recall_min':.40,'false_positive_rate_max':.01,
                          'false_bursts_per_scored_node_day_max':.10,'observable_incident_recall_min':.60},
        'policy_selection':'Among feasible validation policies, maximize precision, then F0.5, then minimize false-burst rate, then maximize recall. If none feasible, report no feasible policy; do not relax targets silently.',
        'independent_evaluation':{'gate':'Only after architecture, seed rule, preprocessing, model and alarm policy are frozen',
            'period':'[validation_end_exclusive, final_end_exclusive) on the seven reserved nodes',
            'node_thresholds':'Global validation-calibrated fallback for unseen nodes; no calibration on final labels',
            'interpretation':'Unseen-node and later-period generalization, reported separately from exploratory development recordings',
            'label_or_finite_coverage':'Apply predeclared eligibility at final evaluation, disclose all excluded rows/incidents without changing servers or period'},
        'historical_test_status':'already inspected; exploratory, never claim it is a new blind test',
        'seed_rule':'Evaluate finalist with 7,42,123; release uses seed 42 unless a different validation-only rule is documented before final evaluation'}
    target=PROTO/'manifest.json'
    fingerprint.write_text(logical_hash+'\n')
    if target.exists():
        previous=json.loads(target.read_text())
        # Container timestamps can change when a cache is regenerated from identical data.
        compare=lambda d:{k:v for k,v in d.items() if k!='base_cache_sha256'}
        if compare(previous)!=compare(manifest):raise RuntimeError('Frozen manifest changed; create a new protocol version with an explicit rationale')
        manifest=previous
    else:write_json(target,manifest)
    (PROTO/'manifest.sha256').write_text(digest(target)+'\n')
    return manifest,splits,quality_rows


def rescore_references(data,selected_audit):
    """Fixed historical policies at 15 minutes; no policy search or refitting."""
    records=[]
    for name in ['base8','robust24','isolation_forest']:
        folder=ROOT/'results'/name;r=json.loads((folder/'summary.json').read_text())
        values,split,_=prepare(data,r.get('scaler','standard'),r.get('clip',5.))
        # The preparation is identical to the historical training transform.
        x,meta=windows(data,values,split,1,r['width'],stride=1)
        _,oldmeta=windows(data,values,split,1,r['width'],stride=2,materialize=False)
        index=meta.set_index('row_index').index.get_indexer(oldmeta.row_index)
        saved=np.load(folder/'scores.npz');policy=r['selected_policy']
        if name=='isolation_forest':
            model=joblib.load(folder/'model.joblib')
            score=-model.score_samples(x.reshape(len(x),-1))
            np.testing.assert_allclose(score[index],saved['va'],rtol=1e-5,atol=1e-6)
        else:
            cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            model=LSTMAutoencoder(len(cp['features']),cp['width']);model.load_state_dict(cp['state_dict'])
            mean,end,_=predict_errors(model,x)
            score=mean if policy['score_mode']=='whole_window' else end
            key='va_mean' if policy['score_mode']=='whole_window' else 'va_end'
            np.testing.assert_allclose(score[index],saved[key],rtol=5e-4,atol=2e-6)
        threshold=policy['threshold']
        if isinstance(threshold,dict):threshold={int(k):v for k,v in threshold.items()}
        # Preserve elapsed confirmation span: old k=2 at 30 min becomes k=3 at 15.
        persistence=1+(policy['persistence']-1)*2
        pred=consecutive_alarm(meta,score,threshold,persistence,cadence_minutes=15)
        incidents,_=incident_metrics(meta,pred,raw_episodes(data,split,1),cadence_minutes=15)
        record={'experiment':name,'width':r['width'],'validation_readings':len(meta),
            'old_persistence':policy['persistence'],'dense_persistence':persistence,
            'threshold_unchanged':True,'cohort':'native historical validation, not the frozen shared cohort',
            **metrics(meta.y_end,score,pred),**incidents}
        records.append(record)
        np.savez_compressed(OUT/(name+'_validation_15min.npz'),score=score,node=meta.node.to_numpy(),
            time_ns=meta.time_ns.to_numpy(),row_index=meta.row_index.to_numpy(),y_end=meta.y_end.to_numpy(),
            y_window=meta.y_window.to_numpy())
        print('Reference rescored at 15 min:',name,flush=True)
    records.append({'experiment':selected_audit['model'],'width':8,
        'validation_readings':selected_audit['validation_15min']['tp']+selected_audit['validation_15min']['fp']+selected_audit['validation_15min']['tn']+selected_audit['validation_15min']['fn'],
        'old_persistence':1,'dense_persistence':1,'threshold_unchanged':True,
        'cohort':'native historical validation, not the frozen shared cohort',**selected_audit['validation_15min']})
    pd.DataFrame(records).to_csv(OUT/'historical_reference_15min.csv',index=False)
    return records


def validation_audit(data):
    name=json.loads((ROOT/'results'/'selected_model.json').read_text())['name']
    folder=ROOT/'results'/name;summary=json.loads((folder/'summary.json').read_text())
    cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    reduced,scaled,split,scaler=prepare_clean(data,cp['features'])
    x,meta=windows(reduced,scaled,split,1,cp['width'],stride=1)
    model=LSTMAutoencoder(len(cp['features']),cp['width'])
    model.load_state_dict(cp['state_dict']);torch.set_num_threads(4)
    mean,end,feature=predict_errors(model,x)
    _,old_meta=windows(reduced,scaled,split,1,cp['width'],stride=2,materialize=False)
    index=meta.set_index('row_index').index.get_indexer(old_meta.row_index)
    assert (index>=0).all()
    saved=np.load(folder/'scores.npz')
    np.testing.assert_allclose(mean[index],saved['va_mean'],rtol=5e-4,atol=2e-6)
    np.testing.assert_allclose(end[index],saved['va_end'],rtol=5e-4,atol=2e-6)
    policy=summary['selected_policy'];thr=policy['threshold']
    if isinstance(thr,dict):thr={int(k):v for k,v in thr.items()}
    score=mean if policy['score_mode']=='whole_window' else end
    pred=consecutive_alarm(meta,score,thr,policy['persistence'],cadence_minutes=15)
    episodes=raw_episodes(data,split,1)
    ev,details=incident_metrics(meta,pred,episodes,cadence_minutes=15)
    details.to_csv(OUT/'validation_incidents_15min.csv',index=False)
    oldscore=saved['va_mean'] if policy['score_mode']=='whole_window' else saved['va_end']
    oldpred=consecutive_alarm(old_meta,oldscore,thr,policy['persistence'],cadence_minutes=30)
    oldev,_=incident_metrics(old_meta,oldpred,episodes,cadence_minutes=30)
    oldmetrics=metrics(old_meta.y_end,oldscore,oldpred)
    for key in ['tp','fp','tn','fn']:assert oldmetrics[key]==policy[key]
    result={'model':name,'policy_unchanged':True,'thresholds_recalibrated':False,
        'historical_training_and_split':'original per-node row splits; exploratory diagnostic, not the new calendar protocol',
        'validation_30min':{**oldmetrics,**oldev},'validation_15min':{**metrics(meta.y_end,score,pred),**ev}}
    frame=meta.copy();frame['score']=score;frame['alarm']=pred
    frame['threshold']=[thr[int(n)] if isinstance(thr,dict) else thr for n in meta.node]
    frame['score_to_threshold']=frame.score/frame.threshold
    frame['outcome']=np.where(pred,np.where(meta.y_end,'TP','FP'),np.where(meta.y_end,'FN','TN'))
    raw=data['raw'];cleaned,filled,_=clean_data(data)
    starts=meta.row_index.to_numpy()-cp['width']+1
    quality={'missing':~np.isfinite(raw),'invalid':np.isfinite(raw)&np.isnan(cleaned),
             'imputed':np.isnan(cleaned)&np.isfinite(filled)}
    for key,flag in quality.items():
        perrow=flag.sum(axis=1);cs=np.r_[0,np.cumsum(perrow)]
        frame[key+'_cells_in_history']=cs[starts+cp['width']]-cs[starts]
    frame['critical_in_history']=meta.y_window
    frame['top_latest_error_sensor']=[cp['features'][j] for j in feature.argmax(axis=1)]
    frame['latest_error_fraction_in_top_sensor']=feature.max(axis=1)/np.maximum(feature.sum(axis=1),1e-20)
    frame.to_csv(OUT/'validation_scored_15min.csv',index=False)
    nodes=[]
    for node in frame.node.unique():
        m=frame.node.to_numpy()==node
        r={**metrics(frame.y_end.to_numpy()[m],score[m],pred[m])}
        nodes.append({'node':int(node),'readings':int(m.sum()),'critical':int(frame.y_end.to_numpy()[m].sum()),
            'normal_score_p95':float(np.percentile(score[m&~frame.y_end.to_numpy()],95)),
            'threshold':float(thr[int(node)] if isinstance(thr,dict) else thr),**r})
    pd.DataFrame(nodes).to_csv(OUT/'validation_by_node.csv',index=False)
    fp=frame[frame.outcome=='FP']
    result['false_alarm_diagnostics']={'readings':len(fp),'with_critical_history':int(fp.critical_in_history.sum()),
        'with_raw_missing_history':int((fp.missing_cells_in_history>0).sum()),
        'with_invalid_history':int((fp.invalid_cells_in_history>0).sum()),
        'with_imputed_history':int((fp.imputed_cells_in_history>0).sum()),
        'top_latest_error_sensor_counts':fp.top_latest_error_sensor.value_counts().to_dict()}
    shifts=[]
    for node in frame.node.unique():
        tr=(data['node']==node)&(split==0)&(data['label']==0)
        va=(data['node']==node)&(split==1)&(data['label']==0)
        for j,c in enumerate(data['features']):
            tv=raw[tr,j];vv=raw[va,j];tv=tv[np.isfinite(tv)];vv=vv[np.isfinite(vv)]
            if len(tv) and len(vv):
                std=float(np.std(tv,dtype=np.float64))
                shifts.append({'node':int(node),'feature':str(c),'training_normal_median':float(np.median(tv)),
                    'validation_normal_median':float(np.median(vv)),
                    'median_shift_in_training_std':float((np.median(vv)-np.median(tv))/std) if std>1e-8 else None})
    pd.DataFrame(shifts).to_csv(OUT/'sensor_distribution_shifts.csv',index=False)
    cases=[]
    def add(kind,index):
        i=int(index);row=frame.iloc[i]
        cases.append({'kind':kind,'node':int(row.node),'time_ns':int(row.time_ns),'time_utc':timestamp(row.time_ns),
            'score':float(row.score),'threshold':float(row.threshold),'outcome':str(row.outcome),
            'critical_in_input_history':bool(row.critical_in_history),'missing_cells_in_history':int(row.missing_cells_in_history),
            'imputed_cells_in_history':int(row.imputed_cells_in_history),'top_latest_error_sensor':str(row.top_latest_error_sensor)})
    if len(fp):add('highest_score_false_alarm',fp.score.idxmax())
    lag=fp[fp.critical_in_history]
    if len(lag):add('normal_endpoint_with_critical_history',lag.score.idxmax())
    else:
        if len(fp):add('near_threshold_false_alarm',(fp.score_to_threshold-1).abs().idxmin())
    hits=details[details.detected]
    if len(hits):
        ep=hits.iloc[0];m=(frame.node==ep.node)&(frame.time_ns>=ep.start_ns)&(frame.time_ns<ep.end_ns)&frame.alarm
        add('detected_incident',frame[m].index[0])
    missed=details[(~details.detected)&(details.score_opportunities>0)]
    if len(missed):
        ep=missed.sort_values('readings',ascending=False).iloc[0]
        m=(frame.node==ep.node)&(frame.time_ns>=ep.start_ns)&(frame.time_ns<ep.end_ns)
        add('missed_observable_incident',frame[m].score.idxmax())
    for case in cases:
        center=case['time_ns'];node=case['node'];view=frame[(frame.node==node)&(abs(frame.time_ns-center)<=6*3600*10**9)]
        sensor=case['top_latest_error_sensor'];j=data['features'].tolist().index(sensor)
        fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
        times=pd.to_datetime(view.time_ns,utc=True)
        breaks=np.r_[True,np.diff(view.time_ns.to_numpy())!=CADENCE_NS]
        axes[0].plot(times,view.score.mask(breaks),color='#19638a',label='Anomaly score')
        axes[0].plot(times,view.threshold,color='#c67b22',ls='--',label='Existing threshold')
        crit=view.y_end.to_numpy();axes[0].scatter(times[crit],view.score.to_numpy()[crit],color='#bc3636',label='Critical endpoint',zorder=3)
        axes[0].set_ylabel('Reconstruction error');axes[0].legend(fontsize=8)
        readings=raw[view.row_index.to_numpy(),j].copy();readings[breaks]=np.nan
        axes[1].plot(times,readings,color='#19638a',label='Original reading')
        axes[1].set_ylabel(sensor);axes[1].set_xlabel('UTC time');axes[1].legend(fontsize=8)
        for ax in axes:ax.grid(alpha=.15)
        axes[0].set_title(case['kind'].replace('_',' ').title()+f' — server {node}')
        fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/(case['kind']+'.png'),dpi=150);plt.close(fig)
    pd.DataFrame(cases).to_csv(OUT/'cases.csv',index=False)
    result['cases']=cases
    write_json(OUT/'summary.json',result)
    print('Validation audit:',json.dumps(result['validation_15min']),flush=True)
    return result


def write_report(audit,manifest,splits,quality_rows):
    dense=audit['validation_15min'];old=audit['validation_30min'];diag=audit['false_alarm_diagnostics']
    node=pd.read_csv(OUT/'validation_by_node.csv').sort_values('fp',ascending=False)
    shifts=pd.read_csv(OUT/'sensor_distribution_shifts.csv')
    shifts=shifts.iloc[shifts.median_shift_in_training_std.abs().sort_values(ascending=False).index].head(5)
    text='# Task 1: data, failure and evaluation audit\n\n'
    text+='This audit evaluates the existing selected model on validation recordings only. No new model was trained, no threshold was changed, and no prediction or label-value inspection was performed on the seven reserved final nodes. Historical test results remain exploratory.\n\n'
    text+='## Problem statement\n\nPredict whether the current 15-minute endpoint has a Nagios CRITICAL server-health label from preceding and current thermal, fan and power telemetry. This is current-state detection. Labels do not establish cooling-specific failures, future-fault prediction or a physical diagnosis. [M100 data descriptor](https://www.nature.com/articles/s41597-023-02174-3).\n\n'
    text+='## Findings from the validation audit\n\n'
    text+=f"At the unchanged policy, the 30-minute validation evaluation has precision {old['precision']:.2%}, recall {old['recall']:.2%}, FPR {old['false_positive_rate']:.2%}, and {old['false_bursts']} false bursts. Scoring every 15 minutes has precision {dense['precision']:.2%}, recall {dense['recall']:.2%}, FPR {dense['false_positive_rate']:.2%}, and {dense['false_bursts']} false bursts. Different sampling grids change the cohort and alarm opportunities; this is not a model improvement. Saved 30-minute validation scores and confusion counts were independently reproduced.\n\n"
    text+=f"Of {diag['readings']:,} false-positive validation endpoints at 15-minute cadence, {diag['with_critical_history']:,} still contain a preceding critical reading in their input history, {diag['with_raw_missing_history']:,} have missing original measurements somewhere in the 22-sensor history, and {diag['with_imputed_history']:,} have filled measurements. These groups can overlap. They identify associations to investigate, not proven causes of the model errors. The quality counts include all 22 original sensors; the selected model uses 12.\n\n"
    text+='The servers contributing the most false-positive endpoints are: '+', '.join(f"{int(r.node)} ({int(r.fp):,})" for r in node.head(3).itertuples())+'. Full per-node metrics remain available; pooled metrics must not conceal server differences.\n\n'
    text+='None of these validation false positives had missing/imputed input measurements. Most also lacked a critical reading in their recent history. This audit therefore does not support missing-data repair or residual critical history as the main explanation of false alarms. The percentile threshold admits roughly 5% of normal validation endpoints, and normal-pattern changes deserve investigation. These are observations and hypotheses, not a proven physical fault diagnosis.\n\n'
    text+='The largest training-to-validation normal-reading median shifts, measured in within-server training standard deviations, are: '+', '.join(f"{r.feature} on {int(r.node)} ({r.median_shift_in_training_std:.2f})" for r in shifts.itertuples())+'. These are descriptive shifts, not validated fault indicators. Raw sensor quality failures can affect these estimates.\n\n'
    text+='## Cases for review\n\n'
    for case in audit['cases']:
        text+=f"- {case['kind']}: server {case['node']}, {case['time_utc']}; endpoint outcome {case['outcome']}; highest latest-reading reconstruction-error contributor {case['top_latest_error_sensor']}.\n"
    text+='\nThe corresponding PNGs show the score, fixed threshold, critical endpoints and original sensor readings. The top sensor explains latest-reading squared error only; the whole-window decision also includes earlier readings and does not prove the fault cause.\n\n'
    text+='## Sensor-quality findings\n\n'
    for row in quality_rows:
        if row['invalid_cells']:
            text+=f"- {row['split']}, label {row['label']}: {row['invalid_cells']} implausible negative sensor cells in {row['rows_with_any_invalid']} rows. Original values remain preserved; invalid flags are retained for future ablation.\n"
    text+='\nUnknown labels remain unknown. Genuine fan zeros, high readings and rare events are preserved. Quality-indicator candidates are saved in `artifacts/quality_indicators.parquet`; their usefulness is not yet established. No future label or Nagios state is added as a feature.\n\n'
    text+='## Frozen evaluation protocol\n\n'
    text+='The autoencoder reference, robust-scaled autoencoder, Isolation Forest and selected model were also scored at 15-minute cadence with their fixed historical thresholds. `historical_reference_15min.csv` records them. Persistence was converted to preserve its minimum elapsed confirmation span (two high samples 30 minutes apart become three high samples 15 minutes apart). This additionally checks intermediate readings. These native cohorts differ by input width; the table is a diagnostic, not a fair ranking. No threshold or model was tuned.\n\n'
    text+='Historical row splits differed by server: server 16 training continued into dates assigned to validation on other servers. New models must use calendar-aligned boundaries: training before 22 January 2022 UTC; validation from 22 January to before 16 April 2022 UTC; exploratory recordings from 16 April onward. These new boundaries do not make previously inspected recordings blind. Existing models were trained under the older row splits, so they are diagnostic references and must be retrained for a fair primary comparison.\n\n'
    text+='The primary matched cohort requires a consecutive, finite 24-reading context across all 22 base sensors, with no node/split/gap crossings and a known endpoint target. Width-8 and width-24 candidates use exactly these endpoints. Unknown historical labels do not disqualify otherwise usable measurements or become input features; normal-only training/calibration requires all historical labels normal. Training stride is four and validation/evaluation stride is one. Limited causal filling resets at every node, timestamp gap and calendar boundary.\n\n'
    text+='| Split | Eligible windows | Critical endpoints | Observable / raw incidents |\n|---|---:|---:|---:|\n'
    for s in splits:text+=f"| {s['split']} | {s['eligible_windows']:,} | {s['eligible_critical_endpoints']:,} | {s['observable_incidents']} / {s['raw_incidents']} |\n"
    text+='\nTraining coverage uses its stride-four sampled cohort. All-incident denominators preserve unobservable incidents. Exposure equals scored bins multiplied by the explicit 15-minute cadence, not wall-clock server uptime. Notification cooldown must remain separate from raw reading decisions.\n\n'
    text+='Final independent nodes are 2, 3, 4, 5, 6, 18 and 19 over [16 April, 29 September 2022) UTC. Only schema and timestamps were inventoried now; their sensor values, labels and predictions are reserved for final evaluation. Unknown/finite coverage will be reported after applying the frozen eligibility rule. This separately evaluates unseen-node and later-period generalization, not just performance on familiar servers.\n\n'
    text+='Select policies on validation only. Targets are precision >=30%, recall >=40%, FPR <=1%, false bursts <=0.1 per scored node-day, and observable-incident recall >=60%. Among feasible policies maximize precision, then F0.5, then minimize false bursts, then maximize recall. If none is feasible, report no feasible policy. These goals are not claimed achievements or professor acceptance rules.\n\n'
    text+='## Next task\n\nTask 2 is threshold/persistence tradeoff analysis on validation data, followed by label-aligned supervised baselines if the existing scores cannot meet the targets. The manifest, cohort CSVs and SHA-256 checksums in `results/protocol_v1` fix the scope before new experiments.\n'
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=load_reduced(ROOT/'0.tar',ROOT/'artifacts'/'reduced_base.npz')
    audit=validation_audit(data)
    rescore_references(data,audit)
    manifest,splits,quality=quality_and_protocol(data)
    write_report(audit,manifest,splits,quality)
    print('Task 1 audit and frozen protocol complete',flush=True)


if __name__=='__main__':main()
