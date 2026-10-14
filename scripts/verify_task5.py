"""Verify frozen recovery inputs/preprocessing, full-window inference and all policies."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import joblib
import numpy as np
import pandas as pd
import torch
from dccad.data import load_reduced,BASE_FEATURES
from dccad.protocol import protocol_clean,constrained_policy_key,TRAIN_END
from dccad.recovery import FEATURES,load_extra,fit_extension,preprocess_recovery_csv
from dccad.recovery_inference import score_recovery_window
from dccad.supervised import sequence_inputs,predict_logits,trailing_statistics,LSTMClassifier
from dccad.alarm_policies import PolicyEvaluator,apply_rule
from dccad.robustness import choose_seed_policy,node_bootstrap
from threadpoolctl import threadpool_limits
from audit_task1 import data_digest,digest


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4);out=ROOT/'results/task5_recovery_v1'
    plan=json.loads((out/'experiment_plan.json').read_text());mp=ROOT/'results/protocol_v1/manifest.json'
    assert sha(mp)==plan['protocol_sha256'];manifest=json.loads(mp.read_text())
    for f,h in plan['implementation_sha256'].items():assert sha(ROOT/f)==h
    for name,files in plan['baseline_artifacts'].items():
        for f,h in files.items():assert sha(ROOT/'results/task3_supervised'/name/f)==h
    for cohort in plan['cohorts'].values():assert sha(ROOT/cohort['path'])==cohort['sha256']
    train=pd.read_csv(ROOT/plan['cohorts']['train']['path']);meta=pd.read_csv(ROOT/plan['cohorts']['validation']['path'])
    data=load_reduced(ROOT/'0.tar',ROOT/'artifacts/reduced_base.npz')
    assert data_digest(data)==(ROOT/'results/protocol_v1/base_data.sha256').read_text().strip()
    assert digest(ROOT/'0.tar')==manifest['source_archive_sha256']
    assert not set(data['node']).intersection(manifest['independent_holdout_nodes'])
    _,filled,split=protocol_clean(data);extras=load_extra(ROOT/'0.tar',data)
    bp=joblib.load(ROOT/'results/task3_supervised/lstm8_p1/preprocessing.joblib')
    values,pre,audit=fit_extension(data,filled,extras,split,bp['scaler'])
    saved=joblib.load(out/'preprocessing.joblib')
    np.testing.assert_array_equal(saved['extra_medians'],pre['extra_medians'])
    np.testing.assert_array_equal(saved['extra_scaler'].mean_,pre['extra_scaler'].mean_)
    np.testing.assert_array_equal(saved['extra_scaler'].scale_,pre['extra_scaler'].scale_)
    np.testing.assert_array_equal(saved['base_scaler'].mean_,bp['scaler'].mean_)
    assert audit==json.loads((out/'preprocessing_audit.json').read_text())
    assert (data['time_ns'][train.row_index]<TRAIN_END).all()
    expected_base=np.clip(bp['scaler'].transform(filled),-10,10).astype(np.float32)
    np.testing.assert_array_equal(values[:,:22],expected_base)
    assert np.isfinite(values[meta.row_index]).all() and values.shape[1]==67
    xv=sequence_inputs(values,meta.row_index,8);scores={}
    episodes=pd.read_csv(ROOT/'results/protocol_v1/validation_incident_coverage.csv')[['node','start_ns','end_ns','readings']]
    engine=PolicyEvaluator(meta,episodes,15)
    for name in ['lstm8_p1','gradient_boosting8']+plan['models']:
        extended=name in plan['models'];folder=out/name if extended else ROOT/'results/task3_supervised'/name
        summary=json.loads((folder/'summary.json').read_text())
        mode=summary.get('score_mode','logit' if summary['kind']=='lstm' else 'score')
        z=np.load(folder/'validation_scores.npz')
        for field in ['node','time_ns','row_index','y_end']:np.testing.assert_array_equal(z[field],meta[field])
        scores[name]=z[mode].astype(np.float64);engine.register_scores(name,scores[name])
        if not extended:continue
        assert summary['training_windows']==len(train) and summary['training_positives']==int(train.y_end.sum())
        assert summary['validation_windows']==len(meta) and summary['validation_positives']==int(meta.y_end.sum())
        assert summary['features']==FEATURES and not summary['reserved_nodes_scored']
        for f,h in summary['artifact_checksums'].items():assert sha(folder/f)==h
        history=pd.read_csv(folder/'history.csv')
        if summary['kind']=='lstm':
            cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            model=LSTMClassifier(len(FEATURES),32,.2);model.load_state_dict(cp['state_dict'])
            actual=predict_logits(model,xv)
            assert len(history)==summary['epochs_run']<=25
            assert int(history.loc[history.validation_ap.idxmax(),'epoch'])==summary['best_epoch']
        else:
            model=joblib.load(folder/'model.joblib')
            with threadpool_limits(limits=4):actual=model.predict_proba(trailing_statistics(xv))[:,1]
            assert int(history.loc[history.validation_ap.idxmax(),'iterations'])==summary['best_iterations']
        np.testing.assert_allclose(actual,z[mode],rtol=1e-7,atol=1e-8)
        example=json.loads((folder/'example_metadata.json').read_text());frame=pd.read_csv(folder/'example_window.csv')
        fresh=score_recovery_window(folder,frame)
        np.testing.assert_allclose(fresh,z[mode][example['validation_position']],rtol=2e-5,atol=2e-6)
        end=example['row_index'];new_values=preprocess_recovery_csv(frame,pre)
        np.testing.assert_allclose(new_values,values[end-23:end+1],rtol=1e-6,atol=1e-7)
    policies=pd.read_csv(out/'policy_comparison.csv');specs=json.loads((out/'policy_specs.json').read_text())
    assert len(policies)==180 and policies.policy_id.is_unique
    for row in policies.to_dict('records'):
        spec=specs[row['policy_id']];t=spec['threshold']
        if isinstance(t,dict):
            t={int(k):v for k,v in t.items()};normal=(meta.history_critical_count==0)&(meta.history_unknown_count==0)
            np.testing.assert_allclose(spec['unseen_node_global_threshold'],np.percentile(scores[row['model']][normal],row['quantile']))
        pred=apply_rule(meta,scores[row['model']],t,spec['rule']);actual=engine.evaluate(pred,row['model'])
        for k,v in actual.items():
            if v is None:assert pd.isna(row[k])
            else:np.testing.assert_allclose(v,row[k],rtol=1e-10,atol=1e-12)
        assert (constrained_policy_key(actual) is not None)==row['feasible']
    comparison=pd.read_csv(out/'model_comparison.csv');nodes=pd.read_csv(out/'per_node_validation.csv')
    intervals=json.loads((out/'node_bootstrap_intervals.json').read_text())
    for name in comparison.model:
        chosen,status=choose_seed_policy(policies[policies.model==name].to_dict('records'))
        saved_row=comparison[comparison.model==name].iloc[0]
        assert saved_row.policy_id==chosen['policy_id'] and saved_row.status==status
        spec=specs[chosen['policy_id']];t=spec['threshold'];t={int(k):v for k,v in t.items()} if isinstance(t,dict) else t
        pred=apply_rule(meta,scores[name],t,spec['rule']);node_actual=[]
        for node in meta.node.unique():
            keep=meta.node==node;m=meta[keep].reset_index(drop=True);e=PolicyEvaluator(m,episodes[episodes.node==node],15)
            e.register_scores('score',scores[name][keep]);actual=e.evaluate(pred[keep],'score');node_actual.append(actual)
            r=nodes[(nodes.model==name)&(nodes.node==node)].iloc[0]
            for k,v in actual.items():
                if v is None:assert pd.isna(r[k])
                else:np.testing.assert_allclose(v,r[k],rtol=1e-10,atol=1e-12)
        assert node_bootstrap(node_actual)==intervals[name]
        if name in plan['models']:
            d=pd.read_csv(out/name/'validation_decisions.csv');np.testing.assert_array_equal(d.pred,pred)
    result=json.loads((out/'summary.json').read_text())
    assert result['feasible_recovery_policies']==int(policies[policies.model.isin(plan['models'])].feasible.sum())
    candidates=[]
    for r in comparison.to_dict('records'):
        if r['model'] in plan['models'] and r['feasible']:
            original=comparison[(~comparison.model.isin(plan['models']))&(comparison.kind==r['kind'])].iloc[0]
            if r['average_precision']>original.average_precision:candidates.append(r)
    assert bool(candidates)==(result['recovery_candidate'] is not None)
    assert not result['independent_holdout_scored'] and not result['exploratory_period_scored'] and not result['dashboard_default_changed']
    checks=dict(passed=True,recovery_models_verified=2,all_validation_scores_freshly_reproduced=True,
        standalone_csv_windows_reproduced=2,policies_reproduced=180,validation_endpoints=len(meta),
        original_base_preprocessing_unchanged=True,extra_preprocessing_training_only=True,
        no_endpoint_excluded_for_extra_missingness=True,node_metrics_and_bootstrap_reproduced=True,
        frozen_input_hashes_verified=True,recovery_selection_gate_reproduced=True,reserved_nodes_scored=False)
    (out/'reproduction_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
