"""Verify training inputs, saved inference and every Task 3 validation policy."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from dccad.data import load_reduced,BASE_FEATURES
from dccad.protocol import protocol_clean,constrained_policy_key,TRAIN_END
from dccad.preprocessing import select_features
from dccad.alarm_policies import PolicyEvaluator,apply_rule
from dccad.task3_inference import score_window
from audit_task1 import data_digest,digest as archive_digest


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4);out=ROOT/'results/task3_supervised'
    plan=json.loads((out/'experiment_plan.json').read_text());mp=ROOT/'results/protocol_v1/manifest.json'
    assert sha(mp)==plan['protocol_sha256']
    manifest=json.loads(mp.read_text())
    repairs=json.loads((out/'implementation_repairs.json').read_text()) if (out/'implementation_repairs.json').exists() else []
    for filename,source_digest in plan['source_checksums'].items():
        current=sha(ROOT/filename)
        if current!=source_digest:
            repair=next(r for r in repairs if r['source']==filename)
            assert repair['pre_training_sha256']==source_digest and repair['repaired_sha256']==current
            assert repair['training_or_policy_changes'] is False
    cohorts={}
    for key in ['train','validation']:
        spec=plan['cohorts'][key];p=ROOT/spec['path'];assert sha(p)==spec['sha256'];cohorts[key]=pd.read_csv(p)
    train=cohorts['train'];meta=cohorts['validation']
    assert not set(meta.node).intersection(manifest['independent_holdout_nodes'])
    data=load_reduced(ROOT/'0.tar',ROOT/'artifacts/reduced_base.npz')
    assert data_digest(data)==(ROOT/'results/protocol_v1/base_data.sha256').read_text().strip()
    assert archive_digest(ROOT/'0.tar')==manifest['source_archive_sha256']
    assert not set(data['node']).intersection(manifest['independent_holdout_nodes'])
    sanitized,filled,split=protocol_clean(data)
    fit=(split==0)&(data['label']==0)&np.isfinite(filled).all(axis=1)
    assert (data['time_ns'][fit]<TRAIN_END).all()
    scaler=StandardScaler().fit(filled[fit]);saved=joblib.load(out/'training_scaler.joblib')
    np.testing.assert_array_equal(saved.mean_,scaler.mean_);np.testing.assert_array_equal(saved.scale_,scaler.scale_)
    selected,_,_=select_features(data,sanitized,filled,split)
    episodes=pd.read_csv(ROOT/'results/protocol_v1/validation_incident_coverage.csv')[['node','start_ns','end_ns','readings']]
    engine=PolicyEvaluator(meta,episodes,15);scores={};inference_checks=[]
    for cfg in plan['configs']:
        folder=out/cfg['name'];r=json.loads((folder/'summary.json').read_text())
        assert r['validation_windows']==len(meta) and r['validation_positives']==int(meta.y_end.sum())
        assert r['reserved_nodes_scored'] is False
        unsupervised=r['kind'] in ['autoencoder','isolation_forest']
        expected=train[(train.history_critical_count==0)&(train.history_unknown_count==0)] if unsupervised else train
        assert r['training_windows']==len(expected) and r['training_positives']==int(expected.y_end.sum())
        for name,digest in r['artifact_checksums'].items():assert sha(folder/name)==digest
        pre=joblib.load(folder/'preprocessing.joblib');np.testing.assert_array_equal(pre['scaler'].mean_,scaler.mean_)
        if cfg['feature_set']=='correlation':assert pre['features']==selected
        z=np.load(folder/'validation_scores.npz')
        for field in ['node','time_ns','row_index','y_end']:np.testing.assert_array_equal(z[field],meta[field])
        for mode in r['score_metrics']:
            key=r['name']+'__'+mode;scores[key]=z[mode].astype(np.float64);engine.register_scores(key,scores[key])
            np.testing.assert_allclose(average_precision_score(meta.y_end,z[mode]),r['score_metrics'][mode]['average_precision'])
        example=json.loads((folder/'example_metadata.json').read_text());actual=score_window(folder,pd.read_csv(folder/'example_window.csv'))
        for mode,value in actual.items():
            np.testing.assert_allclose(value,example['expected_scores'][mode],rtol=2e-5,atol=2e-6)
            np.testing.assert_allclose(value,z[mode][example['validation_position']],rtol=2e-5,atol=2e-6)
        inference_checks.append(r['name'])
        if r['kind'] in ['lstm','autoencoder']:
            history=pd.read_csv(folder/'history.csv');assert len(history)==r['epochs_run']<=25
            criterion='validation_ap' if r['kind']=='lstm' else 'normal_validation_mse'
            index=history[criterion].idxmax() if r['kind']=='lstm' else history[criterion].idxmin()
            assert int(history.loc[index,'epoch'])==r['best_epoch']
    policies=pd.read_csv(out/'policy_comparison.csv');specs=json.loads((out/'policy_specs.json').read_text())
    for r in policies.to_dict('records'):
        spec=specs[r['policy_id']];t=spec['threshold']
        if isinstance(t,dict):
            t={int(k):v for k,v in t.items()};normal=(meta.history_critical_count==0)&(meta.history_unknown_count==0)
            np.testing.assert_allclose(spec['unseen_node_global_threshold'],np.percentile(scores[spec['score_key']][normal],r['quantile']))
        pred=apply_rule(meta,scores[spec['score_key']],t,spec['rule'])
        actual=engine.evaluate(pred,spec['score_key'])
        for key,value in actual.items():
            if value is None:assert pd.isna(r[key])
            else:np.testing.assert_allclose(value,r[key],rtol=1e-10,atol=1e-12,err_msg=r['policy_id']+' '+key)
        assert (constrained_policy_key(actual) is not None)==r['feasible']
    s=json.loads((out/'summary.json').read_text())
    assert int(policies.feasible.sum())==s['feasible_policies']
    neural=policies[(policies.kind=='lstm')&policies.feasible]
    assert len(neural)==s['feasible_neural_policies']
    if not len(neural):assert s['selected_neural_policy'] is None
    else:
        best=max(neural.to_dict('records'),key=constrained_policy_key)
        assert s['selected_neural_policy']['policy_id']==best['policy_id']
    assert not s['independent_holdout_scored'] and not s['exploratory_period_scored']
    checks=dict(passed=True,models_verified=len(inference_checks),standalone_csv_inference_verified=inference_checks,
        policies_reproduced=len(policies),validation_rows=len(meta),training_only_scaler_and_feature_selection_verified=True,
        checkpoint_selection_verified=True,source_and_artifact_hashes_verified=True,reserved_nodes_scored=False)
    (out/'reproduction_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
