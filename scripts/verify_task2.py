"""Reproduce Task 2 decisions from saved validation scores, without model training."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from dccad.alarm_policies import PolicyEvaluator,apply_rule,notifications
from dccad.protocol import constrained_policy_key
from dccad.evaluation import consecutive_alarm


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=ROOT/'results'/'task2_thresholds';plan=json.loads((out/'experiment_plan.json').read_text())
    manifest_path=ROOT/'results'/'protocol_v1'/'manifest.json'
    assert sha(manifest_path)==plan['protocol_sha256']
    manifest=json.loads(manifest_path.read_text());path=ROOT/manifest['cohort_files']['validation']['path']
    assert sha(path)==plan['validation_cohort_sha256']
    meta=pd.read_csv(path);episodes=pd.read_csv(ROOT/'results'/'protocol_v1'/'validation_incident_coverage.csv')[['node','start_ns','end_ns','readings']]
    engine=PolicyEvaluator(meta,episodes,15);scores={}
    assert not set(meta.node).intersection(manifest['independent_holdout_nodes'])
    for name in plan['models']:
        for filename,digest in plan['model_checksums'][name].items():
            assert sha(ROOT/'results'/name/filename)==digest
        z=np.load(out/(name+'_validation_scores.npz'))
        for field in ['node','time_ns','row_index','y_end']:np.testing.assert_array_equal(z[field],meta[field])
        modes=['anomaly_score'] if name=='isolation_forest' else ['whole_window','latest_reading']
        for mode in modes:
            key=name+'__'+mode;scores[key]=z[mode].astype(np.float64);engine.register_scores(key,scores[key])
    policies=pd.read_csv(out/'policy_comparison.csv');specs=json.loads((out/'policy_specs.json').read_text())
    assert len(policies)==298 and policies.policy_id.is_unique
    for row in policies.to_dict('records'):
        spec=specs[row['policy_id']];threshold=spec['threshold']
        if isinstance(threshold,dict):threshold={int(k):v for k,v in threshold.items()}
        score=scores[spec['score_key']]
        if spec['rule']=='consecutive_3':pred=consecutive_alarm(meta,score,threshold,3,cadence_minutes=15)
        else:pred=apply_rule(meta,score,threshold,spec['rule'],15)
        actual=engine.evaluate(pred,spec['score_key'])
        for key,value in actual.items():
            if value is None:assert pd.isna(row[key])
            else:np.testing.assert_allclose(value,row[key],rtol=1e-10,atol=1e-12,err_msg=row['policy_id']+' '+key)
        assert (constrained_policy_key(actual) is not None)==row['feasible']
        if row['scope']=='per_node' and row['threshold_method']=='normal_percentile':
            normal=(meta.history_critical_count==0)&(meta.history_unknown_count==0)
            np.testing.assert_allclose(spec['unseen_node_global_threshold'],np.percentile(score[normal],row['quantile']))
    summary=json.loads((out/'summary.json').read_text());feasible=int(policies.feasible.sum())
    assert summary['feasible_policies']==feasible
    if feasible==0:
        assert summary['status']=='no_feasible_policy' and summary['selected_policy'] is None
    assert summary['holdout_accessed'] is False and summary['models_retrained'] is False
    assert summary['dashboard_default_changed'] is False
    illustration=pd.read_csv(out/'notification_demonstration.csv')
    for row in illustration.to_dict('records'):
        spec=specs[row['policy_id']];t=spec['threshold']
        if isinstance(t,dict):t={int(k):v for k,v in t.items()}
        raw=apply_rule(meta,scores[spec['score_key']],t,spec['rule'])
        before=raw.copy();notice=notifications(meta,raw,60);np.testing.assert_array_equal(raw,before)
        assert notice.sum()==row['displayed_notifications']
        assert row['raw_decisions_unchanged']
    checks={'passed':True,'policies_reproduced':len(policies),'validation_rows':len(meta),
        'all_model_score_arrays_aligned':True,'model_and_cohort_hashes_verified':True,
        'feasibility_flags_reproduced':True,'unseen_node_fallbacks_validation_only':True,
        'cooldown_raw_predictions_unchanged':True,'reserved_nodes_scored':False}
    (out/'reproduction_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
