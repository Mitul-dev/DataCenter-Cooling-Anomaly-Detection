"""Reproduce Task 4 policies, seed/checkpoint integrity, bootstrap and release gate."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from dccad.alarm_policies import PolicyEvaluator,apply_rule
from dccad.protocol import constrained_policy_key
from dccad.robustness import choose_seed_policy,release_gate,node_bootstrap
from dccad.task3_inference import score_window


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4);out=ROOT/'results/task4_robustness';source=ROOT/'results/task3_supervised/lstm8_p1'
    plan=json.loads((out/'experiment_plan.json').read_text());manifest_path=ROOT/'results/protocol_v1/manifest.json'
    assert sha(manifest_path)==plan['protocol_sha256'];manifest=json.loads(manifest_path.read_text())
    for name,hash_value in plan['inputs_sha256'].items():assert sha(source/name)==hash_value
    assert sha(ROOT/'results/task3_supervised/summary.json')==plan['task3_summary_sha256']
    for cohort in plan['cohorts'].values():assert sha(ROOT/cohort['path'])==cohort['sha256']
    meta=pd.read_csv(ROOT/plan['cohorts']['validation']['path'])
    assert not set(meta.node).intersection(manifest['independent_holdout_nodes'])
    episodes=pd.read_csv(ROOT/'results/protocol_v1/validation_incident_coverage.csv')[['node','start_ns','end_ns','readings']]
    engine=PolicyEvaluator(meta,episodes,15);scores={}
    snapshot=json.loads((out/'research_snapshot.json').read_text())
    for seed in plan['seeds']:
        folder=out/f'seed_{seed}';summary=json.loads((folder/'summary.json').read_text())
        assert summary['seed']==seed and summary['configuration']==plan['configuration']
        assert summary['training_windows']==plan['cohorts']['train']['windows']
        assert summary['training_positives']==plan['cohorts']['train']['critical_endpoints']
        for name,hash_value in summary['artifact_checksums'].items():assert sha(folder/name)==hash_value
        assert snapshot['snapshots'][str(seed)]['artifacts']==summary['artifact_checksums']
        z=np.load(folder/'validation_scores.npz')
        for field in ['node','row_index','time_ns','y_end']:np.testing.assert_array_equal(meta[field],z[field])
        scores[str(seed)]=z['logit'].astype(np.float64);engine.register_scores(str(seed),scores[str(seed)])
        np.testing.assert_allclose(summary['average_precision'],engine.score_metrics[str(seed)]['average_precision'])
        history=pd.read_csv(folder/'history.csv');assert len(history)==summary['epochs_run']<=25
        assert history.loc[history.validation_ap.idxmax(),'epoch']==summary['best_epoch']
        example=json.loads((folder/'example_metadata.json').read_text())
        actual=score_window(folder,pd.read_csv(folder/'example_window.csv'))['logit']
        np.testing.assert_allclose(actual,z['logit'][example['validation_position']],rtol=2e-5,atol=2e-6)
        if seed==42:
            np.testing.assert_array_equal(z['logit'],np.load(source/'validation_scores.npz')['logit'])
            cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);old=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
            for k,v in cp['state_dict'].items():torch.testing.assert_close(v,old['state_dict'][k],rtol=0,atol=0)
    policies=pd.read_csv(out/'policy_comparison.csv');specs=json.loads((out/'policy_specs.json').read_text())
    assert len(policies)==135 and policies.policy_id.is_unique
    for r in policies.to_dict('records'):
        spec=specs[r['policy_id']];threshold=spec['threshold']
        if isinstance(threshold,dict):
            threshold={int(k):v for k,v in threshold.items()};normal=(meta.history_critical_count==0)&(meta.history_unknown_count==0)
            np.testing.assert_allclose(spec['unseen_node_global_threshold'],np.percentile(scores[str(r['seed'])][normal],r['quantile']))
        pred=apply_rule(meta,scores[str(r['seed'])],threshold,spec['rule'])
        actual=engine.evaluate(pred,str(r['seed']))
        for k,v in actual.items():
            if v is None:assert pd.isna(r[k])
            else:np.testing.assert_allclose(v,r[k],rtol=1e-10,atol=1e-12)
        assert (constrained_policy_key(actual) is not None)==r['feasible']
    chosen=[];node_rows=pd.read_csv(out/'per_node_validation.csv');intervals=json.loads((out/'node_bootstrap_intervals.json').read_text())
    for seed in plan['seeds']:
        folder=out/f'seed_{seed}';policy=json.loads((folder/'validation_policy.json').read_text())
        expected,status=choose_seed_policy(policies[policies.seed==seed].to_dict('records'))
        assert expected['policy_id']==policy['metrics']['policy_id'] and status==policy['status']
        t=policy['threshold'];t={int(k):v for k,v in t.items()} if isinstance(t,dict) else t
        pred=apply_rule(meta,scores[str(seed)],t,policy['rule'])
        recorded=pd.read_csv(folder/'validation_decisions.csv');np.testing.assert_array_equal(pred,recorded.pred)
        actual_nodes=[]
        for node in meta.node.unique():
            keep=meta.node==node;m=meta[keep].reset_index(drop=True);e=PolicyEvaluator(m,episodes[episodes.node==node],15)
            e.register_scores('score',scores[str(seed)][keep]);actual=e.evaluate(pred[keep],'score');actual_nodes.append(actual)
            saved=node_rows[(node_rows.seed==seed)&(node_rows.node==node)].iloc[0]
            for k,v in actual.items():
                if v is None:assert pd.isna(saved[k])
                else:np.testing.assert_allclose(v,saved[k],rtol=1e-10,atol=1e-12)
        ci=node_bootstrap(actual_nodes,plan['uncertainty']['replicates'],plan['uncertainty']['seed'])
        assert ci==intervals[str(seed)]
        fb=policy.get('unseen_node_global_threshold',t)
        fm=engine.evaluate(apply_rule(meta,scores[str(seed)],fb,policy['rule']),str(seed))
        assert fm==policy['global_fallback_validation']
        chosen.append(dict(expected,status=status,global_fallback_validation=fm))
    release=release_gate(chosen);decision=json.loads((out/'release_decision.json').read_text())
    assert (release is not None)==(decision['status']=='qualified_validation_candidate')
    if release is None:
        assert decision['selected_seed'] is None and decision['selected_policy'] is None
        assert not (out/'frozen_candidate.json').exists()
    assert not decision['independent_holdout_scored'] and not decision['dashboard_default_changed']
    result=json.loads((out/'summary.json').read_text())
    assert result['feasible_policies']==int(policies.feasible.sum())
    for k,stats in result['variation'].items():
        v=np.array([r[k] for r in chosen]);np.testing.assert_allclose(stats['mean'],v.mean());np.testing.assert_allclose(stats['sample_std'],v.std(ddof=1))
    feature_inventory=json.loads((out/'training_feature_inventory.json').read_text())
    assert not feature_inventory['reserved_node_members_opened'] and not feature_inventory['validation_measurements_used'] and not feature_inventory['labels_used']
    assert not set(feature_inventory['development_nodes']).intersection(manifest['independent_holdout_nodes'])
    checks=dict(passed=True,seeds_verified=plan['seeds'],policies_reproduced=len(policies),
        seed42_original_checkpoint_and_scores_exact=True,standalone_csv_inference_verified=3,
        whole_node_bootstrap_reproduced=True,per_node_counts_reproduced=True,release_gate_reproduced=True,
        input_and_model_hashes_verified=True,training_feature_audit_excludes_holdout_and_labels=True,reserved_nodes_scored=False)
    (out/'reproduction_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
