"""Fresh-inference, preprocessing and policy verification for the feature ablation."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from dccad.hvac import read_source,sensor_frame,causal_clean,operation_mask,RAW_FEATURES,WIDTH
from dccad.hvac_feature_study import fit_subset,transform_subset,score_window
from dccad.supervised import LSTMClassifier,sequence_inputs,predict_logits
from dccad.alarm_policies import PolicyEvaluator,apply_rule
from dccad.protocol import constrained_policy_key
from dccad.robustness import choose_seed_policy
from analyze_thresholds import pr_candidates
OUT=ROOT/'results/hvac_feature_v2';OLD=ROOT/'results/hvac_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    plan=json.loads((OUT/'experiment_plan.json').read_text())
    assert sha(ROOT/'results/hvac_sensor_review/review.json')==plan['audit_sha256']
    assert sha(ROOT/'data/hvac_benchmark/afdd_source.zip')==plan['source_sha256']
    for file,h in plan['implementation_sha256'].items():assert sha(ROOT/file)==h
    for name,h in plan['cohort_sha256'].items():assert sha(OLD/f'{name}_cohort.csv')==h
    frame=read_source(ROOT/'data/hvac_benchmark/afdd_source.zip')
    frame=frame[frame.season.isin(['winter','spring','summer'])].reset_index(drop=True)
    raw=sensor_frame(frame).to_numpy(dtype=np.float32)
    clean,filled=causal_clean(raw,frame.date.to_numpy(),frame.timestamp.dt.as_unit('ns').astype('int64').to_numpy())
    fit=frame.season.isin(['winter','spring']).to_numpy()&operation_mask(frame)&(~frame['Fault Detection Ground Truth'].to_numpy(bool))
    val=pd.read_csv(OLD/'validation_cohort.csv');yv=val.y_end.to_numpy(bool)
    models=json.loads((OUT/'models.json').read_text());assert len(models)==9
    selected=pd.read_csv(OUT/'validation_model_comparison.csv')
    policies=pd.read_csv(OUT/'validation_policy_comparison.csv')
    episodes=pd.read_csv(OLD/'validation_fault_days.csv')[['node','start_ns','end_ns','readings']]
    engine=PolicyEvaluator(val,episodes,1);new_scores=0;csv_windows=0;policy_count=0
    for r in models:
        folder=ROOT/r['artifact_path'];name=r['name'];saved=np.load(folder/'validation_scores.npz')
        for key in ['node','time_ns','row_index','y_end']:np.testing.assert_array_equal(saved[key],val[key])
        if r['family']!='engineered32':
            assert r['plan_sha256']==sha(OUT/'experiment_plan.json')
            for file,h in r['checksums'].items():assert sha(folder/file)==h
            features=plan['feature_sets'][r['family']];assert features==r['features']
            indices=[RAW_FEATURES.index(c) for c in features];c=clean[:,indices];f=filled[:,indices]
            expected=fit_subset(c,f,fit,features);pre=json.loads((folder/'preprocessing.json').read_text())
            assert expected==pre
            xv=sequence_inputs(transform_subset(c,f,pre),val.row_index,WIDTH)
            cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            model=LSTMClassifier(len(features),32,.2);model.load_state_dict(cp['state_dict'])
            assert sum(p.numel() for p in model.parameters())==r['parameters']
            fresh=predict_logits(model,xv);np.testing.assert_allclose(fresh,saved['score'],rtol=2e-5,atol=2e-5)
            example=pd.read_csv(folder/'example_window.csv');value=score_window(folder,example)
            np.testing.assert_allclose(value,saved['score'][0],rtol=2e-5,atol=2e-5)
            history=pd.read_csv(folder/'history.csv');assert len(history)==r['epochs_run']<=30
            assert int(history.loc[history.validation_ap.idxmax(),'epoch'])==r['best_epoch']
            new_scores+=1;csv_windows+=1
        else:
            original=json.loads((folder/'summary.json').read_text())
            for file,h in original['checksums'].items():assert sha(folder/file)==h
        score=saved['score'].astype(np.float64);engine.register_scores(name,score)
        candidates,_,_=pr_candidates(yv,score)
        candidates += [(f'normal_percentile_{q}',float(np.percentile(score[~yv],q))) for q in [95,99,99.5,99.9,99.95]]
        candidates += [('default',0.)];rows=[];original_rows=policies[policies.model==name].reset_index(drop=True)
        position=0
        for method,t in candidates:
            for rule in ['immediate','consecutive_2','two_of_three']:
                pred=apply_rule(val,score,t,rule,1);m=engine.evaluate(pred,name)
                old=original_rows.iloc[position];assert old.method==method and old.rule==rule
                for k,v in m.items():
                    if v is None:assert pd.isna(old[k])
                    else:np.testing.assert_allclose(v,old[k],rtol=1e-9,atol=1e-10)
                rows.append(dict(m,policy_id=old.policy_id,feasible=constrained_policy_key(m) is not None));position+=1
        chosen,status=choose_seed_policy(rows);actual=selected[selected.model==name].iloc[0]
        assert actual.policy_id==chosen['policy_id'] and actual.status==status
        policy_count+=position
    result=dict(passed=True,new_full_validation_score_arrays_reproduced=new_scores,fresh_subset_csv_windows_reproduced=csv_windows,
        validation_policies_reproduced=policy_count,original_baseline_artifact_hashes_intact=True,training_only_preprocessing_reproduced=True,
        subset_feature_counts_verified=True,no_new_test_predictions=True,existing_submission_replaced=False)
    assert new_scores==6 and policy_count==270
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
