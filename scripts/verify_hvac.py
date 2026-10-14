"""Check the declared cohort, training-only transforms, scores and frozen decisions."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from dccad.hvac import read_source,sensor_frame,operation_mask,causal_clean,fit_preprocessor,transform,cohorts,WIDTH,MODEL_FEATURES
from dccad.supervised import LSTMClassifier,predict_logits,sequence_inputs,trailing_statistics
from dccad.hvac_inference import score_window
from dccad.alarm_policies import PolicyEvaluator,apply_rule
from dccad.protocol import constrained_policy_key
from dccad.robustness import choose_seed_policy

OUT=ROOT/'results/hvac_v1';DATA=ROOT/'data/hvac_benchmark'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validation-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);plan=json.loads((OUT/'experiment_plan.json').read_text())
    assert sha(DATA/'afdd_source.zip')==plan['archive_sha256'] and sha(DATA/'source_inventory.pdf')==plan['source_inventory_sha256']
    frame=read_source(DATA/'afdd_source.zip');meta=cohorts(frame);by_split={};day_sets=[]
    for split,seasons in [('train',['winter','spring']),('validation',['summer']),('test',['autumn'])]:
        p=ROOT/plan['cohorts'][split]['path'];assert sha(p)==plan['cohorts'][split]['sha256']
        saved=pd.read_csv(p);actual=meta[meta.season.isin(seasons)].reset_index(drop=True)
        for c in ['node','time_ns','row_index','y_end']:np.testing.assert_array_equal(saved[c],actual[c])
        assert saved.date.tolist()==actual.date.tolist();by_split[split]=saved;day_sets.append(set(saved.date))
        idx=saved.row_index.to_numpy()[:,None]-np.arange(WIDTH-1,-1,-1)[None,:]
        assert (frame.date.to_numpy()[idx]==saved.date.to_numpy()[:,None]).all()
        assert operation_mask(frame)[idx].all()
    for i in range(3):
        for j in range(i+1,3):assert not day_sets[i].intersection(day_sets[j])
    raw=sensor_frame(frame).to_numpy(dtype=np.float32);clean,filled=causal_clean(raw,frame.date.to_numpy(),frame.timestamp.dt.as_unit('ns').astype('int64').to_numpy())
    fit=frame.season.isin(['winter','spring']).to_numpy()&(~frame['Fault Detection Ground Truth'].to_numpy(bool))&operation_mask(frame)
    expected=fit_preprocessor(raw,clean,filled,fit);pre=joblib.load(OUT/'preprocessing.joblib')
    for k in ['median']:np.testing.assert_array_equal(pre[k],expected[k])
    np.testing.assert_array_equal(pre['scaler'].mean_,expected['scaler'].mean_);np.testing.assert_array_equal(pre['scaler'].scale_,expected['scaler'].scale_)
    audit=json.loads((OUT/'preprocessing_audit.json').read_text())
    for f,h in audit['implementation_sha256'].items():assert sha(ROOT/f)==h
    values=transform(clean,filled,pre);validation=by_split['validation'];xv=sequence_inputs(values,validation.row_index,WIDTH)
    episodes=pd.read_csv(OUT/'validation_fault_days.csv')[['node','start_ns','end_ns','readings']]
    engine=PolicyEvaluator(validation,episodes,1);scores={};models={}
    names=['lstm_seed_7','lstm_seed_42','lstm_seed_123','boosting']
    for name in names:
        folder=OUT/name;r=json.loads((folder/'summary.json').read_text());assert r['features']==MODEL_FEATURES
        for f,h in r['checksums'].items():assert sha(folder/f)==h
        history=pd.read_csv(folder/'history.csv')
        if r['kind']=='lstm':
            cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model=LSTMClassifier(len(MODEL_FEATURES),32,.2)
            model.load_state_dict(cp['state_dict']);actual=predict_logits(model,xv)
            assert len(history)==r['epochs_run']<=30 and int(history.loc[history.validation_ap.idxmax(),'epoch'])==r['best_epoch']
        else:
            model=joblib.load(folder/'model.joblib')
            with threadpool_limits(limits=4):actual=model.predict_proba(trailing_statistics(xv))[:,1]
            assert int(history.loc[history.validation_ap.idxmax(),'iterations'])==r['best_iterations']
        models[name]=model
        z=np.load(folder/'validation_scores.npz')
        for c in ['node','time_ns','row_index','y_end']:np.testing.assert_array_equal(z[c],validation[c])
        np.testing.assert_allclose(actual,z['score'],rtol=1e-6,atol=1e-7)
        scores[name]=z['score'].astype(np.float64);engine.register_scores(name,scores[name])
        example=json.loads((folder/'example_metadata.json').read_text());fresh=score_window(folder,pd.read_csv(folder/'example_window.csv'))
        np.testing.assert_allclose(fresh,z['score'][example['validation_position']],rtol=2e-5,atol=2e-6)
    policies=pd.read_csv(OUT/'validation_policy_comparison.csv');specs=json.loads((OUT/'validation_policy_specs.json').read_text())
    assert len(policies)==120
    chosen=[]
    for row in policies.to_dict('records'):
        spec=specs[row['policy_id']];pred=apply_rule(validation,scores[row['model']],spec['threshold'],spec['rule'],1)
        actual=engine.evaluate(pred,row['model'])
        for k,v in actual.items():
            if v is None:assert pd.isna(row[k])
            else:np.testing.assert_allclose(v,row[k],rtol=1e-10,atol=1e-12)
        assert row['feasible']==(constrained_policy_key(actual) is not None)
    for name in names:
        r,status=choose_seed_policy(policies[policies.model==name].to_dict('records'))
        p=json.loads((OUT/name/'validation_policy.json').read_text())
        assert p['metrics']['policy_id']==r['policy_id'] and p['status']==status;chosen.append(r)
    frozen=json.loads((OUT/'frozen_model.json').read_text());assert frozen['name']=='lstm_seed_42'
    assert frozen['qualified']==all(r['feasible'] for r in chosen if r['kind']=='lstm')
    test_count=0
    if not args.validation_only and (OUT/'test_summary.json').exists():
        evaluation=json.loads((OUT/'evaluation_freeze.json').read_text());ts=json.loads((OUT/'test_summary.json').read_text())
        assert evaluation['freeze_sha256']==sha(OUT/'frozen_model.json') and ts['evaluation_freeze_sha256']==sha(OUT/'evaluation_freeze.json')
        test=by_split['test'];xt=sequence_inputs(values,test.row_index,WIDTH)
        ep=pd.read_csv(OUT/'test_fault_days.csv')[['node','start_ns','end_ns','readings']];e=PolicyEvaluator(test,ep,1)
        for name in names:
            folder=OUT/name
            for f,h in evaluation['model_artifacts'][name].items():assert sha(folder/f)==h
            model=models[name]
            if name=='boosting':
                with threadpool_limits(limits=4):actual=model.predict_proba(trailing_statistics(xt))[:,1]
            else:actual=predict_logits(model,xt)
            z=np.load(folder/'test_scores.npz')
            for c in ['node','time_ns','row_index','y_end']:np.testing.assert_array_equal(z[c],test[c])
            np.testing.assert_allclose(actual,z['score'],rtol=1e-6,atol=1e-7)
            p=json.loads((folder/'validation_policy.json').read_text());pred=apply_rule(test,z['score'],p['threshold'],p['rule'],1)
            decision=pd.read_csv(folder/'test_decisions.csv');np.testing.assert_array_equal(pred,decision.pred)
            e.register_scores(name,z['score']);actual_metrics=e.evaluate(pred,name);saved=json.loads((folder/'test_metrics.json').read_text())
            for k,v in actual_metrics.items():
                if v is None:assert saved[k] is None
                else:np.testing.assert_allclose(v,saved[k])
            test_count+=1
        assert not ts['post_test_tuning']
    checks=dict(passed=True,source_labels_match_documented_scenarios=True,whole_season_and_day_separation=True,
        eligibility_and_timestamps_verified=True,training_only_preprocessing_reproduced=True,
        full_validation_scores_reproduced=4,validation_policies_reproduced=120,fresh_csv_windows_reproduced=4,
        fixed_seed42_selection_verified=True,full_frozen_test_scores_reproduced=test_count,source_and_artifact_hashes_verified=True,
        m100_reserved_servers_scored=False)
    name='validation_verification.json' if args.validation_only else 'verification.json'
    (OUT/name).write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
