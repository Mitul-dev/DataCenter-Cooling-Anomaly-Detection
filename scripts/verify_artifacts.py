"""Integration checks for the actual trained artifacts."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch,joblib
from predict import predict
torch.set_num_threads(4)


def main():
    checks=[]
    for path in sorted((ROOT/'results').glob('*/summary.json')):
        r=json.loads(path.read_text())
        if 'test' not in r:continue
        test=r['test']
        assert test['observable_episodes']<=test['episodes']
        assert test['detected']<=test['observable_episodes']
        assert all(np.isfinite(test[k]) for k in ['precision','recall','f1','average_precision','false_bursts_per_node_day'])
        if not (path.parent/'model.pt').exists():continue
        example=json.loads((path.parent/'example_metadata.json').read_text())
        actual=predict(path.parent,path.parent/'example_window.csv',example['node'])
        scores=np.load(path.parent/'scores.npz')
        key='te_mean' if r['selected_policy']['score_mode']=='whole_window' else 'te_end'
        expected=float(scores[key][example['test_row']])
        np.testing.assert_allclose(actual['score'],expected,rtol=5e-4,atol=2e-6)
        checks.append({'experiment':r['name'],'saved_score':expected,'fresh_score':actual['score'],'passed':True})
    required={'base24','base8','base48','enhanced24','robust24'}
    assert required.issubset({c['experiment'] for c in checks}),'The original five neural artifacts are required'
    base=ROOT/'results'/'base24';base_meta=json.loads((base/'example_metadata.json').read_text())
    cp=torch.load(base/'model.pt',map_location='cpu',weights_only=True)
    values=pd.read_csv(base/'example_window.csv')[cp['features']].to_numpy(dtype=np.float32)
    scaler=joblib.load(ROOT/'results'/'isolation_forest'/'scaler.joblib')
    flat=np.clip(scaler.transform(values),-5,5).reshape(1,-1)
    forest=joblib.load(ROOT/'results'/'isolation_forest'/'model.joblib')
    baseline_fresh=float(-forest.score_samples(flat)[0])
    baseline_expected=float(np.load(ROOT/'results'/'isolation_forest'/'scores.npz')['te'][base_meta['test_row']])
    np.testing.assert_allclose(baseline_fresh,baseline_expected,rtol=1e-5,atol=1e-6)
    selected=json.loads((ROOT/'results'/'selected_model.json').read_text())['name']
    folder=ROOT/'results'/selected
    frame=pd.read_csv(folder/'example_window.csv')
    frame.loc[1,'timestamp']=str(pd.to_datetime(frame.loc[0,'timestamp'],utc=True)+pd.Timedelta(minutes=31))
    invalid=ROOT/'.build'/'invalid_cadence.csv';frame.to_csv(invalid,index=False)
    try:
        predict(folder,invalid,7)
    except ValueError as error:
        assert 'cadence' in str(error).lower()
    else:
        raise AssertionError('Irregular timestamps must be rejected')
    feature_plan=ROOT/'results'/'feature_selection'/'plan.json'
    if (ROOT/'results'/'feature_selection'/'results.json').exists():
        import polars as pl
        plan=json.loads(feature_plan.read_text())
        reference=pd.read_csv(ROOT/'results'/'feature_reference8'/'window_metadata.csv')
        for config in plan['configs']:
            path=ROOT/'results'/config['name']
            meta=pd.read_csv(path/'window_metadata.csv')
            pd.testing.assert_frame_equal(reference,meta)
            summary=json.loads((path/'summary.json').read_text())
            for key,field in [('train','train_windows'),('validation','val_windows'),('test','test_windows')]:
                assert summary[field]==plan['cohort'][key]['common']
            assert len(np.load(path/'scores.npz')['va_mean'])==plan['cohort']['validation']['common']
        clean=pl.read_parquet(ROOT/'artifacts'/'cleaned_telemetry.parquet')
        reduced=pl.read_parquet(ROOT/'artifacts'/'selected_telemetry.parquet')
        assert clean.height==reduced.height==plan['data_quality']['rows_preserved']
        for column in ['node','time_ns','label','split']:
            assert clean[column].equals(reduced[column])
        assert not clean.select(pl.struct(['node','time_ns']).is_duplicated().any()).item()
        for column in plan['selected_features']:
            np.testing.assert_array_equal(clean[column].to_numpy(),reduced[column].to_numpy())
            values=reduced[column].to_numpy()
            assert np.all(values[np.isfinite(values)]>=0)
        assert int(clean['imputed_sensor_count'].sum())==plan['data_quality']['causally_filled_cells']
        (feature_plan.parent/'verification.json').write_text(json.dumps({'passed':True,
            'matched_test_endpoints':len(reference),'matched_training_and_validation_counts':True,
            'rows_preserved':clean.height,'selected_features':plan['feature_count'],
            'original_labels_splits_and_valid_selected_values_preserved':True},indent=2))
    (ROOT/'results'/'artifact_verification.json').write_text(json.dumps(checks,indent=2))
    print(f'Verified {len(checks)} saved LSTMs and Isolation Forest, finite metrics, incident counts, fresh inference against saved scores and invalid cadence rejection')


if __name__=='__main__':main()
