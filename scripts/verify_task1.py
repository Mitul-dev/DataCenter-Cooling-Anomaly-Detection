"""Verify frozen cohorts and audit outputs without scoring reserved servers."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import polars as pl
from dccad.protocol import protocol_clean
from dccad.data import CADENCE_NS


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'results'/'protocol_v1'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert sha(folder/'manifest.json')==(folder/'manifest.sha256').read_text().strip()
    z=np.load(ROOT/'artifacts'/'reduced_base.npz');data={k:z[k] for k in z.files}
    _,filled,split=protocol_clean(data)
    badstep=np.r_[False,(np.diff(data['time_ns'])!=CADENCE_NS)|(np.diff(data['node'])!=0)|(np.diff(split)!=0)]
    cs_step=np.r_[0,np.cumsum(badstep)]
    cs_bad=np.r_[0,np.cumsum(~np.isfinite(filled).all(axis=1))]
    counts={}
    for which,name in enumerate(['train','validation','exploratory']):
        record=manifest['cohort_files'][name];path=ROOT/record['path']
        assert sha(path)==record['sha256']
        meta=pd.read_csv(path);end=meta.row_index.to_numpy();start=end-23
        assert len(meta)==record['windows'] and meta.row_index.is_unique
        assert (start>=0).all()
        assert np.all(cs_step[end+1]-cs_step[start+1]==0)
        assert np.all(cs_bad[end+1]-cs_bad[start]==0)
        np.testing.assert_array_equal(split[end],np.full(len(end),which))
        np.testing.assert_array_equal(data['node'][end],meta.node)
        np.testing.assert_array_equal(data['time_ns'][end],meta.time_ns)
        assert np.isin(data['label'][end],[0,2]).all()
        np.testing.assert_array_equal(data['label'][end]==2,meta.y_end)
        assert not set(meta.node).intersection(manifest['independent_holdout_nodes'])
        counts[name]=len(meta)
    inventory=pd.read_csv(folder/'holdout_inventory.csv')
    assert set(inventory.node)==set(manifest['independent_holdout_nodes'])
    for flag in ['label_values_read','numeric_sensor_values_read','predictions_made']:
        assert not inventory[flag].any()
    assert inventory.duplicate_timestamps.sum()==0
    assert inventory.missing_source_columns.fillna('').eq('').all()
    indicators=pl.read_parquet(ROOT/'artifacts'/'quality_indicators.parquet')
    assert indicators.height==len(data['label'])
    np.testing.assert_array_equal(indicators['node'].to_numpy(),data['node'])
    np.testing.assert_array_equal(indicators['time_ns'].to_numpy(),data['time_ns'])
    summary=json.loads((ROOT/'results'/'task1_audit'/'summary.json').read_text())
    assert summary['policy_unchanged'] and not summary['thresholds_recalibrated']
    selected=pd.read_csv(ROOT/'results'/'task1_audit'/'validation_scored_15min.csv')
    assert not set(selected.node).intersection(manifest['independent_holdout_nodes'])
    models=pd.read_csv(ROOT/'results'/'task1_audit'/'historical_reference_15min.csv')
    assert len(models)==4 and models.threshold_unchanged.all()
    report={'passed':True,'cohort_counts':counts,'calendar_and_gap_boundaries_verified':True,
        'target_alignment_verified':True,'cohort_and_manifest_hashes_verified':True,
        'reserved_nodes_not_scored':True,'holdout_schema_available':True,
        'quality_rows_preserved':len(data['label']),'historical_fixed_policy_diagnostics':models.experiment.tolist()}
    (ROOT/'results'/'task1_audit'/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
