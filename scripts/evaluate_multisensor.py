"""Evaluate frozen families once, then export honest multi-sensor evidence."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from dccad.multisensor import decisions,classification_metrics,score_window
from prepare_multisensor import OUT,DATA,CLASSES,sha,write
from train_multisensor import full_scores


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',choices=['multisensor_v1','multisensor_v2'],default='multisensor_v2')
    args=parser.parse_args()
    OUT=ROOT/'results'/args.study
    torch.set_num_threads(4)
    frozen=json.loads((OUT/'frozen_model.json').read_text())
    for family,files in frozen['families'].items():
        for name,digest in files.items():assert sha(OUT/family/name)==digest
    if (OUT/'test_summary.json').exists():
        saved=json.loads((OUT/'test_summary.json').read_text())
        assert saved['frozen_model_sha256']==sha(OUT/'frozen_model.json')
        print('Reusing already-inspected test results; no retuning or new predictions.',flush=True)
        return
    scenarios=json.loads((OUT/'scenarios.json').read_text());results={}
    for family in frozen['families']:
        folder=OUT/family;meta,p=full_scores(folder,scenarios,'test')
        policy=json.loads((folder/'validation_policy.json').read_text())
        pred=decisions(meta,p,policy['thresholds'],policy['rule'])
        metrics=classification_metrics(meta.y,pred,p,CLASSES,meta)
        details=meta[['scenario','date','y']].copy()
        details['correct_sensor_alarm']=(pred==meta.y.to_numpy())&(pred!=0)
        details['any_fault_alarm']=pred!=0
        days=details.groupby(['scenario','date','y']).agg(readings=('y','size'),correct_sensor_alarm=('correct_sensor_alarm','sum'),any_fault_alarm=('any_fault_alarm','sum')).reset_index()
        days.to_csv(folder/'test_per_day.csv',index=False)
        np.savez_compressed(folder/'test_scores.npz',probability=p,y=meta.y,node=meta.node,time_ns=meta.time_ns,pred=pred)
        write(folder/'test_metrics.json',metrics);results[family]=metrics
        # Save a fresh, raw window for every modeled class and check standalone inference.
        checks=[]
        for k in range(len(CLASSES)):
            position=int(np.flatnonzero(meta.y.to_numpy()==k)[0]);row=meta.iloc[position]
            frame=pd.read_parquet(DATA/f'{row.scenario}.parquet')
            pre=json.loads((folder/'preprocessing.json').read_text())
            example=frame.iloc[int(row.row_index)-29:int(row.row_index)+1][['timestamp','operating']+pre['features']].copy()
            example['operating']=example.operating.astype(int)
            example.to_csv(folder/f'example_class_{k}.csv',index=False)
            fresh=score_window(folder,pd.read_csv(folder/f'example_class_{k}.csv'))
            np.testing.assert_allclose(list(fresh.values()),p[position],rtol=2e-5,atol=2e-6)
            checks.append(dict(class_id=k,position=position,scores=fresh,passed=True))
        write(folder/'fresh_window_checks.json',checks)
        if family==frozen['family']:
            meta['prediction']=pred
            for k in range(len(CLASSES)):meta[f'score_{k}']=p[:,k]
            chunks=[]
            for record in scenarios:
                g=meta[meta.scenario==record['scenario']].copy()
                raw=pd.read_parquet(DATA/f"{record['scenario']}.parquet")
                for col in ['CSA_TEMP','HSA_TEMP','CSA_SP','HSA_SP']:
                    g[col]=raw[col].iloc[g.row_index].to_numpy()
                chunks.append(g)
            pd.concat(chunks,ignore_index=True).to_parquet(OUT/'test_replay.parquet',index=False)
        print('Test',family,'accuracy',round(metrics['accuracy'],4),'healthy FPR',round(metrics['healthy_fpr'],4),flush=True)
    primary=results[frozen['family']]
    write(OUT/'test_summary.json',dict(study=args.study,primary_family=frozen['family'],seed=42,
          classes=CLASSES,sensor_fault_types=4,source_scenarios=17,
          primary_test=primary,all_families=results,
          test_windows=sum(sum(row) for row in primary['confusion_matrix']),
          frozen_model_sha256=sha(OUT/'frozen_model.json'),
          test_targets_met=primary['healthy_fpr']<=.01 and all(r['recall']>=.4 for r in primary['per_class'][1:]),
          validation_qualified=frozen['qualified'],test_used_for_selection=False,
          limitation='One simulated dual-duct AHU; four single-sensor bias families; one fixed seed; matching calendar weeks held together; not simultaneous or field-validated faults'))


if __name__=='__main__':main()
