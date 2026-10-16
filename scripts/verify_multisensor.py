"""Reproduce saved class scores, alarm policies, splits and fresh CSV inference."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from dccad.multisensor import SensorClassifier,predict,decisions,classification_metrics,fit_preprocessing,score_window
from dccad.supervised import sequence_inputs
from train_multisensor import processed
from prepare_multisensor import DATA,CLASSES,FEATURES,sha,write,partition
OUT=ROOT/'results/multisensor_v2'
OLD=ROOT/'results/multisensor_v1'


def main():
    start=time.perf_counter();torch.set_num_threads(4)
    frozen=json.loads((OUT/'frozen_model.json').read_text())
    assert sha(OUT/'experiment_plan.json')==frozen['plan_sha256']
    plan=json.loads((OUT/'experiment_plan.json').read_text())
    for file,digest in plan['implementation'].items():assert sha(ROOT/file)==digest
    scenarios=json.loads((OUT/'scenarios.json').read_text());models={};saved={};offsets={}
    for family,files in frozen['families'].items():
        folder=OUT/family
        for name,digest in files.items():assert sha(folder/name)==digest
        pre=json.loads((folder/'preprocessing.json').read_text())
        cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        assert cp['classes']==CLASSES and cp['features']==pre['features']
        assert not any(c in pre['features'] for c in ['y','Datetime','timestamp','scenario','SYS_CTL'])
        model=SensorClassifier(len(pre['features']),5);model.load_state_dict(cp['state_dict'])
        models[family]=(model,pre)
        for split in ['validation','test']:
            saved[family,split]=np.load(folder/f'{split}_scores.npz');offsets[family,split]=0
    labels=[];times=[];nodes=[];parts=[]
    for record in scenarios:
        name=record['scenario'];cache=DATA/f'{name}.parquet';cohort=DATA/f'{name}_cohort.parquet'
        assert sha(cache)==record['cache_sha256'] and sha(cohort)==record['cohort_sha256']
        frame=pd.read_parquet(cache);meta=pd.read_parquet(cohort)
        np.testing.assert_array_equal(frame.partition,partition(frame.timestamp))
        assert not frame.groupby(frame.timestamp.dt.normalize()).partition.nunique().gt(1).any()
        for split in ['validation','test']:
            g=meta[meta.partition==split].reset_index(drop=True)
            for family,(model,pre) in models.items():
                values=processed(frame,pre);p=predict(model,sequence_inputs(values,g.row_index,30))
                a=offsets[family,split];b=a+len(g);stored=saved[family,split]
                np.testing.assert_allclose(p,stored['probability'][a:b],rtol=2e-5,atol=2e-6)
                np.testing.assert_array_equal(stored['y'][a:b],g.y)
                np.testing.assert_array_equal(stored['node'][a:b],g.node)
                np.testing.assert_array_equal(stored['time_ns'][a:b],g.time_ns)
                offsets[family,split]=b
        if record['class_id']==0:
            fit=(frame.partition=='train')&frame.operating
            recomputed=fit_preprocessing(frame[FEATURES].to_numpy(np.float32),fit.to_numpy(),FEATURES)
            for _,pre in models.values():
                indices=[FEATURES.index(c) for c in pre['features']]
                for key in ['median','mean','scale']:
                    np.testing.assert_allclose(pre[key],np.asarray(recomputed[key])[indices],rtol=1e-10,atol=1e-10)
        parts.append(meta)
        print('Verified all models and partitions:',name,flush=True)
    meta=pd.concat(parts,ignore_index=True);policies=0;csv_checks=0
    from refine_multisensor import fast_metrics
    for family,(_,pre) in models.items():
        folder=OUT/family
        val=meta[meta.partition=='validation'].reset_index(drop=True);test=meta[meta.partition=='test'].reset_index(drop=True)
        for split,g in [('validation',val),('test',test)]:
            assert offsets[family,split]==len(saved[family,split]['y'])==len(g)
        search=json.loads((folder/'validation_policy_search.json').read_text())
        p=saved[family,'validation']['probability'];y=val.y.to_numpy(int)
        for row,spec in zip(search['policies'],search['specifications']):
            pred=decisions(val,p,spec['thresholds'],spec['rule']);f1,fpr,recall=fast_metrics(y,pred)
            np.testing.assert_allclose([f1,fpr],[row['macro_f1'],row['healthy_fpr']],atol=1e-12)
            np.testing.assert_allclose(recall,row['recall'],atol=1e-12);policies+=1
        policy=json.loads((folder/'validation_policy.json').read_text())
        p=saved[family,'test']['probability'];pred=decisions(test,p,policy['thresholds'],policy['rule'])
        np.testing.assert_array_equal(pred,saved[family,'test']['pred'])
        metrics=classification_metrics(test.y,pred,p,CLASSES,test)
        assert metrics==json.loads((folder/'test_metrics.json').read_text())
        for k in range(5):
            frame=pd.read_csv(folder/f'example_class_{k}.csv');before=score_window(folder,frame)
            frame['y']=99;frame['fault_label']='pressure';frame['severity']=999
            assert score_window(folder,frame)==before;csv_checks+=1
    summary=json.loads((OUT/'test_summary.json').read_text())
    assert summary['frozen_model_sha256']==sha(OUT/'frozen_model.json')
    assert summary['primary_family']==frozen['family']
    report=dict(status='passed',complete_score_arrays=8,validation_policies_reproduced=policies,
                fresh_csv_and_label_exclusion_checks=csv_checks,training_only_preprocessing=True,
                calendar_week_partition_checks=True,classes=CLASSES,test_reselection=False,
                reserved_test_already_inspected=True,seconds=time.perf_counter()-start,
                verified_files={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'test_summary.json',OUT/'frozen_model.json']})
    write(OUT/'verification.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
