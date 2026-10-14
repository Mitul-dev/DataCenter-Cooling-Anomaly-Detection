import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from dccad.data import load_reduced,prepare,windows,raw_episodes
from dccad.evaluation import calibrate,consecutive_alarm,metrics,incident_metrics,selection_key


def main():
    out=ROOT/'results'/'isolation_forest';out.mkdir(parents=True,exist_ok=True)
    data=load_reduced(ROOT/'0.tar',ROOT/'artifacts'/'reduced_base.npz')
    scaled,split,scaler=prepare(data)
    tr,_=windows(data,scaled,split,0,24,4,True)
    va,mva=windows(data,scaled,split,1,24,2)
    te,mte=windows(data,scaled,split,2,24,2)
    norm=~mva.y_window.to_numpy()
    model=IsolationForest(n_estimators=200,random_state=42,n_jobs=2).fit(tr.reshape(len(tr),-1))
    sv=-model.score_samples(va.reshape(len(va),-1));st=-model.score_samples(te.reshape(len(te),-1))
    policies=[]
    for q in [95.,99.,99.5]:
        for per_node in [False,True]:
            threshold=calibrate(mva[norm],sv[norm],q,per_node)
            for persistence in [1,2]:
                pred=consecutive_alarm(mva,sv,threshold,persistence)
                ev,_=incident_metrics(mva,pred,raw_episodes(data,split,1))
                policies.append({'quantile':q,'per_node':per_node,'persistence':persistence,'threshold':threshold,
                                 **metrics(mva.y_end,sv,pred),**ev})
    selected=max(policies,key=selection_key)
    pred=consecutive_alarm(mte,st,selected['threshold'],selected['persistence'])
    ev,detail=incident_metrics(mte,pred,raw_episodes(data,split,2))
    summary={'name':'isolation_forest','width':24,'enhanced':False,'selected_policy':selected,
             'train_windows':len(tr),'val_windows':len(va),'test_windows':len(te),
             'test':{**metrics(mte.y_end,st,pred),**ev}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    detail.to_csv(out/'incident_details.csv',index=False)
    np.savez_compressed(out/'scores.npz',va=sv,te=st)
    joblib.dump(model,out/'model.joblib');joblib.dump(scaler,out/'scaler.joblib')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
