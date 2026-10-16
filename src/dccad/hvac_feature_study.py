"""Sensor-subset preprocessing and fresh inference for the separate feature study."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .hvac import causal_clean,WIDTH
from .supervised import LSTMClassifier


def fit_subset(clean,filled,fit,sensors):
    median=np.nanmedian(filled[fit],axis=0)
    if not np.isfinite(median).all():raise ValueError('A selected sensor lacks training data')
    complete=np.where(np.isfinite(filled),filled,median).astype(np.float32)
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler().fit(complete[fit])
    return dict(sensors=list(sensors),median=median.tolist(),mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),width=WIDTH,clip=10.,fill_limit=5)


def transform_subset(clean,filled,pre):
    complete=np.where(np.isfinite(filled),filled,np.asarray(pre['median']))
    return np.clip((complete-np.asarray(pre['mean']))/np.asarray(pre['scale']),-pre['clip'],pre['clip']).astype(np.float32)


def window_features(frame,pre):
    if len(frame)!=pre['width']:raise ValueError('Exactly thirty one-minute readings required')
    times=pd.to_datetime(frame.timestamp,errors='raise')
    if not (times.diff().dropna()==pd.Timedelta(minutes=1)).all() or times.dt.normalize().nunique()!=1:
        raise ValueError('One contiguous source day is required')
    for c in ['occupied','supply_fan_on']:
        if c not in frame or not (pd.to_numeric(frame[c],errors='raise')==1).all():raise ValueError('Occupied, fan-on operation required')
    for c in ['unit','node']:
        if c in frame and frame[c].nunique(dropna=False)!=1:raise ValueError('One unit required')
    raw=frame[pre['sensors']].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float32)
    clean,filled=causal_clean(raw,np.repeat(str(times.iloc[0].date()),len(frame)),times.dt.as_unit('ns').astype('int64').to_numpy(),pre['fill_limit'])
    return transform_subset(clean,filled,pre)


def score_window(folder,frame):
    folder=Path(folder);pre=json.loads((folder/'preprocessing.json').read_text())
    x=window_features(frame,pre)
    cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    assert cp['features']==pre['sensors']
    model=LSTMClassifier(len(pre['sensors']),32,.2);model.load_state_dict(cp['state_dict']);model.eval()
    with torch.no_grad():return float(model(torch.from_numpy(x[None])).item())
