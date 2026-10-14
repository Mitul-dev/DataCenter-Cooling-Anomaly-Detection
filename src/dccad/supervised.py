"""Training-only sensor preprocessing and current-state LSTM classification."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import BASE_FEATURES
from .preprocessing import sanitize


QUALITY_NAMES=['missing_sensor_fraction','invalid_sensor_fraction','imputed_sensor_fraction']


class LSTMClassifier(nn.Module):
    def __init__(self,features,hidden=32,dropout=.2):
        super().__init__()
        self.encoder=nn.LSTM(features,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden,16),nn.ReLU(),nn.Dropout(dropout),nn.Linear(16,1))

    def forward(self,x):
        _,(hidden,_)=self.encoder(x)
        return self.head(hidden[-1]).squeeze(-1)


def features_from_clean(raw,filled,scaler,names,clip=10.):
    """No labels, server IDs or future values are used as feature inputs."""
    scaled=scaler.transform(filled)
    if clip is not None:scaled=np.clip(scaled,-clip,clip)
    numeric=[BASE_FEATURES.index(c) for c in names if c in BASE_FEATURES]
    result=scaled[:,numeric]
    if any(c in QUALITY_NAMES for c in names):
        cleaned=sanitize(raw,BASE_FEATURES)
        quality=np.column_stack([(~np.isfinite(raw)).mean(axis=1),
            (np.isfinite(raw)&np.isnan(cleaned)).mean(axis=1),
            (np.isnan(cleaned)&np.isfinite(filled)).mean(axis=1)])
        result=np.column_stack([result,quality])
    expected=[BASE_FEATURES[j] for j in numeric]+(QUALITY_NAMES if any(c in QUALITY_NAMES for c in names) else [])
    if list(names)!=expected:raise ValueError('Feature ordering must match the saved preprocessing definition')
    return np.asarray(result,dtype=np.float32)


def sequence_inputs(values,endpoints,width):
    rows=np.asarray(endpoints,dtype=np.int64)[:,None]-np.arange(width-1,-1,-1)[None,:]
    if (rows<0).any():raise ValueError('An endpoint has too little preceding history')
    x=np.ascontiguousarray(values[rows],dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Unresolved missing inputs in a model window')
    return x


def predict_logits(model,x,batch_size=2048):
    model.eval();scores=[]
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            scores.append(model(torch.from_numpy(x[start:start+batch_size])).cpu().numpy())
    return np.concatenate(scores)


def trailing_statistics(x):
    """Same sensor history for the tree baseline: latest, mean, std and slope."""
    t=np.arange(x.shape[1],dtype=np.float32);t-=t.mean()
    slope=np.einsum('ntf,t->nf',x,t)/(t*t).sum()
    return np.ascontiguousarray(np.concatenate([x[:,-1],x.mean(axis=1),x.std(axis=1),slope],axis=1),dtype=np.float32)


def preprocess_csv(frame,preprocessor):
    width=preprocessor['eligibility_width']
    if len(frame)<width:raise ValueError(f'At least {width} readings are required by the frozen eligibility protocol')
    frame=frame.tail(width)
    if 'timestamp' in frame:
        ts=pd.to_datetime(frame.timestamp,utc=True,errors='raise')
        if not (ts.diff().dropna()==pd.Timedelta(minutes=15)).all():raise ValueError('Invalid cadence: readings must be 15 minutes apart')
    raw=frame[BASE_FEATURES].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float32)
    cleaned=sanitize(raw,BASE_FEATURES)
    filled=pd.DataFrame(cleaned).ffill(limit=4).to_numpy(dtype=np.float32)
    if not np.isfinite(filled).all():raise ValueError('Missing or invalid measurements remain after four-reading causal filling')
    return features_from_clean(raw,filled,preprocessor['scaler'],preprocessor['features'],preprocessor['clip'])
