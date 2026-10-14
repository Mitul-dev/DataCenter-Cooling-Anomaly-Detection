"""Compact electrical/thermal extension, causal filling and training-only fallback."""
import io,tarfile
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler
from .data import BASE_FEATURES,CADENCE_NS
from .preprocessing import causal_fill,sanitize
from .protocol import VALIDATION_END

DIRECT=['ps0_input_voltag_avg','ps0_output_curre_avg','ps0_output_volta_avg',
        'ps1_input_voltag_avg','ps1_output_curre_avg','ps1_output_volta_avg',
        'p0_io_power_avg','p1_io_power_avg','p0_mem_power_avg','p1_mem_power_avg',
        'pcie_avg','ambient_std','total_power_std','p0_power_std','p1_power_std',
        'ambient_max','total_power_max']
GROUPS={
    'fan_std_max':([f'fan{i}_{j}_std' for i in range(4) for j in range(2)],'max'),
    'gpu_core_max':([f'gpu{i}_core_temp_max' for i in [0,1,3,4]],'max'),
    'gpu_mem_max':([f'gpu{i}_mem_temp_max' for i in [0,1,3,4]],'max'),
    'gpu_core_spread':([f'gpu{i}_core_temp_avg' for i in [0,1,3,4]],'spread'),
    'gpu_mem_spread':([f'gpu{i}_mem_temp_avg' for i in [0,1,3,4]],'spread')}
EXTRA_FEATURES=DIRECT+list(GROUPS)
SOURCE_COLUMNS=list(dict.fromkeys(DIRECT+[c for cols,_ in GROUPS.values() for c in cols]))
FEATURES=BASE_FEATURES+EXTRA_FEATURES+['unavailable_'+c for c in EXTRA_FEATURES]+['median_fallback_fraction']


def clean_extra(raw):
    x=np.asarray(raw,dtype=np.float32).copy()
    x[(~np.isfinite(x))|(x<0)]=np.nan
    return x


def aggregate_extra(columns):
    """Extrema/spreads require every specified component; flag incomplete groups."""
    result=[clean_extra(np.asarray(columns[c])[:,None])[:,0] for c in DIRECT]
    for names,method in GROUPS.values():
        x=clean_extra(np.column_stack([columns[c] for c in names]))
        valid=np.isfinite(x).all(axis=1)
        y=np.full(len(x),np.nan,dtype=np.float32)
        y[valid]=x[valid].max(axis=1)
        if method=='spread':y[valid]-=x[valid].min(axis=1)
        result.append(y)
    return np.column_stack(result).astype(np.float32)


def load_extra(archive,data):
    """Only development-node members, aligned to original rows before validation end."""
    result=np.full((len(data['raw']),len(EXTRA_FEATURES)),np.nan,dtype=np.float32)
    with tarfile.open(archive) as tar:
        for node in np.unique(data['node']):
            member=tar.extractfile(f'{int(node)}.parquet')
            frame=pl.read_parquet(io.BytesIO(member.read()),columns=['timestamp']+SOURCE_COLUMNS).sort('timestamp')
            times=frame['timestamp'].dt.epoch('ns').to_numpy();keep=times<VALIDATION_END
            frame=frame.filter(pl.Series(keep));times=times[keep]
            idx=np.flatnonzero((data['node']==node)&(data['time_ns']<VALIDATION_END))
            np.testing.assert_array_equal(times,data['time_ns'][idx])
            result[idx]=aggregate_extra({c:frame[c].to_numpy() for c in SOURCE_COLUMNS})
    return result


def fit_extension(data,base_filled,extra,split,base_scaler):
    cleaned=clean_extra(extra)
    filled=causal_fill(cleaned,data['node'],data['time_ns'],split,limit=4)
    fit=(split==0)&(data['label']==0)&np.isfinite(base_filled).all(axis=1)
    medians=np.nanmedian(filled[fit],axis=0)
    if not np.isfinite(medians).all():raise ValueError('An added feature has no usable normal training measurements')
    completed=np.where(np.isfinite(filled),filled,medians).astype(np.float32)
    scaler=StandardScaler().fit(completed[fit])
    pre=dict(base_scaler=base_scaler,extra_scaler=scaler,extra_medians=medians,
        base_features=BASE_FEATURES,extra_features=EXTRA_FEATURES,features=FEATURES,
        width=8,eligibility_width=24,clip=10.,cadence_minutes=15)
    values=transform_extension(base_filled,cleaned,filled,pre)
    return values,pre,dict(fit_rows=int(fit.sum()),medians=medians.tolist(),
        training_unresolved_fraction=np.isnan(filled[fit]).mean(axis=0).tolist())


def transform_extension(base_filled,extra_cleaned,extra_filled,pre):
    fallback=~np.isfinite(extra_filled)
    completed=np.where(fallback,pre['extra_medians'],extra_filled)
    base=np.clip(pre['base_scaler'].transform(base_filled),-pre['clip'],pre['clip'])
    extra=np.clip(pre['extra_scaler'].transform(completed),-pre['clip'],pre['clip'])
    return np.column_stack([base,extra,~np.isfinite(extra_cleaned),fallback.mean(axis=1)]).astype(np.float32)


def preprocess_recovery_csv(frame,pre):
    if len(frame)<24:raise ValueError('At least 24 readings are required')
    frame=frame.tail(24)
    if 'timestamp' not in frame:raise ValueError('Timestamps are required')
    ts=pd.to_datetime(frame.timestamp,utc=True,errors='raise')
    if not (ts.diff().dropna()==pd.Timedelta(minutes=15)).all():raise ValueError('Readings must have a 15-minute cadence')
    for name in ['node','node_id']:
        if name in frame and frame[name].nunique(dropna=False)!=1:raise ValueError('Only one server is allowed per window')
    base=sanitize(frame[BASE_FEATURES].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float32),BASE_FEATURES)
    base=pd.DataFrame(base).ffill(limit=4).to_numpy(dtype=np.float32)
    if not np.isfinite(base).all():raise ValueError('Base sensor measurements remain unresolved')
    extra=clean_extra(frame[EXTRA_FEATURES].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float32))
    filled=pd.DataFrame(extra).ffill(limit=4).to_numpy(dtype=np.float32)
    return transform_extension(base,extra,filled,pre)
