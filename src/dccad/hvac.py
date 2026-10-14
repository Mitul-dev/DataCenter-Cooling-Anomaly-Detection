"""Author-labelled AHU sensor-fault benchmark with whole-season separation."""
import io,zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_MAPPING={
    'supply_temp':'AHU: Supply Air Temperature',
    'supply_setpoint':'AHU: Supply Air Temperature Set Point',
    'outdoor_temp':'AHU: Outdoor Air Temperature',
    'mixed_temp':'AHU: Mixed Air Temperature',
    'return_temp':'AHU: Return Air Temperature',
    'supply_fan_speed':'AHU: Supply Air Fan Speed Control Signal',
    'return_fan_speed':'AHU: Return Air Fan Speed Control Signal',
    'outdoor_damper':'AHU: Outdoor Air Damper Control Signal',
    'return_damper':'AHU: Return Air Damper Control Signal',
    'cooling_valve':'AHU: Cooling Coil Valve Control Signal',
    'heating_valve':'AHU: Heating Coil Valve Control Signal',
    'duct_pressure_setpoint':'AHU: Supply Air Duct Static Pressure Set Point',
    'duct_pressure':'AHU: Supply Air Duct Static Pressure'}
RAW_FEATURES=list(RAW_MAPPING)
DERIVED=['supply_error','mix_return_delta','mix_outdoor_delta','outdoor_return_delta','pressure_error','command_mix_residual']
MEASURED_FEATURES=RAW_FEATURES+DERIVED
MODEL_FEATURES=MEASURED_FEATURES+['missing_'+c for c in RAW_FEATURES]
WIDTH=30
MINUTE_NS=60*10**9
# Dates transcribed from the source inventory, Table 1-2, pages 6-7.
DATES={
    'winter': [('2017-01-30','2017-02-05',0),('2017-02-06','2017-02-12',1),('2017-02-13','2017-02-19',2),('2017-02-20','2017-02-26',4),('2017-02-27','2017-03-05',-1),('2017-03-06','2017-03-12',-2),('2017-03-13','2017-03-19',-4)],
    'spring': [('2017-05-02','2017-05-07',0),('2017-05-09','2017-05-14',1),('2017-05-16','2017-05-21',2),('2017-05-23','2017-05-28',4),('2017-05-30','2017-06-04',-1),('2017-06-06','2017-06-11',-2),('2017-06-13','2017-06-18',-4)],
    'summer': [('2017-07-31','2017-08-06',0),('2017-08-07','2017-08-13',1),('2017-08-14','2017-08-20',2),('2017-08-21','2017-08-27',4),('2017-08-28','2017-09-03',-1),('2017-09-04','2017-09-10',-2),('2017-09-11','2017-09-17',-4)],
    'autumn': [('2017-10-30','2017-11-05',0),('2017-11-06','2017-11-12',1),('2017-11-13','2017-11-19',2),('2017-11-20','2017-11-26',4),('2017-11-27','2017-12-03',-1),('2017-12-04','2017-12-10',-2),('2017-12-11','2017-12-17',-4)]}


def read_source(archive):
    with zipfile.ZipFile(archive) as z:frame=pd.read_csv(io.BytesIO(z.read('MZVAV-1.csv')))
    frame.columns=frame.columns.str.strip()
    frame['timestamp']=pd.to_datetime(frame.Datetime,format='%m/%d/%Y %H:%M')
    if frame.timestamp.duplicated().any():raise ValueError('Duplicate source timestamps')
    frame=frame.sort_values('timestamp').reset_index(drop=True)
    day=frame.timestamp.dt.normalize();frame['date']=day.dt.strftime('%Y-%m-%d')
    frame['season']='';frame['bias_c']=np.nan;frame['scenario']=''
    for season,periods in DATES.items():
        for start,end,bias in periods:
            keep=(day>=pd.Timestamp(start))&(day<=pd.Timestamp(end))
            frame.loc[keep,'season']=season;frame.loc[keep,'bias_c']=bias;frame.loc[keep,'scenario']=f'{season}_{bias:+d}'
    if (frame.season=='').any():raise ValueError('Source dates absent from documented inventory')
    y=frame['Fault Detection Ground Truth'].to_numpy()
    if not np.isin(y,[0,1]).all():raise ValueError('Unknown ground-truth encoding')
    np.testing.assert_array_equal(y==1,frame.bias_c.to_numpy()!=0)
    # Ground truth is read, verified and retained, never reconstructed as a training target.
    return frame


def sensor_frame(frame):
    return pd.DataFrame({alias:pd.to_numeric(frame[column],errors='raise') for alias,column in RAW_MAPPING.items()},index=frame.index)


def operation_mask(frame):
    return ((frame['Occupancy Mode Indicator']==1)&(frame['AHU: Supply Air Fan Status']==1)).to_numpy()


def causal_clean(raw,dates,times,limit=5):
    x=np.asarray(raw,dtype=np.float32).copy();x[~np.isfinite(x)]=np.nan
    cleaned=x.copy()
    reset=np.r_[True,(np.asarray(dates)[1:]!=np.asarray(dates)[:-1])|(np.diff(times)!=MINUTE_NS)]
    bounds=np.r_[np.flatnonzero(reset),len(x)]
    for a,b in zip(bounds[:-1],bounds[1:]):x[a:b]=pd.DataFrame(x[a:b]).ffill(limit=limit).to_numpy(dtype=np.float32)
    return cleaned,x


def augment(raw):
    col={c:raw[:,i] for i,c in enumerate(RAW_FEATURES)}
    return np.column_stack([raw,col['supply_temp']-col['supply_setpoint'],
        col['mixed_temp']-col['return_temp'],col['mixed_temp']-col['outdoor_temp'],
        col['outdoor_temp']-col['return_temp'],col['duct_pressure']-col['duct_pressure_setpoint'],
        col['mixed_temp']-(col['outdoor_damper']*col['outdoor_temp']+(1-col['outdoor_damper'])*col['return_temp'])]).astype(np.float32)


def fit_preprocessor(raw,cleaned,filled,fit):
    median=np.nanmedian(filled[fit],axis=0)
    if not np.isfinite(median).all():raise ValueError('An input lacks finite training observations')
    completed=np.where(np.isfinite(filled),filled,median).astype(np.float32)
    scaler=StandardScaler().fit(augment(completed[fit]))
    return dict(median=median,scaler=scaler,raw_features=RAW_FEATURES,features=MODEL_FEATURES,
        width=WIDTH,cadence_minutes=1,clip=10.,fill_limit=5,task='Operating AHU outdoor-air sensor-bias detection',
        source_temperature_unit='Fahrenheit',temperature_bias_intensity_unit='Celsius',operational_filter='occupied and supply fan on for entire 30-reading window')


def transform(cleaned,filled,pre):
    completed=np.where(np.isfinite(filled),filled,pre['median']).astype(np.float32)
    scaled=np.clip(pre['scaler'].transform(augment(completed)),-pre['clip'],pre['clip'])
    return np.column_stack([scaled,~np.isfinite(cleaned)]).astype(np.float32)


def cohorts(frame,width=WIDTH):
    active=operation_mask(frame);records=[]
    for node,(date,group) in enumerate(frame.groupby('date',sort=True)):
        idx=group.index.to_numpy();times=group.timestamp.dt.as_unit('ns').astype('int64').to_numpy()
        if len(idx)<width:continue
        from numpy.lib.stride_tricks import sliding_window_view
        valid=sliding_window_view(active[idx],width).all(axis=1)
        valid&=(np.diff(sliding_window_view(times,width),axis=1)==MINUTE_NS).all(axis=1)
        end=idx[np.flatnonzero(valid)+width-1]
        if not len(end):continue
        g=frame.loc[end]
        records.append(pd.DataFrame(dict(node=node,date=date,time_ns=g.timestamp.dt.as_unit('ns').astype('int64'),row_index=end,
            y_end=g['Fault Detection Ground Truth'].astype(bool),scenario=g.scenario,bias_c=g.bias_c,season=g.season)))
    return pd.concat(records,ignore_index=True)


def day_episodes(frame):
    rows=[]
    for node,(date,g) in enumerate(frame.groupby('date',sort=True)):
        if g['Fault Detection Ground Truth'].iloc[0]==1:
            rows.append(dict(node=node,start_ns=int(g.timestamp.iloc[0].value),end_ns=int(g.timestamp.iloc[-1].value+MINUTE_NS),readings=len(g),date=date,bias_c=float(g.bias_c.iloc[0]),scenario=g.scenario.iloc[0],season=g.season.iloc[0]))
    return pd.DataFrame(rows)


def uploaded_features(frame,pre):
    if len(frame)<WIDTH:raise ValueError('At least 30 one-minute readings are required')
    frame=frame.tail(WIDTH).copy()
    if 'timestamp' not in frame:raise ValueError('A timestamp column is required')
    times=pd.to_datetime(frame.timestamp,errors='raise')
    if not (times.diff().dropna()==pd.Timedelta(minutes=1)).all():raise ValueError('Exact one-minute cadence is required')
    if times.dt.normalize().nunique()!=1:raise ValueError('Windows cannot cross a source day')
    for c in ['unit','node']:
        if c in frame and frame[c].nunique(dropna=False)!=1:raise ValueError('One equipment unit is required')
    for c in ['occupied','supply_fan_on']:
        if c not in frame or not (pd.to_numeric(frame[c],errors='raise')==1).all():raise ValueError('Model only scores occupied operation with the supply fan on')
    raw=frame[RAW_FEATURES].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float32)
    clean,filled=causal_clean(raw,np.repeat(str(times.iloc[0].date()),WIDTH),times.dt.as_unit('ns').astype('int64').to_numpy(),5)
    return transform(clean,filled,pre)
