"""Recorded replay, corrected experiment metrics and fresh LSTM inference."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dccad.evaluation import consecutive_alarm

st.set_page_config(page_title='Data Center Telemetry Monitor',layout='wide')
st.title('Data Center Telemetry Anomaly Monitor')
st.caption('Deep Learning project | Section B | Rishabh Srivastava, Mitul Seth, Madhav Mittal')
folders=[p for p in (ROOT/'results').glob('*') if (p/'summary.json').exists() and (p/'dashboard.parquet').exists()]
if not folders:
    st.info('Run the experiment workflow to prepare the trained model and replay data.')
    st.code('python scripts/train_experiments.py --configs base24')
    st.stop()
names=sorted(p.name for p in folders)
selection=ROOT/'results'/'selected_model.json'
default=json.loads(selection.read_text())['name'] if selection.exists() else names[0]
name=st.sidebar.selectbox('Experiment',names,index=names.index(default) if default in names else 0)
folder=ROOT/'results'/name

@st.cache_data
def load_experiment(path,modified):
    p=Path(path);raw=pl.read_parquet(p/'dashboard.parquet')
    df=pd.DataFrame({c:raw[c].to_numpy() for c in raw.columns})
    df['time']=pd.to_datetime(df.time_ns,utc=True)
    return df,json.loads((p/'summary.json').read_text())

df,summary=load_experiment(str(folder),(folder/'summary.json').stat().st_mtime)
policy=summary['selected_policy'];threshold=policy['threshold']
if isinstance(threshold,dict):threshold={int(k):v for k,v in threshold.items()}
node=st.sidebar.selectbox('Server',sorted(df.node.unique()))
rule=st.sidebar.selectbox('Alarm rule',['Validation selected','Immediate','Two consecutive high scores'])
persist=policy['persistence'] if rule=='Validation selected' else (1 if rule=='Immediate' else 2)
df['alarm_view']=consecutive_alarm(df,df.score.to_numpy(),threshold,persist)
d=df[df.node==node].copy().reset_index(drop=True)
position=st.sidebar.slider('Replay position',1,len(d),len(d));v=d.iloc[:position]
st.sidebar.caption(f"Input history: {summary['width']} readings. Threshold: {policy['quantile']:g}th percentile, "
                   f"{'per server' if policy['per_node'] else 'global'}. Score: {policy['score_mode'].replace('_',' ')}.")
about,monitor,experiments,inference=st.tabs(['Project','Monitor replay','Experiments','Score new data'])
with about:
    st.subheader('What the model detects')
    st.write('The LSTM autoencoder learns normal relationships between temperature, fan and power readings. High reconstruction error produces an anomaly score. Evaluation uses recorded Nagios server-health alerts, which do not establish that every alert is a cooling fault.')
    st.write('Replay uses previously scored recordings. Score new data runs the saved model on uploaded sensor readings. Incident detection uses original 15-minute alert bins. Missing periods break alarm bursts and persistence. Settings are selected using validation data.')
    st.markdown('[M100 dataset](https://zenodo.org/records/7541722) · [Dataset paper](https://www.nature.com/articles/s41597-023-02174-3)')
    cleaning_plan=ROOT/'results'/'feature_selection'/'plan.json'
    if cleaning_plan.exists():
        with st.expander('Data cleaning and feature selection'):
            plan=json.loads(cleaning_plan.read_text())
            st.write(f"Training-only correlation selection retained {plan['feature_count']} of {plan['original_feature_count']} inputs. Cleaning preserves high readings and zero fan speeds, limits forward filling to four readings, and resets at gaps and split boundaries.")
            st.write('Negative sensor readings are treated as missing under an indoor-rack plausibility assumption. The source readings are preserved. Correlated sensors can still contain useful individual fault signals.')
            st.dataframe(pd.read_csv(cleaning_plan.parent/'feature_decisions.csv'),hide_index=True,width='stretch')
            st.caption('The four feature experiments use identical input-window endpoints. Their model and alarm settings are selected using validation data.')
    if summary.get('cleaning'):
        st.info(f"This experiment uses cleaned readings and {summary['feature_count']} sensor inputs.")
with monitor:
    st.subheader(f'Server {node}')
    now=v.time.iloc[-1];recent=v[v.time>=now-pd.Timedelta(hours=24)]
    st.caption(f'Replay time: {now:%d %b %Y %H:%M} UTC. Recent counts use actual timestamps.')
    cols=st.columns(3)
    cols[0].metric('Readings scored',len(v));cols[1].metric('Alarm readings in last 24 hours',int(recent.alarm_view.sum()))
    cols[2].metric('Recorded critical endpoints',int(v.label_end.sum()))
    if bool(v.alarm_view.iloc[-1]):st.error('The current reading exceeds the alarm rule.')
    else:st.success('The current reading does not raise an alarm.')
    t=threshold[int(node)] if isinstance(threshold,dict) else threshold
    fig=go.Figure();plot=v[['time','score']].copy()
    plot.loc[plot.time.diff()!=pd.Timedelta(minutes=30),'score']=np.nan
    fig.add_trace(go.Scatter(x=plot.time,y=plot.score,mode='lines',name='Anomaly score'))
    a=v[v.label_end==1];fig.add_trace(go.Scatter(x=a.time,y=a.score,mode='markers',name='Critical at this time',marker={'color':'#c43c39','size':6}))
    a=v[v.alarm_view];fig.add_trace(go.Scatter(x=a.time,y=a.score,mode='markers',name='Model alarm',marker={'color':'#df8b24','symbol':'x','size':5}))
    fig.add_hline(y=t,line_dash='dash',line_color='#df8b24',annotation_text='Calibrated threshold')
    fig.update_layout(height=380,yaxis_title='Reconstruction error',margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig,width='stretch')
    errors=[c for c in df.columns if c.startswith('error_')];latest=v.iloc[-1]
    top=sorted(errors,key=lambda c:float(latest[c]),reverse=True)[:3]
    st.markdown('**Sensors contributing most to the latest reading reconstruction error**')
    st.dataframe(pd.DataFrame([{'Sensor':c[6:],'Squared error in scaled units':latest[c],
                              'Recorded reading':latest[c[6:]],'Reconstructed reading':latest['reconstructed_'+c[6:]]}
                             for c in top]),hide_index=True,width='stretch')
    st.caption('These contributions describe the latest reading. A whole-window score also includes earlier readings. They do not prove a physical fault cause.')
    sensors=[c[6:] for c in errors];chosen=st.selectbox('Sensor to inspect',sensors,index=sensors.index(top[0][6:]))
    sensor=go.Figure();sensor.add_trace(go.Scatter(x=recent.time,y=recent[chosen],name='Original reading'))
    sensor.add_trace(go.Scatter(x=recent.time,y=recent['reconstructed_'+chosen],name='Reconstruction'))
    sensor.update_layout(height=300,margin=dict(l=10,r=10,t=20,b=10));st.plotly_chart(sensor,width='stretch')
with experiments:
    st.subheader('Comparison using labels at the scored reading');rows=[]
    for p in sorted((ROOT/'results').glob('*/summary.json')):
        record=json.loads(p.read_text())
        if 'test' not in record:continue
        r=record['test']
        rows.append({'Experiment':record['name'],'Inputs':record.get('feature_count',33 if record.get('enhanced') else 22),'Average precision':r['average_precision'],
                     'Precision':r['precision'],'Recall':r['recall'],'F1':r['f1'],
                     'Incidents caught':f"{r['detected']}/{r['episodes']}",'Observable incidents':r['observable_episodes'],
                     'False bursts per scored node-day':r['false_bursts_per_node_day'],'Median delay in minutes':r['median_delay_minutes']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.caption('The selected LSTM is chosen using validation F1. All raw incidents remain in the total recall denominator. Only incidents with a scored timestamp inside them are observable.')
    if rule!='Validation selected':st.info('Replay uses a manually chosen alarm rule. The table shows the validation-selected rule.')
    if (folder/'per_node.csv').exists():
        st.markdown('**Results by server**');st.dataframe(pd.read_csv(folder/'per_node.csv'),hide_index=True,width='stretch')
    st.caption('Single rack subset and one random seed. General server-health labels and missing coverage limit interpretation.')
    matched=ROOT/'results'/'feature_selection'/'comparison.csv'
    if matched.exists():
        st.markdown('**Cleaning and feature selection: comparison on identical readings**')
        comparison=pd.read_csv(matched)
        st.dataframe(comparison[['experiment','features','validation_f1','precision','recall','f1','average_precision','false_bursts_per_node_day']],hide_index=True,width='stretch')
        st.caption('Original 22 inputs, cleaned 22 inputs, correlation-selected inputs and a predefined cooling subset use the same training, validation and test endpoints. The winner is chosen by validation F1.')
with inference:
    st.subheader('Run the saved LSTM on a new sensor window')
    from dccad.model import LSTMAutoencoder
    import torch,joblib
    torch.set_num_threads(4)
    checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);features=checkpoint['features']
    st.write(f"Upload a CSV with at least {summary['width']} chronological readings and these {len(features)} columns.")
    st.code(', '.join(features),language=None)
    example=folder/'example_window.csv'
    if example.exists():st.download_button('Download example input',example.read_bytes(),'example_window.csv','text/csv')
    uploaded=st.file_uploader('Sensor readings',type=['csv'])
    if uploaded is not None:
        try:
            new=pd.read_csv(uploaded);missing=set(features)-set(new.columns)
            if missing:raise ValueError('Missing columns: '+', '.join(sorted(missing)))
            if len(new)<summary['width']:raise ValueError('There are too few readings for this model.')
            if 'timestamp' in new.columns:
                timestamps=pd.to_datetime(new['timestamp'].tail(summary['width']),utc=True,errors='raise')
                if not (timestamps.diff().dropna()==pd.Timedelta(minutes=15)).all():
                    raise ValueError('Timestamps must be chronological and exactly 15 minutes apart.')
            x=new[features].tail(summary['width']).apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float32)
            if summary.get('cleaning'):
                from dccad.preprocessing import clean_inference
                x=clean_inference(x,features)
            if not np.isfinite(x).all():raise ValueError('The input contains missing or nonfinite readings.')
            scaler=joblib.load(folder/'scaler.joblib');scaled=scaler.transform(x)
            if summary.get('clip',5.) is not None:scaled=np.clip(scaled,-summary.get('clip',5.),summary.get('clip',5.))
            model=LSTMAutoencoder(len(features),summary['width']);model.load_state_dict(checkpoint['state_dict']);model.eval()
            with torch.no_grad():reconstructed=model(torch.from_numpy(scaled.astype(np.float32))[None]).numpy()[0]
            error=(scaled-reconstructed)**2
            score=float(error.mean() if policy['score_mode']=='whole_window' else error[-1].mean())
            st.metric('New window anomaly score',f'{score:.6f}')
            t=threshold[int(node)] if isinstance(threshold,dict) else threshold
            st.write(f"Threshold for server {node}: {t:.6f}. {'Above' if score>t else 'Below'} threshold.")
            st.caption('One uploaded window cannot establish a persistence streak. Readings must be evenly spaced at 15-minute intervals. Enhanced inputs require the training feature definitions.')
            if summary.get('cleaning'):
                st.caption('Cleaning marks negative or nonfinite readings as missing and fills up to four readings from earlier values in this upload. An unresolved missing reading is rejected.')
            st.dataframe(pd.DataFrame({'Sensor':features,'Latest squared error':error[-1]}).sort_values('Latest squared error',ascending=False).head(5),hide_index=True)
        except (ValueError,KeyError,RuntimeError) as error:st.error(str(error))
st.caption('Data: M100 Marconi100 CINECA. Educational anomaly-monitoring prototype.')
