"""Cooling-equipment sensor-fault benchmark, replay and fresh LSTM scoring."""
from pathlib import Path
import io,json,sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dccad.hvac_inference import score_window

OUT=ROOT/'results/hvac_v1';MODEL=OUT/'lstm_seed_42'
st.set_page_config(page_title='Cooling Equipment Fault Detection',layout='wide')
st.title('Cooling Equipment Fault Detection')
st.caption('LSTM sensor-bias detection | Simulated air-handling unit | Section B')
st.write('This course prototype detects outdoor-air temperature sensor faults while the simulated air-handling unit is occupied and running. Results use a reserved autumn season.')

@st.cache_data
def saved_results():
    summary=json.loads((OUT/'test_summary.json').read_text())
    replay=pd.read_parquet(OUT/'test_replay.parquet')
    replay['timestamp']=pd.to_datetime(replay['timestamp'])
    comparison=pd.read_csv(OUT/'test_model_comparison.csv')
    return summary,replay,comparison

try:summary,replay,comparison=saved_results()
except FileNotFoundError:
    st.info('Complete the benchmark training and evidence build to load the demonstration.')
    st.stop()
metrics=summary['primary_test'];policy=json.loads((MODEL/'validation_policy.json').read_text())
cols=st.columns(4)
for col,label,value in zip(cols,['Test accuracy','Alarm precision','Fault-reading recall','False-positive rate'],
        [metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['false_positive_rate']]):col.metric(label,f'{value:.2%}')
st.caption('33,432 scored minutes; 69 false-positive readings in one burst. FPR is 1.44%, above the original 1% goal. This is one simulated unit and one fault family; experimental fault prevalence is 85.71%.')

tabs=st.tabs(['Recorded demonstration','Upload sensor window','Model comparison','Project and report'])
with tabs[0]:
    dates=sorted(replay.date.unique());date=st.selectbox('Autumn test day',dates,index=0)
    day=replay[replay.date==date].reset_index(drop=True)
    position=st.slider('Replay minute',1,len(day),len(day));view=day.iloc[:position];last=view.iloc[-1]
    condition='Faulted source scenario' if bool(last.y_end) else 'Healthy source scenario'
    st.write(f'**Source condition:** {condition}. This author label is used only for evaluating the replay.')
    if last.pred:st.error('Confirmed sensor-fault alarm')
    else:st.success('No confirmed sensor-fault alarm at this minute')
    st.caption(f"Frozen policy: at least two high scores in three consecutive scored minutes; threshold {policy['threshold']:.3f}. State resets at day boundaries and gaps. A quiet minute can still belong to a faulted scenario.")
    chart=go.Figure()
    for name,label in [('outdoor_temp','Outdoor'),('mixed_temp','Mixed air'),('return_temp','Return air'),('supply_temp','Supply air')]:
        chart.add_scatter(x=view.timestamp,y=view[name],name=label,mode='lines')
    chart.update_layout(xaxis_title='Source local time',yaxis_title='Temperature (°F)',height=370,legend=dict(orientation='h'))
    st.plotly_chart(chart,width='stretch')
    score_plot=go.Figure();score_plot.add_scatter(x=view.timestamp,y=view.score,name='LSTM logit',mode='lines')
    score_plot.add_hline(y=policy['threshold'],line_dash='dash',annotation_text='Frozen threshold')
    score_plot.add_scatter(x=view.loc[view.pred,'timestamp'],y=view.loc[view.pred,'score'],mode='markers',name='Confirmed alarm',marker=dict(size=4,color='#db4437'))
    score_plot.update_layout(xaxis_title='Source local time',yaxis_title='Uncalibrated logit',height=280)
    st.plotly_chart(score_plot,width='stretch')
    st.write(f"Day alarms: {int(view.pred.sum())} / {len(view)} scored minutes. The model uses a preceding 30-minute history; it does not predict future faults.")

with tabs[1]:
    st.write('Upload 30 consecutive one-minute readings from one occupied, running unit. Use the example column names; temperatures are in Fahrenheit. Ground-truth labels are optional and never enter the model.')
    st.download_button('Download example sensor window',(MODEL/'example_window.csv').read_bytes(),file_name='ahu_example_window.csv',mime='text/csv')
    uploaded=st.file_uploader('Sensor CSV',type=['csv'])
    if uploaded is not None:
        try:
            frame=pd.read_csv(io.BytesIO(uploaded.getvalue()))
            import torch
            torch.set_num_threads(4)
            score=score_window(MODEL,frame)
            st.metric('Current endpoint logit',f'{score:.4f}')
            st.write('Score above the frozen threshold.' if score>policy['threshold'] else 'Score below the frozen threshold.')
            st.info('One window yields one score. The two-of-three alarm needs three consecutive scored minutes, so a single window cannot confirm or clear a persistent alarm.')
            st.caption('Nonfinite sensor values use causal filling and saved training medians, with missing indicators. No probability calibration is claimed.')
        except (ValueError,KeyError,TypeError) as exc:st.error(f'Input could not be scored: {exc}')

with tabs[2]:
    show=comparison[['model','accuracy','precision','recall','f1','average_precision','false_positive_rate','fp','fn','detected','false_bursts']].copy()
    show=show.rename(columns={'model':'Model','accuracy':'Accuracy','precision':'Precision','recall':'Recall','f1':'F1','average_precision':'AP','false_positive_rate':'FPR','fp':'FP','fn':'FN','detected':'Fault days detected','false_bursts':'False bursts'})
    st.dataframe(show.style.format({k:'{:.2%}' for k in ['Accuracy','Precision','Recall','F1','FPR']}),hide_index=True,width='stretch')
    st.write('Seed 42 was fixed before the autumn evaluation. Seed 7 has higher test accuracy, but the project does not choose a seed using test performance. The tree detects fewer fault readings and has no false positives in this test.')
    st.write('An all-fault classifier would score 85.71% accuracy and 100% FPR on this fault-heavy benchmark. Accuracy is not sufficient evidence by itself.')
    st.caption('35/36 observable fault days detected; 35/42 total fault days. Six fault days have no eligible operating window. Median delay from midnight imposition is 392 minutes, including overnight downtime and warmup.')

with tabs[3]:
    st.write('Training: winter and spring. Validation: summer. Test: autumn. Entire source days and documented scenario blocks stay within one season partition; input windows never cross a day or gap.')
    st.write('The data comes from LBNL’s public AFDD evaluation collection. The selected MZVAV-1 dataset simulates an AHU with outdoor-air sensor bias at ±1, ±2 and ±4 °C; its measurement columns use °F. Labels come from the source authors.')
    st.link_button('Official dataset and inventory','https://data.openei.org/submissions/910')
    for file,label in [('HVAC_DL_Project_Report.pdf','Download project report'),('HVAC_DL_Project_Presentation.pptx','Download presentation')]:
        path=ROOT/'submission'/file
        if path.exists():st.download_button(label,path.read_bytes(),file_name=file)
    st.write('The M100 investigation remains available as supporting research. Its physical telemetry models did not meet the false-alarm targets; its published label query concerns batch-client scheduler state.')
    st.download_button('M100 recovery findings',(ROOT/'results/task5_recovery_v1/REPORT.md').read_bytes(),file_name='M100_recovery_findings.md')
    st.caption('Prototype scope: one simulated unit and one fault family. The study does not demonstrate datacenter deployment, general equipment-fault diagnosis, future-fault prediction or operational fault prevalence.')
