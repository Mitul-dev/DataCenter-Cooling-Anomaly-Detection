"""User-facing four-sensor benchmark replay and saved-model inference."""
import io
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render(root):
    from .multisensor import score_window
    out=Path(root)/'results/multisensor_v2'
    summary=json.loads((out/'test_summary.json').read_text())
    classes=summary['classes'];family=summary['primary_family'];folder=out/family
    policy=json.loads((folder/'validation_policy.json').read_text())
    pre=json.loads((folder/'preprocessing.json').read_text())
    st.title('Fault Detection Across Four Sensors')
    st.write('This LSTM identifies a faulty temperature or pressure sensor in a simulated air-handling unit. It distinguishes healthy operation from four individual sensor-fault types.')
    metric=summary['primary_test'];cols=st.columns(4)
    for col,label,key in zip(cols,['Correct condition','Average class F1','Faults detected','Healthy false-positive rate'],
                              ['accuracy','macro_f1','fault_detection_recall','healthy_fpr']):
        col.metric(label,f'{metric[key]:.2%}')
    st.caption(f"{summary['test_windows']:,} reserved test minutes. The model reads {len(pre['features'])} inputs over 30 minutes. It identifies one fault at a time. Scores describe this simulation benchmark.")
    if not summary['test_targets_met']:
        st.warning('The reserved test did not meet every target. Inspect the sensor-specific results before presenting this as a qualified model.')
    tabs=st.tabs(['Recorded simulation','Upload readings','Sensor results','Project files'])

    @st.cache_data
    def replay_data(path):
        data=pd.read_parquet(path);data['timestamp']=pd.to_datetime(data.timestamp)
        return data

    with tabs[0]:
        replay=replay_data(str(out/'test_replay.parquet'))
        condition=st.selectbox('Recorded condition',range(len(classes)),format_func=lambda k:classes[k])
        subset=replay[replay.y==condition]
        scenario=st.selectbox('Source scenario',sorted(subset.scenario.unique()))
        subset=subset[subset.scenario==scenario]
        date=st.selectbox('Reserved test day',sorted(subset.date.unique()))
        day=subset[subset.date==date].reset_index(drop=True)
        position=st.slider('Replay minute',1,len(day),len(day))
        view=day.iloc[:position];last=view.iloc[-1];label=classes[int(last.prediction)]
        if int(last.prediction)==0:st.success('Model output: healthy / no sensor alarm')
        else:st.error('Model output: '+label+' fault')
        st.caption('The recorded condition is the source scenario label. It never enters the model. A missed alarm can occur during a faulted scenario.')
        temp=go.Figure()
        for col,label in [('CSA_TEMP','Cold-deck air temperature'),('HSA_TEMP','Hot-deck air temperature')]:
            temp.add_scatter(x=view.timestamp,y=view[col],name=label,mode='lines')
        temp.update_layout(yaxis_title='Temperature (°F)',xaxis_title='Source local time',height=310,legend=dict(orientation='h'))
        st.plotly_chart(temp,width='stretch')
        pressure=go.Figure()
        for col,label in [('CSA_SP','Cold-deck pressure'),('HSA_SP','Hot-deck pressure')]:
            pressure.add_scatter(x=view.timestamp,y=view[col],name=label,mode='lines')
        pressure.update_layout(yaxis_title='Duct pressure (native inH₂O)',xaxis_title='Source local time',height=270,legend=dict(orientation='h'))
        st.plotly_chart(pressure,width='stretch')
        st.write(f"Correct condition labels in this view: {(view.prediction==view.y).mean():.2%}. Fault-alarm minutes: {int((view.prediction!=0).sum())} / {len(view)}.")
        st.caption('The model uses current and preceding readings. It does not forecast future failures. Separate temperature and pressure units are preserved.')
    with tabs[1]:
        st.write('Upload exactly 30 consecutive one-minute readings. All rows must come from one operating unit and satisfy the example schema. Temperature readings use Fahrenheit.')
        st.download_button('Download healthy input example',(folder/'example_class_0.csv').read_bytes(),file_name='multisensor_example.csv',mime='text/csv')
        upload=st.file_uploader('Sensor readings',type=['csv'],key='multisensor_upload')
        if upload is not None:
            try:
                import torch
                torch.set_num_threads(4)
                scores=score_window(folder,pd.read_csv(io.BytesIO(upload.getvalue())))
                st.dataframe(pd.DataFrame({'Condition':list(scores),'Model score':list(scores.values())}).style.format({'Model score':'{:.2%}'}),hide_index=True)
                if policy['rule']=='immediate':
                    active=[(name,scores[name]) for name,t in zip(classes[1:],policy['thresholds']) if scores[name]>t]
                    label=max(active,key=lambda item:item[1])[0] if active else 'Healthy / no sensor alarm'
                    st.write('**Frozen policy output:** '+label)
                else:st.info('The saved policy requires multiple consecutive endpoint scores to confirm an alarm. This window supplies one score.')
                st.caption('These are uncalibrated softmax scores. The fault condition, filename and fault severity never enter model inference.')
            except (ValueError,KeyError,TypeError) as exc:st.error('Readings could not be scored: '+str(exc))
    with tabs[2]:
        table=pd.DataFrame(metric['per_class'])[['sensor','support','precision','recall','f1','one_vs_rest_fpr']]
        st.dataframe(table.rename(columns={'sensor':'Condition','support':'Test minutes','precision':'Precision','recall':'Correct-condition recall','f1':'F1','one_vs_rest_fpr':'One-vs-rest FPR'}).style.format({c:'{:.2%}' for c in ['Precision','Correct-condition recall','F1','One-vs-rest FPR']}),hide_index=True,width='stretch')
        st.write('Sensor recall requires identifying the correct faulty sensor. Fault detection recall only requires raising any fault alarm. Healthy FPR counts alarms among healthy minutes.')
        comparisons=[]
        for name,m in summary['all_families'].items():
            comparisons.append({'Model':name,'Inputs':len(json.loads((out/name/'preprocessing.json').read_text())['features']),
                                'Accuracy':m['accuracy'],'Macro F1':m['macro_f1'],'Healthy FPR':m['healthy_fpr']})
        st.dataframe(pd.DataFrame(comparisons).style.format({c:'{:.2%}' for c in ['Accuracy','Macro F1','Healthy FPR']}),hide_index=True,width='stretch')
        st.caption('Training and validation selected the primary model before test inference. Five inputs reduce model size but can lose information needed to identify a sensor fault.')
    with tabs[3]:
        st.write('The source contains 114 recorded channels. This study uses 26 AHU inputs: 21 measurement channels and five control commands. Training-only feature selection also tested five raw inputs.')
        st.write('Fault coverage: cold-deck temperature, hot-deck temperature, cold-deck pressure and hot-deck pressure. Each native scenario contains one imposed sensor bias. Simultaneous faults and other equipment faults require additional evidence.')
        st.link_button('Official simulation dataset','https://faultdetection.lbl.gov/dataset/simulated-dd-ahu-dataset/')
        for name,label in [('HVAC_Multisensor_PBL_Report.pdf','Download updated report'),('HVAC_Multisensor_PBL_Report.docx','Download editable updated report'),('HVAC_Multisensor_Presentation.pptx','Download updated presentation')]:
            path=Path(root)/'submission'/name
            if path.exists():st.download_button(label,path.read_bytes(),file_name=name)
        report=out/'REPORT.md'
        if report.exists():st.download_button('Download experiment findings',report.read_bytes(),file_name='multisensor_findings.md')
        st.caption('The raw archive is downloaded separately for retraining. Saved models, replay data and input examples allow the demonstration to run offline.')
