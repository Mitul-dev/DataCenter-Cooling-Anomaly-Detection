"""Generate submission evidence from completed runs, without inventing results."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from dccad.data import CADENCE_NS,prepare
from dccad.evaluation import bursts,metrics,consecutive_alarm

FIG=ROOT/'results'/'figures';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','figure.dpi':150})


def read_dash(path):
    p=pl.read_parquet(path)
    return pd.DataFrame({c:p[c].to_numpy() for c in p.columns})


def main():
    selected=json.loads((ROOT/'results'/'selected_model.json').read_text())['name']
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*/summary.json')) if 'test' in json.loads(p.read_text())]
    chosen=next(r for r in records if r['name']==selected)
    baseline=next(r for r in records if r['name']=='isolation_forest')
    summaries=[]
    for r in records:
        summaries.append({'experiment':r['name'],'width':r['width'],'features':r.get('feature_count',33 if r.get('enhanced') else 22),
                          'validation_f1':r['selected_policy']['f1'],**r['test']})
    comparison=pd.DataFrame(summaries);comparison.to_csv(ROOT/'results'/'experiment_comparison.csv',index=False)
    baseline_meta=pd.read_csv(ROOT/'results'/'base24'/'window_metadata.csv')
    baseline_score=np.load(ROOT/'results'/'isolation_forest'/'scores.npz')['te']
    bp=baseline['selected_policy'];bt=bp['threshold']
    if isinstance(bt,dict):bt={int(k):v for k,v in bt.items()}
    baseline_pred=consecutive_alarm(baseline_meta,baseline_score,bt,bp['persistence'])
    node_baseline=[]
    for node in baseline_meta.node.unique():
        mask=baseline_meta.node.to_numpy()==node
        node_baseline.append({'node':int(node),'samples':int(mask.sum()),'positives':int(baseline_meta.y_end.to_numpy()[mask].sum()),
                              **metrics(baseline_meta.y_end.to_numpy()[mask],baseline_score[mask],baseline_pred[mask])})
    pd.DataFrame(node_baseline).to_csv(ROOT/'results'/'isolation_forest'/'per_node.csv',index=False)
    for r in records:
        if r['name']=='isolation_forest':continue
        folder=ROOT/'results'/r['name']
        z=np.load(ROOT/'artifacts'/f"reduced_{'enhanced' if r['enhanced'] else 'base'}.npz",allow_pickle=False)
        meta=pd.read_csv(folder/'window_metadata.csv')
        for m in meta.itertuples():
            end=int(m.row_index);idx=np.arange(end-r['width']+1,end+1)
            if np.isfinite(z['raw'][idx]).all() and np.all(np.diff(z['time_ns'][idx])==CADENCE_NS):
                example=pd.DataFrame(z['raw'][idx],columns=z['features'])
                example.insert(0,'timestamp',pd.to_datetime(z['time_ns'][idx],utc=True).astype(str))
                example.to_csv(folder/'example_window.csv',index=False)
                (folder/'example_metadata.json').write_text(json.dumps({'node':int(m.node),'test_row':int(m.Index)}))
                break
    hist=pd.read_csv(ROOT/'results'/selected/'history.csv')
    fig,ax=plt.subplots(figsize=(9,3.6));ax.plot(hist.epoch,hist.train_loss,label='Normal training',color='#19638a')
    ax.plot(hist.epoch,hist.val_loss,label='Normal validation',color='#c67b22');ax.set(xlabel='Epoch',ylabel='Scaled reconstruction MSE',title=f'Training history for {selected}')
    ax.legend();ax.grid(alpha=.15);fig.tight_layout();fig.savefig(FIG/'training.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    labels=comparison.experiment.tolist();x=np.arange(len(labels))
    axes[0].bar(x,comparison.average_precision,color=['#19638a' if n==selected else '#92a8b5' for n in labels]);axes[0].set_title('Average precision at scored endpoints')
    axes[1].bar(x,comparison.false_bursts_per_node_day,color='#c67b22');axes[1].set_title('False bursts per scored node-day')
    for ax in axes:ax.set_xticks(x,labels,rotation=30,ha='right');ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(FIG/'comparison.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,3.7))
    for name,color in [(selected,'#19638a'),('isolation_forest','#c67b22')]:
        if name=='isolation_forest':
            meta=pd.read_csv(ROOT/'results'/'base24'/'window_metadata.csv');scores=np.load(ROOT/'results'/name/'scores.npz')['te']
        else:
            meta=pd.read_csv(ROOT/'results'/name/'window_metadata.csv');ss=np.load(ROOT/'results'/name/'scores.npz')
            scores=ss['te_mean' if chosen['selected_policy']['score_mode']=='whole_window' else 'te_end']
        p,recall,_=precision_recall_curve(meta.y_end,scores);ax.plot(recall,p,label=name,color=color)
    ax.set(xlabel='Recall',ylabel='Precision',title='Endpoint precision recall curves',xlim=(0,1),ylim=(0,1));ax.legend();ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(FIG/'pr_curve.png',bbox_inches='tight');plt.close(fig)
    dash=read_dash(ROOT/'results'/selected/'dashboard.parquet');dash['time']=pd.to_datetime(dash.time_ns,utc=True)
    details=pd.read_csv(ROOT/'results'/selected/'incident_details.csv')
    cases=[]
    detected=details[details.detected & (details.score_opportunities>0)]
    missed=details[(~details.detected)&(details.score_opportunities>0)]
    for kind,pool in [('successful_detection',detected),('missed_incident',missed)]:
        if len(pool):
            ep=pool.iloc[0];cases.append({'kind':kind,'node':int(ep.node),'anchor_ns':int(ep.start_ns),'episode_start_ns':int(ep.start_ns),'episode_end_ns':int(ep.end_ns),'delay_minutes':None if pd.isna(ep.delay_minutes) else float(ep.delay_minutes)})
    all_bursts=bursts(dash,dash.alarm.to_numpy().astype(bool))
    false=[]
    for a,b in all_bursts:
        valid=details[details.node==dash.node.iloc[a]]
        if not any(np.any((dash.time_ns.iloc[a:b+1]>=ep.start_ns)&(dash.time_ns.iloc[a:b+1]<ep.end_ns)) for ep in valid.itertuples()):
            false.append((float(dash.score.iloc[a:b+1].max()),a,b))
    if false:
        _,a,b=max(false);cases.append({'kind':'false_alarm','node':int(dash.node.iloc[a]),'anchor_ns':int(dash.time_ns.iloc[a]),'burst_readings':b-a+1})
    for case in cases:
        node=case['node'];anchor=pd.to_datetime(case['anchor_ns'],utc=True)
        view=dash[(dash.node==node)&(dash.time>=anchor-pd.Timedelta(hours=8))&(dash.time<=anchor+pd.Timedelta(hours=12))]
        error_cols=[c for c in view if c.startswith('error_')]
        sensor=max(error_cols,key=lambda c:view[c].mean())[6:];case['sensor']=sensor
        thr=chosen['selected_policy']['threshold'];thr=thr[str(node)] if isinstance(thr,dict) else thr
        fig,axes=plt.subplots(2,1,figsize=(10.5,4.8),sharex=True)
        axes[0].plot(view.time,view.score,color='#19638a',label='Anomaly score');axes[0].axhline(thr,color='#c67b22',ls='--',label='Validation threshold')
        axes[0].set_ylabel('Score');axes[0].set_yscale('log');axes[0].legend(loc='upper right',fontsize=8)
        axes[1].plot(view.time,view[sensor],color='#19638a',label='Original reading')
        axes[1].plot(view.time,view['reconstructed_'+sensor],color='#c67b22',ls='--',label='Reconstruction')
        axes[1].set_ylabel(sensor);axes[1].legend(loc='upper right',fontsize=8)
        if 'episode_start_ns' in case:
            for ax in axes:ax.axvspan(pd.to_datetime(case['episode_start_ns'],utc=True),pd.to_datetime(case['episode_end_ns'],utc=True),color='#c43c39',alpha=.15)
        for ax in axes:ax.grid(alpha=.15)
        axes[0].set_title(f"{case['kind'].replace('_',' ').title()} on node {node}");axes[1].set_xlabel('UTC time')
        fig.autofmt_xdate();fig.tight_layout();fig.savefig(FIG/(case['kind']+'.png'),bbox_inches='tight');plt.close(fig)
    base=np.load(ROOT/'artifacts'/'reduced_base.npz');data={k:base[k] for k in base.files};_,split,_=prepare(data)
    dataset={'rows':len(split),'nodes':sorted(np.unique(data['node']).tolist()),'features':data['features'].tolist(),
             'raw_missing_pct':np.isnan(data['raw']).mean(axis=0).tolist(),
             'labels':{str(int(k)):int((data['label']==k).sum()) for k in np.unique(data['label'])},
             'split_rows':{'train':int((split==0).sum()),'validation':int((split==1).sum()),'test':int((split==2).sum())},
             'start':str(pd.to_datetime(data['time_ns'].min(),utc=True)),'end':str(pd.to_datetime(data['time_ns'].max(),utc=True))}
    per_node=pd.read_csv(ROOT/'results'/selected/'per_node.csv')
    manifest={'selected':selected,'chosen':chosen,'baseline':baseline,'experiments':records,'comparison':summaries,
              'per_node':per_node.to_dict('records'),'baseline_per_node':node_baseline,
              'macro_ap':float(per_node.average_precision.mean()),'baseline_macro_ap':float(pd.DataFrame(node_baseline).average_precision.mean()),
              'dataset':dataset,'cases':cases,'history':hist.to_dict('records'),
              'original_audit':json.loads((ROOT/'results'/'original_audit'/'summary.json').read_text())}
    (ROOT/'results'/'submission_evidence.json').write_text(json.dumps(manifest,indent=2))
    print('Evidence prepared for',selected,'cases:',[c['kind'] for c in cases],flush=True)


if __name__=='__main__':main()
