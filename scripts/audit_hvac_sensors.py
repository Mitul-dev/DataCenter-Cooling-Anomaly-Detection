"""Audit available channels and rank sensor groups using original training seasons only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib,json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,log_loss
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dccad.hvac import read_source,sensor_frame,operation_mask,causal_clean,cohorts,RAW_FEATURES,RAW_MAPPING,WIDTH
from dccad.supervised import sequence_inputs,trailing_statistics
OUT=ROOT/'results/hvac_sensor_review'


def main():
    OUT.mkdir(exist_ok=True)
    archive=ROOT/'data/hvac_benchmark/afdd_source.zip'
    frame=read_source(archive)
    # All measured summaries, ranking labels and selector training remain winter/spring.
    frame=frame[frame.season.isin(['winter','spring'])].reset_index(drop=True)
    # Keep the initial season-transfer screen as a diagnostic, not a hidden retry.
    if (OUT/'review.json').exists():
        prior=json.loads((OUT/'review.json').read_text())
        if prior.get('selector_fit')=='original training winter':
            for old,new in [('review.json','winter_spring_transfer_diagnostic.json'),('training_only_sensor_ranking.csv','winter_spring_transfer_ranking.csv')]:
                if not (OUT/new).exists():(OUT/new).write_bytes((OUT/old).read_bytes())
    active=operation_mask(frame)
    inventory=[]
    for c in frame.columns[:18]:
        role='model input' if c in RAW_MAPPING.values() else 'eligibility/status' if c in ['Occupancy Mode Indicator','AHU: Supply Air Fan Status','AHU: Return Air Fan Status'] else 'label excluded' if c=='Fault Detection Ground Truth' else 'timestamp excluded'
        inventory.append(dict(column=c,role=role,unique_training=int(frame[c].nunique()),unique_active_training=int(frame.loc[active,c].nunique()),missing_training=int(frame[c].isna().sum())))
    pd.DataFrame(inventory).to_csv(OUT/'training_sensor_inventory.csv',index=False)
    raw=sensor_frame(frame).to_numpy(dtype=np.float32)
    clean,filled=causal_clean(raw,frame.date.to_numpy(),frame.timestamp.dt.as_unit('ns').astype('int64').to_numpy())
    train=pd.read_csv(ROOT/'results/hvac_v1/train_cohort.csv')
    reconstructed=cohorts(frame)
    for key in ['row_index','time_ns','y_end']:np.testing.assert_array_equal(train[key],reconstructed[key])
    fit_days=[];rank_days=[]
    for scenario,g in train.groupby('scenario',sort=True):
        days=sorted(g.date.unique());assert len(days)>=4
        fit_days.extend(days[:-2]);rank_days.extend(days[-2:])
    assert not set(fit_days).intersection(rank_days)
    fit=frame.date.isin(fit_days).to_numpy()&active&(~frame['Fault Detection Ground Truth'].to_numpy(bool))
    median=np.nanmedian(filled[fit],axis=0)
    assert np.isfinite(median).all()
    completed=np.where(np.isfinite(filled),filled,median)
    # Raw trailing statistics need no normalization for the tree selector.
    x=trailing_statistics(sequence_inputs(completed,train.row_index,WIDTH))
    winter=train.date.isin(fit_days).to_numpy();spring=train.date.isin(rank_days).to_numpy()
    xt=x[winter];xr=x[spring];yt=train.loc[winter,'y_end'].to_numpy(bool);yr=train.loc[spring,'y_end'].to_numpy(bool)
    weight=(~yt).sum()/yt.sum();wr=np.where(yr,(~yr).sum()/yr.sum(),1.)
    model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=100,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.,early_stopping=False,random_state=42)
    with threadpool_limits(limits=4):
        model.fit(xt,yt,sample_weight=np.where(yt,weight,1.))
        base=model.predict_proba(xr)[:,1]
        base_loss=log_loss(yr,base,sample_weight=wr);base_ap=average_precision_score(yr,base)
        # Shuffle complete donor-day trajectories with identical scored minute patterns.
        meta=train.loc[spring].reset_index(drop=True)
        day_rows={int(n):g.index.to_numpy() for n,g in meta.groupby('node',sort=True)}
        groups={}
        for n,idx in day_rows.items():
            signature=tuple((meta.loc[idx,'time_ns'].to_numpy()//(60*10**9))%1440)
            groups.setdefault(signature,[]).append(n)
        assert all(len(g)>1 for g in groups.values()),'Insufficient exchangeable whole-day blocks'
        rng=np.random.default_rng(2026);permutations=[]
        for repeat in range(5):
            permutation=np.arange(len(meta))
            for nodes in groups.values():
                donors=rng.permutation(nodes)
                for recipient,donor in zip(nodes,donors):permutation[day_rows[recipient]]=day_rows[donor]
            permutations.append(permutation)
        records=[];channels=len(RAW_FEATURES)
        assert xr.shape[1]==channels*4
        for j,name in enumerate(RAW_FEATURES):
            losses=[];aps=[];cols=[j,j+channels,j+2*channels,j+3*channels]
            for permutation in permutations:
                changed=xr.copy();changed[:,cols]=xr[permutation][:,cols]
                prob=model.predict_proba(changed)[:,1]
                losses.append(log_loss(yr,prob,sample_weight=wr)-base_loss)
                aps.append(base_ap-average_precision_score(yr,prob))
            records.append(dict(sensor=name,balanced_log_loss_increase=float(np.mean(losses)),repeat_sd=float(np.std(losses,ddof=1)),ap_decrease=float(np.mean(aps)),unique_active_training=int(pd.Series(raw[active,j]).nunique())))
    ranking=pd.DataFrame(records).sort_values(['balanced_log_loss_increase','sensor'],ascending=[False,True]).reset_index(drop=True)
    ranking.insert(0,'rank',np.arange(1,len(ranking)+1));ranking.to_csv(OUT/'training_only_sensor_ranking.csv',index=False)
    selected=ranking[ranking.unique_active_training>1].head(5).sensor.tolist();assert len(selected)==5
    joblib.dump(dict(model=model,median=median,sensors=RAW_FEATURES,fit_days=fit_days,rank_days=rank_days),OUT/'selector.joblib')
    result=dict(stage='training_only_sensor_review',source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),original_inputs=13,engineered_channels=32,
        selector_fit='winter/spring training days, first eligible days in each documented scenario',importance_partition='last two eligible training days per scenario',validation_or_test_used_for_selection=False,
        fit_days=fit_days,importance_days=rank_days,internal_partition='whole days within the original training partition only',
        method='Grouped whole-day permutation of latest/mean/std/slope for one raw sensor; five fixed repeats; rank by balanced log-loss increase',
        primary_selector_seed=42,permutation_seed=2026,internal_screening_ap=float(base_ap),internal_screening_balanced_loss=float(base_loss),
        positive_importance_sensors=int((ranking.balanced_log_loss_increase>0).sum()),
        candidate_top_five=selected,top_five_lstm_trained=False,improvement_claimed=False,
        caveats=['Importance is conditional on the tree and other correlated sensors; it is not a physical causal ranking.',
                 'Top-five LSTM must be retrained and compared with matched raw13 and existing32 feature sets.',
                 'Autumn was already inspected; any repeated use is retrospective, not a new untouched evaluation.',
                 'New sensor types need a richer aligned source or an explicitly separate synthetic simulation.'])
    (OUT/'review.json').write_text(json.dumps(result,indent=2))
    lines=['# Sensor expansion and top-five review','',
        'This audit uses only the original winter/spring training seasons. The existing trained model and final report are unchanged.','',
        'The source already supplies all thirteen measurement/control inputs used by the project. The remaining fan-status/occupancy columns are constant during eligible operation; timestamps and labels are excluded. Additional humidity, airflow or power channels are not present in MZVAV-1. Channels from a different unit cannot be appended as if they were simultaneous observations of this AHU.','',
        '## Provisional top five','',
        ', '.join(selected), '',
        'Ranking uses whole-day partitions within the original winter/spring training set: the last two eligible days of each documented scenario are used only for screening, and the remaining training days fit the selector. Each sensor is permuted together with its latest/mean/std/slope summaries. Five fixed repeats preserve the scored minute-of-day pattern within donor days. Balanced log loss prevents the fault majority from hiding healthy-class errors. Correlations can mask or distribute importance, so this ranking needs a retrained LSTM ablation.','',
        'The first diagnostic selector fit winter and screened spring. It had AP 0.8539 and poor balanced loss, so its ranking was not treated as reliable. Its diagnostic and candidate ranking are preserved separately. Neither screen reads the original summer validation or autumn test outcomes.','',
        '## Controlled comparison','',
        'Compare three predeclared sets: existing engineered32, raw13 and selected raw5. Keep endpoints, histories, architecture family, training budget, seeds and validation-only policy selection matched. Report recall/FPR/false bursts, parameter count and inference cost. Five sensors do not guarantee higher accuracy.','',
        'The original autumn data have already been inspected. A comparison there must be labelled retrospective. An independent improvement claim requires new reserved scenario runs, weather periods or equipment units. Any expanded simulator must be a separate, documented synthetic study with faults injected by scenario and labels kept out of model inputs.','',
        'No top-five LSTM training or accuracy-improvement claim has been made by this audit.']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
