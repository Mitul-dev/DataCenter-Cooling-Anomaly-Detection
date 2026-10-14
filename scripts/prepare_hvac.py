"""Audit source suitability and freeze a season-separated, author-labelled AHU task."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from dccad.hvac import read_source,cohorts,day_episodes,RAW_FEATURES,MODEL_FEATURES,DATES

OUT=ROOT/'results/hvac_v1'
DATA=ROOT/'data/hvac_benchmark'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True);inventory=[]
    with zipfile.ZipFile(DATA/'afdd_source.zip') as z:
        for name in z.namelist():
            if not name.endswith('.csv'):continue
            # Suitability uses timestamps and source labels, not sensor performance.
            header=pd.read_csv(io.BytesIO(z.read(name)),nrows=0).columns
            time_col=header[0];label_col=header[-1]
            m=pd.read_csv(io.BytesIO(z.read(name)),usecols=[time_col,label_col])
            inventory.append(dict(file=name,rows=len(m),healthy=int((m[label_col]==0).sum()),faulty=int((m[label_col]==1).sum()),
                dates=int(pd.to_datetime(m[time_col],format='%m/%d/%Y %H:%M').dt.date.nunique())))
    pd.DataFrame(inventory).to_csv(OUT/'source_suitability.csv',index=False)
    frame=read_source(DATA/'afdd_source.zip');meta=cohorts(frame);episodes=day_episodes(frame)
    groups=frame.groupby('date').agg(season=('season','first'),scenario=('scenario','first'),bias_c=('bias_c','first'),
        label=('Fault Detection Ground Truth','first'),raw_rows=('timestamp','size')).reset_index()
    groups['split']=groups.season.map({'winter':'train','spring':'train','summer':'validation','autumn':'test'})
    groups['scored_endpoints']=groups.date.map(meta.groupby('date').size()).fillna(0).astype(int)
    groups.to_csv(OUT/'day_groups.csv',index=False);files={}
    for name,seasons in [('train',['winter','spring']),('validation',['summer']),('test',['autumn'])]:
        m=meta[meta.season.isin(seasons)].reset_index(drop=True);p=OUT/(name+'_cohort.csv')
        if p.exists():
            pd.testing.assert_frame_equal(pd.read_csv(p),m,check_dtype=False)
        else:m.to_csv(p,index=False,lineterminator='\n')
        ep=episodes[episodes.season.isin(seasons)];ep.to_csv(OUT/(name+'_fault_days.csv'),index=False)
        files[name]=dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),windows=len(m),faulty=int(m.y_end.sum()),
            scored_days=int(m.date.nunique()),raw_days=int(groups[groups.split==name].shape[0]),raw_fault_days=len(ep))
    plan=dict(experiment='hvac_v1',source_url='https://data.openei.org/submissions/910',source_doi='10.25984/1824861',
        archive_sha256=sha(DATA/'afdd_source.zip'),source_inventory_sha256=sha(DATA/'source_inventory.pdf'),
        csv_member='MZVAV-1.csv',task='Operating simulated air-handling-unit outdoor-air sensor-bias detection',
        generation='EnergyPlus-Dymola co-simulation; one AHU, one modeled Chicago building; no datacenter deployment claim',
        labels='Original Fault Detection Ground Truth: 0 unfaulted, 1 faulted; all 189 dates verified against inventory Table 1-2',
        scenario_dates=DATES,split='Whole seasons and documented scenario blocks: winter/spring train, summer validation, autumn heldout test',
        no_random_window_split=True,window_width=30,cadence_minutes=1,
        eligibility='Every reading in input history must be occupied and have supply fan on; histories cannot cross a day or time gap',
        preprocessing='Forward fill <=5 minutes reset at each day/gap; remaining missing values use normal training medians; StandardScaler on normal active training rows; +/-10 clip; original missing flags retained',
        raw_features=RAW_FEATURES,model_features=MODEL_FEATURES,derived_note='Temperature/pressure differences and a command-weighted mixing residual; damper command is a proxy, not measured mixing fraction',
        no_model_inputs=['reference label','date/time','day ID','season','scenario','bias intensity'],cohorts=files,
        seeds=[7,42,123],primary_seed=42,neural=dict(hidden=32,dropout=.2,learning_rate=.001,batch=512,max_epochs=30,patience=6,
            positive_weight='normal/faulty training endpoint ratio; balances training loss only',early_stopping='highest validation AP'),
        boosting=dict(learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.,checkpoints=[50,100,200],
            early_stopping=False,training_class_weights='balanced using training endpoint counts only',history='latest/mean/std/slope'),
        policy=dict(normal_validation_percentiles=[95,99,99.5,99.9,99.95],global_only=True,pr_recall_floors=[.4,.6,.8],best_f0_5=True,
            rules=['immediate','consecutive_2','two_of_three'],reset='day/unit/gap'),
        targets=dict(precision_min=.3,recall_min=.4,fpr_max=.01,false_bursts_per_scored_day_max=.1,observable_fault_day_recall_min=.6),
        selection='feasible policy key; primary seed 42 for release; all three neural seeds must pass before operational qualification; tree reported separately',
        test_access='Only after model/preprocessing/policy hashes frozen; score once; never tune against test',
        limitations=['Scenario prevalence is experimental, not operational fault prevalence','Single simulated unit and one fault family','Healthy validation/test are one documented scenario week each','Within-season days are correlated; seasonal transfer is the main challenge'],
        additional_architecture_search=False)
    p=OUT/'experiment_plan.json';plan=json.loads(json.dumps(plan))
    if p.exists() and json.loads(p.read_text())!=plan:raise RuntimeError('Frozen HVAC protocol changed; create a new version')
    if not p.exists():write(p,plan)
    lines=['# HVAC benchmark preflight and frozen protocol','',
        'The source archive contains six collections. Selection was based on label/independent-day availability and documented provenance, before model fitting. RTU has no native normal labels; SZCAV has only one healthy day; the smaller collections have fewer independent runs. The largest MZVAV-1 file is selected and its native labels are verified against the inventory for every date. Labels are not corrected or synthesized.','',
        'The task is sensor-bias detection in an operating simulated AHU, not confirmed datacenter cooling failure. Occupied/fan-on filtering is declared before training and applied identically to all partitions. Nights and nonoperating days are reported as unscored; all raw condition days remain in the fault-day coverage denominator.','',
        '| Partition | Seasons | Raw days | Scored days | Endpoints | Fault endpoints |','|---|---|---:|---:|---:|---:|']
    for name,v in files.items():lines.append(f"| {name} | {'winter + spring' if name=='train' else 'summer' if name=='validation' else 'autumn'} | {v['raw_days']} | {v['scored_days']} | {v['windows']} | {v['faulty']} |")
    lines += ['', 'Scenarios and whole seasons do not cross partitions. No shuffled overlapping-window evaluation is allowed. The final season is reserved until preprocessing, checkpoint and policy selection are frozen. Source times are local/unspecified, not asserted as UTC. Temperature inputs are Fahrenheit; imposed sensor-bias levels in the inventory are Celsius.',
        '', 'The physical consistency features use measurements/control commands only. They do not determine labels. Model training uses the author-provided ground truth. Scenario/calendar fields are metadata only and never model inputs.',
        '', 'The dataset is strongly fault-heavy because it was designed for equipment-fault experiments. Accuracy alone is misleading: an all-fault classifier would have high accuracy but 100% FPR. Endpoint precision, recall, FPR, AP, false bursts and observed/total fault-day coverage are required. No operational prevalence claim is made.',
        '', '[Official data record](https://data.openei.org/submissions/910). Fault/unit/schedule definitions are in the accompanying source inventory, pages 3–7.']
    (OUT/'PREFLIGHT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(files,indent=2))

if __name__=='__main__':main()
