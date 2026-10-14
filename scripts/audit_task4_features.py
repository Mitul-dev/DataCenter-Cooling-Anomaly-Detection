"""Inventory additional measurements using development-server training rows only."""
import io,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from dccad.data import NODES
from dccad.protocol import TRAIN_END

OUT=ROOT/'results/task4_robustness'


def candidates(columns):
    chosen={}
    for c in columns:
        if c in {'timestamp','value','__index_level_0__'}:continue
        if c.endswith('_avg') and (c.startswith(('p0_io_power','p1_io_power','p0_mem_power','p1_mem_power','gv100card','pcie'))):
            chosen[c]='additional per-component measurements'
        elif c.endswith('_avg') and c.startswith(('ps0_input_voltag','ps1_input_voltag','ps0_output_curre','ps1_output_curre','ps0_output_volta','ps1_output_volta')):
            chosen[c]='power-supply voltage/current'
        elif c.endswith('_std') and (c.startswith(('ambient','fan','total_power','p0_power','p1_power'))):
            chosen[c]='variation within aggregation intervals'
        elif c.endswith(('_min','_max')) and c.startswith(('ambient','total_power','gpu')):
            chosen[c]='extrema hidden by averaged/pooled features'
    return chosen


def main():
    OUT.mkdir(parents=True,exist_ok=True);schemas={};records=[];cutoff=pd.Timestamp(TRAIN_END,tz='UTC').to_pydatetime()
    with tarfile.open(ROOT/'0.tar') as tar:
        for node in NODES:
            member=tar.extractfile(f'{node}.parquet')
            payload=io.BytesIO(member.read());schema=pq.read_schema(payload)
            schemas[str(node)]=schema.names;wanted=candidates(schema.names)
            payload.seek(0)
            frame=pl.read_parquet(payload,columns=['timestamp']+list(wanted)).filter(pl.col('timestamp')<pl.lit(cutoff))
            for c,group in wanted.items():
                values=frame[c].to_numpy().astype(np.float64);finite=values[np.isfinite(values)]
                records.append(dict(node=node,feature=c,group=group,training_rows=len(values),finite_rows=len(finite),
                    finite_fraction=len(finite)/len(values),negative_fraction=float((finite<0).mean()) if len(finite) else None,
                    training_std=float(finite.std()) if len(finite) else None,training_min=float(finite.min()) if len(finite) else None,
                    training_max=float(finite.max()) if len(finite) else None))
            print('Audited training source features:',node,len(wanted),flush=True)
    table=pd.DataFrame(records);table.to_csv(OUT/'additional_features_training_availability.csv',index=False)
    common=set.intersection(*(set(v) for v in schemas.values()))
    aggregate=[]
    for feature,rows in table.groupby('feature'):
        aggregate.append(dict(feature=feature,group=rows.group.iloc[0],available_nodes=len(rows),
            training_finite_fraction=float(rows.finite_rows.sum()/rows.training_rows.sum()),
            minimum_node_finite_fraction=float(rows.finite_fraction.min()),nonconstant_nodes=int((rows.training_std>1e-8).sum()),
            present_in_all_development_schemas=feature in common))
    a=pd.DataFrame(aggregate);a.to_csv(OUT/'additional_features_summary.csv',index=False)
    inventory=dict(development_nodes=NODES,training_end_exclusive=pd.Timestamp(TRAIN_END,tz='UTC').isoformat(),
        schema_columns=schemas,common_source_column_count=len(common),candidate_measurements=len(a),
        reserved_node_members_opened=False,validation_measurements_used=False,labels_used=False,
        field_interpretation='Literal source names; units and manufacturer semantics are not inferred for unfamiliar/truncated names.',
        no_utilization_or_network_fields_found=not any('util' in c.lower() or 'network' in c.lower() or 'usage' in c.lower() for c in common))
    (OUT/'training_feature_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    lines=['# Training-only additional-feature audit','',
        f"Inspected only the nine development-node archive members and their rows before {inventory['training_end_exclusive']}. No labels are read by this audit. The development schemas share {len(common)} columns including metadata and target; {len(a)} candidate source measurements are inventoried.",'',
        '| Candidate group | Measurements | Min finite fraction across nodes |','|---|---:|---:|']
    for group,rows in a.groupby('group'):lines.append(f"| {group} | {len(rows)} | {rows.minimum_node_finite_fraction.min():.2%} |")
    lines += ['', 'The archive offers additional per-component electrical readings, power-supply voltage/current, within-interval variability, and individual sensor extrema. Averaging multiple sensors can obscure a faulty device, but that is a hypothesis to test, not evidence of improved model accuracy. No CPU-utilization, memory-usage or network-health fields were found in these schemas. The broad Nagios target may describe faults that this thermal/electrical data cannot explain.','',
        '## Concrete bounded follow-up','',
        '1. Check original label construction and, if accessible, alert-specific fault types against the physical sensor task. Do not substitute sensor-derived labels for the reference outcome.',
        '2. Before training another configuration, declare one extended input set from this training-only inventory: power-supply voltage/current, memory/I/O electrical inputs and compact aggregates of sensor maxima and within-interval variability. Confirm units and sensor validity rules for unfamiliar names.',
        '3. Fit feature preprocessing on the fixed training period. Retain the existing endpoints for a primary matched-cohort comparison; do not let extra missing sensors silently exclude difficult validation cases. Compare the extended supervised LSTM with an extended tree baseline under the same training and alarm constraints.',
        '4. Require useful later-period validation ranking and a feasible operating point before consuming the independent test recordings. If the richer telemetry still fails, the defensible options are a clearly reported negative study or a separately agreed project/dataset change.',
        '', 'This audit does not rank features using validation labels, train an extra model, alter the frozen targets or access reserved servers. Availability and variance do not prove predictive value. The complete source names and per-node training counts are saved in the adjacent JSON/CSV files.']
    (OUT/'TRAINING_FEATURE_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Inventoried',len(a),'additional measurements on training data only')

if __name__=='__main__':main()
