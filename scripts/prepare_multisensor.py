"""Audit and partition native four-sensor AHU fault scenarios by calendar week."""
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'results/multisensor_v1'
DATA = ROOT / 'data/dda_benchmark'
FEATURES = ['CSA_TEMP', 'HSA_TEMP', 'CSA_SP', 'HSA_SP', 'OA_TEMP', 'MA_TEMP', 'RA_TEMP',
            'OA_HUMD', 'RA_HUMD', 'CSA_HUMD', 'HSA_HUMD', 'CSA_CFM', 'HSA_CFM',
            'OA_CFM', 'RA_CFM', 'CSF_SPD', 'HSF_SPD', 'RF_SPD', 'CSF_WAT', 'HSF_WAT',
            'RF_WAT', 'OA_DMPR_DM', 'RA_DMPR_DM', 'CHWC_VLV_DM', 'HWC_VLV_DM', 'EA_DMPR_DM']
CLASSES = ['Healthy', 'Cold-deck temperature sensor', 'Hot-deck temperature sensor',
           'Cold-deck pressure sensor', 'Hot-deck pressure sensor']
CODES = ['normal', 'CSA', 'HSA', 'CSP', 'HSP']


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def partition(times):
    week = times.dt.isocalendar().week.to_numpy(int)
    return np.where(week % 6 == 0, 'test', np.where(week % 6 == 5, 'validation', 'train'))


def endpoints(frame):
    active = frame.operating.to_numpy(bool)
    time = frame.timestamp.dt.as_unit('ns').astype('int64').to_numpy()
    date = frame.timestamp.dt.normalize().to_numpy()
    from numpy.lib.stride_tricks import sliding_window_view
    valid = sliding_window_view(active, 30).all(axis=1)
    valid &= (time[29:]-time[:-29] == 29*60*10**9)
    valid &= date[29:] == date[:-29]
    return np.flatnonzero(valid) + 29


def main():
    OUT.mkdir(exist_ok=True)
    archive_hash = sha(DATA / 'source.zip')
    source = json.loads((DATA / 'download.json').read_text())
    assert archive_hash == source['sha256']
    plan = dict(study='multisensor_v1', created='2026-10-16', source=source,
                source_page='https://faultdetection.lbl.gov/dataset/simulated-dd-ahu-dataset/',
                reference='https://doi.org/10.1038/s41597-023-02197-w',
                scope='Native simulated single-sensor biases on one dual-duct AHU; no simultaneous faults',
                classes=CLASSES, candidate_inputs=FEATURES, width=30, cadence_minutes=1,
                partition='ISO week modulo 6: 0 test, 5 validation, all others train; identical calendar split across every scenario',
                training_stride_minutes=60, checkpoint_validation_stride_minutes=15,
                policy_and_final_evaluation_stride_minutes=1,
                operating_filter='SYS_CTL == 1 and both cold/hot supply fan speeds > 0',
                primary_seed=42, seeds=[42], hidden=32, batch=512, learning_rate=.001,
                maximum_epochs=20, patience=4, criterion='maximum validation fault macro average precision',
                feature_selection='Training-only internal whole-week holdout permutation importance; top five raw inputs',
                threshold_selection='Validation only; immediate/consecutive_2/two_of_three; healthy percentile thresholds shared quantile across fault types',
                selection_targets=dict(healthy_fpr=.01, minimum_recall_each_sensor=.4),
                model_family_selection='Highest validation macro F1 among feasible families; else highest macro F1 diagnostic, not qualified',
                test_not_used_for_selection=True, original_model_retained=True,
                pressure_severity_note='Native filenames say +/-2inwg and +/-4inwg; paper Table 4 documents +/-0.2 and +/-0.4 inH2O. Filename tokens retained without claiming verified magnitudes.')
    path = OUT / 'experiment_plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan, 'Existing study protocol differs'
    else:
        write(path, plan)
    rows = []; ranges = []; cohort_counts = []
    with zipfile.ZipFile(DATA / 'source.zip') as archive:
        all_members = [i for i in archive.infolist() if i.filename.endswith('.csv')]
        inventory = [dict(member=i.filename, bytes=i.file_size, crc=i.CRC) for i in all_members]
        write(OUT / 'archive_inventory.json', inventory)
        members = [i for i in all_members if 'SensorBias_' in i.filename or 'FaultFree' in i.filename]
        assert len(members) == 17
        signatures = {(i.file_size, i.CRC) for i in members}
        assert len(signatures) == 17, 'Selected scenario files duplicate archive content'
        for number, info in enumerate(members):
            name = Path(info.filename).stem
            found = re.search(r'SensorBias_(CSA|HSA|CSP|HSP)_([+-][24])', name)
            label = CODES.index(found[1]) if found else 0
            cache = DATA / f'{name}.parquet'
            if not cache.exists():
                with archive.open(info.filename) as handle:
                    frame = pd.read_csv(handle, usecols=['Datetime', 'SYS_CTL']+FEATURES,
                                        dtype={c:'float32' for c in FEATURES+['SYS_CTL']})
                frame['timestamp'] = pd.to_datetime(frame.pop('Datetime'), format='%m/%d/%Y %H:%M')
                assert not frame.timestamp.duplicated().any()
                assert frame.timestamp.is_monotonic_increasing
                assert (frame.timestamp.diff().dropna()==pd.Timedelta(minutes=1)).all()
                frame['operating'] = (frame.SYS_CTL==1) & (frame.CSF_SPD>0) & (frame.HSF_SPD>0)
                frame['partition'] = partition(frame.timestamp)
                frame.to_parquet(cache, index=False)
            frame = pd.read_parquet(cache)
            end = endpoints(frame)
            meta = pd.DataFrame(dict(row_index=end, timestamp=frame.timestamp.iloc[end].to_numpy(),
                                     y=label, partition=frame.partition.iloc[end].to_numpy()))
            meta['node'] = number*1000 + meta.timestamp.dt.dayofyear
            meta['time_ns'] = meta.timestamp.dt.as_unit('ns').astype('int64')
            meta['date'] = meta.timestamp.dt.strftime('%Y-%m-%d')
            meta['scenario'] = name
            meta.to_parquet(DATA / f'{name}_cohort.parquet', index=False)
            for split, g in meta.groupby('partition'):
                cohort_counts.append(dict(scenario=name, sensor=CLASSES[label], partition=split,
                                          raw_days=int(frame.loc[frame.partition==split, 'timestamp'].dt.normalize().nunique()),
                                          scored_days=g.date.nunique(), endpoints=len(g)))
            # Range summaries use training rows only, avoiding held-out value inspection.
            active = frame.operating & (frame.partition=='train')
            for column in FEATURES:
                values = frame.loc[active, column].to_numpy()
                ranges.append(dict(scenario=name, sensor=CLASSES[label], column=column,
                                   finite=int(np.isfinite(values).sum()), missing=int((~np.isfinite(values)).sum()),
                                   minimum=float(np.nanmin(values)), median=float(np.nanmedian(values)),
                                   maximum=float(np.nanmax(values)), std=float(np.nanstd(values))))
            rows.append(dict(member=info.filename, scenario=name, class_id=label, sensor=CLASSES[label],
                             severity_filename_token=found[2] if found else None,
                             rows=len(frame), first=str(frame.timestamp.iloc[0]), last=str(frame.timestamp.iloc[-1]),
                             raw_inputs=len(FEATURES), cache_sha256=sha(cache),
                             cohort_sha256=sha(DATA / f'{name}_cohort.parquet')))
            print('Prepared', name, len(frame), 'rows;', len(end), 'eligible windows', flush=True)
    write(OUT / 'scenarios.json', rows)
    pd.DataFrame(ranges).to_csv(OUT / 'training_ranges.csv', index=False)
    pd.DataFrame(cohort_counts).to_csv(OUT / 'cohort_counts.csv', index=False)
    write(OUT / 'preparation.json', dict(scenarios=17, sensor_fault_types=4, classes=5,
                                        source_csv_files=len(all_members), source_archive_sha256=archive_hash,
                                        plan_sha256=sha(path), no_random_row_split=True,
                                        models_trained=False, test_predictions_made=False))
    print(pd.DataFrame(cohort_counts).groupby('partition').endpoints.sum().to_string(), flush=True)


if __name__ == '__main__':
    main()
