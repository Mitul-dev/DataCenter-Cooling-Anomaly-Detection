"""Audit the public source before accepting its documented sensor-fault labels."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'data/sda_benchmark'
    out = ROOT / 'results/sda_source_audit'
    out.mkdir(exist_ok=True)
    inventory = []
    differences = []
    statistics = []
    with zipfile.ZipFile(folder / 'source.zip') as archive:
        for info in archive.infolist():
            if info.filename.endswith('.csv'):
                inventory.append(dict(member=info.filename, bytes=info.file_size, crc=info.CRC))
        base_name = next(n for n in archive.namelist() if n.endswith('/AHU_annual.csv'))
        with archive.open(base_name) as handle:
            base = pd.read_csv(handle)
        for record in inventory:
            name = record['member']
            if not ('bias_' in name or name == base_name):
                continue
            with archive.open(name) as handle:
                frame = pd.read_csv(handle)
            digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()
            times = pd.to_datetime(frame.Datetime)
            active = (frame.SYS_CTL == 1) & (frame.SF_SPD_DM == 1)
            record.update(rows=len(frame), first=str(times.iloc[0]), last=str(times.iloc[-1]),
                          duplicate_timestamps=int(times.duplicated().sum()),
                          nonminute_gaps=int((times.diff().dropna() != pd.Timedelta(minutes=1)).sum()),
                          active_rows=int(active.sum()), table_sha256=digest)
            if len(base) != len(frame) or not base.Datetime.equals(frame.Datetime):
                raise ValueError('Source calendars do not match')
            for column in frame.columns[1:]:
                values = frame[column].to_numpy(float)
                delta = values - base[column].to_numpy(float)
                selected = values[active]
                d = delta[active]
                differences.append(dict(member=name, column=column, delta_mean=float(np.nanmean(d)),
                                        delta_sd=float(np.nanstd(d)),
                                        fraction_changed=float((np.abs(d) > 1e-4).mean())))
                statistics.append(dict(member=name, column=column, missing=int((~np.isfinite(values)).sum()),
                                       minimum=float(np.nanmin(selected)), p01=float(np.nanpercentile(selected, 1)),
                                       median=float(np.nanmedian(selected)), p99=float(np.nanpercentile(selected, 99)),
                                       maximum=float(np.nanmax(selected))))
            print(Path(name).name, record['rows'], record['active_rows'], digest[:12], flush=True)
    pd.DataFrame(inventory).to_csv(out / 'archive_inventory.csv', index=False)
    pd.DataFrame(differences).to_csv(out / 'healthy_case_differences.csv', index=False)
    pd.DataFrame(statistics).to_csv(out / 'active_input_ranges.csv', index=False)
    groups = {}
    for record in inventory:
        groups.setdefault((record['bytes'], record['crc']), []).append(Path(record['member']).name)
    duplicates = [members for members in groups.values() if len(members) > 1]
    summary = dict(archive=json.loads((folder / 'download.json').read_text()),
                   csv_count=len(inventory), duplicate_content_groups=duplicates,
                   documented_sa_members_present=False,
                   undocumented_coi_bias_members_present=True,
                   training_started=False, labels_accepted=False)
    (out / 'audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
