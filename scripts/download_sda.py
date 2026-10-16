"""Download the unmodified public LBNL single-duct AHU source."""
import hashlib
import json
from pathlib import Path
import urllib.request
import argparse

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_SDAHU/LBNL_FDD_Data_Sets_SDAHU.zip'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dual', action='store_true')
    args = parser.parse_args()
    url = URL.replace('SDAHU', 'DDAHU') if args.dual else URL
    folder = ROOT / ('data/dda_benchmark' if args.dual else 'data/sda_benchmark')
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / 'source.zip'
    if not dest.exists():
        partial = dest.with_suffix('.partial')
        with urllib.request.urlopen(url, timeout=90) as response, partial.open('wb') as out:
            total = int(response.headers.get('Content-Length', 0))
            count = 0
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
                count += len(chunk)
                if count % (32 * 1024 * 1024) == 0:
                    print(f'Downloaded {count / 1024**2:.0f} / {total / 1024**2:.0f} MiB', flush=True)
            if total and count != total:
                raise ValueError('Incomplete source download')
        partial.replace(dest)
    digest = hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest()
    record = dict(url=url, bytes=dest.stat().st_size, sha256=digest,
                  doi='10.25984/1881324', downloaded_date='2026-10-16')
    (folder / 'download.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
