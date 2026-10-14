"""Package the runnable project and evidence, excluding local tooling and raw data."""
from pathlib import Path
import zipfile,json
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'submission'/'DCCAD_Project_Submission.zip'


def main():
    files=[]
    for folder in ['src','scripts','tests','results','dashboard']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':files.append(path)
    for name in ['README.md','requirements.txt','requirements-tested.txt','run_dashboard.ps1','Start Dashboard.bat',
                 'M100_submission.ipynb','M100_explore.ipynb','scores.npz','lstm_threshold_results.csv','node10_error_plot.png']:
        files.append(ROOT/name)
    for name in ['cleaned_telemetry.parquet','selected_telemetry.parquet','quality_indicators.parquet']:
        path=ROOT/'artifacts'/name
        if path.exists():files.append(path)
    files.extend(p for p in (ROOT/'submission').iterdir() if p.is_file() and p.suffix in ['.pdf','.pptx','.md'])
    assert all(p.exists() for p in files)
    staging=ROOT/'.build'/'DCCAD_Project_Submission.zip'
    with zipfile.ZipFile(staging,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(set(files)):z.write(path,'DCCAD/'+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(staging) as z:
        assert z.testzip() is None
        required=['DCCAD/README.md','DCCAD/M100_submission.ipynb','DCCAD/submission/DCCAD_Final_Report.pdf',
                  'DCCAD/submission/DCCAD_Final_Presentation_Reviewed.pptx','DCCAD/results/selected_model.json']
        assert all(n in z.namelist() for n in required)
    staging.replace(OUTPUT)
    (ROOT/'results'/'package_manifest.json').write_text(json.dumps({'file_count':len(set(files)),'bytes':OUTPUT.stat().st_size,
                                                                'raw_archive_omitted':True},indent=2))
    print(f'Packaged {len(set(files))} files, {OUTPUT.stat().st_size/1024**2:.1f} MiB; source archive linked in README')


if __name__=='__main__':main()
