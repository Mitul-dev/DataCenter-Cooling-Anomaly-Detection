"""Bundle the primary submission without private runtimes or M100 archives."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'submission/HVAC_Project_Submission.zip'


def main():
    files = set()
    for folder in ['src', 'tests', 'results/hvac_v1', 'data/hvac_benchmark']:
        files.update(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    names = ['README.md', 'PROJECT_IMPROVEMENT_PLAN.md', 'requirements.txt', 'requirements-tested.txt',
             'Start Dashboard.bat', 'run_dashboard.ps1', 'HVAC_submission.ipynb', 'dashboard/app.py',
             'docs/M100_RESEARCH.md', 'results/project_progress.json',
             'submission/HVAC_DL_Project_Report.pdf', 'submission/HVAC_DL_Project_Presentation.pptx',
             'submission/HVAC_artifact_checks.json']
    names += ['scripts/' + n + '.py' for n in ['prepare_hvac', 'train_hvac', 'evaluate_hvac',
              'verify_hvac', 'build_hvac_evidence', 'analyze_thresholds', 'package_hvac', 'run_tests']]
    names += [str(p.relative_to(ROOT)) for p in (ROOT / 'results').glob('task*_*/*.md')]
    for name in names:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.add(path)
    inventory = []
    with ZipFile(OUTPUT, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            archive.writestr('Cooling_Equipment_DL_Project/' + relative, data)
            inventory.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    with ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
    manifest = dict(bundle=OUTPUT.name, bytes=OUTPUT.stat().st_size,
                    sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), files=inventory)
    (ROOT / 'submission/HVAC_package_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Created {OUTPUT.name}: {len(files)} files, {OUTPUT.stat().st_size / 1024**2:.2f} MiB; ZIP integrity passed.')


if __name__ == '__main__':
    main()
