"""Audit, clean, select features and run matched-cohort LSTM experiments."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import polars as pl
from dccad.data import load_reduced, prepare, windows, CADENCE_NS
from dccad.preprocessing import clean_data, select_features, prepare_clean
from dccad.evaluation import selection_key
from train_experiments import run

OUT = ROOT / 'results' / 'feature_selection'


def prepare_workflow():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_reduced(ROOT / '0.tar', ROOT / 'artifacts' / 'reduced_base.npz')
    sanitized, filled, split = clean_data(data)
    selected, decisions, corr = select_features(data, sanitized, filled, split)
    decisions.to_csv(OUT / 'feature_decisions.csv', index=False)
    corr.to_csv(OUT / 'training_correlations.csv')
    names = data['features'].tolist()
    audit = []
    for j, name in enumerate(names):
        raw = data['raw'][:, j]
        newly_invalid = np.isfinite(raw) & np.isnan(sanitized[:, j])
        for value, label in [(0, 'train'), (1, 'validation'), (2, 'test')]:
            mask = split == value
            audit.append({'feature': name, 'split': label, 'readings': int(mask.sum()),
                          'original_missing': int(np.isnan(raw[mask]).sum()),
                          'original_infinite': int(np.isinf(raw[mask]).sum()),
                          'implausible_negative': int(newly_invalid[mask].sum()),
                          'causally_filled': int((np.isnan(sanitized[mask, j]) & np.isfinite(filled[mask, j])).sum()),
                          'remaining_missing': int(np.isnan(filled[mask, j]).sum())})
    pd.DataFrame(audit).to_csv(OUT / 'cleaning_audit.csv', index=False)
    columns = {'node': data['node'], 'time_ns': data['time_ns'], 'label': data['label'],
               'split': np.array(['train', 'validation', 'test'])[split],
               'invalid_sensor_count': (np.isfinite(data['raw']) & np.isnan(sanitized)).sum(axis=1),
               'imputed_sensor_count': (np.isnan(sanitized) & np.isfinite(filled)).sum(axis=1),
               'remaining_missing_count': np.isnan(filled).sum(axis=1)}
    columns.update({name: filled[:, j] for j, name in enumerate(names)})
    pl.DataFrame(columns).write_parquet(ROOT / 'artifacts' / 'cleaned_telemetry.parquet')
    reduced_columns = {k: v for k, v in columns.items() if k not in names or k in selected}
    selected_idx = [names.index(c) for c in selected]
    reduced_columns['invalid_sensor_count'] = (np.isfinite(data['raw'][:, selected_idx]) & np.isnan(sanitized[:, selected_idx])).sum(axis=1)
    reduced_columns['imputed_sensor_count'] = (np.isnan(sanitized[:, selected_idx]) & np.isfinite(filled[:, selected_idx])).sum(axis=1)
    reduced_columns['remaining_missing_count'] = np.isnan(filled[:, selected_idx]).sum(axis=1)
    pl.DataFrame(reduced_columns).write_parquet(ROOT / 'artifacts' / 'selected_telemetry.parquet')
    scaled_old, split_old, _ = prepare(data)
    _, scaled_clean, _, _ = prepare_clean(data)
    cohorts, counts = {}, {}
    for which, key, stride, normal in [(0, 'train', 4, True), (1, 'validation', 2, False), (2, 'test', 2, False)]:
        _, old = windows(data, scaled_old, split_old, which, 8, stride, normal, materialize=False)
        _, new = windows(data, scaled_clean, split, which, 8, stride, normal, materialize=False)
        cohorts[key] = np.intersect1d(old.row_index.to_numpy(), new.row_index.to_numpy())
        counts[key] = {'original_eligible': len(old), 'cleaned_eligible': len(new),
                       'common': len(cohorts[key]),
                       'common_critical_endpoints': int(new.loc[new.row_index.isin(cohorts[key]), 'y_end'].sum())}
    np.savez_compressed(OUT / 'common_cohort.npz', **cohorts)
    cooling = ['ambient', 'fan0_0', 'fan1_0', 'fan1_1', 'p0_vdd_temp', 'p1_vdd_temp',
               'dimm_temp', 'p0_core_temp', 'p1_core_temp', 'gpu_core_temp', 'gpu_mem_temp', 'total_power']
    common = {'width': 8, 'enhanced': False, 'common_cohort_path': 'results/feature_selection/common_cohort.npz'}
    configs = [dict(common, name='feature_reference8'),
               dict(common, name='clean22_8', cleaning=True),
               dict(common, name='correlation8', cleaning=True, selected_features=selected),
               dict(common, name='cooling12_8', cleaning=True, selected_features=cooling)]
    plan = {'selection_fit': 'normal training rows only; neither validation nor test statistics',
            'data_quality': {'rows_preserved': len(data['label']), 'duplicate_node_timestamp_keys': 0,
                             'original_missing_cells': int(np.isnan(data['raw']).sum()),
                             'implausible_negative_cells': int((np.isfinite(data['raw']) & np.isnan(sanitized)).sum()),
                             'causally_filled_cells': int((np.isnan(sanitized) & np.isfinite(filled)).sum()),
                             'remaining_missing_cells': int(np.isnan(filled).sum())},
            'correlation_limit': .95, 'training_missing_limit': .20, 'variance_minimum': 1e-8,
            'selected_features': selected, 'feature_count': len(selected), 'original_feature_count': len(names),
            'cohort': counts, 'configs': configs,
            'rules': ['nonfinite values become missing',
                      'negative temperatures, fan speeds and powers become missing under an indoor-rack plausibility rule',
                      'zero fan speeds/power and high measurements remain unchanged',
                      'forward fill at most four consecutive 15-minute readings; reset at gaps, node and split boundaries',
                      'no backfill, interpolation, removal of rare high readings or invented labels',
                      'unknown labels remain recorded but are excluded from labelled evaluation windows'],
            'limitations': ['removing correlated sensors can hide individual fan faults',
                           'implausible temperature readings can themselves signal a sensor-health anomaly',
                           'previous test results were inspected; new experiments are exploratory',
                           'one seed and one rack subset; no guaranteed performance improvement']}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
    print('TRAINING-ONLY FEATURES:', selected, flush=True)
    print('MATCHED COHORT:', counts, flush=True)
    return plan


def save_example(record):
    folder = ROOT / 'results' / record['name']
    data = load_reduced(ROOT / '0.tar', ROOT / 'artifacts' / 'reduced_base.npz')
    columns = record['features']; indices = [data['features'].tolist().index(c) for c in columns]
    meta = pd.read_csv(folder / 'window_metadata.csv')
    for m in meta.itertuples():
        idx = np.arange(int(m.row_index) - record['width'] + 1, int(m.row_index) + 1)
        raw = data['raw'][idx][:, indices]
        # An entirely valid raw window verifies inference independently of prior fill history.
        if np.isfinite(raw).all() and (raw >= 0).all() and np.all(np.diff(data['time_ns'][idx]) == CADENCE_NS):
            frame = pd.DataFrame(raw, columns=columns)
            frame.insert(0, 'timestamp', pd.to_datetime(data['time_ns'][idx], utc=True).astype(str))
            frame.to_csv(folder / 'example_window.csv', index=False)
            (folder / 'example_metadata.json').write_text(json.dumps({'node': int(m.node), 'test_row': int(m.Index)}))
            return
    raise ValueError('No fresh-inference verification window found')


def complete(records, plan):
    rows = [{'experiment': r['name'], 'features': r['feature_count'], 'validation_f1': r['selected_policy']['f1'],
             'validation_average_precision': r['selected_policy']['average_precision'], **r['test']} for r in records]
    pd.DataFrame(rows).to_csv(OUT / 'comparison.csv', index=False)
    winner = max(records, key=lambda r: selection_key(r['selected_policy']))
    previous = json.loads((ROOT / 'results' / 'selected_model.json').read_text())
    prior_result = json.loads((OUT / 'results.json').read_text()) if (OUT / 'results.json').exists() else {}
    original_default = prior_result.get('previous_default', previous.get('previous_selection', previous['name']))
    previous_record = json.loads((ROOT / 'results' / previous['name'] / 'summary.json').read_text())
    original_record = json.loads((ROOT / 'results' / original_default / 'summary.json').read_text())
    update_default = selection_key(winner['selected_policy']) > selection_key(previous_record['selected_policy'])
    improved = selection_key(winner['selected_policy']) > selection_key(original_record['selected_policy'])
    experiments = list(dict.fromkeys(previous.get('experiments', []) + [r['name'] for r in records]))
    if update_default:
        selected = {'name': winner['name'], 'criterion': 'Validation endpoint F1, then incident recall, then false bursts',
                    'experiments': experiments,
                    'previous_selection': original_default, 'feature_selection_winner': winner['name']}
    else:
        selected = dict(previous, experiments=experiments, feature_selection_winner=winner['name'])
    (ROOT / 'results' / 'selected_model.json').write_text(json.dumps(selected, indent=2))
    result = {'winner_on_matched_cohort': winner['name'], 'default_changed': improved and selected['name']==winner['name'],
              'previous_default': original_default, 'default': selected['name'],
              'comparison': rows, 'plan': plan,
              'selection_note': 'Matched experiment winner and dashboard default use validation only. Existing experiments with other widths may have different coverage.'}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    all_records = [json.loads(path.read_text()) for path in sorted((ROOT / 'results').glob('*/summary.json'))]
    all_rows = [{'experiment': r['name'], 'width': r['width'],
                 'features': r.get('feature_count', 33 if r.get('enhanced') else 22),
                 'validation_f1': r['selected_policy']['f1'], **r['test']}
                for r in all_records if 'test' in r]
    pd.DataFrame(all_rows).to_csv(ROOT / 'results' / 'experiment_comparison.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = [r['experiment'] for r in rows]
    positions = np.arange(len(rows))
    axes[0].bar(positions-.18, [r['validation_f1'] for r in rows], width=.36, label='Validation F1', color='#19638a')
    axes[0].bar(positions+.18, [r['f1'] for r in rows], width=.36, label='Test F1', color='#c67b22')
    axes[0].set_title('Matched-cohort F1')
    axes[1].bar(positions, [r['average_precision'] for r in rows], color='#19638a')
    axes[1].set_title('Test average precision')
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=25, ha='right')
        ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.15)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(OUT / 'comparison.png', dpi=160); plt.close(fig)
    table = ['| Experiment | Features | Validation F1 | Test precision | Test recall | Test F1 | Test AP | False bursts/day |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        table.append(f"| {r['experiment']} | {r['features']} | {r['validation_f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['average_precision']:.4f} | {r['false_bursts_per_node_day']:.3f} |")
    text = '# Data cleaning and feature selection experiment\n\n'
    text += 'All four models use width 8, seed 42, the same model architecture and training settings, and exactly the same training, validation and test endpoints. Feature removal reduces the input/output parameter count. Configuration selection uses validation F1; test results remain exploratory because this recording has already been examined.\n\n'
    text += '## Cleaning\n\n' + '\n'.join('- ' + rule for rule in plan['rules']) + '\n\n'
    text += 'Original readings remain in the source archive and original caches. Cleaned Parquet exports retain every row, labels, split, counts of invalid/imputed/remaining missing sensors, and the sensor values after limited causal filling. Duplicates fail validation for manual reconciliation; none exist in the current dataset. Remaining missing readings are not invented.\n\n'
    text += 'Measured data quality: ' + json.dumps(plan['data_quality']) + '.\n\n'
    text += '## Feature selection\n\n'
    text += f"Normal-training data removed constants (standard deviation <=1e-8), inputs with over 20% missing measurements, and redundant inputs with absolute Pearson correlation >=0.95. The deterministic greedy order follows the original feature list. This produced {len(plan['selected_features'])} inputs: " + ', '.join(plan['selected_features']) + '.\n\n'
    text += 'The cooling candidate is a predefined 12-feature subset: ambient, three distinct fan channels, seven internal temperatures and total power. Correlation is pooled across nodes and may obscure node-specific relationships; selection is not proof that a removed sensor is unnecessary for a physical diagnosis.\n\n'
    text += '## Measured results\n\n' + '\n'.join(table) + '\n\n'
    text += f"Matched-cohort validation winner: **{winner['name']}**. Dashboard default: **{result['default']}**. "
    text += ('The default changed because validation performance improved; test results did not select it.\n\n' if result['default_changed'] else 'The existing default is retained because the new winner did not improve its validation score.\n\n')
    reference = next(r for r in records if r['name'] == 'feature_reference8')
    text += f"The validation F1 difference between the reference ({reference['selected_policy']['f1']:.4f}) and the winner ({winner['selected_policy']['f1']:.4f}) is small and comes from one seed. This is not evidence of a reliable general improvement. Compared with the reference on identical test readings, the winner changes falsely flagged readings from {reference['test']['fp']:,} to {winner['test']['fp']:,}, false-alarm bursts from {reference['test']['false_bursts']:,} to {winner['test']['false_bursts']:,}, incidents caught from {reference['test']['detected']} to {winner['test']['detected']}, and average precision from {reference['test']['average_precision']:.4f} to {winner['test']['average_precision']:.4f}. Fewer false-positive readings can form more separate bursts when alarms are less continuous. Performance therefore involves a tradeoff rather than a universal accuracy improvement.\n\n"
    text += 'Cohort counts: ' + json.dumps(plan['cohort']) + '.\n\n'
    text += '## Limits\n\n' + '\n'.join('- ' + item for item in plan['limitations']) + '\n\n'
    text += 'The existing final PDF and PowerPoint describe the earlier experiments; this file is a supplementary experiment report. Their previous results have not been silently replaced.\n'
    (OUT / 'REPORT.md').write_text(text, encoding='utf-8')
    print('MATCHED-COHORT WINNER', winner['name'], 'DASHBOARD DEFAULT', result['default'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--summarize-only', action='store_true', help='Rebuild the audit and report from completed experiments without retraining')
    p.add_argument('--epochs', type=int, default=30)
    args = p.parse_args()
    plan = prepare_workflow()
    if args.prepare_only:
        return
    records = []
    for config in plan['configs']:
        if args.summarize_only:
            record = json.loads((ROOT / 'results' / config['name'] / 'summary.json').read_text())
        else:
            record = run(config, args.epochs, 512, 42, 4)
        save_example(record)
        records.append(record)
    complete(records, plan)


if __name__ == '__main__':
    main()
