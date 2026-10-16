"""Leakage-safe multiclass sequence models and sensor-specific alarm decisions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, average_precision_score
from .hvac import causal_clean
from .alarm_policies import apply_rule, segment_starts


class SensorClassifier(nn.Module):
    def __init__(self, features, classes, hidden=32):
        super().__init__()
        self.encoder = nn.LSTM(features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Dropout(.2), nn.Linear(16, classes))

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        return self.head(h[-1])


def fit_preprocessing(values, fit, features):
    selected = np.asarray(values, np.float32)[np.asarray(fit)]
    selected = np.where(np.isfinite(selected), selected, np.nan)
    median = np.nanmedian(selected, axis=0)
    if not np.isfinite(median).all():
        raise ValueError('A model input has no finite normal training values')
    complete = np.where(np.isfinite(selected), selected, median)
    scale = complete.std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.
    return dict(features=list(features), median=median.tolist(),
                mean=complete.mean(axis=0, dtype=np.float64).tolist(), scale=scale.tolist(),
                width=30, clip=10., fill_limit=5, cadence_minutes=1)


def transform(values, dates, times, pre):
    raw, filled = causal_clean(values, dates, times, pre['fill_limit'])
    complete = np.where(np.isfinite(filled), filled, np.asarray(pre['median']))
    return np.clip((complete - np.asarray(pre['mean'])) / np.asarray(pre['scale']),
                   -pre['clip'], pre['clip']).astype(np.float32)


def predict(model, x, batch=2048):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            result.append(torch.softmax(model(torch.from_numpy(x[start:start+batch])), dim=1).numpy())
    return np.concatenate(result)


def decisions(meta, probabilities, thresholds, rule='two_of_three'):
    """Per-sensor persistence, then highest softmax score if two sensors qualify."""
    p = np.asarray(probabilities)
    if p.ndim != 2 or len(p) != len(meta) or len(thresholds) != p.shape[1] - 1:
        raise ValueError('Aligned class scores and one threshold per fault type required')
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1., atol=1e-5):
        raise ValueError('Finite normalized class scores required')
    hits = np.column_stack([apply_rule(meta, p[:, j+1], t, rule, 1) for j, t in enumerate(thresholds)])
    eligible_scores = np.where(hits, p[:, 1:], -1.)
    return np.where(hits.any(axis=1), eligible_scores.argmax(axis=1)+1, 0).astype(np.int8)


def classification_metrics(y, pred, probabilities, classes, meta=None):
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    matrix = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    rows = []
    for k, label in enumerate(classes):
        tp = int(matrix[k, k]); fn = int(matrix[k].sum()-tp); fp = int(matrix[:, k].sum()-tp)
        tn = int(matrix.sum()-tp-fn-fp)
        precision = tp/(tp+fp) if tp+fp else 0.
        recall = tp/(tp+fn) if tp+fn else 0.
        rows.append(dict(sensor=label, support=tp+fn, tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=precision, recall=recall,
                         f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                         one_vs_rest_fpr=fp/(fp+tn) if fp+tn else None,
                         ap=float(average_precision_score(y==k, probabilities[:, k]))))
    normal = y == 0
    healthy_fp = int(((pred != 0) & normal).sum())
    result = dict(accuracy=float((y==pred).mean()), macro_f1=float(np.mean([r['f1'] for r in rows])),
                  balanced_accuracy=float(np.mean([r['recall'] for r in rows])),
                  fault_macro_ap=float(np.mean([r['ap'] for r in rows[1:]])),
                  healthy_fpr=healthy_fp/int(normal.sum()) if normal.any() else None,
                  healthy_false_readings=healthy_fp, healthy_readings=int(normal.sum()),
                  fault_detection_recall=float((pred[y != 0] != 0).mean()),
                  confusion_matrix=matrix.tolist(), per_class=rows)
    if meta is not None:
        alarm = pred != 0
        reset = segment_starts(meta, 1)
        rising = alarm & (reset | ~np.r_[False, alarm[:-1]])
        result['healthy_false_bursts'] = int((rising & normal).sum())
        result['healthy_scored_days'] = int(normal.sum()) / 1440
        result['false_bursts_per_healthy_scored_day'] = result['healthy_false_bursts'] / result['healthy_scored_days']
    return result


def score_window(folder, frame):
    folder = Path(folder)
    pre = json.loads((folder / 'preprocessing.json').read_text())
    if len(frame) != pre['width']:
        raise ValueError('Exactly 30 consecutive one-minute readings required')
    times = pd.to_datetime(frame.timestamp, errors='raise')
    if times.isna().any() or times.dt.normalize().nunique() != 1 or not (times.diff().dropna() == pd.Timedelta(minutes=1)).all():
        raise ValueError('One contiguous source day required')
    for field in ['unit', 'scenario']:
        if field in frame and frame[field].nunique(dropna=False) != 1:
            raise ValueError('One equipment unit and scenario required')
    if 'operating' not in frame or not (pd.to_numeric(frame.operating, errors='raise') == 1).all():
        raise ValueError('All 30 readings must satisfy the recorded operating filter')
    values = frame[pre['features']].apply(pd.to_numeric, errors='raise').to_numpy(np.float32)
    x = transform(values, times.dt.strftime('%Y-%m-%d').to_numpy(), times.dt.as_unit('ns').astype('int64').to_numpy(), pre)
    checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
    if checkpoint['features'] != pre['features']:
        raise ValueError('Model and preprocessing features differ')
    model = SensorClassifier(len(pre['features']), len(checkpoint['classes']))
    model.load_state_dict(checkpoint['state_dict'])
    return dict(zip(checkpoint['classes'], predict(model, x[None])[0].astype(float)))
