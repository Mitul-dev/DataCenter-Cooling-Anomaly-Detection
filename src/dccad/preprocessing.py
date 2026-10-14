"""Conservative telemetry cleaning and training-only feature selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import CADENCE_NS, BASE_FEATURES


def chronological_split(data):
    split = np.zeros(len(data['raw']), dtype=np.int8)
    for node in np.unique(data['node']):
        idx = np.flatnonzero(data['node'] == node)
        split[idx[int(.7 * len(idx)):int(.8 * len(idx))]] = 1
        split[idx[int(.8 * len(idx)):]] = 2
    return split


def sanitize(values, features):
    """Mark invalid measurements; retain high readings and genuine zero speeds."""
    x = np.asarray(values, dtype=np.float32).copy()
    invalid = ~np.isfinite(x)
    for j, feature in enumerate(features):
        # Project-specific indoor-rack plausibility rules, not sensor specifications.
        if feature in BASE_FEATURES:
            invalid[:, j] |= x[:, j] < 0
    x[invalid] = np.nan
    return x


def causal_fill(values, nodes, times, split, limit=4):
    """No interpolation/backfill; reset at nodes, gaps and split boundaries."""
    filled = values.copy()
    for node in np.unique(nodes):
        idx = np.flatnonzero(nodes == node)
        breaks = (np.diff(times[idx]) != CADENCE_NS) | (np.diff(split[idx]) != 0)
        segments = np.r_[0, np.flatnonzero(breaks) + 1, len(idx)]
        for a, b in zip(segments[:-1], segments[1:]):
            filled[idx[a:b]] = pd.DataFrame(values[idx[a:b]]).ffill(limit=limit).to_numpy()
    return filled


def clean_data(data):
    keys = pd.MultiIndex.from_arrays([data['node'], data['time_ns']])
    if keys.has_duplicates:
        raise ValueError('Duplicate node/timestamp keys require reconciliation before training')
    for node in np.unique(data['node']):
        times = data['time_ns'][data['node'] == node]
        if np.any(np.diff(times) <= 0):
            raise ValueError('Readings must be strictly chronological within each node')
    split = chronological_split(data)
    sanitized = sanitize(data['raw'], data['features'])
    filled = causal_fill(sanitized, data['node'], data['time_ns'], split)
    return sanitized, filled, split


def select_features(data, sanitized, filled, split, correlation_limit=.95, missing_limit=.20):
    """Greedy interpretable selection; no validation/test statistics or labels."""
    normal_train = (split == 0) & (data['label'] == 0)
    train = filled[normal_train]
    names = [str(x) for x in data['features']]
    missing = np.isnan(sanitized[normal_train]).mean(axis=0)
    rows, candidates = [], []
    for j, name in enumerate(names):
        finite = train[:, j][np.isfinite(train[:, j])]
        std = float(np.std(finite, dtype=np.float64)) if len(finite) else 0.
        reason = 'missing_training_fraction' if missing[j] > missing_limit else ('constant' if std <= 1e-8 else None)
        rows.append({'feature': name, 'training_missing_fraction': float(missing[j]),
                     'training_std': std, 'selected': reason is None, 'reason': reason or 'retained',
                     'correlated_with': None, 'absolute_correlation': None})
        if reason is None:
            candidates.append(j)
    # Pairwise finite training readings. Raw units do not affect Pearson correlation.
    corr = pd.DataFrame(train[:, candidates], columns=[names[j] for j in candidates]).corr().abs()
    kept = []
    for j in candidates:
        match = next((k for k in kept if corr.loc[names[j], names[k]] >= correlation_limit), None)
        if match is None:
            kept.append(j)
        else:
            rows[j].update(selected=False, reason='redundant_training_correlation',
                           correlated_with=names[match], absolute_correlation=float(corr.loc[names[j], names[match]]))
    if not kept:
        raise ValueError('Feature selection removed every input')
    return [names[j] for j in kept], pd.DataFrame(rows), corr


def prepare_clean(data, features=None, clip=5.):
    sanitized, filled, split = clean_data(data)
    names = data['features'].tolist() if features is None else features
    indices = [data['features'].tolist().index(c) for c in names]
    values = filled[:, indices]
    fit = (split == 0) & (data['label'] == 0) & np.isfinite(values).all(axis=1)
    scaler = StandardScaler().fit(values[fit])
    scaled = scaler.transform(values)
    if clip is not None:
        scaled = np.clip(scaled, -clip, clip)
    reduced = {k: v for k, v in data.items() if k not in ('raw', 'features')}
    reduced.update(raw=data['raw'][:, indices], features=np.array(names))
    return reduced, scaled.astype(np.float32), split, scaler


def clean_inference(values, features):
    """An uploaded window has no history before its first row."""
    x = sanitize(values, features)
    x = pd.DataFrame(x).ffill(limit=4).to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Missing or invalid readings remain after causal filling (maximum four intervals)')
    return x
