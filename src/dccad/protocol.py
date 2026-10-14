"""Frozen, calendar-aligned development protocol; no holdout predictions here."""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from .data import CADENCE_NS
from .preprocessing import sanitize, causal_fill

TRAIN_END = int(pd.Timestamp('2022-01-22T00:00:00Z').value)
VALIDATION_END = int(pd.Timestamp('2022-04-16T00:00:00Z').value)
FINAL_END = int(pd.Timestamp('2022-09-29T00:00:00Z').value)
HOLDOUT_NODES = [2, 3, 4, 5, 6, 18, 19]


def calendar_split(data, train_end=TRAIN_END, validation_end=VALIDATION_END):
    if train_end >= validation_end:
        raise ValueError('Training must end before validation')
    return np.where(data['time_ns'] < train_end, 0,
                    np.where(data['time_ns'] < validation_end, 1, 2)).astype(np.int8)


def protocol_clean(data):
    split = calendar_split(data)
    sanitized = sanitize(data['raw'], data['features'])
    filled = causal_fill(sanitized, data['node'], data['time_ns'], split)
    return sanitized, filled, split


def cohort_metadata(data, filled, split, which, width=24, stride=1):
    """Known endpoint target; unknown historical labels are not input features."""
    records = []
    for node in np.unique(data['node']):
        idx = np.flatnonzero((data['node'] == node) & (split == which))
        if len(idx) < width:
            continue
        tw = sliding_window_view(data['time_ns'][idx], width)
        labels = sliding_window_view(data['label'][idx], width)
        finite = sliding_window_view(np.isfinite(filled[idx]).all(axis=1), width).all(axis=1)
        keep = finite & (np.diff(tw, axis=1) == CADENCE_NS).all(axis=1)
        keep &= np.isin(labels[:, -1], [0, 2])
        keep &= np.arange(len(keep)) % stride == 0
        starts = np.flatnonzero(keep)
        records.append(pd.DataFrame({'node': node, 'time_ns': tw[keep, -1],
            'row_index': idx[starts+width-1], 'y_end': labels[keep, -1] == 2,
            'history_critical_count': (labels[keep] == 2).sum(axis=1),
            'history_unknown_count': (~np.isin(labels[keep], [0, 2])).sum(axis=1)}))
    columns = ['node','time_ns','row_index','y_end','history_critical_count','history_unknown_count']
    return pd.concat(records,ignore_index=True) if records else pd.DataFrame(columns=columns)


def constrained_policy_key(row):
    """No feasible policy returns None; never silently weaken the fixed targets."""
    required = ['precision','recall','false_positive_rate','false_bursts_per_node_day','event_recall_observable']
    if any(row.get(k) is None or not np.isfinite(row[k]) for k in required):
        return None
    feasible = (row['precision'] >= .30 and row['recall'] >= .40
        and row['false_positive_rate'] <= .01 and row['false_bursts_per_node_day'] <= .10
        and row['event_recall_observable'] >= .60)
    if not feasible:
        return None
    return (row['precision'],row.get('f0_5',0.),-row['false_bursts_per_node_day'],row['recall'])
