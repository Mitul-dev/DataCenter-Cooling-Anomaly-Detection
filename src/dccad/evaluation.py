from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def consecutive_alarm(meta, score, thresholds, persistence=1, cadence_minutes=30):
    """Reset at node changes and gaps; emit only after k consecutive highs."""
    per_node = isinstance(thresholds, dict)
    if cadence_minutes <= 0 or persistence < 1 or int(persistence) != persistence:
        raise ValueError('Cadence must be positive and persistence a positive integer')
    if len(meta) != len(score):
        raise ValueError('Metadata and scores must have the same length')
    pred = np.zeros(len(score), dtype=bool)
    streak = 0
    previous_node, previous_time = None, None
    for i, (node, time_ns) in enumerate(zip(meta["node"], meta["time_ns"])):
        if node != previous_node or previous_time is None or time_ns - previous_time != cadence_minutes * 60 * 10**9:
            streak = 0
        threshold = thresholds[int(node)] if per_node else thresholds
        streak = streak + 1 if score[i] > threshold else 0
        pred[i] = streak >= persistence
        previous_node, previous_time = node, time_ns
    return pred


def metrics(y, score, pred):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    score = np.asarray(score)
    if y.ndim != 1 or pred.shape != y.shape or score.shape != y.shape:
        raise ValueError('Labels, scores and predictions must be aligned one-dimensional arrays')
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    fp = int((pred & ~y).sum()); tn = int((~pred & ~y).sum())
    tp = int((pred & y).sum())
    return {"roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, score)) if y.any() else None,
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "f0_5": float(1.25*p*r/(.25*p+r)) if .25*p+r else 0.,
            "false_positive_rate": fp/(fp+tn) if fp+tn else None,
            "false_discovery_ratio": fp/(tp+fp) if tp+fp else None,
            "tp": int((pred & y).sum()), "fp": int((pred & ~y).sum()),
            "fn": int((~pred & y).sum()), "tn": int((~pred & ~y).sum())}


def bursts(meta, flag, cadence_minutes=30):
    if cadence_minutes <= 0 or len(meta) != len(flag):
        raise ValueError('Positive cadence and aligned alarm metadata are required')
    result, i = [], 0
    nodes, times = meta["node"].to_numpy(), meta["time_ns"].to_numpy()
    while i < len(flag):
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flag) and flag[j + 1] and nodes[j + 1] == nodes[i] and times[j + 1] - times[j] == cadence_minutes * 60 * 10**9:
            j += 1
        result.append((i, j))
        i = j + 1
    return result


def incident_metrics(meta, pred, episodes, cadence_minutes=30):
    pred = np.asarray(pred, dtype=bool)
    nodes, times = meta["node"].to_numpy(), meta["time_ns"].to_numpy()
    detected, observable, delays = 0, 0, []
    details = []
    for ep in episodes.to_dict("records"):
        inside = (nodes == ep["node"]) & (times >= ep["start_ns"]) & (times < ep["end_ns"])
        opportunities = int(inside.sum())
        hits = np.flatnonzero(inside & pred)
        observable += opportunities > 0
        detected += len(hits) > 0
        delay = float((times[hits[0]] - ep["start_ns"]) / (60 * 10**9)) if len(hits) else None
        if delay is not None:
            delays.append(delay)
        details.append({**ep, "score_opportunities": opportunities, "detected": bool(len(hits)), "delay_minutes": delay})
    alarm_bursts = bursts(meta, pred, cadence_minutes)
    true_bursts = 0
    for a, b in alarm_bursts:
        for ep in episodes.to_dict("records"):
            if nodes[a] == ep["node"] and np.any((times[a:b+1] >= ep["start_ns"]) & (times[a:b+1] < ep["end_ns"])):
                true_bursts += 1
                break
    # Exposure is the duration of scored bins, excluding missing periods.
    node_days = len(meta) * cadence_minutes / (24 * 60)
    false_bursts = len(alarm_bursts) - true_bursts
    return {"cadence_minutes": cadence_minutes,"episodes": len(episodes), "observable_episodes": int(observable), "detected": int(detected),
            "event_recall_all": detected / len(episodes) if len(episodes) else 0.,
            "event_recall_observable": detected / observable if observable else 0.,
            "alarm_bursts": len(alarm_bursts), "false_bursts": false_bursts,
            "burst_precision": true_bursts / len(alarm_bursts) if alarm_bursts else 0.,
            "scored_node_days": node_days, "false_bursts_per_node_day": false_bursts / node_days if node_days else None,
            "median_delay_minutes": float(np.median(delays)) if delays else None}, pd.DataFrame(
                details, columns=['node','start_ns','end_ns','readings','score_opportunities','detected','delay_minutes'])


def calibrate(normal_meta, normal_scores, quantile=99, per_node=False):
    if not per_node:
        return float(np.percentile(normal_scores, quantile))
    return {int(n): float(np.percentile(normal_scores[normal_meta["node"].to_numpy() == n], quantile))
            for n in normal_meta["node"].unique()}


def selection_key(row):
    """Validation-only objective, with deterministic tie breaking."""
    return (row["f1"], row["event_recall_all"], -row["false_bursts_per_node_day"])
