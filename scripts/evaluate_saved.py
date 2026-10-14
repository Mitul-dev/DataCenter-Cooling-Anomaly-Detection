"""Audit the original TensorFlow scores against timestamped labels."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import json
import numpy as np
import pandas as pd
import polars as pl
from dccad.data import load_reduced, prepare, windows, raw_episodes
from dccad.evaluation import calibrate, consecutive_alarm, metrics, incident_metrics


def main():
    out = ROOT / "results" / "original_audit"
    out.mkdir(parents=True, exist_ok=True)
    data = load_reduced(ROOT / "0.tar", ROOT / "artifacts" / "reduced_base.npz")
    scaled, split, scaler = prepare(data)
    _, va = windows(data, scaled, split, 1, stride=4, normal_only=True, materialize=False)
    _, te = windows(data, scaled, split, 2, stride=2, materialize=False)
    saved = np.load(ROOT / "scores.npz", allow_pickle=False)
    dash = pl.read_parquet(ROOT / "dashboard" / "dashboard_data.parquet")
    assert len(va) == len(saved["err_va"]), "Validation score alignment cannot be verified"
    np.testing.assert_array_equal(te.node, saved["nte"])
    np.testing.assert_array_equal(te.y_window, saved["yte"])
    np.testing.assert_array_equal(te.time_ns, dash["time"].dt.epoch("ns").to_numpy())
    ep = raw_episodes(data, split)
    ep.to_csv(out / "raw_incidents.csv", index=False)
    te.to_csv(out / "window_metadata.csv", index=False)
    rows, node_rows = [], []
    for name, tag in [("Original LSTM", "err"), ("Original Isolation Forest", "iso")]:
        sv, st = saved[f"{tag}_va"], saved[f"{tag}_te"]
        for q in [95, 99, 99.5]:
            for per_node in [False, True]:
                threshold = calibrate(va, sv, q, per_node)
                for persistence in [1, 2]:
                    pred = consecutive_alarm(te, st, threshold, persistence)
                    events, detail = incident_metrics(te, pred, ep)
                    rows.append({"model": name, "quantile": q, "per_node": per_node,
                                 "persistence": persistence, **metrics(te.y_end, st, pred), **events})
                    if q == 95 and not per_node and persistence == 1:
                        detail.to_csv(out / f"{tag}_incident_details.csv", index=False)
        for n in te.node.unique():
            mask = te.node.to_numpy() == n
            row = {"model":name,"node":int(n),"samples":int(mask.sum()),
                   "positive_endpoints":int(te.y_end.to_numpy()[mask].sum())}
            row.update(metrics(te.y_end.to_numpy()[mask], st[mask], st[mask] > np.percentile(sv,99)))
            # Kept in a separate file to avoid confusing policy and node metrics.
            node_rows.append(row)
    pd.DataFrame(rows).to_csv(out / "alarm_sensitivity.csv", index=False)
    pd.DataFrame(node_rows).to_csv(out / "per_node.csv", index=False)
    summary = {"test_windows":len(te),"window_positives":int(te.y_window.sum()),
               "endpoint_positives":int(te.y_end.sum()),"raw_incidents":len(ep),
               "alignment_verified":True,
               "status":"Fixed policy sensitivity analysis on previously inspected test scores; not validation-selected improvements"}
    (out / "summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
