from __future__ import annotations

import io
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

NODES = [7, 8, 9, 10, 11, 13, 14, 16, 17]
CADENCE_NS = 15 * 60 * 1_000_000_000
BASE_FEATURES = ["ambient"] + [f"fan{i}_{j}" for i in range(4) for j in range(2)] + [
    "fan_disk_power", "p0_vdd_temp", "p1_vdd_temp", "dimm_temp", "p0_core_temp",
    "p1_core_temp", "gpu_core_temp", "gpu_mem_temp", "p0_power", "p1_power",
    "total_power", "ps0_input_power", "ps1_input_power",
]


def load_reduced(archive: Path, cache: Path, enhanced: bool = False):
    """Read archive members directly without extracting arbitrary paths."""
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    blocks = []
    feature_names = BASE_FEATURES.copy()
    extra = ["cpu_temp_max", "gpu_temp_max", "fan_min", "fan_spread",
             "cpu_above_ambient", "gpu_above_ambient", "cpu_temp_change",
             "fan_change", "power_change", "ps0_output_voltage", "ps1_output_voltage"]
    if enhanced:
        feature_names += extra
    with tarfile.open(archive) as tar:
        for node in NODES:
            source = tar.extractfile(f"{node}.parquet")
            if source is None:
                raise FileNotFoundError(f"Node {node} missing from archive")
            raw = pl.read_parquet(io.BytesIO(source.read())).sort("timestamp")
            numeric = {c: raw[c].to_numpy().astype(np.float32) for c in raw.columns
                       if c not in ("timestamp", "value")}

            def column(name):
                return numeric.get(name, np.full(len(raw), np.nan, dtype=np.float32))

            def aggregate(prefix, suffix, operation="mean"):
                cols = [v for c, v in numeric.items() if c.startswith(prefix) and c.endswith(suffix)]
                if not cols:
                    return np.full(len(raw), np.nan, dtype=np.float32)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    fn = np.nanmean if operation == "mean" else np.nanmax
                    return fn(np.column_stack(cols), axis=1).astype(np.float32)

            features = {"ambient": column("ambient_avg")}
            features.update({f"fan{i}_{j}": column(f"fan{i}_{j}_avg") for i in range(4) for j in range(2)})
            for c in ["fan_disk_power", "p0_vdd_temp", "p1_vdd_temp", "p0_power", "p1_power",
                      "total_power", "ps0_input_power", "ps1_input_power"]:
                features[c] = column(c + "_avg")
            for name, prefix, suffix in [("dimm_temp", "dimm", "_temp_avg"),
                                         ("p0_core_temp", "p0_core", "_temp_avg"),
                                         ("p1_core_temp", "p1_core", "_temp_avg"),
                                         ("gpu_core_temp", "gpu", "core_temp_avg"),
                                         ("gpu_mem_temp", "gpu", "mem_temp_avg")]:
                features[name] = aggregate(prefix, suffix)
            times = raw["timestamp"].dt.epoch("ns").to_numpy()
            if enhanced:
                fans = np.column_stack([features[f"fan{i}_{j}"] for i in range(4) for j in range(2)])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    features["cpu_temp_max"] = np.fmax(aggregate("p0_core", "_temp_max", "max"),
                                                         aggregate("p1_core", "_temp_max", "max"))
                    features["gpu_temp_max"] = aggregate("gpu", "core_temp_max", "max")
                    features["fan_min"] = np.nanmin(fans, axis=1)
                    features["fan_spread"] = np.nanmax(fans, axis=1) - features["fan_min"]
                features["cpu_above_ambient"] = np.fmax(features["p0_core_temp"], features["p1_core_temp"]) - features["ambient"]
                features["gpu_above_ambient"] = features["gpu_core_temp"] - features["ambient"]
                for dst, src in [("cpu_temp_change", "p0_core_temp"), ("fan_change", "fan0_0"),
                                 ("power_change", "total_power")]:
                    change = np.r_[np.nan, np.diff(features[src])].astype(np.float32)
                    change[np.r_[True, np.diff(times) != CADENCE_NS]] = np.nan
                    features[dst] = change
                features["ps0_output_voltage"] = column("ps0_output_volta_avg")
                features["ps1_output_voltage"] = column("ps1_output_volta_avg")
            blocks.append({"node": np.full(len(raw), node, dtype=np.int16), "time_ns": times,
                           "label": raw["value"].fill_null(-1).to_numpy().astype(np.int8),
                           "raw": np.column_stack([features[c] for c in feature_names]).astype(np.float32)})
            print(f"Loaded node {node}: {len(raw):,} readings", flush=True)
    result = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    result["features"] = np.array(feature_names)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **result)
    return result


def prepare(data, scaler_kind="standard", clip=5.0):
    from sklearn.preprocessing import RobustScaler
    filled = data["raw"].copy()
    split = np.zeros(len(filled), dtype=np.int8)
    for node in NODES:
        idx = np.flatnonzero(data["node"] == node)
        # Causal filling only. A large timestamp gap resets the fill history.
        segments = np.r_[0, np.flatnonzero(np.diff(data["time_ns"][idx]) != CADENCE_NS) + 1, len(idx)]
        for a, b in zip(segments[:-1], segments[1:]):
            filled[idx[a:b]] = pd.DataFrame(filled[idx[a:b]]).ffill(limit=4).to_numpy()
        k1, k2 = int(.7 * len(idx)), int(.8 * len(idx))
        split[idx[k1:k2]] = 1
        split[idx[k2:]] = 2
    fit = (split == 0) & (data["label"] == 0) & np.isfinite(filled).all(axis=1)
    scaler = (StandardScaler() if scaler_kind == "standard" else RobustScaler()).fit(filled[fit])
    scaled = scaler.transform(filled)
    if clip is not None:
        scaled = np.clip(scaled, -clip, clip)
    return scaled.astype(np.float32), split, scaler


def windows(data, scaled, split, which, width=24, stride=2, normal_only=False, materialize=True):
    xs, records = [], []
    for node in NODES:
        idx = np.flatnonzero((data["node"] == node) & (split == which))
        if len(idx) < width:
            continue
        f = scaled[idx]
        label = data["label"][idx]
        times = data["time_ns"][idx]
        fw = sliding_window_view(f, width, axis=0).transpose(0, 2, 1)
        lw = sliding_window_view(label, width)
        tw = sliding_window_view(times, width)
        keep = np.isfinite(fw).all(axis=(1, 2)) & np.isin(lw, [0, 2]).all(axis=1)
        keep &= (np.diff(tw, axis=1) == CADENCE_NS).all(axis=1)
        keep &= (np.arange(len(keep)) % stride) == 0
        y_window = (lw == 2).any(axis=1)
        if normal_only:
            keep &= ~y_window
        starts = np.flatnonzero(keep)
        if materialize:
            xs.append(fw[keep].copy())
        records.append(pd.DataFrame({"node": node, "time_ns": tw[keep, -1],
                                     "row_index": idx[starts + width - 1],
                                     "y_end": lw[keep, -1] == 2, "y_window": y_window[keep]}))
    return np.concatenate(xs) if xs else None, pd.concat(records, ignore_index=True)


def raw_episodes(data, split, which=2):
    episodes = []
    for node in NODES:
        idx = np.flatnonzero((data["node"] == node) & (split == which))
        flag = data["label"][idx] == 2
        times = data["time_ns"][idx]
        i = 0
        while i < len(idx):
            if not flag[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(idx) and flag[j + 1] and times[j + 1] - times[j] == CADENCE_NS:
                j += 1
            # Each original label describes one 15-minute aggregation bin.
            episodes.append({"node": node, "start_ns": int(times[i]),
                             "end_ns": int(times[j] + CADENCE_NS), "readings": j - i + 1})
            i = j + 1
    return pd.DataFrame(episodes)
