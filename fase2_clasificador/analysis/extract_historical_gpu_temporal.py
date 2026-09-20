"""Extrae descriptores temporales NVML de corridas GPU ya existentes.

No usa identidad, etiqueta, OI ni ridge para recortar la serie. Conserva el
intervalo entre la primera y última muestra con actividad NVML observable y
produce una sola fila por corrida, compatible con validación por familia.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SIGNALS = ("gpu_util_pct", "gpu_mem_util_pct", "gpu_power_mw", "gpu_sm_clock_mhz")
USECOLS = ("run_id", "timestamp_ns", *SIGNALS)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return 0.0
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _lag_corr(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    if lag > 0:
        return _safe_corr(a[:-lag], b[lag:])
    if lag < 0:
        return _safe_corr(a[-lag:], b[:lag])
    return _safe_corr(a, b)


def _burst_stats(mask: np.ndarray) -> tuple[float, float, float]:
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    lengths = ends - starts
    if not len(lengths):
        return 0.0, 0.0, 0.0
    return float(len(lengths)), float(np.mean(lengths)), float(np.max(lengths))


def temporal_features(samples: pd.DataFrame) -> dict[str, float]:
    gpu = samples.dropna(subset=list(SIGNALS), how="all").sort_values("timestamp_ns").reset_index(drop=True)
    if len(gpu) < 3:
        raise ValueError("menos de tres muestras GPU")
    util = pd.to_numeric(gpu["gpu_util_pct"], errors="coerce").to_numpy(float)
    mem = pd.to_numeric(gpu["gpu_mem_util_pct"], errors="coerce").to_numpy(float)
    # Regla no supervisada: elimina únicamente colas totalmente inactivas.
    active = np.nan_to_num(util) > 0
    active |= np.nan_to_num(mem) > 0
    if active.any():
        indices = np.flatnonzero(active)
        gpu = gpu.iloc[indices[0]:indices[-1] + 1].reset_index(drop=True)
    arrays = {signal: pd.to_numeric(gpu[signal], errors="coerce").to_numpy(float) for signal in SIGNALS}
    ts = pd.to_numeric(gpu["timestamp_ns"], errors="coerce").to_numpy(float)
    result: dict[str, float] = {
        "temporal_n_samples": float(len(gpu)),
        "temporal_duration_s": float((ts[-1] - ts[0]) / 1e9),
    }
    dt = np.diff(ts) / 1e6
    result["temporal_cadence_ms_median"] = float(np.nanmedian(dt))
    result["temporal_cadence_ms_iqr"] = float(np.nanpercentile(dt, 75) - np.nanpercentile(dt, 25))
    for signal, values in arrays.items():
        finite = values[np.isfinite(values)]
        diffs = np.diff(values)
        diffs = diffs[np.isfinite(diffs)]
        prefix = f"temporal_{signal}"
        result[f"{prefix}_mean_abs_diff"] = float(np.mean(np.abs(diffs))) if len(diffs) else 0.0
        result[f"{prefix}_change_fraction"] = float(np.mean(np.abs(diffs) > 0)) if len(diffs) else 0.0
        result[f"{prefix}_lag1_autocorr"] = _lag_corr(values[:-1], values[1:], 0) if len(values) > 2 else 0.0
        if len(finite) > 2 and np.std(finite) > 0:
            x = np.linspace(-1.0, 1.0, len(values))
            valid = np.isfinite(values)
            result[f"{prefix}_trend_per_span"] = float(np.polyfit(x[valid], values[valid], 1)[0])
        else:
            result[f"{prefix}_trend_per_span"] = 0.0
    util, mem, power = arrays["gpu_util_pct"], arrays["gpu_mem_util_pct"], arrays["gpu_power_mw"]
    result["temporal_gpu_active_fraction"] = float(np.mean(np.nan_to_num(util) > 0))
    result["temporal_gpu_high_fraction"] = float(np.mean(np.nan_to_num(util) >= 80))
    result["temporal_mem_active_fraction"] = float(np.mean(np.nan_to_num(mem) > 0))
    result["temporal_mem_high_fraction"] = float(np.mean(np.nan_to_num(mem) >= 50))
    for name, mask in (("gpu", np.nan_to_num(util) > 0), ("mem", np.nan_to_num(mem) > 0)):
        count, mean_length, max_length = _burst_stats(mask)
        result[f"temporal_{name}_burst_count_per_1k"] = count * 1000.0 / len(mask)
        result[f"temporal_{name}_burst_mean_fraction"] = mean_length / len(mask)
        result[f"temporal_{name}_burst_max_fraction"] = max_length / len(mask)
    result["temporal_corr_util_mem"] = _safe_corr(util, mem)
    result["temporal_corr_util_power"] = _safe_corr(util, power)
    result["temporal_corr_mem_power"] = _safe_corr(mem, power)
    lag_values = [_lag_corr(util, power, lag) for lag in range(-5, 6)]
    result["temporal_util_power_best_abs_lag_corr"] = float(max(lag_values, key=abs))
    result["temporal_util_power_best_lag"] = float(range(-5, 6)[int(np.argmax(np.abs(lag_values)))])
    return result


def build_dataset(raw_root: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    by_run = metadata.set_index("run_id", drop=False)
    target_run_dirs = {Path(str(value)).parent.name for value in metadata["source_file"]}
    paths_by_run_dir: dict[str, Path] = {}
    for path in raw_root.rglob("samples.csv"):
        if path.parent.name not in target_run_dirs:
            continue
        if path.parent.name in paths_by_run_dir:
            raise ValueError(f"directorio de corrida duplicado: {path.parent.name}")
        paths_by_run_dir[path.parent.name] = path
    rows = []
    for metadata_row in metadata.itertuples():
        run_dir = Path(str(metadata_row.source_file)).parent.name
        path = paths_by_run_dir.get(run_dir)
        if path is None:
            continue
        samples = pd.read_csv(path, usecols=list(USECOLS), low_memory=False)
        run_ids = samples["run_id"].dropna().astype(str).unique()
        if len(run_ids) != 1 or run_ids[0] not in by_run.index:
            continue
        base = by_run.loc[run_ids[0]].to_dict()
        base.update(temporal_features(samples))
        base["temporal_source_file"] = str(path)
        rows.append(base)
    result = pd.DataFrame(rows)
    if len(result) != len(metadata) or result["run_id"].nunique() != len(metadata):
        raise ValueError(f"cobertura temporal incompleta: {len(result)}/{len(metadata)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-family", action="append", default=[])
    args = parser.parse_args()
    metadata = pd.read_csv(args.metadata, low_memory=False)
    metadata = metadata[~metadata["kernel_family"].isin(args.exclude_family)].copy()
    result = build_dataset(args.raw_root, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"{len(result)} corridas temporales escritas en {args.output}")


if __name__ == "__main__":
    main()
