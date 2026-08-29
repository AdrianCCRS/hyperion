"""Construccion auditable del dataset cold/warm del selector CPU/GPU."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import math
import re
import subprocess

import numpy as np
import pandas as pd
import yaml


IDLE_GPU_POWER_W = 34.8379
IDLE_GPU_POWER_SOURCE_JOB = "6714"
CONTRACT_VERSION = "cold_warm_v1"
EXPECTED_REPETITIONS = 3
CPU_LEVELS = ("REF", "F0", "F1", "F2", "F3", "F4", "F5", "F6")
GPU_HOST_LEVELS = ("REF", "F0", "F3", "F6")
GPU_LEVELS = CPU_LEVELS

CPU_TELEMETRY = (
    "ipc", "mpki", "llc_miss_rate", "stall_backend_ratio", "ips",
    "freq_khz_observed", "running_ratio",
)
GPU_TELEMETRY = (
    "gpu_power_mw", "gpu_util_pct", "gpu_mem_util_pct",
    "gpu_sm_clock_mhz", "gpu_temperature_c",
)


@dataclass(frozen=True)
class BuildConfig:
    cpu_campaign_dir: Path
    catalog_path: Path
    output_dir: Path
    cpu_manifest_path: Path | None = None
    gpu_campaign_dir: Path | None = None
    gpu_manifest_path: Path | None = None
    mode: str = "cpu-provisional"
    idle_gpu_power_w: float = IDLE_GPU_POWER_W
    idle_gpu_power_source_job: str = IDLE_GPU_POWER_SOURCE_JOB
    expected_repetitions: int = EXPECTED_REPETITIONS

    def __post_init__(self) -> None:
        if self.mode not in {"cpu-provisional", "final"}:
            raise ValueError(f"modo desconocido: {self.mode}")
        if self.mode == "final" and self.gpu_campaign_dir is None:
            raise ValueError("modo final exige gpu_campaign_dir")
        if self.expected_repetitions <= 0:
            raise ValueError("expected_repetitions debe ser positivo")


class DatasetContractError(RuntimeError):
    """Los artefactos no cumplen el contrato necesario para etiquetar."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise DatasetContractError(f"JSON no es objeto: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise DatasetContractError(f"YAML no es objeto: {path}")
    return payload


def _catalog_entries(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_yaml(path)
    entries = payload.get("kernels")
    if not isinstance(entries, list):
        raise DatasetContractError(f"catalogo sin lista kernels: {path}")
    return {str(entry["id"]): entry for entry in entries if isinstance(entry, dict) and "id" in entry}


def _operation_and_size(config_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"([a-z0-9_]+)_N(\d+)", config_id)
    if not match:
        raise DatasetContractError(f"config_id dual invalido: {config_id}")
    return match.group(1), int(match.group(2))


def _iterations(entry: Mapping[str, Any], metadata: Mapping[str, Any]) -> int:
    value = metadata.get("iterations")
    # ``metadata.iterations`` pertenece al launcher historico y vale 0 en
    # modo --exec; las iteraciones reales del binario dual viven en exec_args.
    if value is not None and int(value) > 0:
        result = int(value)
    else:
        match = re.search(r"(?:^|\s)--iterations\s+(\d+)(?:\s|$)", str(entry.get("exec_args", "")))
        if not match:
            raise DatasetContractError(f"kernel sin --iterations: {entry.get('id')}")
        result = int(match.group(1))
    if result <= 0:
        raise DatasetContractError(f"iterations no positivo: {result}")
    return result


def action_id(device: str, cpu_level: str, gpu_level: str | None = None) -> str:
    if device == "cpu":
        if gpu_level not in {None, "", "nan"}:
            raise DatasetContractError("accion CPU no puede declarar nivel GPU")
        return f"cpu:{cpu_level}"
    if device == "gpu" and gpu_level:
        return f"gpu:{cpu_level}:{gpu_level}"
    raise DatasetContractError(f"accion invalida: device={device}, gpu_level={gpu_level}")


def expected_actions(mode: str) -> tuple[str, ...]:
    cpu = tuple(action_id("cpu", level) for level in CPU_LEVELS)
    if mode == "cpu-provisional":
        return cpu
    gpu = tuple(action_id("gpu", cpu_level, gpu_level)
                for cpu_level in GPU_HOST_LEVELS for gpu_level in GPU_LEVELS)
    return cpu + gpu


def configured_actions(config: BuildConfig) -> tuple[str, ...]:
    """Deriva el espacio de acciones de los manifiestos y verifica el contrato."""
    if config.cpu_manifest_path is None:
        actions = expected_actions(config.mode)
    else:
        cpu_manifest = _load_yaml(config.cpu_manifest_path)
        cpu_levels = tuple(str(item["id"]) for item in cpu_manifest.get("frequency_levels", []))
        actions = tuple(action_id("cpu", level) for level in cpu_levels)
        if config.mode == "final":
            if config.gpu_manifest_path is None:
                raise DatasetContractError("modo final exige gpu_manifest_path para derivar acciones")
            gpu_manifest = _load_yaml(config.gpu_manifest_path)
            host_levels = tuple(str(item["id"]) for item in gpu_manifest.get("frequency_levels", []))
            gpu_levels = tuple(str(item["id"]) for item in gpu_manifest.get("gpu_frequency_levels", []))
            actions += tuple(action_id("gpu", host, gpu) for host in host_levels for gpu in gpu_levels)
    canonical = set(expected_actions(config.mode))
    if set(actions) != canonical:
        raise DatasetContractError(
            f"espacio de acciones del manifiesto no coincide: observado={sorted(actions)}, esperado={sorted(canonical)}"
        )
    return actions


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _overlap_weights(frame: pd.DataFrame, t0_ns: int, t1_ns: int) -> tuple[pd.Series, pd.Series]:
    starts = _numeric(frame, "t_start_ns")
    ends = _numeric(frame, "t_end_ns")
    interval = ends - starts
    overlap = np.maximum(0.0, np.minimum(ends, float(t1_ns)) - np.maximum(starts, float(t0_ns)))
    overlap = pd.Series(overlap, index=frame.index, dtype=float)
    fraction = pd.Series(np.where(interval > 0, overlap / interval, 0.0), index=frame.index, dtype=float)
    return overlap, fraction.clip(lower=0.0, upper=1.0)


def _with_gpu_interval_starts(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye el inicio del delta NVML desde la muestra previa.

    ``postprocess.py`` persiste las filas GPU con ``t_end_ns=timestamp`` y
    ``t_start_ns`` vacio. El delta de energia, sin embargo, corresponde al
    intervalo entre esa muestra y la GPU anterior. La reconstruccion usa
    exclusivamente el orden de las filas NVML; no mezcla ticks CPU.
    """
    out = frame.copy()
    if "t_start_ns" not in out:
        out["t_start_ns"] = np.nan
    gpu_rows = _numeric(out, "gpu_power_mw").notna() & _numeric(out, "t_end_ns").notna()
    ordered = out.loc[gpu_rows].sort_values("t_end_ns").index
    previous = _numeric(out.loc[ordered], "t_end_ns").shift(1)
    missing = _numeric(out.loc[ordered], "t_start_ns").isna()
    out.loc[ordered[missing], "t_start_ns"] = previous[missing].to_numpy()
    return out


def _interval_union_ns(frame: pd.DataFrame, mask: pd.Series, t0_ns: int, t1_ns: int) -> float:
    intervals: list[tuple[float, float]] = []
    starts = _numeric(frame, "t_start_ns")
    ends = _numeric(frame, "t_end_ns")
    for start, end in zip(starts[mask], ends[mask]):
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        left, right = max(float(start), t0_ns), min(float(end), t1_ns)
        if right > left:
            intervals.append((left, right))
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    left, right = intervals[0]
    for next_left, next_right in intervals[1:]:
        if next_left > right:
            total += right - left
            left, right = next_left, next_right
        else:
            right = max(right, next_right)
    return total + right - left


def _weighted_mean(frame: pd.DataFrame, column: str, weights: pd.Series) -> float:
    values = _numeric(frame, column)
    valid = values.notna() & np.isfinite(values) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def integrate_region(
    windows: pd.DataFrame,
    *,
    t0_ns: int,
    t1_ns: int,
    device: str,
    iterations: int,
    region: str,
    interval_ns: int,
    gpu_interval_ns: int,
    idle_gpu_power_w: float = IDLE_GPU_POWER_W,
) -> dict[str, Any]:
    """Integra una region usando la fraccion real de cada ventana."""
    if region not in {"cold", "warm"} or t1_ns <= t0_ns:
        raise DatasetContractError(f"region invalida {region}: {t0_ns}..{t1_ns}")
    if device not in {"cpu", "gpu"}:
        raise DatasetContractError(f"device invalido: {device}")
    windows = _with_gpu_interval_starts(windows)
    overlap, fraction = _overlap_weights(windows, t0_ns, t1_ns)
    duration_s = (t1_ns - t0_ns) / 1e9

    energy_valid = _bool_series(windows.get("energy_valid", pd.Series(False, index=windows.index)))
    pkg_uj = _numeric(windows, "pkg_delta_uj")
    dram_uj = _numeric(windows, "dram_delta_uj")
    rapl_values_uj = pkg_uj + dram_uj
    rapl_mask = energy_valid & pkg_uj.notna() & dram_uj.notna() & (overlap > 0)
    if not rapl_mask.any():
        raise DatasetContractError("region sin energia RAPL package+DRAM valida")
    rapl_j = float((rapl_values_uj[rapl_mask] * fraction[rapl_mask]).sum() / 1e6)

    gpu_valid = _bool_series(windows.get("gpu_energy_valid", pd.Series(False, index=windows.index)))
    gpu_delta_mj = _numeric(windows, "gpu_energy_delta_mj")
    gpu_mask = gpu_valid & gpu_delta_mj.notna() & (overlap > 0)
    raw_gpu_j = float((gpu_delta_mj[gpu_mask] * fraction[gpu_mask]).sum() / 1e3)
    if device == "cpu":
        gpu_j = idle_gpu_power_w * duration_s
        gpu_source = "idle_baseline"
    else:
        if not gpu_mask.any():
            raise DatasetContractError("corrida GPU sin energia NVML valida en la region")
        gpu_j = raw_gpu_j
        gpu_source = "nvml_integrated"

    total_j = rapl_j + gpu_j
    dispatches = 1 if region == "cold" else iterations
    time_per_dispatch = duration_s / dispatches
    energy_per_dispatch = total_j / dispatches
    rapl_coverage = _interval_union_ns(windows, rapl_mask, t0_ns, t1_ns) / (t1_ns - t0_ns)
    gpu_coverage = _interval_union_ns(windows, gpu_mask, t0_ns, t1_ns) / (t1_ns - t0_ns)
    resolution_ns = gpu_interval_ns if device == "gpu" else interval_ns

    result: dict[str, Any] = {
        "region": region,
        "region_t0_ns": t0_ns,
        "region_t1_ns": t1_ns,
        "duration_total_s": duration_s,
        "dispatch_count": dispatches,
        "time_per_dispatch_s": time_per_dispatch,
        "rapl_energy_total_j": rapl_j,
        "gpu_energy_total_j": gpu_j,
        "gpu_energy_raw_observer_j": raw_gpu_j,
        "total_energy_j": total_j,
        "energy_per_dispatch_j": energy_per_dispatch,
        "edp_per_dispatch_js": energy_per_dispatch * time_per_dispatch,
        "gpu_energy_source": gpu_source,
        "rapl_coverage_fraction": min(1.0, rapl_coverage),
        "gpu_coverage_fraction": min(1.0, gpu_coverage),
        "sampling_resolution_ns": resolution_ns,
        "region_to_sampling_ratio": (t1_ns - t0_ns) / resolution_ns,
        "energy_resolution_status": "low" if (t1_ns - t0_ns) < resolution_ns else "nominal",
    }
    for column in (*CPU_TELEMETRY, *GPU_TELEMETRY):
        result[column] = _weighted_mean(windows, column, overlap)
    return result


def _validate_timing(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[int, ...]]:
    if metadata.get("dispatch_timing_contract_valid") is not True:
        raise DatasetContractError("dispatch_timing_contract_valid != true")
    timing = metadata.get("dispatch_timing")
    if not isinstance(timing, dict) or timing.get("contract_version") != CONTRACT_VERSION:
        raise DatasetContractError("contrato temporal ausente o incompatible")
    names = ("cold_t0_ns", "setup_complete_ns", "cold_t1_ns", "warm_t0_ns", "warm_t1_ns")
    values = tuple(int(timing[name]) for name in names)
    if values != tuple(sorted(values)) or values[0] == values[2] or values[3] == values[4]:
        raise DatasetContractError(f"marcadores no monotonicos: {values}")
    return timing, values


def _accepted_run_dirs(campaign_dir: Path) -> list[Path]:
    campaign_meta = _read_json(campaign_dir / "campaign_metadata.json")
    accepted = campaign_meta.get("accepted_run_ids")
    if not isinstance(accepted, list):
        raise DatasetContractError(f"campaign_metadata sin accepted_run_ids: {campaign_dir}")
    if len(set(map(str, accepted))) != len(accepted):
        raise DatasetContractError(f"accepted_run_ids duplicados: {campaign_dir}")
    result: list[Path] = []
    for run_id in accepted:
        run_dir = campaign_dir / str(run_id)
        verdict = _read_json(run_dir / "verdict.json")
        if verdict.get("accepted") is not True:
            raise DatasetContractError(f"accepted_run_ids contradice verdict: {run_id}")
        result.append(run_dir)
    return result


def read_campaign_runs(
    campaign_dir: Path,
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    idle_gpu_power_w: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    accepted_dirs = _accepted_run_dirs(campaign_dir)
    seen_keys: set[tuple[str, str, int]] = set()
    for run_dir in accepted_dirs:
        try:
            metadata = _read_json(run_dir / "metadata.json")
            kernel_ref = str(metadata["kernel_ref"])
            entry = catalog.get(kernel_ref)
            if not entry or not entry.get("config_id"):
                raise DatasetContractError(f"kernel no dual/no catalogado: {kernel_ref}")
            config_id = str(entry["config_id"])
            operation, size = _operation_and_size(config_id)
            device = str(entry.get("device", "cpu"))
            cpu_level = str(metadata["freq_level_id"])
            gpu_value = metadata.get("gpu_freq_level_id")
            gpu_level = None if gpu_value in {None, "", "nan"} else str(gpu_value)
            action = action_id(device, cpu_level, gpu_level)
            repetition = int(metadata["repetition_index"])
            duplicate_key = (config_id, action, repetition)
            if duplicate_key in seen_keys:
                raise DatasetContractError(f"corrida duplicada: {duplicate_key}")
            seen_keys.add(duplicate_key)
            timing, values = _validate_timing(metadata)
            iterations = _iterations(entry, metadata)
            windows = pd.read_csv(run_dir / "windows.csv", low_memory=False)
            common = {
                "run_id": str(metadata["run_id"]),
                "campaign_id": str(metadata.get("campaign_id", campaign_dir.name)),
                "kernel_ref": kernel_ref,
                "config_id": config_id,
                "operation": operation,
                "size": size,
                "family": "vector" if operation in {"axpy", "spmv"} else "matrix",
                "device": device,
                "action_id": action,
                "cpu_level": cpu_level,
                "gpu_level": gpu_level,
                "repetition": repetition,
                "iterations": iterations,
                "binary_checksum": metadata.get("binary_checksum"),
                "node_id": metadata.get("node_id"),
                "freq_khz_requested": metadata.get("freq_khz_requested"),
                "freq_khz_applied": metadata.get("freq_khz_applied"),
                "contract_version": timing["contract_version"],
            }
            regions = {
                "cold": (values[0], values[2]),
                "warm": (values[3], values[4]),
            }
            for region, (t0_ns, t1_ns) in regions.items():
                integrated = integrate_region(
                    windows, t0_ns=t0_ns, t1_ns=t1_ns, device=device,
                    iterations=iterations, region=region,
                    interval_ns=int(metadata.get("interval_ns", 1_000_000)),
                    gpu_interval_ns=int(metadata.get("gpu_interval_ns", 5_000_000)),
                    idle_gpu_power_w=idle_gpu_power_w,
                )
                rows.append({**common, **integrated})
        except Exception as error:
            exclusions.append({"run_id": run_dir.name, "reason": f"{type(error).__name__}: {error}"})
    frame = pd.DataFrame(rows)
    return frame, {
        "campaign_dir": str(campaign_dir),
        "accepted_run_ids": len(accepted_dirs),
        "integrated_runs": int(frame["run_id"].nunique()) if not frame.empty else 0,
        "exclusions": exclusions,
    }


def _mean_std_cv(series: pd.Series) -> tuple[float, float, float, float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return (float("nan"),) * 5
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    cv = 100.0 * std / abs(mean) if mean else float("nan")
    return mean, std, cv, float(values.min()), float(values.max())


def aggregate_candidates(run_regions: pd.DataFrame, expected_repetitions: int) -> pd.DataFrame:
    keys = ["config_id", "operation", "size", "family", "device", "action_id", "cpu_level", "gpu_level", "region"]
    records: list[dict[str, Any]] = []
    for key, group in run_regions.groupby(keys, dropna=False, observed=True):
        record = dict(zip(keys, key))
        record["n_repetitions"] = int(group["repetition"].nunique())
        for source, prefix in (
            ("time_per_dispatch_s", "time"),
            ("energy_per_dispatch_j", "energy"),
            ("edp_per_dispatch_js", "edp"),
            ("rapl_energy_total_j", "rapl_total"),
            ("gpu_energy_total_j", "gpu_total"),
        ):
            mean, std, cv, minimum, maximum = _mean_std_cv(group[source])
            record.update({f"{prefix}_mean": mean, f"{prefix}_std": std,
                           f"{prefix}_cv_pct": cv, f"{prefix}_min": minimum,
                           f"{prefix}_max": maximum})
        record["eligible_repetitions"] = record["n_repetitions"] >= expected_repetitions
        record["gpu_energy_source"] = ",".join(sorted(group["gpu_energy_source"].dropna().astype(str).unique()))
        records.append(record)
    return pd.DataFrame(records)


def _static_descriptors(operation: str, n: int) -> dict[str, float]:
    if operation == "gemm":
        flops, logical_bytes = 2.0 * n**3, 24.0 * n**2
    elif operation == "fft":
        flops, logical_bytes = 5.0 * n**2 * math.log2(n**2), 32.0 * n**2
    elif operation == "axpy":
        flops, logical_bytes = 2.0 * n, 24.0 * n
    elif operation == "stencil":
        flops, logical_bytes = 5.0 * max(0, n - 2) ** 2, 16.0 * n**2
    elif operation == "cholesky":
        flops, logical_bytes = n**3 / 3.0, 16.0 * n**2
    elif operation == "spmv":
        flops, logical_bytes = 14.0 * n, 104.0 * n + 4.0
    else:
        raise DatasetContractError(f"operacion sin descriptores: {operation}")
    return {
        "log10_n": math.log10(n),
        "flops_per_dispatch_analytic": flops,
        "log10_flops_per_dispatch": math.log10(max(flops, 1.0)),
        "logical_bytes_per_dispatch": logical_bytes,
        "log10_logical_bytes": math.log10(max(logical_bytes, 1.0)),
        "arithmetic_intensity_analytic": flops / logical_bytes,
    }


def _level_fraction(level: str | float | None) -> float:
    mapping = {"REF": 1.0, "F0": 1.0, "F1": 0.833, "F2": 0.667,
               "F3": 0.5, "F4": 0.333, "F5": 0.167, "F6": 0.0}
    return mapping.get(str(level), float("nan"))


def _decorate_candidate(record: dict[str, Any]) -> dict[str, Any]:
    record.update(_static_descriptors(str(record["operation"]), int(record["size"])))
    record["candidate_device"] = record["device"]
    record["candidate_cpu_fraction"] = _level_fraction(record["cpu_level"])
    record["candidate_gpu_fraction"] = _level_fraction(record.get("gpu_level")) if record["device"] == "gpu" else 0.0
    record["candidate_cpu_is_ref"] = int(record["cpu_level"] == "REF")
    record["candidate_gpu_is_ref"] = int(record["device"] == "gpu" and record.get("gpu_level") == "REF")
    return record


def _label_groups(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["is_optimal"] = 0
    out["margin_edp_pct"] = np.nan
    out["optimum_stability"] = "undefined"
    groupby_key: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols
    for _, positions in out.groupby(groupby_key, observed=True).groups.items():
        group = out.loc[positions].sort_values(
            ["edp_mean", "energy_mean", "time_mean", "action_id"], kind="mergesort"
        )
        if len(group) < 1 or not np.isfinite(group.iloc[0]["edp_mean"]):
            continue
        best_index = group.index[0]
        out.loc[best_index, "is_optimal"] = 1
        if len(group) > 1:
            best, second = group.iloc[0], group.iloc[1]
            delta = float(second["edp_mean"] - best["edp_mean"])
            margin = 100.0 * delta / float(best["edp_mean"]) if best["edp_mean"] else float("nan")
            se_best = float(best["edp_std"]) / math.sqrt(max(1, int(best["n_repetitions"])))
            se_second = float(second["edp_std"]) / math.sqrt(max(1, int(second["n_repetitions"])))
            combined_se = math.sqrt(se_best**2 + se_second**2)
            stability = "uncertain" if delta <= combined_se else "separated"
            out.loc[positions, "margin_edp_pct"] = margin
            out.loc[positions, "optimum_stability"] = stability
    return out


def _complete_candidate_slice(
    candidates: pd.DataFrame,
    *,
    actions: Iterable[str],
    region: str | None = None,
) -> pd.DataFrame:
    expected = set(actions)
    work = candidates[candidates["eligible_repetitions"]].copy()
    if region is not None:
        work = work[work["region"] == region]
    complete_configs = [
        config_id for config_id, group in work.groupby("config_id", observed=True)
        if set(group["action_id"]) >= expected
    ]
    return work[work["config_id"].isin(complete_configs) & work["action_id"].isin(expected)].copy()


def build_strategy_a(candidates: pd.DataFrame, actions: Iterable[str]) -> pd.DataFrame:
    cold = _complete_candidate_slice(candidates, actions=actions, region="cold")
    records = []
    for row in cold.to_dict("records"):
        record = _decorate_candidate(row)
        record.update({
            "strategy": "A", "resource_state": "none_ready",
            "decision_group_id": f"A:{record['config_id']}",
            "requires_cold_start": 1,
        })
        records.append(record)
    return _label_groups(pd.DataFrame(records), ["decision_group_id"])


def _probe_summary(run_regions: pd.DataFrame, probe_action: str, probe_device: str) -> pd.DataFrame:
    work = run_regions[(run_regions["region"] == "cold") & (run_regions["action_id"] == probe_action)].copy()
    telemetry = CPU_TELEMETRY if probe_device == "cpu" else GPU_TELEMETRY
    if work.empty:
        return pd.DataFrame()
    # La estrategia C representa una unica ejecucion de sondeo, no el promedio
    # retrospectivo de las tres repeticiones disponibles en el experimento.
    first = (
        work.sort_values(["config_id", "repetition", "run_id"], kind="mergesort")
        .groupby("config_id", observed=True, as_index=False)
        .first()
    )
    first["avg_power_w"] = first["energy_per_dispatch_j"] / first["time_per_dispatch_s"]
    # Si cold dura menos que el intervalo de muestreo, tiempo y energia siguen
    # siendo estimaciones integradas, pero no se fabrica una observacion puntual
    # de telemetria a partir de una fraccion de ventana.
    low_resolution = first["region_to_sampling_ratio"] < 1.0
    for column in telemetry:
        if column in first:
            first.loc[low_resolution, column] = np.nan
    numeric = [
        "time_per_dispatch_s", "energy_per_dispatch_j", "rapl_energy_total_j",
        "gpu_energy_total_j", "avg_power_w", "region_to_sampling_ratio", *telemetry,
    ]
    return first[["config_id", *(column for column in numeric if column in first)]].copy()


def build_strategy_c(
    candidates: pd.DataFrame,
    run_regions: pd.DataFrame,
    actions: Iterable[str],
    *,
    probe_devices: Iterable[str] = ("cpu", "gpu"),
) -> pd.DataFrame:
    expected = set(actions)
    complete = _complete_candidate_slice(candidates, actions=expected, region=None)
    configs_by_region = {
        region: set(complete.loc[complete["region"] == region, "config_id"].unique())
        for region in ("cold", "warm")
    }
    complete_configs = configs_by_region["cold"] & configs_by_region["warm"]
    complete = complete[complete["config_id"].isin(complete_configs)]
    records: list[dict[str, Any]] = []
    probe_actions = {"cpu": "cpu:REF", "gpu": "gpu:REF:REF"}
    probes = tuple((device, probe_actions[device]) for device in probe_devices)
    for probe_device, probe_action in probes:
        probe = _probe_summary(run_regions, probe_action, probe_device)
        if probe.empty:
            continue
        probe_map = probe.set_index("config_id").to_dict("index")
        state = f"{probe_device}_ready"
        for config_id in sorted(complete_configs & set(probe_map)):
            for action in sorted(expected):
                candidate_device = action.split(":", 1)[0]
                target_region = "warm" if candidate_device == probe_device else "cold"
                match = complete[(complete["config_id"] == config_id)
                                 & (complete["action_id"] == action)
                                 & (complete["region"] == target_region)]
                if len(match) != 1:
                    continue
                record = _decorate_candidate(match.iloc[0].to_dict())
                record.update({
                    "strategy": "C", "probe_device": probe_device,
                    "resource_state": state,
                    "decision_group_id": f"C:{state}:{config_id}",
                    "target_region": target_region,
                    "requires_cold_start": int(target_region == "cold"),
                })
                for column, value in probe_map[config_id].items():
                    record[f"probe_{column}"] = value
                    if isinstance(value, (int, float, np.number)):
                        record[f"probe_{column}_missing"] = int(not np.isfinite(value))
                records.append(record)
    return _label_groups(pd.DataFrame(records), ["decision_group_id"])


def completeness_report(run_regions: pd.DataFrame, candidates: pd.DataFrame, mode: str, expected_repetitions: int, actions: Iterable[str] | None = None) -> dict[str, Any]:
    actions = set(actions or expected_actions(mode))
    configs = sorted(run_regions[run_regions["device"] == "cpu"]["config_id"].unique())
    details: dict[str, Any] = {}
    for config_id in configs:
        observed = candidates[(candidates["config_id"] == config_id)
                              & (candidates["region"] == "cold")
                              & candidates["eligible_repetitions"]]
        present = set(observed["action_id"])
        details[config_id] = {
            "present_actions": sorted(present & actions),
            "missing_actions": sorted(actions - present),
            "complete": actions <= present,
        }
    return {
        "mode": mode,
        "expected_repetitions": expected_repetitions,
        "expected_actions_per_config": len(actions),
        "config_count": len(configs),
        "complete_config_count": sum(int(item["complete"]) for item in details.values()),
        "details": details,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except Exception:
        return None


def _environment_versions() -> dict[str, str | None]:
    """Versiones del entorno que materializo el dataset."""
    import platform

    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", None),
    }
    for package, key in (("sklearn", "scikit_learn"), ("optuna", "optuna"), ("xgboost", "xgboost")):
        try:
            module = __import__(package)
            versions[key] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[key] = None
    return versions


def build_selector_datasets(config: BuildConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = _catalog_entries(config.catalog_path)
    all_frames: list[pd.DataFrame] = []
    campaign_reports: list[dict[str, Any]] = []
    for campaign_dir in (config.cpu_campaign_dir, config.gpu_campaign_dir):
        if campaign_dir is None:
            continue
        frame, report = read_campaign_runs(
            campaign_dir, catalog, idle_gpu_power_w=config.idle_gpu_power_w,
        )
        all_frames.append(frame)
        campaign_reports.append(report)
    exclusions = [item for report in campaign_reports for item in report["exclusions"]]
    if exclusions:
        preview = exclusions[:3]
        raise DatasetContractError(f"corridas aceptadas no integrables ({len(exclusions)}): {preview}")
    run_regions = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if run_regions.empty:
        raise DatasetContractError("ninguna corrida dual pudo integrarse")
    if run_regions.duplicated(["run_id", "region"]).any():
        raise DatasetContractError("run_id/region duplicado entre campañas")

    candidates = aggregate_candidates(run_regions, config.expected_repetitions)
    actions = configured_actions(config)
    completeness = completeness_report(
        run_regions, candidates, config.mode, config.expected_repetitions, actions,
    )
    if config.mode == "cpu-provisional":
        cpu_runs = run_regions[run_regions["device"] == "cpu"]
        if cpu_runs["run_id"].nunique() != 68 * 8 * config.expected_repetitions:
            raise DatasetContractError(
                f"CPU provisional incompleto: {cpu_runs['run_id'].nunique()} corridas, esperadas {68*8*config.expected_repetitions}"
            )
    if config.mode == "final" and completeness["complete_config_count"] != 68:
        raise DatasetContractError(
            f"matriz final incompleta: {completeness['complete_config_count']}/68 config_id"
        )

    strategy_a = build_strategy_a(candidates, actions)
    strategy_c = build_strategy_c(
        candidates, run_regions, actions,
        probe_devices=("cpu",) if config.mode == "cpu-provisional" else ("cpu", "gpu"),
    )
    paths = {
        "run_regions": config.output_dir / "run_regions.csv",
        "candidate_summary": config.output_dir / "candidate_summary.csv",
        "strategy_a": config.output_dir / "strategy_a_candidates.csv",
        "strategy_c": config.output_dir / "strategy_c_candidates.csv",
        "completeness": config.output_dir / "completeness.json",
        "provenance": config.output_dir / "provenance.json",
    }
    run_regions.to_csv(paths["run_regions"], index=False)
    candidates.to_csv(paths["candidate_summary"], index=False)
    strategy_a.to_csv(paths["strategy_a"], index=False)
    strategy_c.to_csv(paths["strategy_c"], index=False)
    _write_json(paths["completeness"], completeness)
    provenance = {
        "builder": "classifier.selector.dataset",
        "git_commit": _git_commit(),
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key, value in asdict(config).items()},
        "energy_rule": {
            "cpu_gpu_idle_power_w": config.idle_gpu_power_w,
            "source_job": config.idle_gpu_power_source_job,
            "cpu": "rapl_package_dram_plus_idle_gpu_baseline",
            "gpu": "rapl_package_dram_plus_nvml_integrated",
        },
        "campaigns": campaign_reports,
        "environment": _environment_versions(),
        "counts": {
            "run_regions": len(run_regions),
            "unique_runs": int(run_regions["run_id"].nunique()),
            "candidate_summaries": len(candidates),
            "strategy_a_rows": len(strategy_a),
            "strategy_c_rows": len(strategy_c),
        },
    }
    _write_json(paths["provenance"], provenance)
    return paths
