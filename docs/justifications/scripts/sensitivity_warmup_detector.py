#!/usr/bin/env python3
"""Sweep E (docs/justifications): sensibilidad de measure_warmup.py a
CV_THRESHOLD_PCT, MARGIN, min_mean_floor y plateau_ratio -- reprocesamiento
puro de datos YA recolectados (docs/justifications/data/raw_for_sweep_e/),
sin usar computo nuevo de pacca.

NOTA (ARC-91): min_relative_gain y max_depth (parametros de
detect_changepoints) NO se varian aqui -- se pasan fijos (0.10, 6) en
_detect() mas abajo. Quedan fuera del alcance de este script; siguen
"sin justificar" en la tabla maestra de docs/justifications/report/
sections/master_table.tex, no confundir con "auditados".

Corre localmente. Requiere que docs/justifications/data/raw_for_sweep_e/
ya tenga los windows.csv descargados (ver docs/justifications/scripts/
fetch_sweep_e_data.sh, o el historial de comandos del reporte).
"""
import csv
import importlib
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import scripts.pacca.measure_warmup as mw  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_for_sweep_e"
OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "sensitivity_warmup_detector.csv"

GPU_KERNELS = {
    "rodinia_hotspot": RAW_DIR / "rodinia_hotspot_fine.csv",
    "rodinia_heartwall": RAW_DIR / "rodinia_heartwall_fine.csv",
    "rodinia_lavamd": RAW_DIR / "rodinia_lavamd_fine.csv",
}
CPU_KERNELS = {
    "npb_bt": RAW_DIR / "npb_bt_rep01.csv",
    "npb_cg": RAW_DIR / "npb_cg_rep01.csv",
    "rodinia_lud": RAW_DIR / "rodinia_lud_rep01.csv",
}


def _load_series(path: Path, is_gpu: bool):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if is_gpu:
        return sorted(
            (int(r["t_end_ns"]), float(r["gpu_util_pct"]))
            for r in rows
            if r.get("quality_status") == "gpu_telemetry"
            and r.get("t_end_ns") not in (None, "") and r.get("gpu_util_pct") not in (None, "")
        )
    return sorted(
        (int(r["t_start_ns"]), float(r["ipc"]))
        for r in rows
        if r.get("t_start_ns") not in (None, "") and r.get("ipc") not in (None, "")
    )


def _detect(series, *, cv_threshold_pct, margin, min_mean_floor, plateau_ratio, min_relative_gain, max_depth):
    original_cv = mw.CV_THRESHOLD_PCT
    mw.CV_THRESHOLD_PCT = cv_threshold_pct
    try:
        t_ns, ok = mw.detect_warmup_ns(series, min_mean_floor=min_mean_floor)
        method = "cv_threshold"
        if not ok:
            t_ns, ok = mw.detect_warmup_via_changepoints(series, plateau_ratio=plateau_ratio)
            # nota: min_relative_gain/max_depth se aplican dentro de detect_changepoints,
            # invocado por detect_warmup_via_changepoints con sus propios defaults --
            # se registran aqui solo para trazabilidad del punto de barrido, no se
            # pasan (la firma actual de detect_warmup_via_changepoints no los expone).
            method = "changepoint"
    finally:
        mw.CV_THRESHOLD_PCT = original_cv
    if not ok:
        return None, None, method
    raw_s = t_ns / 1e9
    return raw_s, raw_s * margin, method


def main():
    rows = []

    param_grid = {
        "cv_threshold_pct": [1.0, 2.0, 5.0, 8.0, 10.0],
        "margin": [1.0, 1.1, 1.2, 1.3, 1.5],
        "min_mean_floor": [0.0, 2.0, 5.0, 8.0, 12.0],
        "plateau_ratio": [0.6, 0.7, 0.8, 0.9],
    }
    defaults = {"cv_threshold_pct": 5.0, "margin": 1.2, "min_mean_floor": 5.0, "plateau_ratio": 0.8}

    all_kernels = {**{(k, True): v for k, v in GPU_KERNELS.items()}, **{(k, False): v for k, v in CPU_KERNELS.items()}}

    for (kernel_ref, is_gpu), path in all_kernels.items():
        if not path.exists():
            print(f"AVISO: falta {path}, se omite {kernel_ref}")
            continue
        series = _load_series(path, is_gpu)
        floor_for_kernel = defaults["min_mean_floor"] if is_gpu else 0.0

        for varied_param, values in param_grid.items():
            if varied_param == "min_mean_floor" and not is_gpu:
                continue  # el piso de ruido solo aplica a senales GPU
            for value in values:
                kwargs = dict(defaults)
                kwargs["min_mean_floor"] = floor_for_kernel
                kwargs[varied_param] = value
                raw_s, proposed_s, method = _detect(series, cv_threshold_pct=kwargs["cv_threshold_pct"],
                                                     margin=kwargs["margin"], min_mean_floor=kwargs["min_mean_floor"],
                                                     plateau_ratio=kwargs["plateau_ratio"],
                                                     min_relative_gain=0.10, max_depth=6)
                rows.append({
                    "kernel_ref": kernel_ref, "device": "gpu" if is_gpu else "cpu",
                    "varied_param": varied_param, "varied_value": value,
                    "raw_warmup_s": raw_s, "proposed_warmup_s": proposed_s, "method": method,
                    "n_samples": len(series),
                })
                print(f"{kernel_ref:<20} {varied_param}={value:<6} -> raw={raw_s} proposed={proposed_s} method={method}")

    fieldnames = sorted({key for row in rows for key in row})
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen escrito en {OUT_CSV} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
