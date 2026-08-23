#!/usr/bin/env python3
"""T0.2d: validar el binado por progreso de instrucciones contra la
etiqueta de verdad real de `phasic` (ARC-187).

POR QUE. El Anexo B del plan maestro dejó abierta una duda: el alpha por
TRAMO sigue dando >1 en varios kernels incluso después de quitar el sesgo
del filtro de calidad, y la hipótesis más probable era desalineación de
tramos -- la invariancia del 0.34% de `delta_instructions` es sobre la
CORRIDA COMPLETA, pero una celda de C2 dura ~1% de eso, así que una deriva
pequeña podría hacer que el centil `k` en F4 no cubra el mismo trabajo que
en F0.

Con esto se puede medir directamente en vez de conjeturar. `phasic`
imprime marcas `PHASE <offset_s> <C|M>` ancladas a `T0_MONOTONIC_NS`
(ARC-177), en el MISMO reloj que el colector estampa `t_start_ns`
/`t_end_ns`. Eso da, para cada ventana, la fase REAL en la que cayó -- sin
pasar por OI ni por uncore, así que sirve aunque el uncore esté roto
(exactamente el caso: estos datos son del job 6420, rechazado por
VAL-09/I10 por el problema de CAP_PERFMON, pero el running_ratio es 1.0 y
las marcas de verdad son reales).

QUE MIDE. Para cada kernel `phasic_*`, se alinean las ventanas con
`add_instruction_progress`/`assign_progress_bins` (la misma coordenada que
usa C2) y se les asigna la fase REAL (interpolando el timestamp de cada
ventana contra las marcas PHASE). Luego se compara, centil a centil, si la
fase dominante en F0 coincide con la fase dominante en F4. Una alta
concordancia respalda que el binado es fiel; una baja concordancia
confirma la desalineación y explica el alpha > 1 residual.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.features import align  # noqa: E402

PHASE_LINE_RE = re.compile(r"^PHASE\s+([0-9.]+)\s+([CM])\s*$", re.MULTILINE)
T0_RE = re.compile(r"^T0_MONOTONIC_NS\s+(\d+)\s*$", re.MULTILINE)


def load_ground_truth(stdout_path: Path) -> tuple[int, list[tuple[float, str]]] | None:
    text = stdout_path.read_text(errors="replace")
    t0_match = T0_RE.search(text)
    if not t0_match:
        return None
    t0_ns = int(t0_match.group(1))
    marks = [(float(off), kind) for off, kind in PHASE_LINE_RE.findall(text)]
    if not marks:
        return None
    marks.sort(key=lambda m: m[0])
    return t0_ns, marks


def phase_at(marks: list[tuple[float, str]], t_offset_s: float) -> str | None:
    """Fase vigente en el instante `t_offset_s` (offset desde T0), por
    búsqueda binaria sobre las marcas ordenadas."""
    if t_offset_s < marks[0][0]:
        return None
    idx = np.searchsorted([m[0] for m in marks], t_offset_s, side="right") - 1
    if idx < 0:
        return None
    return marks[idx][1]


def analyze_kernel(run_dirs: dict[str, list[Path]], kernel: str, n_bins: int) -> dict:
    """`run_dirs`: nivel -> lista de directorios de corrida (una por rep)."""
    frames = []
    for level, dirs in run_dirs.items():
        for rep_idx, run_dir in enumerate(sorted(dirs), start=1):
            windows_path = run_dir / "windows.csv"
            stdout_path = run_dir / "stdout.txt"
            if not windows_path.exists() or not stdout_path.exists():
                continue
            gt = load_ground_truth(stdout_path)
            if gt is None:
                continue
            t0_ns, marks = gt

            df = pd.read_csv(windows_path, usecols=[
                "window_index", "t_start_ns", "t_end_ns", "delta_t_ns",
                "delta_instructions", "quality_status",
            ])
            df = df[df["quality_status"] == "ok"].copy()
            if df.empty:
                continue
            # Punto medio de la ventana, en offset desde T0, mismo reloj
            # (CLOCK_MONOTONIC) que las marcas de verdad.
            mid_ns = (pd.to_numeric(df["t_start_ns"], errors="coerce")
                     + pd.to_numeric(df["t_end_ns"], errors="coerce")) / 2.0
            df["real_phase"] = [
                phase_at(marks, (m - t0_ns) / 1e9) if np.isfinite(m) else None
                for m in mid_ns
            ]
            df["freq_level_id"] = level
            df["repetition"] = rep_idx
            df["kernel_ref"] = kernel
            frames.append(df)

    if not frames:
        return {"kernel": kernel, "error": "sin datos utilizables"}
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["real_phase"])

    work = align.add_instruction_progress(data)
    work = align.assign_progress_bins(work, n_bins=n_bins)

    # Fase dominante real por (nivel, repeticion, bin).
    dominant = (
        work.groupby(["freq_level_id", "repetition", "progress_bin"], observed=True)["real_phase"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )
    # Pureza del bin: fracción de ventanas que coinciden con la fase
    # dominante -- un bin puro respalda que el binado captura una fase
    # real; uno mixto es evidencia directa de desalineación.
    purity = (
        work.groupby(["freq_level_id", "repetition", "progress_bin"], observed=True)["real_phase"]
        .agg(lambda s: float((s == s.value_counts().idxmax()).mean()))
    )

    levels = sorted(dominant["freq_level_id"].unique())
    agreement = None
    if "F0" in levels and "F4" in levels:
        f0 = dominant[dominant["freq_level_id"] == "F0"].groupby("progress_bin")["real_phase"].agg(
            lambda s: s.value_counts().idxmax())
        f4 = dominant[dominant["freq_level_id"] == "F4"].groupby("progress_bin")["real_phase"].agg(
            lambda s: s.value_counts().idxmax())
        common = f0.index.intersection(f4.index)
        if len(common):
            agreement = float((f0.loc[common] == f4.loc[common]).mean())

    return {
        "kernel": kernel,
        "n_windows_con_fase_real": int(len(data.dropna(subset=["real_phase"]))),
        "levels": levels,
        "mean_bin_purity": round(float(purity.mean()), 4),
        "p05_bin_purity": round(float(np.percentile(purity, 5)), 4),
        "f0_vs_f4_bin_agreement": round(agreement, 4) if agreement is not None else None,
        "n_bins_comparados": int(len(common)) if agreement is not None else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir)
    print(f"nodo de análisis: {platform.node()}\n", flush=True)

    results = []
    for kernel in ("phasic_p010", "phasic_p100", "phasic_p1000"):
        run_dirs: dict[str, list[Path]] = {}
        for path in sorted(campaign_dir.iterdir()):
            if not path.is_dir() or "__baseline" in path.name:
                continue
            if f"__{kernel}__" not in path.name:
                continue
            m = re.search(r"__(REF|F\d+)__rep(\d+)$", path.name)
            if not m or m.group(2) == "00":
                continue
            run_dirs.setdefault(m.group(1), []).append(path)
        if not run_dirs:
            print(f"{kernel}: sin corridas encontradas", flush=True)
            continue
        result = analyze_kernel(run_dirs, kernel, args.n_bins)
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    Path(args.out).write_text(json.dumps({
        "analysis_node": platform.node(), "per_kernel": results,
    }, indent=2))
    print(f"\nreporte -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
