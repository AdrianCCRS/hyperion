"""F1-XDEV-002 -- calibración trazable de `warmup_seconds` por candidato.

`warmup_seconds` NO ordena al harness una corrida de calentamiento aparte: la
telemetría se captura desde el inicio y, al postprocesar, las ventanas/muestras
anteriores a `primera_muestra + warmup_seconds` quedan `warmup_excluded`. Un
valor heredado o puesto conservadoramente puede descartar datos válidos o dejar
dentro un transitorio.

Procedimiento (Seguimiento_Cambios_Plan_Director.md, F1-XDEV-002):

1. mini-campaña con `warmup_seconds: 0` para cada candidato/dispositivo, mismo
   binario/checksum/args/tamaño/hilos/pinning/nodo/colector/frecuencias que la
   campaña posterior;
2. >= 3 repeticiones, cubriendo REF y los extremos de frecuencia a usar;
3. detección sobre IPC (CPU) o `gpu_util_pct` (GPU): dos ventanas móviles
   consecutivas con CV <= 5%; si no, segmentación por puntos de cambio y primer
   segmento que alcanza 80% de la meseta de actividad;
4. `warmup_seconds = 1.2 * t_detectado`;
5. como el catálogo admite un único valor por kernel, se adopta el **máximo**
   robustamente detectado entre esas condiciones;
6. estados explícitos: `measured`, `insufficient_signal`, `documented_fallback`,
   `not_suitable`;
7. artefacto CSV/JSON con trazabilidad;
8. propuesta al catálogo SIN reemplazo silencioso (requiere `--apply`).

La lógica de detección se portó de `old/scripts/pacca/measure_warmup.py`
(auditada, no copiada a ciegas). Este módulo NO inventa valores: opera sobre
`windows.csv` reales de la mini-campaña.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

CV_THRESHOLD_PCT = 5.0
MARGIN = 1.2  # +20% sobre el instante detectado
PLATEAU_RATIO = 0.8
MIN_REPETITIONS = 3

STATUS_MEASURED = "measured"
STATUS_INSUFFICIENT = "insufficient_signal"
STATUS_FALLBACK = "documented_fallback"
STATUS_NOT_SUITABLE = "not_suitable"


# ----------------------------------------------------------- detección (portada)

def _cv_pct(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    return (statistics.pstdev(values) / abs(mean)) * 100.0


def _adaptive_window_size(n: int) -> int:
    # GPU: pocas muestras (cadencia gruesa) -> ventana chica.
    # CPU: miles de muestras (~1 ms) -> 15 alcanza.
    return max(3, min(15, n // 4))


def detect_warmup_ns_cv(series: list[tuple[int, float]], *, min_mean_floor: float = 0.0
                        ) -> tuple[int | None, bool]:
    """Dos ventanas móviles consecutivas con CV <= umbral. `min_mean_floor`
    evita que un reposo inicial "estable en cero" (CV%=0) cuente como fin de
    arranque."""
    n = len(series)
    window = _adaptive_window_size(n)
    if n < window * 2:
        return None, False
    values = [v for _, v in series]
    t0 = series[0][0]
    for i in range(n - window * 2 + 1):
        w1, w2 = values[i:i + window], values[i + window:i + 2 * window]
        c1, c2 = _cv_pct(w1), _cv_pct(w2)
        if c1 is None or c2 is None:
            continue
        if c1 <= CV_THRESHOLD_PCT and c2 <= CV_THRESHOLD_PCT:
            if statistics.fmean(w1 + w2) < min_mean_floor:
                continue
            return series[i][0] - t0, True
    return None, False


def _sse(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = statistics.fmean(values)
    return sum((v - mean) ** 2 for v in values)


def _best_split(values: list[float], min_size: int) -> tuple[int, float] | None:
    n = len(values)
    if n < 2 * min_size:
        return None
    total = _sse(values)
    best_i, best_gain = None, 0.0
    for i in range(min_size, n - min_size + 1):
        gain = total - _sse(values[:i]) - _sse(values[i:])
        if gain > best_gain:
            best_gain, best_i = gain, i
    return (best_i, best_gain) if best_i is not None else None


def detect_changepoints(values: list[float], *, min_size: int = 10,
                        min_relative_gain: float = 0.10, max_depth: int = 6) -> list[int]:
    """Segmentación binaria recursiva (Scott & Knott). Separa fin-de-arranque
    de cambios de fase legítimos: los detecta a todos como changepoints."""
    cps: list[int] = []

    def recurse(lo: int, hi: int, depth: int) -> None:
        if depth <= 0 or hi - lo < 2 * min_size:
            return
        seg = values[lo:hi]
        split = _best_split(seg, min_size)
        if split is None:
            return
        i, gain = split
        total = _sse(seg)
        if total == 0 or gain / total < min_relative_gain:
            return
        cps.append(lo + i)
        recurse(lo, lo + i, depth - 1)
        recurse(lo + i, hi, depth - 1)

    recurse(0, len(values), max_depth)
    return sorted(cps)


def detect_warmup_ns_changepoint(series: list[tuple[int, float]], *,
                                 plateau_ratio: float = PLATEAU_RATIO) -> tuple[int | None, bool]:
    """Primer segmento cuya media alcanza `plateau_ratio` de la meseta (media
    máxima entre segmentos). Descarta blips de arranque sin umbral de duración."""
    values = [v for _, v in series]
    if not values:
        return None, False
    min_size = max(5, len(series) // 50)
    cps = detect_changepoints(values, min_size=min_size)
    if not cps:
        return None, False
    bounds = [0, *cps, len(values)]
    segments = list(zip(bounds, bounds[1:]))
    means = [statistics.fmean(values[a:b]) for a, b in segments]
    plateau = max(means)
    if plateau <= 0:
        return None, False
    threshold = plateau_ratio * plateau
    for (a, _b), m in zip(segments, means):
        if m >= threshold:
            if a == 0:
                return None, False
            return series[a][0] - series[0][0], True
    return None, False


# ----------------------------------------------------------- análisis por corrida

def _load_series(windows_csv: Path) -> tuple[list[tuple[int, float]], str, bool]:
    with windows_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    is_gpu = any(r.get("gpu_power_mw") not in (None, "") for r in rows)
    if is_gpu:
        series = sorted(
            (int(r["t_end_ns"]), float(r["gpu_util_pct"]))
            for r in rows
            if r.get("quality_status") == "gpu_telemetry"
            and r.get("t_end_ns") not in (None, "")
            and r.get("gpu_util_pct") not in (None, "")
        )
        return series, "gpu_util_pct", True
    series = sorted(
        (int(r["t_start_ns"]), float(r["ipc"]))
        for r in rows
        if r.get("t_start_ns") not in (None, "") and r.get("ipc") not in (None, "")
    )
    return series, "ipc", False


@dataclass
class RunWarmup:
    windows_csv: str
    run_id: str | None
    freq_level_id: str | None
    signal: str
    n_samples: int
    total_span_s: float | None
    detected: bool
    method: str | None = None
    raw_warmup_s: float | None = None
    proposed_warmup_s: float | None = None
    fraction_of_run: float | None = None
    reason: str = ""


def analyze_run(windows_csv: Path, *, run_id: str | None = None,
                freq_level_id: str | None = None) -> RunWarmup:
    series, signal, is_gpu = _load_series(windows_csv)
    r = RunWarmup(str(windows_csv), run_id, freq_level_id, signal, len(series), None, False)
    if not series:
        r.reason = "sin muestras de la señal"
        return r
    r.total_span_s = (series[-1][0] - series[0][0]) / 1e9
    floor = 5.0 if is_gpu else 0.0
    t_ns, ok = detect_warmup_ns_cv(series, min_mean_floor=floor)
    method = "cv_threshold"
    if not ok:
        t_ns, ok = detect_warmup_ns_changepoint(series)
        method = "changepoint" if ok else method
    r.detected = ok
    if ok:
        raw = t_ns / 1e9
        r.method = method
        r.raw_warmup_s = round(raw, 4)
        r.proposed_warmup_s = round(raw * MARGIN, 4)
        r.fraction_of_run = round(raw / r.total_span_s, 4) if r.total_span_s else None
    else:
        r.reason = "sin transitorio resoluble (señal insuficiente o cadencia demasiado gruesa)"
    return r


# ----------------------------------------------------------- calibración por kernel

@dataclass
class KernelWarmup:
    kernel_ref: str
    device: str
    binary_checksum: str | None
    status: str
    warmup_seconds: float | None
    method: str | None
    raw_warmup_s_max: float | None
    n_runs_analyzed: int
    n_runs_detected: int
    freq_levels_covered: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    per_run: list[dict] = field(default_factory=list)
    fallback_reason: str | None = None
    fallback_risk: str | None = None
    notes: list[str] = field(default_factory=list)


def calibrate_kernel(
    runs: list[tuple[Path, str | None, str | None]],
    *,
    kernel_ref: str,
    device: str,
    binary_checksum: str | None = None,
    fallback_seconds: float | None = None,
    fallback_reason: str | None = None,
    fallback_risk: str | None = None,
) -> KernelWarmup:
    """`runs`: lista de (windows_csv, run_id, freq_level_id) de una mini-campaña
    con `warmup_seconds: 0`. Adopta el MÁXIMO `proposed_warmup_s` entre corridas
    detectadas (criterio robusto: un único valor por kernel debe cubrir el peor
    caso observado)."""
    analyzed = [analyze_run(p, run_id=rid, freq_level_id=lvl) for p, rid, lvl in runs]
    detected = [a for a in analyzed if a.detected]
    levels = sorted({a.freq_level_id for a in analyzed if a.freq_level_id})
    out = KernelWarmup(
        kernel_ref=kernel_ref, device=device, binary_checksum=binary_checksum,
        status=STATUS_INSUFFICIENT, warmup_seconds=None, method=None,
        raw_warmup_s_max=None, n_runs_analyzed=len(analyzed), n_runs_detected=len(detected),
        freq_levels_covered=levels, run_ids=[a.run_id for a in analyzed if a.run_id],
        per_run=[asdict(a) for a in analyzed], fallback_reason=fallback_reason,
        fallback_risk=fallback_risk,
    )
    if len(analyzed) < MIN_REPETITIONS:
        out.notes.append(f"solo {len(analyzed)} corridas (< {MIN_REPETITIONS}); "
                         "el criterio robusto exige al menos 3")
    if detected:
        worst = max(detected, key=lambda a: a.proposed_warmup_s or 0.0)
        # Solo se declara 'measured' con >= MIN_REPETITIONS corridas Y al menos
        # 3 detecciones; si no, queda insufficient aunque haya algún valor.
        if len(analyzed) >= MIN_REPETITIONS and len(detected) >= MIN_REPETITIONS:
            out.status = STATUS_MEASURED
            out.warmup_seconds = worst.proposed_warmup_s
            out.method = worst.method
            out.raw_warmup_s_max = worst.raw_warmup_s
        else:
            out.notes.append("hay detección pero no en >=3 corridas: no se declara 'measured'")
            out.raw_warmup_s_max = worst.raw_warmup_s
    if out.status != STATUS_MEASURED and fallback_seconds is not None:
        if not (fallback_reason and fallback_risk):
            raise ValueError("un documented_fallback exige fallback_reason y fallback_risk")
        out.status = STATUS_FALLBACK
        out.warmup_seconds = float(fallback_seconds)
        out.notes.append("valor de fallback documentado -- NO equivale a warmup medido")
    if out.status == STATUS_INSUFFICIENT and device == "gpu":
        out.notes.append("GPU sin señal suficiente: alargar la carga o declarar "
                         "not_suitable; nunca sustituir por un valor arbitrario")
    return out


# ----------------------------------------------------------- catálogo (sin pisar)

def propose_catalog_updates(results: list[KernelWarmup]) -> list[dict]:
    """Propuestas, no escrituras. Solo kernels con `status == measured` o un
    `documented_fallback` completo."""
    props = []
    for r in results:
        if r.status not in (STATUS_MEASURED, STATUS_FALLBACK) or r.warmup_seconds is None:
            continue
        props.append({
            "kernel_ref": r.kernel_ref, "device": r.device,
            "proposed_warmup_seconds": round(r.warmup_seconds, 4),
            "status": r.status, "method": r.method,
            "binary_checksum": r.binary_checksum,
            "freq_levels_covered": r.freq_levels_covered,
            "run_ids": r.run_ids,
            "fallback_reason": r.fallback_reason if r.status == STATUS_FALLBACK else None,
        })
    return props


def apply_proposals_to_catalog(catalog_path: Path, proposals: list[dict], *,
                               apply: bool = False) -> dict:
    """Toca SOLO `warmup_seconds` del kernel indicado. Sin `apply=True` solo
    devuelve el diff. Con `apply=True` hace un backup `.bak` primero. Falla si
    el checksum del catálogo no coincide con el de la propuesta."""
    import yaml  # dependencia ya del proyecto

    doc = yaml.safe_load(catalog_path.read_text())
    kernels = doc.get("kernels", doc)
    diff = []
    for p in proposals:
        entry = kernels.get(p["kernel_ref"]) if isinstance(kernels, dict) else \
            next((k for k in kernels if k.get("id") == p["kernel_ref"]), None)
        if entry is None:
            diff.append({"kernel_ref": p["kernel_ref"], "action": "skip", "reason": "no está en el catálogo"})
            continue
        cur = entry.get("warmup_seconds")
        cat_ck = entry.get("binary_checksum")
        if p.get("binary_checksum") and cat_ck and p["binary_checksum"] != cat_ck:
            diff.append({"kernel_ref": p["kernel_ref"], "action": "skip",
                         "reason": f"checksum distinto (catálogo {cat_ck} vs propuesta {p['binary_checksum']})"})
            continue
        diff.append({"kernel_ref": p["kernel_ref"], "action": "update",
                     "from": cur, "to": p["proposed_warmup_seconds"], "status": p["status"]})
        if apply:
            entry["warmup_seconds"] = p["proposed_warmup_seconds"]
    result = {"applied": apply, "diff": diff}
    if apply and any(d["action"] == "update" for d in diff):
        bak = catalog_path.with_suffix(catalog_path.suffix + ".bak")
        if not bak.exists():
            bak.write_text(catalog_path.read_text())  # backup byte-a-byte del original
        catalog_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return result


def write_artifact(results: list[KernelWarmup], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "f1-xdev-002/warmup_calibration/1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cv_threshold_pct": CV_THRESHOLD_PCT, "margin": MARGIN,
        "min_repetitions": MIN_REPETITIONS,
        "per_kernel": {r.kernel_ref: asdict(r) for r in results},
        "proposals": propose_catalog_updates(results),
    }
    json_path = out_dir / "warmup_calibration.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    csv_path = out_dir / "warmup_calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kernel_ref", "device", "status", "warmup_seconds", "method",
                    "raw_warmup_s_max", "n_runs_analyzed", "n_runs_detected",
                    "freq_levels_covered", "binary_checksum"])
        for r in results:
            w.writerow([r.kernel_ref, r.device, r.status, r.warmup_seconds, r.method,
                        r.raw_warmup_s_max, r.n_runs_analyzed, r.n_runs_detected,
                        ";".join(r.freq_levels_covered), r.binary_checksum or ""])
    return json_path, csv_path


# ----------------------------------------------------------- CLI

def _discover_runs(campaign_dir: Path, kernel_ref: str) -> list[tuple[Path, str | None, str | None]]:
    """Busca subdirectorios `*__<kernel_ref>__<LEVEL>__rep*` con windows.csv."""
    runs = []
    for d in sorted(campaign_dir.glob(f"*__{kernel_ref}__*__rep*")):
        wc = d / "windows.csv"
        if not wc.exists():
            continue
        parts = d.name.split("__")
        level = parts[2] if len(parts) >= 3 else None
        runs.append((wc, d.name, level))
    return runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign-dir", type=Path, required=True,
                   help="Directorio de la mini-campaña de calibración (warmup_seconds: 0).")
    p.add_argument("--kernel", action="append", dest="kernels", required=True,
                   help="kernel_ref a calibrar (repetible).")
    p.add_argument("--device", choices=["cpu", "gpu"], required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--catalog", type=Path, default=None,
                   help="Si se da, escribe el diff de propuestas contra este catálogo.")
    p.add_argument("--apply", action="store_true",
                   help="Aplica las propuestas al catálogo (backup .bak primero). Sin esto, solo diff.")
    a = p.parse_args(argv)

    results: list[KernelWarmup] = []
    for k in a.kernels:
        runs = _discover_runs(a.campaign_dir, k)
        if not runs:
            print(f"AVISO: sin corridas para {k} bajo {a.campaign_dir}")
            results.append(KernelWarmup(
                kernel_ref=k, device=a.device, binary_checksum=None,
                status=STATUS_NOT_SUITABLE, warmup_seconds=None, method=None,
                raw_warmup_s_max=None, n_runs_analyzed=0, n_runs_detected=0,
                notes=["sin windows.csv de mini-campaña -- no se calibra"],
            ))
            continue
        results.append(calibrate_kernel(runs, kernel_ref=k, device=a.device))

    json_path, csv_path = write_artifact(results, a.out_dir)
    for r in results:
        print(f"{r.kernel_ref:<32}{r.status:<22}warmup={r.warmup_seconds}  "
              f"({r.n_runs_detected}/{r.n_runs_analyzed} corridas, método={r.method})")
    print(f"\nartefacto: {json_path}  |  {csv_path}")

    if a.catalog:
        diff = apply_proposals_to_catalog(a.catalog, propose_catalog_updates(results), apply=a.apply)
        print(f"\ncatálogo {'APLICADO' if a.apply else '(solo diff)'}:")
        for d in diff["diff"]:
            print(f"  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
