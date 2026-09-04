"""F1-GPU-002 Etapa A -- comparación de cadencias NVML y elección de `q_produccion`.

El probe `common/telemetry/experiments/gpu_clock_transition_probe.cpp` se corre
con `--probe-interval-ns` de 5, 10, 50 y 100 ms (con `--dry-run-actuation`)
sobre la misma carga; cada corrida escribe un `gpu_clock_transition_summary.json`
con `observed_cadence` y `signal_step_analysis`. Este módulo agrega esos JSON y
recomienda la cadencia más gruesa que, frente al baseline de 5 ms, NO pierde
escalones observables ni reduce materialmente la resolución temporal.

No toca ninguna campaña: solo produce la recomendación versionable.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

BASELINE_NS = 5_000_000
_SIGNALS = (
    "power_mw",
    "util_pct",
    "mem_util_pct",
    "sm_clock_mhz",
    "graphics_clock_mhz",
    "temperature_c",
    "energy_mj",
)
# Señales online que deben conservar resolución para elegir q_produccion. La
# temperatura cambia mucho más lentamente y la energía es acumulativa: se
# reportan para conocer su frescura, pero no convierten por sí solas una
# cadencia en inválida. graphics_clock_mhz verifica la actuación; la feature
# que consume el modelo es sm_clock_mhz.
_GATING_SIGNALS = (
    "power_mw",
    "util_pct",
    "mem_util_pct",
    "sm_clock_mhz",
    "graphics_clock_mhz",
)
# tolerancia: una cadencia candidata es aceptable si conserva al menos este
# porcentaje de los escalones observados por señal frente al baseline de 5 ms.
STEP_RETENTION_MIN = 0.8


def _load(paths_or_globs: list[str]) -> list[dict]:
    out = []
    for pg in paths_or_globs:
        matches = glob.glob(pg) if any(c in pg for c in "*?[") else [pg]
        for m in matches:
            p = Path(m)
            if p.is_dir():
                for f in sorted(p.rglob("gpu_clock_transition_summary.json")):
                    d = json.loads(f.read_text()); d["_path"] = str(f); out.append(d)
            elif p.exists():
                d = json.loads(p.read_text()); d["_path"] = str(p); out.append(d)
    return out


def compare_cadences(summaries: list[dict]) -> dict:
    """Agrupa por `probe_interval_ns_requested` y compara contra 5 ms."""
    by_interval: dict[int, list[dict]] = {}
    for s in summaries:
        q = int(s.get("probe_interval_ns_requested") or 0)
        by_interval.setdefault(q, []).append(s)

    def steps_for(s: dict, sig: str) -> int:
        sa = (s.get("signal_step_analysis") or {}).get(sig) or {}
        return int(sa.get("n_consecutive_changes_lower_bound") or 0)

    def agg(rows: list[dict], sig: str) -> float:
        vals = [steps_for(r, sig) for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    baseline_rows = by_interval.get(BASELINE_NS, [])
    per_interval = {}
    for q, rows in sorted(by_interval.items()):
        entry = {
            "n_runs": len(rows),
            "observed_delta_p50_ns": _mean(rows, ("observed_cadence", "p50_delta_ns")),
            "observed_delta_p95_ns": _mean(rows, ("observed_cadence", "p95_delta_ns")),
            "steps_by_signal": {sig: agg(rows, sig) for sig in _SIGNALS},
        }
        if baseline_rows and q != BASELINE_NS:
            entry["step_retention_vs_5ms"] = {
                sig: (agg(rows, sig) / agg(baseline_rows, sig)) if agg(baseline_rows, sig) else None
                for sig in _SIGNALS
            }
        per_interval[q] = entry

    # q_produccion: la más gruesa (excluye <=5ms) cuya retención de escalones
    # >= STEP_RETENTION_MIN en TODAS las señales con baseline no nulo.
    recommended = BASELINE_NS
    reason = "sin datos para superar el baseline de 5 ms"
    if baseline_rows:
        candidates = sorted(q for q in by_interval if q > BASELINE_NS)
        for q in candidates:
            ret = per_interval[q].get("step_retention_vs_5ms", {})
            checks = [ret[sig] for sig in _GATING_SIGNALS if ret.get(sig) is not None]
            if checks and all(v >= STEP_RETENTION_MIN for v in checks):
                recommended = q
                reason = (f"{q/1e6:.0f} ms conserva >= {STEP_RETENTION_MIN:.0%} de los "
                          "escalones observados en todas las señales de decisión frente a 5 ms")
            else:
                break  # más grueso solo empeora
        if recommended == BASELINE_NS:
            reason = "ninguna cadencia más gruesa conserva suficientes escalones; se mantiene 5 ms"

    return {
        "schema": "f1-gpu-002/cadence_sweep/1",
        "baseline_ns": BASELINE_NS,
        "step_retention_min": STEP_RETENTION_MIN,
        "signals_reported": list(_SIGNALS),
        "signals_used_for_decision": list(_GATING_SIGNALS),
        "per_interval": per_interval,
        "q_produccion_ns": recommended,
        "q_produccion_reason": reason,
        "note": ("El nº de escalones es una COTA INFERIOR de actualizaciones físicas "
                 "del sensor. q_produccion no convierte lecturas redundantes en "
                 "observaciones independientes; solo reduce redundancia sin perder "
                 "resolución temporal observable. Temperatura y energía se "
                 "informan, pero no gobiernan q_produccion por ser lenta y "
                 "acumulativa, respectivamente."),
    }


def _mean(rows: list[dict], path: tuple[str, ...]) -> float | None:
    vals = []
    for r in rows:
        cur = r
        for k in path:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        if isinstance(cur, (int, float)):
            vals.append(cur)
    return sum(vals) / len(vals) if vals else None


def runbook(workload_cmd: str, gpu: str, mid_mhz: int, out_dir: Path) -> Path:
    intervals = [5_000_000, 10_000_000, 50_000_000, 100_000_000]
    lines = [
        "#!/usr/bin/env bash",
        "# F1-GPU-002 Etapa A -- barrido de cadencia NVML (paccaA100, dentro de Slurm)",
        "set -euo pipefail",
        f'PROBE=./common/telemetry/build/gpu_clock_transition_probe',
        f'OUT="$PWD/etapaA"; mkdir -p "$OUT"',
        "",
    ]
    for q in intervals:
        lines.append(
            f'"$PROBE" --workload-cmd {workload_cmd!r} --gpu {gpu} '
            f'--from-clock REF --to-clock {mid_mhz} --tolerance-mhz 15 '
            f'--probe-interval-ns {q} --dry-run-actuation '
            f'--warmup-ns 2000000000 --workload-min-active-ns 6000000000 --max-wait-ns 1000000000 '
            f'--out-dir "$OUT/q_{q}"'
        )
    lines += [
        "",
        "python3 -m fase1_telemetria.gpu_transition.cadence_sweep \"$OUT\" --out \"$OUT/cadence_sweep.json\"",
    ]
    path = out_dir / "etapaA_cadence_sweep_runbook.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="summary.json, globs o directorios del barrido de cadencia.")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    summaries = _load(a.inputs)
    if not summaries:
        print("sin summary.json -- correr primero el probe con --dry-run-actuation a 5/10/50/100 ms")
        return 1
    report = compare_cadences(summaries)
    for q, e in sorted(report["per_interval"].items()):
        ret = e.get("step_retention_vs_5ms")
        print(f"  {q/1e6:>5.0f} ms  runs={e['n_runs']}  p50={e['observed_delta_p50_ns']}  "
              f"retención={ret}")
    print(f"\nq_produccion = {report['q_produccion_ns']/1e6:.0f} ms -- {report['q_produccion_reason']}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"JSON: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
