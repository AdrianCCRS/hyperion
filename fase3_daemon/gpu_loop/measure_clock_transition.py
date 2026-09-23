#!/usr/bin/env python3
"""Mide T_transicion_gpu (Plan_Fase3_Daemon.md, Bloque D; §2.4.1 del plan de
realineacion, hasta hoy sin medicion en ningun lado): cuanto cuesta cambiar el
reloj SM de la GPU con el MISMO codigo que usa el daemon
(`common.hpc.gpu_freqctl.apply_gpu_frequency`) y cuanto tarda el reloj
observado en llegar al objetivo. Debe correrse con la GPU BAJO CARGA (el
lanzador shell mantiene un `gpu_phase_target bench_*` en marcha): con la GPU
ociosa el reloj SM cae a un nivel de reposo y la medicion no significa nada.

Por cada ciclo alterna entre dos relojes y registra:
  * `raw_ms`     : solo `nvidia-smi -lgc t,t` (el comando).
  * `apply_ms`   : `apply_gpu_frequency` completo, que incluye la espera de
                   asentamiento de NVML (`_UTILIZATION_SETTLE_SECONDS`) y las
                   relecturas -- lo que de verdad paga el daemon por cambio.
  * `observed_ms`: desde que se emitio el comando hasta que `clocks.sm` leido
                   coincide con el objetivo (+-15 MHz).
Restaura con `-rgc` siempre, incluso ante excepcion.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.hpc import environment as environment_module  # noqa: E402
from common.hpc import gpu_freqctl  # noqa: E402


def _query_clock(gpu: str) -> int | None:
    return gpu_freqctl._default_query_sm_clock_mhz(gpu)


def _wait_observed(gpu: str, target: int, t0: float, timeout_s: float = 3.0) -> float | None:
    while time.monotonic() - t0 < timeout_s:
        c = _query_clock(gpu)
        if c is not None and abs(c - target) <= 15:
            return (time.monotonic() - t0) * 1000.0
    return None


def _stats(name: str, v: list[float]) -> dict:
    if not v:
        return {"name": name, "n": 0}
    v = sorted(v)
    return {"name": name, "n": len(v), "p50_ms": round(statistics.median(v), 1),
            "p95_ms": round(v[min(len(v) - 1, int(0.95 * len(v)))], 1), "max_ms": round(v[-1], 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-index", default="0")
    ap.add_argument("--clocks", default="1410,1260", help="dos relojes MHz entre los que alternar")
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--delegated-cpus", default="0")
    ap.add_argument("--settle-s", type=float, default=None, help="espera de asentamiento dentro de apply_gpu_frequency "
                    "(default: la de gpu_freqctl, 1.5 s)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    lo_hi = [int(x) for x in a.clocks.split(",")]

    env = environment_module.detect_environment(a.delegated_cpus)
    if not env.gpu_frequency_write_capable:
        print("gpu_frequency_write_capable=False: exportar HYPERION_GPU_FREQ_WRITE_CAPABLE=1", file=sys.stderr)
        return 2
    print(f"relojes disponibles: {env.gpu_available_clocks_mhz}")

    raw, full, observed, unobserved = [], [], [], 0
    try:
        for i in range(a.cycles):
            for target in lo_hi:
                # 1) solo el comando
                t0 = time.monotonic()
                r = gpu_freqctl._default_run_nvidia_smi(["-lgc", f"{target},{target}"], gpu_index=a.gpu_index)
                if r.returncode != 0:
                    print(f"-lgc {target} fallo: {r.stderr.strip()}", file=sys.stderr)
                    return 3
                raw.append((time.monotonic() - t0) * 1000.0)
                obs = _wait_observed(a.gpu_index, target, t0)
                if obs is None: unobserved += 1
                else: observed.append(obs)
                # 2) apply_gpu_frequency completo (lo que paga el daemon), hacia el otro reloj y de vuelta
                lv = SimpleNamespace(id=f"t{target}", mode="fixed",
                                     fraction=(target - min(env.gpu_available_clocks_mhz)) /
                                              (max(env.gpu_available_clocks_mhz) - min(env.gpu_available_clocks_mhz)))
                t1 = time.monotonic()
                extra = {} if a.settle_s is None else {"sleep": lambda _s: time.sleep(a.settle_s)}
                applied = gpu_freqctl.apply_gpu_frequency(lv, env, gpu_index=a.gpu_index, **extra)
                full.append((time.monotonic() - t1) * 1000.0)
                if applied.applied_mhz != target:
                    print(f"aplicado {applied.applied_mhz} != pedido {target} (observado {applied.observed_sm_mhz})")
    finally:
        ok = gpu_freqctl.restore_gpu_state(env, gpu_index=a.gpu_index)
        print(f"restore_gpu_state: {ok}")
    result = {"clocks": lo_hi, "cycles": a.cycles, "unobserved": unobserved,
              "raw_command": _stats("raw -lgc", raw), "apply_gpu_frequency": _stats("apply_gpu_frequency", full),
              "observed_after_command": _stats("clocks.sm == objetivo", observed)}
    print(json.dumps(result, indent=1))
    if a.out:
        a.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
