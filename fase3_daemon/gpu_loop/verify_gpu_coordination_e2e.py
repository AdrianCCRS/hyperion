#!/usr/bin/env python3
"""Bloque C4 (Plan_Fase3_Daemon.md §0.1): verifica de punta a punta que la
señal de coordinación CPU-GPU (`coordination.py`/`gpu_active_reader.hpp`)
funciona entre dos PROCESOS reales y separados -- Python escribiendo,
C++ leyendo -- no solo que cada lado pasa sus propios tests unitarios en
aislamiento.

Lanza `gpu_active_signal_probe` (binario C++ real, compilado aparte, sin
PMU/ONNX) como subproceso mientras este script escribe una secuencia
programada de transiciones con `GpuActiveSignalWriter` y esperas reales
(`time.sleep`, no inyectadas -- esto SÍ necesita reloj de pared real,
porque lo que se prueba es la sincronización entre dos procesos del
sistema operativo). Al final, parsea la salida del binario y confirma que
observó las transiciones en el orden esperado.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fase3_daemon.gpu_loop.coordination import GpuActiveSignalWriter  # noqa: E402

# (segundos de espera antes de escribir, valor a escribir)
DEFAULT_SCHEDULE: tuple[tuple[float, bool], ...] = (
    (0.3, True),
    (0.5, False),
    (0.5, True),
    (0.5, False),
)


def parse_probe_output(stdout: str) -> list[tuple[int, bool]]:
    """`ts_ns,active` por línea -> lista de (ts_ns, bool)."""
    readings: list[tuple[int, bool]] = []
    for line in stdout.strip().splitlines():
        ts_str, active_str = line.split(",")
        readings.append((int(ts_str), active_str == "1"))
    return readings


def observed_transitions(readings: list[tuple[int, bool]]) -> list[bool]:
    """Colapsa lecturas repetidas consecutivas -- lo que importa es la
    SECUENCIA de estados distintos que el proceso C++ observó, no cuántas
    veces releyó el mismo valor entre dos transiciones."""
    transitions: list[bool] = []
    for _ts, active in readings:
        if not transitions or transitions[-1] != active:
            transitions.append(active)
    return transitions


def run_e2e(
    probe_path: Path, signal_path: Path, *, schedule: tuple[tuple[float, bool], ...] = DEFAULT_SCHEDULE,
    probe_duration_s: float = 3.0, probe_interval_ms: int = 20, sleep_fn=time.sleep,
    run_fn=subprocess.run,
) -> tuple[list[bool], subprocess.CompletedProcess]:
    proc = None

    def launch_probe():
        nonlocal proc
        import subprocess as _sp
        proc = _sp.Popen(
            [str(probe_path), "--signal-path", str(signal_path),
             "--duration-s", str(probe_duration_s), "--interval-ms", str(probe_interval_ms)],
            stdout=_sp.PIPE, stderr=_sp.PIPE, text=True,
        )

    launch_probe()
    writer = GpuActiveSignalWriter(signal_path)
    for delay_s, active in schedule:
        sleep_fn(delay_s)
        writer.write(active)

    stdout, stderr = proc.communicate(timeout=probe_duration_s + 10.0)
    completed = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    readings = parse_probe_output(stdout)
    return observed_transitions(readings), completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-path", type=Path, required=True)
    parser.add_argument("--signal-path", type=Path, required=True)
    parser.add_argument("--probe-duration-s", type=float, default=3.0)
    args = parser.parse_args()

    transitions, completed = run_e2e(
        args.probe_path, args.signal_path, probe_duration_s=args.probe_duration_s,
    )
    print(f"sonda C++: exit_code={completed.returncode}")
    print(f"transiciones observadas por el proceso C++: {transitions}")
    if completed.stderr:
        print(f"stderr de la sonda: {completed.stderr}", file=sys.stderr)

    expected = [step[1] for step in DEFAULT_SCHEDULE]
    # El proceso C++ arranca leyendo el estado por defecto (archivo ausente
    # -> False) ANTES de la primera escritura -- se antepone al esperado.
    expected_with_initial = [False, *expected]
    # Colapsar repeticiones consecutivas del propio expected tambien, por
    # si el schedule pidiera escribir el mismo valor dos veces seguidas.
    collapsed_expected: list[bool] = []
    for value in expected_with_initial:
        if not collapsed_expected or collapsed_expected[-1] != value:
            collapsed_expected.append(value)

    ok = completed.returncode == 0 and transitions == collapsed_expected
    print(f"esperado: {collapsed_expected}")
    print(f"verificación C4 (señal de coordinación CPU-GPU de punta a punta): {'OK' if ok else 'FALLÓ'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
