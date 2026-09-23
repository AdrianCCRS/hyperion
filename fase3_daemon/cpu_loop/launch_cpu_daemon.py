"""Lanzador del loop de CPU (Bloque C8): resuelve `policy_table.yaml` a los
flags de `cpu_loop_main` y lo ejecuta.

`cpu_loop_main` no parsea YAML a proposito (ver su docstring): este script,
que ya tiene `yaml`, traduce la tabla a `--compute-actuar/--compute-freq-khz`
y `--memory-actuar/--memory-freq-khz`. Hace `exec` sobre el binario para que
las senales (SIGINT/SIGTERM del job o del usuario) lleguen directo al proceso
C++ que restaura la frecuencia y el turbo; un intermediario Python romperia esa
cadena.

Reglas que hace cumplir (todas fallan antes de ejecutar nada):
  * Solo `actuar` o `actuar_experimental` con `resolved_freq_khz` generan flags;
    `no_actuar` no genera ninguno.
  * `memory_bound` en actuar exige `compute_bound` en actuar: el daemon activo
    siempre fija un nivel base (F0) y memory solo decide si baja de el. Sin
    base, el turbo quedaria apagado con el rango nativo, un estado que ningun
    brazo del experimento define.
  * El brazo `sombra` recibe los mismos flags que `activo` (hace el mismo
    trabajo y solo se detiene antes de escribir; `cpu_loop_main` ya lo garantiza).

Uso:
  python3 launch_cpu_daemon.py --policy-table fase3_daemon/policy_table.yaml \\
      --binary build/cpu_loop_main --arm activo -- --perf-cpus 0,1,2,3 ...
  (--print-only imprime el comando y no ejecuta.)
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

import yaml

ACTIONABLE = ("actuar", "actuar_experimental")


def cpu_flags(policy: dict) -> list[str]:
    flags: list[str] = []
    enabled = {}
    for cls, prefix in (("compute_bound", "compute"), ("memory_bound", "memory")):
        entry = policy.get(f"cpu-{cls}")
        if entry is None:
            raise ValueError(f"policy_table.yaml no tiene la entrada cpu-{cls}")
        action = entry.get("action")
        if action == "no_actuar":
            enabled[cls] = False
            continue
        if action not in ACTIONABLE:
            raise ValueError(f"cpu-{cls}: accion desconocida {action!r}")
        khz = entry.get("resolved_freq_khz")
        if not isinstance(khz, int) or khz <= 0:
            raise ValueError(f"cpu-{cls}: action={action!r} sin resolved_freq_khz entero positivo")
        enabled[cls] = True
        flags += [f"--{prefix}-actuar", f"--{prefix}-freq-khz", str(khz)]
    if enabled["memory_bound"] and not enabled["compute_bound"]:
        raise ValueError("memory_bound actua pero compute_bound no: falta el nivel base del daemon activo")
    return flags


def build_command(policy_doc: dict, binary: str, arm: str, passthrough: list[str]) -> list[str]:
    if arm not in ("sombra", "activo"):
        raise ValueError(f"--arm debe ser 'sombra' o 'activo', no {arm!r}")
    return [binary, "--arm", arm, *cpu_flags(policy_doc["policy"]), *passthrough]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-table", type=Path, required=True)
    ap.add_argument("--binary", required=True, help="ruta a cpu_loop_main")
    ap.add_argument("--arm", required=True, choices=["sombra", "activo"])
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("passthrough", nargs=argparse.REMAINDER, help="tras '--': flags que se pasan tal cual a cpu_loop_main")
    a = ap.parse_args(argv)
    rest = a.passthrough[1:] if a.passthrough[:1] == ["--"] else a.passthrough
    try:
        cmd = build_command(yaml.safe_load(a.policy_table.read_text()), a.binary, a.arm, rest)
    except (ValueError, KeyError) as exc:
        print(f"launch_cpu_daemon: {exc}", file=sys.stderr)
        return 2
    print("comando:", shlex.join(cmd), file=sys.stderr)
    if a.print_only:
        return 0
    os.execv(cmd[0], cmd)  # no retorna


if __name__ == "__main__":
    sys.exit(main())
