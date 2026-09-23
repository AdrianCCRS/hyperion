"""Lanzador del daemon de GPU en C++: resuelve `policy_table.yaml` a los flags de
`gpu_loop_main` y lo ejecuta con `exec` (las senales llegan directo al proceso
C++, que libera el reloj). Igual que `fase3_daemon/cpu_loop/launch_cpu_daemon.py`:
el binario no parsea YAML.

Reglas (fallan antes de ejecutar nada):
  * `gpu-*` con action `actuar` exige `resolved_clock_mhz` entero positivo; `no_actuar`
    da reloj 0 (el daemon libera el candado en esa clase).
  * Los flags del usuario tras `--` se pasan tal cual; los de politica no se pueden repetir alli.
Uso:
  python3 launch_gpu_daemon.py --policy-table fase3_daemon/policy_table.yaml --binary build/gpu_loop_main \\
      --arm activo -- --model gpu_rf_sin_reloj.onnx --features gpu_rf_sin_reloj.features.txt --min-dwell-ns 3700000000
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

import yaml


def gpu_flags(policy: dict) -> list[str]:
    flags: list[str] = []
    for cls, flag in (("compute_bound", "--compute-clock-mhz"), ("memory_bound", "--memory-clock-mhz")):
        entry = policy.get(f"gpu-{cls}")
        if entry is None:
            raise ValueError(f"policy_table.yaml no tiene la entrada gpu-{cls}")
        action = entry.get("action")
        if action == "no_actuar":
            mhz = 0
        elif action == "actuar":
            mhz = entry.get("resolved_clock_mhz")
            if not isinstance(mhz, (int, float)) or mhz <= 0:
                raise ValueError(f"gpu-{cls}: action='actuar' sin resolved_clock_mhz positivo")
            mhz = int(round(mhz))
        else:
            raise ValueError(f"gpu-{cls}: accion desconocida {action!r}")
        flags += [flag, str(mhz)]
    return flags


def build_command(policy_doc: dict, binary: str, arm: str, passthrough: list[str]) -> list[str]:
    if arm not in ("sombra", "activo"):
        raise ValueError(f"--arm debe ser 'sombra' o 'activo', no {arm!r}")
    for f in ("--compute-clock-mhz", "--memory-clock-mhz", "--arm"):
        if f in passthrough:
            raise ValueError(f"{f} lo fija la politica/el lanzador, no puede repetirse tras '--'")
    return [binary, "--arm", arm, *gpu_flags(policy_doc["policy"]), *passthrough]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-table", type=Path, required=True)
    ap.add_argument("--binary", required=True, help="ruta a gpu_loop_main")
    ap.add_argument("--arm", required=True, choices=["sombra", "activo"])
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("passthrough", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    rest = a.passthrough[1:] if a.passthrough[:1] == ["--"] else a.passthrough
    try:
        cmd = build_command(yaml.safe_load(a.policy_table.read_text()), a.binary, a.arm, rest)
    except (ValueError, KeyError) as exc:
        print(f"launch_gpu_daemon: {exc}", file=sys.stderr)
        return 2
    print("comando:", shlex.join(cmd), file=sys.stderr)
    if a.print_only:
        return 0
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
