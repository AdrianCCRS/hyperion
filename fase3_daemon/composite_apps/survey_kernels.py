#!/usr/bin/env python3
"""Sondeo de duracion de kernels del catalogo SIN exigir `phase_label_hint` (los candidatos a aplicacion compuesta B de
GPU, p.ej. los ERT probes o BabelStream, no lo traen): verifica el checksum como Fase 1 (CAT-07), corre cada kernel una
vez y reporta duracion y exito. No clasifica nada; solo dice si duran lo suficiente para la ventana de decision del
daemon (~3 s de actividad sostenida). Correr en pacca con GPU libre."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from common.hpc.catalog import load_catalog, verify_binary  # noqa: E402
from fase3_daemon.composite_apps.composite_known import DEFAULT_CATALOG_PATH, _check_success  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node-id", required=True)
    ap.add_argument("--kernels-root", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    ap.add_argument("kernels", nargs="+")
    a = ap.parse_args()
    catalog = load_catalog(str(a.catalog))
    rc = 0
    for spec in a.kernels:
        # "id" usa los argumentos del catalogo; "id::argumentos" los reemplaza (para medir tamanos escalados)
        kid, _, override = spec.partition("::")
        if kid not in catalog:
            print(f"{kid}: NO esta en el catalogo")
            rc = 1
            continue
        e = catalog[kid]
        if override:
            e = replace(e, exec_args=override)
        exe = str(a.kernels_root / e.exec_path)
        if not verify_binary(replace(e, exec_path=exe), node_id=a.node_id):
            print(f"{kid}: checksum NO verificado")
            rc = 1
            continue
        t0 = time.monotonic()
        p = subprocess.run([exe, *shlex.split(e.exec_args)], cwd=str(a.kernels_root), capture_output=True, text=True)
        dt = time.monotonic() - t0
        print(f"{spec} ({e.phase_label_hint or 'sin hint'}) {'OK' if _check_success(e, p.returncode, p.stdout) else 'FALLO'} en {dt:.2f}s", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
