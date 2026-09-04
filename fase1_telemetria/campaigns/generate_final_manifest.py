"""F1-XDEV-003 / brecha G -- generador de manifiestos de barrido DEFINITIVO.

Los manifiestos de cribado (`F1-XDEV-001`) conservan pocos niveles de
frecuencia; el barrido definitivo usa una rejilla FINA. Este generador NO
inventa una selección de kernels: **exige** la lista congelada que sale del
cribado real (`--kernels-file`). Produce un manifiesto de esquema válido
tomando un manifiesto de cribado como plantilla (para heredar nodo, cores,
RAPL/uncore, bloque GPU, etc.) y sustituyendo únicamente la lista de kernels y
la rejilla de frecuencia.

El esquema de manifiesto solo admite niveles `fixed` por `fraction` en [0,1],
no por MHz explícito. La rejilla del plan está en MHz, así que hay que
resolverla contra el rango real del nodo:

- CPU: `--cpu-freq-range-khz MIN MAX` (de `scaling_min_freq`/`scaling_max_freq`);
- GPU: `--gpu-supported-clocks C1,C2,...` (de `nvidia-smi -q -d SUPPORTED_CLOCKS`).

Sin esos datos, el manifiesto se emite igual pero marcado
`frequency_grid_status: assumed_range_pending_node_verification`, y
`verify_grid_against_node()` (o el gate H) debe fallar hasta resolverlo en
paccaA100. Nunca se produce un manifiesto "final" con una rejilla sin verificar
presentada como verdad.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

# Rejillas del plan (§ brecha G del encargo). MHz.
CPU_GRID_MHZ_DEFAULT = [3200, 3100, 3000, 2900, 2800, 2600, 2400, 2200, 2000, 1400, 800]
GPU_GRID_MHZ_DEFAULT = [1410, 1350, 1290, 1230, 1170, 1110, 810, 510, 210]

# Rangos ASUMIDOS solo para emitir algo válido sin el nodo; deben verificarse.
_ASSUMED_CPU_RANGE_KHZ = (800_000, 3_200_000)   # Gold 5315Y aprox
_ASSUMED_GPU_CLOCKS_MHZ = [210, 510, 810, 1110, 1170, 1230, 1290, 1350, 1410]  # A100 aprox


def _fraction_for_mhz(mhz: int, low_mhz: float, high_mhz: float) -> float:
    if high_mhz <= low_mhz:
        return 0.0
    return round(min(1.0, max(0.0, (mhz - low_mhz) / (high_mhz - low_mhz))), 6)


def _nearest(mhz: int, supported: list[int]) -> int:
    return min(supported, key=lambda c: abs(c - mhz))


def build_frequency_levels(
    grid_mhz: list[int],
    *,
    device: str,
    cpu_range_khz: tuple[int, int] | None = None,
    gpu_supported_clocks_mhz: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Devuelve (frequency_levels, metadata). Un único `native_governor` (REF)
    y un `fixed` por cada punto de la rejilla, resuelto a `fraction`."""
    resolved = True
    if device == "cpu":
        rng = cpu_range_khz or _ASSUMED_CPU_RANGE_KHZ
        resolved = cpu_range_khz is not None
        low_mhz, high_mhz = rng[0] / 1000.0, rng[1] / 1000.0
        pts = [(f"FG_{i}", m, _fraction_for_mhz(m, low_mhz, high_mhz))
               for i, m in enumerate(grid_mhz)]
    else:
        supported = gpu_supported_clocks_mhz or _ASSUMED_GPU_CLOCKS_MHZ
        resolved = gpu_supported_clocks_mhz is not None
        low_mhz, high_mhz = float(min(supported)), float(max(supported))
        pts = []
        for i, m in enumerate(grid_mhz):
            snapped = _nearest(m, supported)
            pts.append((f"FG_{i}", snapped, _fraction_for_mhz(snapped, low_mhz, high_mhz)))

    levels = [{"id": "REF", "mode": "native_governor"}]
    for lvl_id, mhz, frac in pts:
        levels.append({"id": lvl_id, "mode": "fixed", "fraction": frac})
    meta = {
        "frequency_grid_mhz": grid_mhz,
        "frequency_grid_resolved_points": [
            {"id": i, "requested_mhz": m, "used_mhz": u, "fraction": f}
            for (i, m), (_, u, f) in zip(enumerate(grid_mhz), pts)
        ],
        "frequency_grid_status": "resolved_against_node" if resolved
        else "assumed_range_pending_node_verification",
    }
    return levels, meta


def generate(
    template_path: Path,
    kernels: list[str],
    *,
    device: str,
    campaign_id: str,
    grid_mhz: list[int] | None = None,
    cpu_range_khz: tuple[int, int] | None = None,
    gpu_supported_clocks_mhz: list[int] | None = None,
    catalog_path: str | None = None,
) -> dict:
    if not kernels:
        raise ValueError("F1-XDEV-003: se requiere la lista congelada de kernels "
                         "del cribado; este generador no inventa una selección")
    doc = yaml.safe_load(template_path.read_text())
    out = copy.deepcopy(doc)
    out["campaign_id"] = campaign_id
    out["kernels"] = [{"kernel_ref": k} for k in kernels]
    if catalog_path is not None:
        out["catalog_path"] = catalog_path

    grid = grid_mhz or (CPU_GRID_MHZ_DEFAULT if device == "cpu" else GPU_GRID_MHZ_DEFAULT)
    levels, meta = build_frequency_levels(
        grid, device=device, cpu_range_khz=cpu_range_khz,
        gpu_supported_clocks_mhz=gpu_supported_clocks_mhz,
    )
    if device == "cpu":
        out["frequency_levels"] = levels
    else:
        # eje CPU fijo en REF; el barrido fino va por el eje GPU
        out["frequency_levels"] = [{"id": "REF", "mode": "native_governor"}]
        out["gpu_frequency_levels"] = levels
        out.setdefault("gpu", {})["enabled"] = True

    out.setdefault("metadata", {})
    out["metadata"].update({
        "generated_by": "fase1_telemetria/campaigns/generate_final_manifest.py",
        "source_template": template_path.name,
        "device_axis": device,
        "frozen_kernel_list_size": len(kernels),
        **meta,
    })
    return out


def verify_grid_against_node(manifest: dict) -> tuple[bool, str]:
    """El gate previo a lanzar la campaña: no lanzar si la rejilla no fue
    resuelta contra el nodo real."""
    status = manifest.get("metadata", {}).get("frequency_grid_status")
    if status == "resolved_against_node":
        return True, "rejilla resuelta contra el nodo"
    return False, (f"frequency_grid_status={status!r}: resolver contra "
                   "scaling_min/max_freq (CPU) o SUPPORTED_CLOCKS (GPU) del nodo real "
                   "y regenerar antes de lanzar")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", type=Path, required=True,
                   help="Manifiesto de cribado a usar como plantilla.")
    p.add_argument("--kernels-file", type=Path, required=True,
                   help="Lista congelada (un kernel_ref por línea; '#' comenta).")
    p.add_argument("--device", choices=["cpu", "gpu"], required=True)
    p.add_argument("--campaign-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cpu-freq-range-khz", type=int, nargs=2, default=None,
                   metavar=("MIN", "MAX"))
    p.add_argument("--gpu-supported-clocks", default=None,
                   help="MHz separados por coma (nvidia-smi -q -d SUPPORTED_CLOCKS).")
    p.add_argument("--grid-mhz", default=None, help="Rejilla MHz separada por coma (opcional).")
    p.add_argument("--catalog-path", default=None,
                   help="Sobrescribe catalog_path del template (útil si el manifiesto "
                        "final no vive junto al catálogo).")
    a = p.parse_args(argv)

    kernels = [ln.strip() for ln in a.kernels_file.read_text().splitlines()
               if ln.strip() and not ln.strip().startswith("#")]
    grid = [int(x) for x in a.grid_mhz.split(",")] if a.grid_mhz else None
    gpu_clocks = [int(x) for x in a.gpu_supported_clocks.split(",")] if a.gpu_supported_clocks else None
    cpu_range = tuple(a.cpu_freq_range_khz) if a.cpu_freq_range_khz else None

    manifest = generate(
        a.template, kernels, device=a.device, campaign_id=a.campaign_id,
        grid_mhz=grid, cpu_range_khz=cpu_range, gpu_supported_clocks_mhz=gpu_clocks,
        catalog_path=a.catalog_path,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    ok, msg = verify_grid_against_node(manifest)
    print(f"escrito {a.out}  ({len(kernels)} kernels, "
          f"{len(manifest.get('gpu_frequency_levels') or manifest['frequency_levels'])} niveles)")
    print(f"gate rejilla: {'OK' if ok else 'BLOCKED'} -- {msg}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
