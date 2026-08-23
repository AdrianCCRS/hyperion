"""alpha en GPU de los kernels de calibración: prueba directa de la hipótesis.

En pacca el DVFS de GPU escala SOLO el reloj de SM (``nvidia-smi -lgc``);
el reloj de memoria queda intacto. La predicción física que se sigue de
eso es que un kernel limitado por ancho de banda de DRAM debe ser casi
insensible al reloj de SM (alpha ~ 0), mientras que uno limitado por
cómputo debe escalar casi 1:1 (alpha ~ 1).

Los kernels de calibración de la campaña ya corrieron en los 6 niveles
GPU, así que la hipótesis se puede falsear sin gastar un segundo de nodo:

- ``gpu_stream_bw``      -> puro ancho de banda   -> se espera alpha bajo
- ``gpu_ert_probe_fp64`` -> puro cómputo FP64     -> se espera alpha alto

Si la predicción se cumple, alpha queda validado como instrumento de
tamizaje barato para elegir kernels GPU con margen de DVFS, y el criterio
para ampliar el catálogo deja de ser adivinanza.

Advertencia de lectura: es 1 repetición por nivel (rep00), no 3. Sirve
como señal de tamizaje, no como medición definitiva.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from classifier.analysis.gpu_oracle_headroom import fit_alpha, read_summary

CID = "pacca_gpu_nucleo_activo_20260823"
DEFAULT_BASE = Path.home() / f"hyperion-results/campaigns/{CID}"
FIXED_LEVELS = ["F0", "F1", "F2", "F3", "F4"]

KERNELS = {
    "gpu_stream_bw": "ancho de banda puro -> se espera alpha BAJO",
    "gpu_ert_probe_fp64": "cómputo FP64 puro -> se espera alpha ALTO",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()

    print("=" * 70)
    print("alpha GPU de kernels de calibración (1 rep/nivel, tamizaje)")
    print("=" * 70)
    print("Solo se escala el reloj de SM; el de memoria queda fijo.")
    print()

    for kernel, expectation in KERNELS.items():
        print(f"{kernel}  --  {expectation}")
        points = []
        reference = None
        rows = []
        for level in FIXED_LEVELS:
            run_dir = args.base / f"{CID}__{kernel}__{level}__rep00"
            summary_path = run_dir / "summary.txt"
            metadata_path = run_dir / "metadata.json"
            if not (summary_path.exists() and metadata_path.exists()):
                print(f"   {level}: falta la corrida, se omite")
                continue
            summary = read_summary(summary_path)
            metadata = json.loads(metadata_path.read_text())
            mhz = float(metadata.get("gpu_freq_mhz_applied") or 0.0)
            elapsed = summary.get("telemetry_elapsed_ns_mean", 0.0) / 1e9
            if mhz <= 0 or elapsed <= 0:
                print(f"   {level}: datos incompletos (MHz={mhz}, t={elapsed})")
                continue
            if level == "F0":
                reference = (mhz, elapsed)
            rows.append((level, mhz, elapsed))

        if reference is None:
            print("   sin nivel F0 de referencia, no se puede ajustar\n")
            continue

        ref_mhz, ref_t = reference
        for level, mhz, elapsed in rows:
            ratio = elapsed / ref_t
            print(f"   {level}  {mhz:>5.0f} MHz   t={elapsed:>8.3f} s   T/Tref={ratio:>6.3f}")
            points.append((ref_mhz / mhz, ratio))

        alpha, intercept, r2 = fit_alpha(points)
        print(f"   => alpha={alpha:.3f}   intercepto={intercept:.3f}   r2={r2:.4f}")
        print()

    print("Umbral de referencia heredado del eje CPU: alpha < 0.226 => el DVFS paga.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
