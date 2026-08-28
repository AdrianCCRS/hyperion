"""Diagnostico NO invasivo (no toca el pipeline de recoleccion): revisa
gpu_sm_clock_mhz/gpu_temperature_c ya grabados en windows.csv para detectar
si el throttling termico de GPU fue solo un hueco teorico o si de verdad
paso durante una campana.

Contexto (2026-08-28): apply_gpu_frequency() solo detecta "el candado no se
aplico" (observado > techo con carga real, gpu_freqctl.py:140-247); nunca
detecto "observado < techo con carga real", que es la firma de throttling
termico. No existe un classify_frequency_window() para GPU (omision
documentada en postprocess.py:1065-1067) que hubiera filtrado esas ventanas
del dataset. Este script es el chequeo posterior que reemplaza, por ahora,
a ese gate que no existe.

Uso:
    python3 diagnose_gpu_throttling.py <output_dir_de_la_campana>
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

# Relojes SM disponibles en la A100 de pacca, min/max por nivel de la
# rejilla GPU_LEVELS_FULL/GPU (fraction 1.0=max, 0.0=min) -- si el
# gpu_freq_khz_requested no esta en windows.csv (no se grabo en pases viejos),
# se cae a comparar contra este mapa por gpu_freq_level_id.
KNOWN_LEVELS_MHZ = {"REF": None, "F0": None, "F1": None, "F2": None,
                     "F3": None, "F4": None, "F5": None, "F6": None}

TOLERANCE_FRACTION = 0.05


def load_windows(campaign_dir: Path):
    rows = []
    for run_dir in sorted(campaign_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        wpath = run_dir / "windows.csv"
        if not wpath.exists():
            continue
        with wpath.open() as f:
            for row in csv.DictReader(f):
                if row.get("gpu_sm_clock_mhz") not in (None, "", "None"):
                    rows.append(row)
    return rows


def main(campaign_dir: str) -> None:
    rows = load_windows(Path(campaign_dir))
    print(f"ventanas GPU con gpu_sm_clock_mhz grabado: {len(rows)}")
    if not rows:
        print("Nada que analizar -- revisa la ruta o si la campana ya escribio windows.csv.")
        return

    by_level = defaultdict(list)
    temps = []
    for row in rows:
        level = row.get("gpu_freq_level_id") or "?"
        clk = row.get("gpu_sm_clock_mhz")
        temp = row.get("gpu_temperature_c")
        if clk not in (None, "", "None"):
            by_level[level].append(float(clk))
        if temp not in (None, "", "None"):
            temps.append(float(temp))

    if temps:
        print(f"\ntemperatura GPU: min={min(temps):.0f}C  max={max(temps):.0f}C  media={sum(temps)/len(temps):.1f}C")
        hot = sum(1 for t in temps if t >= 83)
        if hot:
            print(f"  *** {hot}/{len(temps)} ventanas con gpu_temperature_c >= 83C "
                  f"(la A100-PCIe throttlea termicamente en torno a 85-88C) ***")

    print("\nreloj SM observado por nivel declarado (fixed != REF, deberia ser estable):")
    for level, clocks in sorted(by_level.items()):
        if not clocks or level == "REF":
            continue
        mn, mx, mean = min(clocks), max(clocks), sum(clocks) / len(clocks)
        spread_pct = 100 * (mx - mn) / mean if mean else 0.0
        flag = " <-- dispersion alta, revisar" if spread_pct > TOLERANCE_FRACTION * 100 else ""
        print(f"  {level:6s} n={len(clocks):5d}  min={mn:6.0f}  max={mx:6.0f}  media={mean:6.0f}  "
              f"dispersion={spread_pct:5.1f}%{flag}")

    print("\nSin gate automatico de throttling en el pipeline (ver docstring) --"
          " esta salida es lectura manual, no un veredicto pass/fail.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1])
