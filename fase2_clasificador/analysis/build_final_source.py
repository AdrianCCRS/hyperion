"""Une la matriz CPU de 24 familias con las seis familias compute nuevas.

Entrada: feature_contract_source.csv (campaña final 2026-09-13) y el CSV de
intervalos de la campaña compute (`build_power_source.py`). Excluye `ptrchase`
(latency-bound, fuera del dominio del clasificador, F1-CPU-004) y las sondas de
calibración. Salida: una sola matriz con las columnas de la primera.
"""
from __future__ import annotations

import argparse

import pandas as pd

EXCLUDE = ("ptrchase", "stream_official", "ert_probe")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_csv")
    ap.add_argument("new_csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-new", default=None, help="CSV solo con las familias nuevas ya filtradas (prueba externa)")
    a = ap.parse_args()
    base = pd.read_csv(a.base_csv, low_memory=False)
    new = pd.read_csv(a.new_csv, low_memory=False)
    new = new[~new["kernel_ref"].isin(EXCLUDE)]
    # mismo criterio de elegibilidad que la matriz base: telemetría correcta y frecuencia utilizable
    new = new[(new["training_quality_status"] == "ok")
              & new["frequency_quality_status"].isin(["valid", "not_applicable_native"])
              & new["phase_label_train"].notna() & (new["phase_label_train"] != "")]
    if a.out_new:
        new.to_csv(a.out_new, index=False)
    out = pd.concat([base, new[[c for c in base.columns if c in new.columns]]], ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"base={len(base)} nuevas={len(new)} total={len(out)} kernels={out['kernel_ref'].nunique()}")


if __name__ == "__main__":
    main()
