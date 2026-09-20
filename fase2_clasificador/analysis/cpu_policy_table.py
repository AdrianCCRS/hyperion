"""Tabla clase -> frecuencia de menor EDP para CPU (§3.4 y §3.5 del plan).

Unidad de análisis: la corrida completa. Toda corrida de un kernel ejecuta el
mismo trabajo (mismos argumentos e iteraciones) en todos los niveles, así que
EDP = (energía RAPL paquete + DRAM) x tiempo es comparable entre niveles. El
EDP por ventana de 1 ms no lo es: con ventana fija solo mide potencia.

Entrada: CSV por corrida (run, kernel_ref, level, rep, elapsed_ns, pkg_uj,
dram_uj, n_compute, n_memory, accepted), extraído de los metadata.json y
windows.csv de la campaña final. La clase de un kernel es la mayoritaria entre
sus ventanas etiquetadas (todas las frecuencias). Prueba: Wilcoxon pareado
por kernel entre REF y cada nivel (common.stats).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from common.stats import paired_significance_test  # noqa: E402

CALIBRATION = ("stream_official", "ert_probe")  # sondas de calibración, no kernels del dataset
LEVELS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.accepted.astype(bool) & df.level.isin(["REF"] + LEVELS)
           & ~df.kernel_ref.isin(CALIBRATION)].copy()
    d["energy_j"] = (d.pkg_uj + d.dram_uj) / 1e6
    d["time_s"] = d.elapsed_ns / 1e9
    d["edp"] = d.energy_j * d.time_s
    return d


def kernel_class(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("kernel_ref")[["n_compute", "n_memory"]].sum()
    g["compute_share"] = g.n_compute / (g.n_compute + g.n_memory)
    g["class"] = np.where(g.compute_share >= 0.5, "compute_bound", "memory_bound")
    return g


def policy(runs: pd.DataFrame, kc: pd.DataFrame, alpha: float = 0.05) -> tuple[dict, pd.DataFrame]:
    med = runs.groupby(["kernel_ref", "level"])[["edp", "energy_j", "time_s"]].median().reset_index()
    rows, out = [], {}
    for cls in ("compute_bound", "memory_bound"):
        ks = kc.index[kc["class"] == cls]
        sub = med[med.kernel_ref.isin(ks)]
        ref = sub[sub.level == "REF"].set_index("kernel_ref")
        best = None
        for lv in LEVELS:
            c = sub[sub.level == lv].set_index("kernel_ref")
            common = sorted(set(ref.index) & set(c.index))
            if len(common) < 2:
                continue
            r, x = ref.loc[common], c.loc[common]
            gain = 1 - x.edp.sum() / r.edp.sum()
            t = paired_significance_test(r.edp.to_numpy(), x.edp.to_numpy(), alpha=alpha)
            rel = (x.edp / r.edp)
            rows.append(dict(cls=cls, level=lv, n_kernels=len(common), gain_agg=gain,
                             median_ratio=float(rel.median()), frac_better=float((rel < 1).mean()),
                             energy_ratio=float((x.energy_j / r.energy_j).median()),
                             time_ratio=float((x.time_s / r.time_s).median()),
                             test=t.test_name, p=t.p_value, significant=bool(t.significant)))
            if t.significant and gain > 0 and (best is None or gain > best[1]):
                best = (lv, gain, t)
        out[cls] = ({"action": "no_actuar", "n_kernels": int(len(ks))} if best is None else
                    {"action": "actuar", "level": best[0], "gain": round(float(best[1]), 4),
                     "p": float(best[2].p_value), "n_kernels": int(len(ks))})
    return out, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.runs_csv)
    runs = per_run(df)
    kc = kernel_class(runs)
    pol, tab = policy(runs, kc)
    a.out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.out / "policy_by_level.csv", index=False)
    kc.to_csv(a.out / "kernel_class.csv")
    (a.out / "policy_cpu.json").write_text(json.dumps(pol, indent=1))
    pd.set_option("display.width", 200)
    print(kc["class"].value_counts().to_dict())
    print(tab.round(3).to_string(index=False))
    print(json.dumps(pol, indent=1))
    # mejor nivel por kernel
    med = runs.groupby(["kernel_ref", "level"]).edp.median().unstack()
    best = med[LEVELS].dropna(how="all").idxmin(axis=1)
    ratio = (med[LEVELS].min(axis=1) / med["REF"])
    pk = pd.DataFrame({"class": kc["class"], "best_level": best, "edp_vs_ref": ratio})
    pk.to_csv(a.out / "best_level_by_kernel.csv")
    print(pk.sort_values(["class", "best_level"]).round(3).to_string())


if __name__ == "__main__":
    main()
