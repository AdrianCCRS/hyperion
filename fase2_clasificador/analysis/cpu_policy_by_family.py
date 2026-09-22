"""Politica de frecuencia CPU con la familia como unidad (complemento de cpu_policy_table.py).

Criterio declarado antes de mirar resultados:
- Por (kernel, nivel): razon EDP_nivel / EDP_REF sobre la mediana de repeticiones de la corrida completa.
- Por (familia, clase): media de log(razon) de sus kernels (cada kernel pesa igual; la familia pesa uno).
- Agregado por nivel y clase: media geometrica entre familias (ganancia = 1 - exp(media log)), IC95 por
  bootstrap de familias y Wilcoxon pareado sobre log-razones por familia.
- Validacion leave-one-family-out: la regla "elegir el nivel de mayor ganancia media" se aplica sobre las
  demas familias y se evalua en la familia excluida; se reporta la ganancia realizada y la de "no actuar" (0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from fase2_clasificador.analysis.cpu_policy_table import LEVELS, kernel_class, per_run  # noqa: E402
from fase2_clasificador.eval.protocol import derive_kernel_family  # noqa: E402


def family_log_ratios(runs: pd.DataFrame, kc: pd.DataFrame) -> pd.DataFrame:
    med = runs.groupby(["kernel_ref", "level"]).edp.median().unstack()
    ratio = med[LEVELS].div(med["REF"], axis=0).dropna(how="all")
    lr = np.log(ratio)
    lr["cls"] = kc["class"].reindex(lr.index)
    lr["family"] = [derive_kernel_family(k) for k in lr.index]
    return lr.groupby(["cls", "family"])[LEVELS].mean()


def boot_gain(x: np.ndarray, reps: int = 4000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    g = 1 - np.exp(x[idx].mean(axis=1))
    return [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]


def analyse(fl: pd.DataFrame, min_effect: float) -> dict:
    out = {}
    for cls, sub in fl.groupby(level="cls"):
        sub = sub.droplevel("cls")
        levels = {}
        for lv in LEVELS:
            x = sub[lv].dropna().to_numpy()
            if len(x) < 3:
                continue
            gain = float(1 - np.exp(x.mean()))
            lo, hi = boot_gain(x)
            p = float(wilcoxon(x, alternative="less").pvalue) if np.any(x != 0) else 1.0
            levels[lv] = {"n_familias": int(len(x)), "gain": round(gain, 4), "gain_ci95": [round(lo, 4), round(hi, 4)],
                          "frac_familias_mejoran": round(float((x < 0).mean()), 3), "p_wilcoxon_mejora": round(p, 4)}
        # regla LOFO: mejor nivel en el resto de familias; ganancia realizada en la excluida
        realized, chosen = [], []
        fams = list(sub.index)
        for f in fams:
            rest = sub.drop(index=f)
            g = {lv: 1 - np.exp(rest[lv].dropna().mean()) for lv in LEVELS if rest[lv].notna().sum() >= 2}
            lv = max(g, key=g.get)
            if g[lv] < min_effect or pd.isna(sub.loc[f, lv]):
                lv = None
            chosen.append(lv)
            realized.append(0.0 if lv is None else float(1 - np.exp(sub.loc[f, lv])))
        realized = np.array(realized)
        out[cls] = {"n_familias": len(fams), "min_effect": min_effect, "levels": levels,
                    "lofo": {"nivel_elegido_por_pliegue": {f: c for f, c in zip(fams, chosen)},
                             "ganancia_media_realizada": round(float(realized.mean()), 4),
                             "ganancia_ic95_bootstrap_familias": [round(v, 4) for v in np.percentile(
                                 realized[np.random.default_rng(1).integers(0, len(realized), (4000, len(realized)))].mean(axis=1), [2.5, 97.5])],
                             "familias_que_empeoran": int((realized < 0).sum()),
                             "familias_que_actuan": int(sum(c is not None for c in chosen)),
                             "peor_familia": round(float(realized.min()), 4)}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_csv", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-effect", type=float, default=0.01)
    a = ap.parse_args()
    runs = per_run(pd.concat([pd.read_csv(f) for f in a.runs_csv], ignore_index=True))
    kc = kernel_class(runs)
    fl = family_log_ratios(runs, kc)
    res = analyse(fl, a.min_effect)
    a.out.mkdir(parents=True, exist_ok=True)
    fl.to_csv(a.out / "family_log_ratios.csv")
    (a.out / "policy_by_family.json").write_text(json.dumps(res, indent=1))
    for cls, r in res.items():
        print(cls, "familias:", r["n_familias"])
        print(pd.DataFrame(r["levels"]).T.to_string())
        print({k: v for k, v in r["lofo"].items() if k != "nivel_elegido_por_pliegue"})
        print(pd.Series(r["lofo"]["nivel_elegido_por_pliegue"]).value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
