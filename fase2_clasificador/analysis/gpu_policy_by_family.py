"""Politica de frecuencia GPU con la familia como unidad (complemento de
gpu_policy_table.py), espejo de cpu_policy_by_family.py.

Por que hace falta ademas de la tabla por kernel: 3 de los 9 kernels
memory_bound de la tabla por kernel son en realidad el mismo algoritmo en
dos tamanos de problema (dual_axpy, dual_spmv, dual_stencil) -- tratarlos
como observaciones independientes infla la n real de 9 a un efectivo ~6.
La razon de log-medias por familia colapsa cada par a un solo punto antes
de agregar, y la validacion leave-one-family-out evalua si la regla
"elegir el nivel de mayor ganancia media" generaliza a una familia que el
propio calculo de "mejor nivel" nunca vio.

Familia: `kernel_family` del dataset fusionado, con el mismo split de
rajaperf_cuda en gemm/heat_3d/jacobi_2d que usa gpu_quality_report.py (son
kernels de intensidad aritmetica y comportamiento de clase distintos,
agruparlos infringiria el mismo principio que motiva el split).
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
from fase2_clasificador.analysis.gpu_policy_table import LEVELS, kernel_class, per_run  # noqa: E402

_RAJAPERF_CUDA_SPLIT = {
    "gpu_rajaperf_gemm": "rajaperf_cuda_gemm",
    "gpu_rajaperf_heat_3d": "rajaperf_cuda_heat_3d",
    "gpu_rajaperf_jacobi_2d": "rajaperf_cuda_jacobi_2d",
}


def kernel_family(kernel_ref: str, raw_family: str) -> str:
    return _RAJAPERF_CUDA_SPLIT.get(kernel_ref, raw_family)


def family_log_ratios(runs: pd.DataFrame, kc: pd.DataFrame, raw_family: pd.Series) -> pd.DataFrame:
    med = runs.groupby(["kernel_ref", "level"]).edp.median().unstack()
    have_ref = med.index[med["REF"].notna()] if "REF" in med.columns else []
    med = med.loc[med.index.isin(have_ref)]
    lv_cols = [c for c in LEVELS if c in med.columns]
    ratio = med[lv_cols].div(med["REF"], axis=0).dropna(how="all")
    lr = np.log(ratio)
    lr["cls"] = kc["class"].reindex(lr.index)
    lr["family"] = [kernel_family(k, raw_family.get(k, k)) for k in lr.index]
    return lr.groupby(["cls", "family"])[lv_cols].mean()


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
        for lv in sub.columns:
            x = sub[lv].dropna().to_numpy()
            if len(x) < 3:
                continue
            gain = float(1 - np.exp(x.mean()))
            lo, hi = boot_gain(x)
            p = float(wilcoxon(x, alternative="less").pvalue) if np.any(x != 0) else 1.0
            levels[lv] = {"n_familias": int(len(x)), "gain": round(gain, 4), "gain_ci95": [round(lo, 4), round(hi, 4)],
                          "frac_familias_mejoran": round(float((x < 0).mean()), 3), "p_wilcoxon_mejora": round(p, 4)}
        realized, chosen = [], []
        fams = list(sub.index)
        for f in fams:
            rest = sub.drop(index=f)
            g = {lv: 1 - np.exp(rest[lv].dropna().mean()) for lv in sub.columns if rest[lv].notna().sum() >= 2}
            lv = max(g, key=g.get) if g else None
            if lv is None or g[lv] < min_effect or pd.isna(sub.loc[f, lv]):
                lv = None
            chosen.append(lv)
            realized.append(0.0 if lv is None else float(1 - np.exp(sub.loc[f, lv])))
        realized = np.array(realized)
        out[cls] = {"n_familias": len(fams), "familias": fams, "min_effect": min_effect, "levels": levels,
                    "lofo": {"nivel_elegido_por_pliegue": {f: c for f, c in zip(fams, chosen)},
                             "ganancia_media_realizada": round(float(realized.mean()), 4),
                             "ganancia_ic95_bootstrap_familias": [round(v, 4) for v in np.percentile(
                                 realized[np.random.default_rng(1).integers(0, len(realized), (4000, len(realized)))].mean(axis=1), [2.5, 97.5])] if len(realized) > 1 else [None, None],
                             "familias_que_empeoran": int((realized < 0).sum()),
                             "familias_que_actuan": int(sum(c is not None for c in chosen)),
                             "peor_familia": round(float(realized.min()), 4) if len(realized) else None}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-effect", type=float, default=0.01)
    a = ap.parse_args()
    df = pd.read_csv(a.dataset_csv, low_memory=False)
    raw_family = df.groupby("kernel_ref")["kernel_family"].first()
    runs = per_run(df)
    kc = kernel_class(runs)
    fl = family_log_ratios(runs, kc, raw_family)
    res = analyse(fl, a.min_effect)
    a.out.mkdir(parents=True, exist_ok=True)
    fl.to_csv(a.out / "family_log_ratios.csv")
    (a.out / "policy_by_family.json").write_text(json.dumps(res, indent=1))
    for cls, r in res.items():
        print(cls, "familias:", r["n_familias"], r["familias"])
        print(pd.DataFrame(r["levels"]).T.to_string())
        print({k: v for k, v in r["lofo"].items() if k != "nivel_elegido_por_pliegue"})
        print(pd.Series(r["lofo"]["nivel_elegido_por_pliegue"]).value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
