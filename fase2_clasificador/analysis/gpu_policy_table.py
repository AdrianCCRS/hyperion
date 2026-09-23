"""Tabla clase -> frecuencia de menor EDP para GPU, espejo exacto de
`cpu_policy_table.py` (misma unidad de analisis, misma prueba, mismo
bootstrap). Reemplaza un primer intento (2026-09-22) que corria Wilcoxon
sobre las 3 repeticiones de cada kernel por separado -- eso responde si el
ruido intra-kernel es distinguible de cero, no si la politica generaliza
entre kernels, que es la pregunta que decide una tabla de accion.

Unidad de analisis: el kernel (mediana de sus repeticiones aceptadas en
cada nivel), igual que CPU. EDP = energia GPU (mJ) x duracion cubierta
(ns), comparable entre niveles porque cada kernel ejecuta el mismo trabajo
en todos los niveles (el reloj SM no cambia el tamano del problema).

Clase del kernel: mayoria de `phase_label_train` entre sus propias filas
(esta ya viene de OI medido por `ncu` vs el punto de ridge del nivel, no es
una eleccion nueva de este script). A diferencia de CPU (donde la clase es
una proporcion continua n_compute/n_memory), en GPU la clase puede cambiar
de nivel a nivel porque el ridge se desplaza con la frecuencia mientras la
OI del kernel es constante (fenomeno fisico real, no ruido) -- se reporta
el margen de mayoria por kernel y se marcan como ambiguos (margen < 0.75)
en vez de tratarlos como si la clase fuera inequivoca.

Excluye minibude_cuda_bm1 (una sola repeticion medida a un solo nivel, sin
REF) y cualquier kernel sin filas en REF (no se puede parear).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from common.stats import paired_significance_test  # noqa: E402

EXCLUDED_KERNELS = ("minibude_cuda_bm1",)  # una sola repeticion, un solo nivel (REF), sin pares posibles
LEVELS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
AMBIGUOUS_MARGIN = 0.75  # margen de mayoria de clase por debajo del cual el kernel se marca ambiguo


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.training_eligible.astype(bool) & df.gpu_freq_level_id.isin(["REF"] + LEVELS)
           & ~df.kernel_ref.isin(EXCLUDED_KERNELS)].copy()
    d["energy_j"] = d.gpu_energy_delta_mj_sum / 1e3
    d["time_s"] = d.covered_duration_ns / 1e9
    d["edp"] = d.gpu_energy_delta_mj_sum * d.covered_duration_ns
    d = d.rename(columns={"gpu_freq_level_id": "level"})
    return d


def kernel_class(df: pd.DataFrame) -> pd.DataFrame:
    vc = df.groupby("kernel_ref")["phase_label_train"].value_counts(normalize=True).unstack(fill_value=0.0)
    g = pd.DataFrame(index=vc.index)
    g["class"] = vc.idxmax(axis=1)
    g["margin"] = vc.max(axis=1)
    g["ambiguous"] = g["margin"] < AMBIGUOUS_MARGIN
    return g


def _gain_ci(ref_edp: np.ndarray, lv_edp: np.ndarray, reps: int = 4000, seed: int = 0) -> list[float]:
    """IC95 de la ganancia agregada de EDP por bootstrap de kernels (positivo = mejora frente a REF)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ref_edp), size=(reps, len(ref_edp)))
    gains = 1 - lv_edp[idx].sum(axis=1) / ref_edp[idx].sum(axis=1)
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]


def policy(runs: pd.DataFrame, kc: pd.DataFrame, alpha: float = 0.05, min_effect: float = 0.0) -> tuple[dict, pd.DataFrame]:
    med = runs.groupby(["kernel_ref", "level"])[["edp", "energy_j", "time_s"]].median().reset_index()
    rows, out = [], {}
    for cls in ("compute_bound", "memory_bound"):
        ks_all = kc.index[kc["class"] == cls]
        ks = [k for k in ks_all if (med.kernel_ref == k).any() and "REF" in med[med.kernel_ref == k].level.values]
        n_ambiguous = int(kc.loc[ks, "ambiguous"].sum()) if ks else 0
        sub = med[med.kernel_ref.isin(ks)]
        ref = sub[sub.level == "REF"].set_index("kernel_ref")
        best = None
        sig = None
        for lv in LEVELS:
            c = sub[sub.level == lv].set_index("kernel_ref")
            common = sorted(set(ref.index) & set(c.index))
            if len(common) < 2:
                continue
            r, x = ref.loc[common], c.loc[common]
            gain = 1 - x.edp.sum() / r.edp.sum()
            t = paired_significance_test(r.edp.to_numpy(), x.edp.to_numpy(), alpha=alpha)
            rel = x.edp / r.edp
            ci = _gain_ci(r.edp.to_numpy(), x.edp.to_numpy())
            rows.append(dict(cls=cls, level=lv, n_kernels=len(common), gain_agg=gain,
                             gain_ci95_lo=ci[0], gain_ci95_hi=ci[1],
                             median_ratio=float(rel.median()), frac_better=float((rel < 1).mean()),
                             test=t.test_name, p=t.p_value, significant=bool(t.significant)))
            if t.significant and gain > 0 and (sig is None or gain > sig[1]):
                sig = (lv, gain, t)
        best = sig if sig is not None and sig[1] >= min_effect else None
        tab_cls = pd.DataFrame([r_ for r_ in rows if r_["cls"] == cls])
        levels_tested = {r_["level"]: {"gain_agg": round(float(r_["gain_agg"]), 4),
                                       "gain_ci95": [round(r_["gain_ci95_lo"], 4), round(r_["gain_ci95_hi"], 4)],
                                       "median_ratio": round(float(r_["median_ratio"]), 4), "p": round(float(r_["p"]), 4),
                                       "significant": bool(r_["significant"])} for _, r_ in tab_cls.iterrows()}
        base = {"n_kernels": int(len(ks)), "n_ambiguous_class_excluded_note": n_ambiguous,
                "reference_level": "REF", "sample_unit": "kernel (mediana de sus repeticiones aceptadas)",
                "test": "wilcoxon/t pareado por kernel contra REF (common.stats.paired_significance_test)",
                "alpha": alpha, "min_effect": min_effect, "levels_tested": levels_tested}
        positive = [r_ for _, r_ in tab_cls.iterrows() if r_["gain_agg"] > 0]
        top = max(positive, key=lambda r_: r_["gain_agg"]) if positive else None
        base["chosen_level"] = best[0] if best else None
        if best is None:
            if sig is not None:
                top = tab_cls[tab_cls.level == sig[0]].iloc[0]
                code, txt = ("efecto_bajo_minimo",
                             f"mejora significativa pero menor que el efecto minimo relevante declarado ({min_effect:.0%})")
            elif top is None:
                code, txt = "ningun_nivel_mejora_edp", "ningun nivel bajo mejora el EDP agregado frente a REF"
            else:
                code, txt = ("mejora_no_significativa",
                             "algun nivel mejora el EDP agregado frente a REF pero sin significancia (prueba pareada por kernel)")
            out[f"gpu-{cls}"] = {"action": "no_actuar", "reason": code, "reason_detail": txt,
                                 "best_candidate": None if top is None else {
                                     "level": top["level"], "gain_agg": round(float(top["gain_agg"]), 4),
                                     "gain_ci95": [round(top["gain_ci95_lo"], 4), round(top["gain_ci95_hi"], 4)],
                                     "p": round(float(top["p"]), 4)}, **base}
        else:
            out[f"gpu-{cls}"] = {"action": "actuar", "gain": round(float(best[1]), 4),
                                 "gain_ci95": levels_tested[best[0]]["gain_ci95"], "p": float(best[2].p_value),
                                 "n_pairs": int(best[2].n_pairs), **base}
    return out, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-effect", type=float, default=0.01,
                    help="ganancia minima de EDP para actuar (fraccion); decision declarada posterior al resultado")
    a = ap.parse_args()
    df = pd.read_csv(a.dataset_csv, low_memory=False)
    runs = per_run(df)
    kc = kernel_class(runs)
    pol, tab = policy(runs, kc, min_effect=a.min_effect)
    a.out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.out / "policy_by_level.csv", index=False)
    kc.to_csv(a.out / "kernel_class.csv")
    doc = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
           "device": "gpu", "reference_level": "REF",
           "inputs": {a.dataset_csv.name: hashlib.sha256(a.dataset_csv.read_bytes()).hexdigest()},
           "excluded_kernels": {"motivo": "una sola repeticion, un solo nivel (REF), sin pares posibles",
                                "kernels": list(EXCLUDED_KERNELS)},
           "ambiguous_class_note": "kernels con margen de mayoria de clase < 0.75 (el ridge se desplaza con el "
                                    "nivel de frecuencia mientras la OI ncu es constante por kernel; ver kernel_class.csv col. margin/ambiguous) "
                                    "se mantienen en la clase mayoritaria pero no se excluyen: excluirlos seria seleccionar datos por resultado",
           "policy": pol}
    (a.out / "policy_gpu.json").write_text(json.dumps(doc, indent=1))
    pd.set_option("display.width", 200)
    print(kc["class"].value_counts().to_dict())
    print(kc.round(3).to_string())
    print(tab.round(4).to_string(index=False))
    print(json.dumps(doc, indent=1))
    med = runs.groupby(["kernel_ref", "level"]).edp.median().unstack()
    lv_cols = [c for c in LEVELS if c in med.columns]
    best = med[lv_cols].dropna(how="all").idxmin(axis=1)
    ratio = med[lv_cols].min(axis=1) / med["REF"]
    pk = pd.DataFrame({"class": kc["class"], "margin": kc["margin"], "best_level": best, "edp_vs_ref": ratio})
    pk.to_csv(a.out / "best_level_by_kernel.csv")
    print(pk.sort_values(["class", "best_level"]).round(3).to_string())


if __name__ == "__main__":
    main()
