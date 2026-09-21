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

Salida policy_cpu.json: mismo sobre que fase3_daemon/policy/derive_policy_table.py
(schema_version, generated_at_utc, policy con llaves "cpu-compute_bound" y
"cpu-memory_bound"), para que el daemon la cargue sin tabla en su codigo (§3.5
pasos 6 y 7). Cada entrada lleva accion, nivel elegido (None si no actuar),
motivo codificado, campanas de origen, tamano de muestra, prueba y el IC95 de
la ganancia de EDP de cada nivel probado.
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


def _gain_ci(ref_edp: np.ndarray, lv_edp: np.ndarray, reps: int = 4000, seed: int = 0) -> list[float]:
    """IC95 de la ganancia agregada de EDP por bootstrap de kernels (positivo = mejora frente a REF)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ref_edp), size=(reps, len(ref_edp)))
    gains = 1 - lv_edp[idx].sum(axis=1) / ref_edp[idx].sum(axis=1)
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]


def policy(runs: pd.DataFrame, kc: pd.DataFrame, alpha: float = 0.05) -> tuple[dict, pd.DataFrame]:
    med = runs.groupby(["kernel_ref", "level"])[["edp", "energy_j", "time_s"]].median().reset_index()
    campaign = runs.groupby("kernel_ref")["run"].agg(lambda r: sorted({x.split("__")[0] for x in r})) if "run" in runs else None
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
                             gain_ci95_lo=_gain_ci(r.edp.to_numpy(), x.edp.to_numpy())[0],
                             gain_ci95_hi=_gain_ci(r.edp.to_numpy(), x.edp.to_numpy())[1],
                             median_ratio=float(rel.median()), frac_better=float((rel < 1).mean()),
                             energy_ratio=float((x.energy_j / r.energy_j).median()),
                             time_ratio=float((x.time_s / r.time_s).median()),
                             test=t.test_name, p=t.p_value, significant=bool(t.significant)))
            if t.significant and gain > 0 and (best is None or gain > best[1]):
                best = (lv, gain, t)
        tab_cls = pd.DataFrame([r_ for r_ in rows if r_["cls"] == cls])
        campaigns = sorted({c for k in ks for c in campaign.loc[k]}) if campaign is not None else []
        levels_tested = {r_["level"]: {"gain_agg": round(float(r_["gain_agg"]), 4),
                                       "gain_ci95": [round(r_["gain_ci95_lo"], 4), round(r_["gain_ci95_hi"], 4)],
                                       "median_ratio": round(float(r_["median_ratio"]), 4), "p": round(float(r_["p"]), 4),
                                       "significant": bool(r_["significant"])} for _, r_ in tab_cls.iterrows()}
        base = {"n_kernels": int(len(ks)), "campaign_ids": campaigns, "reference_level": "REF",
                "sample_unit": "kernel (mediana de sus repeticiones aceptadas)",
                "test": "wilcoxon pareado por kernel contra REF", "alpha": alpha, "levels_tested": levels_tested}
        positive = [r_ for _, r_ in tab_cls.iterrows() if r_["gain_agg"] > 0]
        top = max(positive, key=lambda r_: r_["gain_agg"]) if positive else None
        base["chosen_level"] = best[0] if best else None
        if best is None:
            if top is None:
                code, txt = "ningun_nivel_mejora_edp", "ningun nivel bajo mejora el EDP agregado frente a REF"
            else:
                code, txt = ("mejora_no_significativa",
                             "algun nivel mejora el EDP agregado frente a REF pero sin significancia (Wilcoxon pareado)")
            out[f"cpu-{cls}"] = {"action": "no_actuar", "reason": code, "reason_detail": txt,
                                 "best_candidate": None if top is None else {
                                     "level": top["level"], "gain_agg": round(float(top["gain_agg"]), 4),
                                     "gain_ci95": [round(top["gain_ci95_lo"], 4), round(top["gain_ci95_hi"], 4)],
                                     "p": round(float(top["p"]), 4)}, **base}
        else:
            out[f"cpu-{cls}"] = {"action": "actuar", "gain": round(float(best[1]), 4),
                                 "gain_ci95": levels_tested[best[0]]["gain_ci95"], "p": float(best[2].p_value),
                                 "n_pairs": int(best[2].n_pairs), **base}
    return out, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_csv", type=Path, nargs="+", help="uno o mas CSV por corrida (se concatenan)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in a.runs_csv], ignore_index=True)
    runs = per_run(df)
    kc = kernel_class(runs)
    pol, tab = policy(runs, kc)
    a.out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.out / "policy_by_level.csv", index=False)
    kc.to_csv(a.out / "kernel_class.csv")
    doc = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
           "device": "cpu", "reference_level": "REF",
           "inputs": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in a.runs_csv},
           "policy": pol}
    (a.out / "policy_cpu.json").write_text(json.dumps(doc, indent=1))
    pd.set_option("display.width", 200)
    print(kc["class"].value_counts().to_dict())
    print(tab.round(3).to_string(index=False))
    print(json.dumps(doc, indent=1))
    # mejor nivel por kernel
    med = runs.groupby(["kernel_ref", "level"]).edp.median().unstack()
    best = med[LEVELS].dropna(how="all").idxmin(axis=1)
    ratio = (med[LEVELS].min(axis=1) / med["REF"])
    pk = pd.DataFrame({"class": kc["class"], "best_level": best, "edp_vs_ref": ratio})
    pk.to_csv(a.out / "best_level_by_kernel.csv")
    print(pk.sort_values(["class", "best_level"]).round(3).to_string())


if __name__ == "__main__":
    main()
