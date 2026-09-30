"""Contraste de gobernador nativo powersave frente a performance para el resto de la Fase 4: matriz inicial (A), confirmatorio
de E-A, Escenario D (LAMMPS) y CloverLeaf. Completa el TODO(job 7789); el contraste de E-B (job 7779/7705) ya estaba resuelto
en generar_figuras_fase4_20260929.py.

Entradas:
  docs/libro/datos/fase4_20260930/fase4_A_powersave.csv           job 7789 (2026-09-29/30), etapa A: misma matriz que
                                                                   fase4_20260926/fase4_A.csv (CPU/GPU/conjunto, vistos e
                                                                   ineditos), con los 12 nucleos delegados en powersave/EPP
                                                                   default, turbo apagado, rango 0.8-3.2 GHz.
  docs/libro/datos/fase4_20260930/fase4_EA_powersave.csv          etapa EA_confirmatorio: mismo confirmatorio prospectivo de
                                                                   cinco bloques que fase4_20260926/fase4_EA.csv.
  docs/libro/datos/fase4_20260930/fase4_D_powersave.csv           etapa D: mismas cuatro entradas LAMMPS que
                                                                   fase4_20260926/fase4_D.csv.
  docs/libro/datos/fase4_20260930/fase4_cloverleaf_powersave.csv  etapa cloverleaf: mismo confirmatorio de cinco bloques que
                                                                   fase4_20260924/cloverleaf_confirm_7692.csv.
Los cuatro se comparan contra su contraparte bajo performance (job 7696/7692) para aislar el efecto del gobernador de la base
(REF_powersave/REF_performance) del efecto del agente dentro de cada gobernador.
Solo se usan celdas con state_ok = 1.
Reproduce: python3 docs/libro/scripts/generar_tablas_fase4_powersave_20260930.py
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from fase4_evaluacion.analyze_matrix import load_results, summarize  # noqa: E402

L = ROOT / "docs" / "libro"
D26, D24, D30 = L / "datos" / "fase4_20260926", L / "datos" / "fase4_20260924", L / "datos" / "fase4_20260930"


def valid(path):
    return [r for r in load_results(path) if r["state_ok"] == "1"]


def _geomean_ci(ratios):
    logs = np.log(ratios)
    gm = float(np.exp(np.mean(logs)))
    se = np.std(logs, ddof=1) / np.sqrt(len(logs))
    return gm, (float(np.exp(np.mean(logs) - 1.96 * se)), float(np.exp(np.mean(logs) + 1.96 * se)))


def _sign_p(ratios):
    n_fav = sum(x < 1 for x in ratios)
    n = len(ratios)
    return min(1.0, 2 * sum(1 for s in product((0, 1), repeat=n) if sum(s) >= max(n_fav, n - n_fav)) / 2 ** n)


def matriz_inicial():
    """A: EDP/REF por (alcance, kernels, activo) bajo cada gobernador, y REF_ps/REF_perf."""
    s_perf = {(s["scope"], s["set"], s["arm"]): s for s in summarize(valid(D26 / "fase4_A.csv"))}
    s_ps = {(s["scope"], s["set"], s["arm"]): s for s in summarize(valid(D30 / "fase4_A_powersave.csv"))}
    rows = []
    for sc in ("cpu", "gpu", "joint"):
        for ks in ("known", "unseen"):
            a = s_perf[(sc, ks, "activo")]["ratio_EDP"]
            b = s_ps[(sc, ks, "activo")]["ratio_EDP"]
            rp = {k: np.median([r[k] for r in valid(D26 / "fase4_A.csv") if r["scope"] == sc and r["set"] == ks and r["arm"] == "base"]) for k in ("wall_s", "e_cpu_j", "edp")}
            rq = {k: np.median([r[k] for r in valid(D30 / "fase4_A_powersave.csv") if r["scope"] == sc and r["set"] == ks and r["arm"] == "base"]) for k in ("wall_s", "e_cpu_j", "edp")}
            rows.append((sc, ks, a, b, {k: round(rq[k] / rp[k], 4) for k in rp}))
    (D30 / "tabla_A_powersave.tex").write_text("\n".join(
        f"{sc} & {ks} & {a:.3f} & {b:.3f} & {ref['wall_s']:.3f} & {ref['e_cpu_j']:.3f} & {ref['edp']:.3f} \\\\"
        for sc, ks, a, b, ref in rows) + "\n")
    return rows


def escenario_ea():
    """Confirmatorio E-A bajo powersave: mismo calculo por bloque que resumen_ea() en generar_figuras_fase4_20260926.py."""
    g = {}
    for r in load_results(D30 / "fase4_EA_powersave.csv"):
        g.setdefault(int(r["rep"]), {})[r["arm"]] = r | {"ok": r["state_ok"] == "1"}
    out = {}
    for arm in ("sombra", "activo_gpu", "fijo_gpu_f1"):
        val = [b for b in g if g[b][arm]["ok"] and g[b]["base"]["ok"]]
        ratios = {k: [g[b][arm][k] / g[b]["base"][k] for b in val] for k in ("wall_s", "e_gpu_j", "edp")}
        if not val:
            out[arm] = {"validos": 0}
            continue
        gm_edp, ci_edp = _geomean_ci(ratios["edp"])
        gm_gpu, ci_gpu = _geomean_ci(ratios["e_gpu_j"])
        out[arm] = {"validos": len(val), "p_signo": _sign_p(ratios["edp"]),
                    "edp_mediana": round(float(np.median(ratios["edp"])), 4), "edp_geomean": round(gm_edp, 4), "edp_ci95": tuple(round(x, 4) for x in ci_edp),
                    "gpu_geomean": round(gm_gpu, 4), "gpu_ci95": tuple(round(x, 4) for x in ci_gpu)}
    # REF powersave / REF performance
    perf_base = [r for r in load_results(D26 / "fase4_EA.csv") if r["state_ok"] == "1" and r["arm"] == "base"]
    refp = {k: np.median([r[k] for r in perf_base]) for k in ("wall_s", "e_cpu_j", "edp")}
    refq = {k: np.median([g[b]["base"][k] for b in g if g[b]["base"]["ok"]]) for k in ("wall_s", "e_cpu_j", "edp")}
    out["ref_ps_sobre_perf"] = {k: round(refq[k] / refp[k], 4) for k in refp}
    return out


def escenario_d():
    s_perf = {(s["scope"], s["set"], s["arm"]): s for s in summarize(valid(D26 / "fase4_D.csv"))}
    s_ps = {(s["scope"], s["set"], s["arm"]): s for s in summarize(valid(D30 / "fase4_D_powersave.csv"))}
    out = {}
    for e in ("chain", "eam", "lj", "rhodo"):
        out[e] = {"activo_perf": s_perf[("lammps", e, "activo_gpu")]["ratio_EDP"],
                   "activo_ps": s_ps[("lammps", e, "activo_gpu")]["ratio_EDP"],
                   "cpuobs_perf": s_perf[("lammps", e, "activo_gpu_cpuobs")]["ratio_EDP"],
                   "cpuobs_ps": s_ps[("lammps", e, "activo_gpu_cpuobs")]["ratio_EDP"]}
    return out


def cloverleaf():
    import csv

    def load(p):
        rows = list(csv.DictReader(open(p)))
        for r in rows:
            r["block"] = int(r["block"])
            for k in ("wall_s", "e_cpu_j", "e_gpu_j"):
                r[k] = float(r[k])
            r["edp"] = (r["e_cpu_j"] + r["e_gpu_j"]) * r["wall_s"]
        return rows

    perf = load(D24 / "cloverleaf_confirm_7692.csv")
    ps = load(D30 / "fase4_cloverleaf_powersave.csv")
    gp = {b: {} for b in range(1, 6)}
    for r in ps:
        gp[r["block"]][r["arm"]] = r
    edp = [gp[b]["activo"]["edp"] / gp[b]["ref"]["edp"] for b in gp]
    gpu = [gp[b]["activo"]["e_gpu_j"] / gp[b]["ref"]["e_gpu_j"] for b in gp]
    sombra = [gp[b]["sombra"]["edp"] / gp[b]["ref"]["edp"] for b in gp]
    gm_edp, ci_edp = _geomean_ci(edp)
    gm_gpu, ci_gpu = _geomean_ci(gpu)
    gm_so, ci_so = _geomean_ci(sombra)
    refp = {k: np.median([r[k] for r in perf if r["arm"] == "ref"]) for k in ("wall_s", "e_cpu_j", "edp")}
    refq = {k: np.median([r[k] for r in ps if r["arm"] == "ref"]) for k in ("wall_s", "e_cpu_j", "edp")}
    return {"p_signo": _sign_p(edp), "edp_geomean": round(gm_edp, 4), "edp_ci95": tuple(round(x, 4) for x in ci_edp),
            "gpu_geomean": round(gm_gpu, 4), "gpu_ci95": tuple(round(x, 4) for x in ci_gpu),
            "sombra_geomean": round(gm_so, 4), "sombra_ci95": tuple(round(x, 4) for x in ci_so),
            "ref_ps_sobre_perf": {k: round(refq[k] / refp[k], 4) for k in refp}}


if __name__ == "__main__":
    print("A:", matriz_inicial())
    print("EA:", escenario_ea())
    print("D:", escenario_d())
    print("CloverLeaf:", cloverleaf())
