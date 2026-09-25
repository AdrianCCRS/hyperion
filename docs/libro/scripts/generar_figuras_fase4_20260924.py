"""Figuras y tablas de la Fase 4 (evaluacion del agente frente a REF), ronda 2026-09-24.

Entradas (docs/libro/datos/fase4_20260924/):
  fase4_matrix_main.csv  matriz completa, muestreo de 1 ms (job 7639)
  fase4_cpu_10ms.csv, fase4_joint_10ms.csv  repeticion a 10 ms (jobs 7642, 7643)
  fase4_cpu_diag.csv     configuraciones de inferencia del agente de CPU (job 7645)
  clasificacion.json     puntuacion de clasificacion contra las fronteras reales de fase
  fase4_C_ort1.csv, fase4_C_noturbo.csv, fase4_E.csv  escenarios C (agente con un hilo de inferencia), REF sin turbo y E
  clasificacion_CE.json  puntuacion de clasificacion de esas tres corridas
  ../gpu_calidad_20260922/politica/policy_by_family.json  ganancia por nivel de GPU (Fase 2)
Los tiempos de fase salen de las celdas base de la matriz (medias de 6 a 10 fases por kernel).
Reproduce: python3 docs/libro/scripts/generar_figuras_fase4_20260924.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from fase4_evaluacion.analyze_matrix import load_results, summarize  # noqa: E402

L = ROOT / "docs" / "libro"
D, F = L / "datos" / "fase4_20260924", L / "figuras"
COMPUTE, MEMORY, REF = "#2b6cb0", "#dd6b20", "#a0aec0"
ARMS = {"base": ("REF", REF), "base_noturbo": ("REF sin turbo", "#cbd5e0"), "sombra": ("Sombra", "#718096"), "activo": ("Activo", "#2f855a"),
        "activo_f0": ("Activo F0", "#68d391"), "activo_nofloor": ("Activo sin piso", "#9ae6b4"),
        "activo_gpu": ("Activo (sin agente de CPU)", "#2f855a"), "activo_gpu_cpuobs": ("Activo + agente de CPU en observación", "#9ae6b4")}
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#8a8a8a", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dcdcdc", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 300,
})

# (alcance, conjunto) -> [(kernel, clase, duracion_s)] de un ciclo, medidas en las celdas base
APPS = {
    ("cpu", "known"): [("dgemm", "c", 4.2), ("npb_cg", "m", 10.1)],
    ("cpu", "unseen"): [("rsbench", "c", 65.7), ("xsbench", "m", 12.0)],
    ("gpu", "known"): [("cutlass dgemm", "c", 31.5), ("stream_triad", "m", 18.0)],
    ("gpu", "unseen"): [("BabelStream", "m", 8.2), ("lavamd", "c", 8.7)],
    ("joint", "known"): [("dgemm", "c", 4.2), ("npb_cg", "m", 10.1), ("cutlass", "c", 31.5), ("triad", "m", 18.0)],
    ("joint", "unseen"): [("rsbench", "c", 65.6), ("xsbench", "m", 12.0), ("BabelStream", "m", 8.3), ("lavamd", "c", 8.7)],
}
NOM = {"cpu": "CPU", "gpu": "GPU", "joint": "Conjunto", "gpumem": "GPU (memoria)"}
CONJ = {"known": "A (vistos)", "unseen": "B (inéditos)"}


def summ(name):
    return {(s["scope"], s["set"], s["arm"]): s for s in summarize(load_results(D / f"{name}.csv"))}


def aplicaciones():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    keys = list(APPS)[::-1]
    for i, k in enumerate(keys):
        left = 0
        for name, cl, dur in APPS[k]:
            ax.barh(i, dur, left=left, color=COMPUTE if cl == "c" else MEMORY, edgecolor="white", height=0.62)
            txt = f"{name}\n{dur:.1f} s" if dur >= 12 else f"{dur:.1f}"
            ax.text(left + dur / 2, i, txt, ha="center", va="center", color="white", fontsize=7 if dur >= 12 else 6.5)
            left += dur
        mem = sum(d for _, c, d in APPS[k] if c == "m") / left
        ax.text(left + 2, i, f"{100 * mem:.0f} % memoria", va="center", fontsize=8, color="#4a5568")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{NOM[a]} {CONJ[s].split()[0]}" for a, s in keys])
    ax.set_xlabel("Duración de un ciclo de la aplicación (s)")
    ax.set_xlim(0, 125)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COMPUTE, label="compute_bound"), Patch(color=MEMORY, label="memory_bound")],
              frameon=False, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(F / "fig_fase4_aplicaciones_20260924.png")
    plt.close(fig)


def edp_alcance():
    s = summ("fase4_matrix_main")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=True)
    for ax, sc in zip(axes, ("cpu", "gpu", "joint")):
        arms = [a for a in ARMS if (sc, "known", a) in s]
        w = 0.8 / len(arms)
        for j, a in enumerate(arms):
            vals = [s[(sc, ks, a)]["ratio_EDP"] for ks in ("known", "unseen")]
            b = ax.bar(np.arange(2) + (j - (len(arms) - 1) / 2) * w, vals, w, color=ARMS[a][1], label=ARMS[a][0])
            for x, v in zip(np.arange(2) + (j - (len(arms) - 1) / 2) * w, vals):
                ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90)
        ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
        ax.set_xticks(range(2)); ax.set_xticklabels(["A", "B"])
        ax.set_title(NOM[sc], fontsize=10)
        ax.legend(frameon=False, fontsize=6.5, loc="upper right")
        ax.set_ylim(0, 2.6)
    axes[0].set_ylabel("EDP del nodo relativo a REF")
    fig.tight_layout()
    fig.savefig(F / "fig_fase4_edp_alcance_20260924.png")
    plt.close(fig)


def costo_daemon():
    d = summ("fase4_cpu_diag")
    m = summ("fase4_matrix_main"); t = summ("fase4_cpu_10ms")
    cfg = [("Por defecto\n1 ms", m[("cpu", "known", "sombra")]),
           ("Por defecto\n10 ms", t[("cpu", "known", "sombra")]),
           ("Por defecto\n(diagnóstico)", d[("cpu", "known", "sombra")]),
           ("Un hilo de\ninferencia", d[("cpu", "known", "sombra_ort1")])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, key, tit in zip(axes, ("ratio_T", "ratio_E_cpu"), ("Tiempo", "Energía de CPU")):
        vals = [c[1][key] for c in cfg]
        ax.bar(range(len(cfg)), vals, color=["#718096"] * 3 + ["#2f855a"])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
        ax.set_xticks(range(len(cfg))); ax.set_xticklabels([c[0] for c in cfg], fontsize=7.5)
        ax.set_title(f"{tit} del brazo sombra frente a REF", fontsize=9)
        ax.set_ylim(0, 1.9)
    fig.tight_layout()
    fig.savefig(F / "fig_fase4_costo_agente_cpu_20260924.png")
    plt.close(fig)


def gpu_niveles():
    pol = json.load(open(L / "datos" / "gpu_calidad_20260922" / "politica" / "policy_by_family.json"))["memory_bound"]["levels"]
    lv = [k for k in pol if int(k[1:]) <= 6]
    g = np.array([pol[k]["gain"] for k in lv]) * 100
    lo = np.array([pol[k]["gain_ci95"][0] for k in lv]) * 100
    hi = np.array([pol[k]["gain_ci95"][1] for k in lv]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.errorbar(range(len(lv)), g, yerr=[g - lo, hi - g], fmt="o", color=MEMORY, capsize=3)
    ax.axhline(0, color="#4a5568", lw=0.8, ls=":")
    ax.set_xticks(range(len(lv))); ax.set_xticklabels(lv)
    ax.set_xlabel("Nivel de reloj de GPU (F0 = referencia, F1 = 1260 MHz)")
    ax.set_ylabel("Ganancia de EDP en memory_bound (%)")
    ax.set_title("Fase 2: ganancia por nivel", fontsize=9.5)
    s = summ("fase4_matrix_main")
    share = {"known": 18.0 / 49.5, "unseen": 8.2 / 16.9}
    ax2 = axes[1]
    x = np.arange(2)
    exp = [share[k] * g[1] for k in ("known", "unseen")]
    med = [100 * (1 - s[("gpu", k, "activo")]["ratio_EDP"]) for k in ("known", "unseen")]
    ax2.bar(x - 0.18, exp, 0.36, color="#cbd5e0", label="Esperada (fracción en memoria × ganancia de F1)")
    ax2.bar(x + 0.18, med, 0.36, color="#2f855a", label="Medida (activo frente a REF)")
    for xi, v in zip(x - 0.18, exp): ax2.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x + 0.18, med): ax2.text(xi, max(v, 0) + 0.1, f"{v:.1f}", ha="center", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(["A", "B"])
    ax2.set_ylabel("Ganancia de EDP del nodo (%)")
    ax2.set_title("Fase 4, alcance GPU", fontsize=9.5)
    ax2.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax2.set_ylim(0, 6.5)
    fig.tight_layout()
    fig.savefig(F / "fig_fase4_gpu_potencial_20260924.png")
    plt.close(fig)


def wrap(spec, head, body):
    return ("\\begin{tabular}{@{}" + spec + "@{}}\n\\toprule\n" + " & ".join(f"\\textbf{{{h}}}" for h in head)
            + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def _per_rep(name):
    from collections import defaultdict
    g = defaultdict(list)
    for r in load_results(D / f"{name}.csv"):
        g[(r["scope"], r["set"], r["arm"])].append(r)
    return g


def escenario_c():
    s = summ("fase4_C_ort1")
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    groups = [("cpu", "known"), ("cpu", "unseen"), ("joint", "known"), ("joint", "unseen")]
    arms = ["sombra", "activo", "activo_f0", "activo_nofloor"]
    w = 0.2
    for j, a in enumerate(arms):
        xs, vals = [], []
        for i, (sc, ks) in enumerate(groups):
            if (sc, ks, a) in s:
                xs.append(i + (j - 1.5) * w); vals.append(s[(sc, ks, a)]["ratio_EDP"])
        ax.bar(xs, vals, w, color=ARMS[a][1], label=ARMS[a][0])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.004, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90)
    ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{NOM[a]} {CONJ[k].split()[0]}" for a, k in groups])
    ax.set_ylim(0.95, 1.22)
    ax.set_ylabel("EDP del nodo relativo a REF")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(F / "fig_fase4_C_edp_20260924.png")
    plt.close(fig)


def escenario_e():
    g = _per_rep("fase4_E")
    arms = ["base", "base_noturbo", "sombra", "activo_gpu", "activo_gpu_cpuobs"]
    lab = ["REF", "REF sin\nturbo", "Sombra", "Activo", "Activo +\nagente de CPU"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex=True)
    for col, ks in enumerate(("known", "unseen")):
        refs = {"edp": np.median([r["edp"] for r in g[("gpumem", ks, "base")]]), "e_gpu_j": np.median([r["e_gpu_j"] for r in g[("gpumem", ks, "base")]])}
        for row, (key, tit) in enumerate((("edp", "EDP del nodo"), ("e_gpu_j", "Energía de GPU"))):
            ax = axes[row, col]
            for i, a in enumerate(arms):
                v = [r[key] / refs[key] for r in g[("gpumem", ks, a)]]
                ax.bar(i, np.median(v), 0.6, color=ARMS[a][1] if a not in ("activo_gpu", "activo_gpu_cpuobs") else ("#2f855a" if a == "activo_gpu" else "#9ae6b4"))
                ax.plot([i] * len(v), v, "o", color="#1a202c", ms=3)
                ax.text(i, max(v) + 0.012, f"{np.median(v):.3f}", ha="center", fontsize=7)
            ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
            ax.set_ylim(0.85, 1.28)
            ax.set_title(f"{tit}, {'A (vistos)' if ks == 'known' else 'B (inéditos)'}", fontsize=9)
    for ax in axes[1]:
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(lab, fontsize=7)
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    fig.text(0.012, 0.5, "Relativo a REF (mediana y cada repetición)", rotation=90, va="center", fontsize=9)
    fig.savefig(F / "fig_fase4_E_20260924.png")
    plt.close(fig)


def escenario_d():
    entradas = ["chain", "eam", "lj", "rhodo"]
    brazos = {
        "REF": [1.000, 1.000, 1.000, 1.000],
        "Sombra": [1.003, 1.004, 0.999, 1.004],
        "Activo GPU": [1.001, 0.998, 1.002, 1.008],
        "Activo + CPU obs.": [1.030, 1.043, 1.036, 1.034],
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(entradas))
    offsets = np.linspace(-0.27, 0.27, len(brazos))
    colors = [REF, "#718096", "#2f855a", "#9ae6b4"]
    for (label, med), offset, color in zip(brazos.items(), offsets, colors):
        ax.bar(x + offset, med, width=0.16, color=color, label=label, zorder=2)
    ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(entradas)
    ax.set_ylabel("EDP del nodo relativo a REF")
    ax.set_ylim(0.97, 1.06)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(F / "fig_fase4_D_20260924.png")
    plt.close(fig)


def tablas_ce():
    nom = {"base": "REF", "base_noturbo": "REF sin turbo", "sombra": "Sombra", "activo": "Activo", "activo_f0": "Activo F0",
           "activo_nofloor": "Activo sin piso", "activo_gpu": "Activo", "activo_gpu_cpuobs": "Activo + CPU obs."}
    # escenario C
    s = summ("fase4_C_ort1")
    rows = []
    for sc in ("cpu", "joint"):
        for ks in ("known", "unseen"):
            for a in ("base", "sombra", "activo", "activo_f0", "activo_nofloor"):
                r = s.get((sc, ks, a))
                if r is None:
                    continue
                rows.append(f"{NOM[sc]} & {CONJ[ks].split()[0]} & {nom[a]} & {r['n']} & {r['wall_s']:.1f} & {r['e_cpu_j']:.0f} & {r['e_gpu_j']:.0f} & "
                            f"{r['ratio_T']:.3f} & {r['ratio_E_cpu']:.3f} & {r['ratio_E_tot']:.3f} & {r['ratio_EDP']:.3f} \\\\")
            rows.append("\\midrule")
    (D / "tabla_C.tex").write_text(wrap("llllrrrrrrr", ["Alc.", "Kern.", "Brazo", "$n$", "$T$ (s)", "$E_{\\mathrm{CPU}}$ (J)", "$E_{\\mathrm{GPU}}$ (J)",
                                                         "$T$/REF", "$E_{\\mathrm{CPU}}$/REF", "$E_{\\mathrm{tot}}$/REF", "EDP/REF"], "\n".join(rows[:-1])))
    # REF sin turbo (C_noturbo y E)
    rows = []
    for name in ("fase4_C_noturbo", "fase4_E"):
        s2 = summ(name)
        for (sc, ks, a), r in sorted(s2.items()):
            if a == "base_noturbo":
                rows.append(f"{NOM[sc]} & {CONJ[ks].split()[0]} & {r['n']} & {r['ratio_T']:.3f} & {r['ratio_E_cpu']:.3f} & {r['ratio_E_gpu']:.3f} & {r['ratio_EDP']:.3f} \\\\"
                            if "ratio_T" in r else "")
    # ratios de base_noturbo frente a REF: se calculan a partir de los datos (base_noturbo es el brazo, base la referencia)
    rows = []
    for name in ("fase4_C_noturbo", "fase4_E"):
        g = _per_rep(name)
        for (sc, ks, a) in sorted(k for k in g if k[2] == "base_noturbo"):
            f = lambda m, arm: np.median([r[m] for r in g[(sc, ks, arm)]])
            rows.append(f"{NOM[sc]} & {CONJ[ks].split()[0]} & {len(g[(sc, ks, a)])} & " + " & ".join(
                f"{f(m, 'base_noturbo') / f(m, 'base'):.3f}" for m in ("wall_s", "e_cpu_j", "e_gpu_j", "edp")) + " \\\\")
    (D / "tabla_noturbo.tex").write_text(wrap("llccccc", ["Alcance", "Kernels", "$n$", "$T$", "$E_{\\mathrm{CPU}}$", "$E_{\\mathrm{GPU}}$", "EDP"], "\n".join(rows)))
    # escenario E
    s = summ("fase4_E")
    rows = []
    for ks in ("known", "unseen"):
        for a in ("base", "base_noturbo", "sombra", "activo_gpu", "activo_gpu_cpuobs"):
            r = s.get(("gpumem", ks, a))
            rows.append(f"{CONJ[ks].split()[0]} & {nom[a]} & {r['n']} & {r['wall_s']:.1f} & {r['e_cpu_j']:.0f} & {r['e_gpu_j']:.0f} & "
                        f"{r['ratio_T']:.3f} & {r['ratio_E_gpu']:.3f} & {r['ratio_EDP']:.3f} & {r['ratio_EDPgpu']:.3f} & "
                        f"{r['ratio_EDP_nt']:.3f} \\\\")
        rows.append("\\midrule")
    (D / "tabla_E.tex").write_text(wrap("llrrrrrrrrr", ["Kern.", "Brazo", "$n$", "$T$ (s)", "$E_{\\mathrm{CPU}}$ (J)", "$E_{\\mathrm{GPU}}$ (J)", "$T$/REF",
                                                        "$E_{\\mathrm{GPU}}$/REF", "EDP/REF", "EDP GPU/REF", "EDP/REF sin turbo"], "\n".join(rows[:-1])))
    # clasificacion de C y E
    js = json.load(open(D / "clasificacion_CE.json"))
    out = []
    for name, lab in (("fase4_C_ort1", "C"), ("fase4_E", "E")):
        for r in js[name]["score"]:
            if r["arm"] not in ("activo", "sombra", "activo_gpu") or (r["ok"] + r["wrong"] + r["abst"] == 0):
                continue
            out.append(f"{lab} & {NOM[r['scope']]} & {CONJ[r['set']].split()[0]} & {r['device'].upper()} & {nom[r['arm']]} & {r['ok']} & {r['wrong']} & {r['abst']} & "
                       f"{r['acc_decided']:.3f} & {r['covered']}/{r['phases']} \\\\")
    (D / "tabla_clasificacion_CE.tex").write_text(wrap("lllllrrrcc", ["Esc.", "Alcance", "Kernels", "Agente", "Brazo", "Correctas", "Erróneas", "Abst.", "Exactitud", "Cobertura"], "\n".join(out)))


def tablas():
    s = summ("fase4_matrix_main")
    rows = []
    for sc in ("cpu", "gpu", "joint"):
        for ks in ("known", "unseen"):
            for a in ARMS:
                r = s.get((sc, ks, a))
                if r is None:
                    continue
                p = f"{r['p_edp']:.4f}" if r.get("p_edp") is not None else "--"
                rows.append(f"{NOM[sc]} & {CONJ[ks].split()[0]} & {ARMS[a][0]} & {r['n']} & {r['wall_s']:.1f} & {r['e_cpu_j']:.0f} & {r['e_gpu_j']:.0f} & "
                            f"{r['ratio_T']:.3f} & {r['ratio_E_tot']:.3f} & {r['ratio_EDP']:.3f} & {p} \\\\")
            rows.append("\\midrule")
    (D / "tabla_resultados.tex").write_text(wrap("llllrrrrrrr", ["Alc.", "Kern.", "Brazo", "$n$", "$T$ (s)", "$E_{\\mathrm{CPU}}$ (J)", "$E_{\\mathrm{GPU}}$ (J)", "$T/T_{\\mathrm{REF}}$", "$E_{\\mathrm{tot}}$/REF", "EDP/REF", "$p$"], "\n".join(rows[:-1])))
    sc_ = json.load(open(D / "clasificacion.json"))["fase4_matrix_main"]["score"]
    out = []
    for r in sc_:
        if r["arm"] not in ("activo", "sombra"):
            continue
        dev = r["device"].upper()
        out.append(f"{NOM[r['scope']]} & {CONJ[r['set']].split()[0]} & {dev} & {ARMS[r['arm']][0]} & {r['ok']} & {r['wrong']} & {r['abst']} & "
                   f"{r['acc_decided']:.3f} & {r['covered']}/{r['phases']} \\\\")
    (D / "tabla_clasificacion.tex").write_text(wrap("llllrrrcc", ["Alcance", "Kernels", "Agente", "Brazo", "Correctas", "Erróneas", "Abst.", "Exactitud", "Cobertura"], "\n".join(out)))

    d = summ("fase4_cpu_diag"); m = summ("fase4_matrix_main"); t = summ("fase4_cpu_10ms")
    acc = {r["arm"]: r["acc_decided"] for r in json.load(open(D / "clasificacion.json"))["fase4_cpu_diag"]["score"]}
    accm = {(r["scope"], r["set"], r["arm"]): r["acc_decided"] for r in json.load(open(D / "clasificacion.json"))["fase4_matrix_main"]["score"] if r["device"] == "cpu"}
    acct = {(r["scope"], r["set"], r["arm"]): r["acc_decided"] for r in json.load(open(D / "clasificacion.json"))["fase4_cpu_10ms"]["score"]}
    cfg = [("Pool por defecto, 1 ms", m[("cpu", "known", "sombra")], accm[("cpu", "known", "sombra")]),
           ("Pool por defecto, 10 ms", t[("cpu", "known", "sombra")], acct[("cpu", "known", "sombra")]),
           ("Pool por defecto, 10 ms (diagnóstico)", d[("cpu", "known", "sombra")], acc["sombra"]),
           ("Consumidor fijado a un núcleo", d[("cpu", "known", "sombra_pin")], acc["sombra_pin"]),
           ("Un hilo de inferencia", d[("cpu", "known", "sombra_ort1")], acc["sombra_ort1"]),
           ("Un hilo y consumidor fijado", d[("cpu", "known", "sombra_pin_ort1")], acc["sombra_pin_ort1"])]
    (D / "tabla_costo_agente.tex").write_text(wrap("lcccc", ["Configuración", "$T$/REF", "$E_{\\mathrm{CPU}}$/REF", "EDP/REF", "Exactitud"], "\n".join(
        f"{n} & {r['ratio_T']:.3f} & {r['ratio_E_cpu']:.3f} & {r['ratio_EDP']:.3f} & {a:.3f} \\\\" for n, r, a in cfg)))
    ac = {(r["scope"], r["set"], r["arm"]): r for r in summarize(load_results(D / "fase4_matrix_main.csv"))}
    rel = []
    for sc in ("cpu", "joint"):
        for ks in ("known", "unseen"):
            a, b = ac[(sc, ks, "activo")], ac[(sc, ks, "sombra")]
            rel.append(f"{NOM[sc]} & {CONJ[ks].split()[0]} & {b['ratio_EDP']:.3f} & {a['ratio_EDP']:.3f} & {a['ratio_EDP'] / b['ratio_EDP']:.3f} \\\\")
    (D / "tabla_activo_vs_sombra.tex").write_text(wrap("llccc", ["Alcance", "Kernels", "Sombra/REF", "Activo/REF", "Activo/sombra"], "\n".join(rel)))


if __name__ == "__main__":
    aplicaciones(); edp_alcance(); costo_daemon(); gpu_niveles(); tablas(); escenario_c(); escenario_e(); escenario_d(); tablas_ce()
    print("ok")
