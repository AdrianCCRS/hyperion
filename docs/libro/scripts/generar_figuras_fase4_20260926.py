"""Figuras y tablas de la Fase 4 (evaluacion del agente frente a REF), campana final del job 7696 (2026-09-25).

Entradas (docs/libro/datos/fase4_20260926/), un CSV por etapa con una fila por celda:
  fase4_A.csv        matriz principal (CPU, GPU y conjunto; vistos e ineditos)
  fase4_C.csv        repeticion independiente del alcance CPU y conjunto
  fase4_noturbo.csv  REF con y sin turbo en CPU y conjunto
  fase4_E.csv        escenario E (GPU dominada por memoria)
  fase4_EA.csv       confirmatorio prospectivo de E-A (cinco bloques)
  fase4_D.csv        LAMMPS
  clasificacion.json puntuacion de clasificacion contra las fronteras reales de fase, por etapa
  ../gpu_calidad_20260922/politica/policy_by_family.json  ganancia por nivel de GPU (Fase 2)
Solo se usan las celdas con state_ok = 1 (estado de frecuencia restaurado y, en el brazo F1 fijo, reloj sostenido bajo carga).
Los tiempos de fase salen de las celdas base (medias de 6 a 10 fases por kernel).
Reproduce: python3 docs/libro/scripts/generar_figuras_fase4_20260926.py
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
D, F = L / "datos" / "fase4_20260926", L / "figuras"
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


def valid(name):
    return [r for r in load_results(D / f"{name}.csv") if r["state_ok"] == "1"]


def summ(name):
    return {(s["scope"], s["set"], s["arm"]): s for s in summarize(valid(name))}


def aplicaciones():
    from matplotlib.patches import Patch
    CPU_C, GPU_C = "#2d3748", "#38a169"
    canon = {"cutlass": "cutlass dgemm", "triad": "stream_triad"}
    orden = []
    for ph in APPS.values():
        for name, _, _ in ph:
            name = canon.get(name, name)
            if name not in orden:
                orden.append(name)
    letra = {n: chr(ord("a") + m) for m, n in enumerate(orden)}
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    keys = list(APPS)[::-1]
    paso, alto, banda = 1.0, 0.76, 0.07
    for i, k in enumerate(keys):
        y = i * paso
        left = 0.0
        for n, (name, cl, dur) in enumerate(APPS[k]):
            dev = k[0] if k[0] != "joint" else ("cpu" if n < 2 else "gpu")
            ax.barh(y, dur, left=left, color=COMPUTE if cl == "c" else MEMORY, edgecolor="white", height=alto)
            ax.barh(y - alto / 2 - banda, dur, left=left, color=CPU_C if dev == "cpu" else GPU_C, edgecolor="white",
                    height=banda * 1.6)
            l = letra[canon.get(name, name)]
            ax.text(left + dur / 2, y + 0.11, l, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            ax.text(left + dur / 2, y - 0.16, f"{dur:.1f} s", ha="center", va="center", color="white", fontsize=6.8 if dur > 6 else 5.8)
            left += dur
        mem = sum(d for _, c, d in APPS[k] if c == "m") / left
        ax.text(left + 1.5, y, f"{100 * mem:.0f} % memoria", va="center", ha="left", fontsize=8, color="#4a5568")
    ax.set_yticks([i * paso for i in range(len(keys))])
    ax.set_yticklabels([f"{NOM[a]} {CONJ[s].split()[0]}" for a, s in keys], fontsize=9.5)
    ax.set_xlabel("Duración de un ciclo (s)", fontsize=9.5)
    ax.set_xlim(0, 110); ax.set_xticks(range(0, 101, 20)); ax.set_ylim(-0.7, (len(keys) - 1) * paso + 0.55)
    ax.grid(axis="y", visible=False)
    clave = "    ".join(f"{letra[n]} = {n}" for n in orden[:4]) + "\n" + "    ".join(f"{letra[n]} = {n}" for n in orden[4:])
    fig.text(0.5, 0.115, clave, ha="center", va="center", fontsize=8.3, linespacing=1.7)
    fig.legend(handles=[Patch(color=COMPUTE, label="compute_bound"), Patch(color=MEMORY, label="memory_bound"),
                        Patch(color=CPU_C, label="Corre en CPU (franja inferior)"), Patch(color=GPU_C, label="Corre en GPU (franja inferior)")],
               loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.115, right=0.99, top=0.98, bottom=0.28)
    fig.savefig(F / "fig_fase4_aplicaciones_20260926.png")
    plt.close(fig)


def edp_alcance():
    s = summ("fase4_A")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=True)
    for ax, sc in zip(axes, ("cpu", "gpu", "joint")):
        arms = [a for a in ARMS if (sc, "known", a) in s]
        w = 0.8 / len(arms)
        for j, a in enumerate(arms):
            vals = [s[(sc, ks, a)]["ratio_EDP"] for ks in ("known", "unseen")]
            b = ax.bar(np.arange(2) + (j - (len(arms) - 1) / 2) * w, vals, w, color=ARMS[a][1], label=ARMS[a][0])
            for x, v in zip(np.arange(2) + (j - (len(arms) - 1) / 2) * w, vals):
                ax.text(x, v + 0.004, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90)
        ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
        ax.set_xticks(range(2)); ax.set_xticklabels(["A", "B"])
        ax.set_title(NOM[sc], fontsize=10)
        ax.legend(frameon=False, fontsize=6.5, loc="upper right")
        ax.set_ylim(0.9, 1.25)
    axes[0].set_ylabel("EDP del nodo relativo a REF")
    fig.tight_layout()
    fig.savefig(F / "fig_fase4_edp_alcance_20260926.png")
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
    s = summ("fase4_A")
    share = {"known": 18.0 / 49.5, "unseen": 8.2 / 16.9}
    ax2 = axes[1]
    x = np.arange(2)
    exp = [share[k] * g[1] for k in ("known", "unseen")]
    med_g = [100 * (1 - s[("gpu", k, "activo")]["ratio_EDPgpu"]) for k in ("known", "unseen")]
    med_n = [100 * (1 - s[("gpu", k, "activo")]["ratio_EDP"]) for k in ("known", "unseen")]
    ax2.bar(x - 0.27, exp, 0.27, color="#cbd5e0", label="Esperada (EDP de GPU)")
    ax2.bar(x, med_g, 0.27, color="#2f855a", label="Medida, EDP de GPU")
    ax2.bar(x + 0.27, med_n, 0.27, color="#4a5568", label="Medida, EDP del nodo")
    for off, vals in ((-0.27, exp), (0, med_g), (0.27, med_n)):
        for xi, v in zip(x + off, vals): ax2.text(xi, max(v, 0) + 0.1, f"{v:.1f}", ha="center", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(["A", "B"])
    ax2.set_ylabel("Ganancia de EDP (%)")
    ax2.set_title("Fase 4, alcance GPU", fontsize=9.5)
    ax2.legend(frameon=False, fontsize=7, loc="upper left")
    ax2.set_ylim(0, 7)
    fig.tight_layout()
    fig.savefig(F / "fig_fase4_gpu_potencial_20260926.png")
    plt.close(fig)


def wrap(spec, head, body):
    return ("\\begin{tabular}{@{}" + spec + "@{}}\n\\toprule\n" + " & ".join(f"\\textbf{{{h}}}" for h in head)
            + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def _per_rep(name):
    from collections import defaultdict
    g = defaultdict(list)
    for r in valid(name):
        g[(r["scope"], r["set"], r["arm"])].append(r)
    return g


def escenario_c():
    s = summ("fase4_C")
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
    fig.savefig(F / "fig_fase4_C_edp_20260926.png")
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
            ax.set_ylim(0.8, 1.28)
            ax.set_title(f"{tit}, {'A (vistos)' if ks == 'known' else 'B (inéditos)'}", fontsize=9)
    for ax in axes[1]:
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(lab, fontsize=7)
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    fig.text(0.012, 0.5, "Relativo a REF (mediana y cada repetición)", rotation=90, va="center", fontsize=9)
    fig.savefig(F / "fig_fase4_E_20260926.png")
    plt.close(fig)


def escenario_d():
    s = summ("fase4_D")
    entradas = ["chain", "eam", "lj", "rhodo"]
    brazos = {"REF": "base", "Sombra": "sombra", "Activo GPU": "activo_gpu", "Activo + CPU obs.": "activo_gpu_cpuobs"}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(entradas))
    offsets = np.linspace(-0.27, 0.27, len(brazos))
    colors = [REF, "#718096", "#2f855a", "#9ae6b4"]
    for (label, arm), offset, color in zip(brazos.items(), offsets, colors):
        med = [s[("lammps", e, arm)]["ratio_EDP"] for e in entradas]
        ax.bar(x + offset, med, width=0.16, color=color, label=label, zorder=2)
    ax.axhline(1, color="#4a5568", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(entradas)
    ax.set_ylabel("EDP del nodo relativo a REF")
    ax.set_ylim(0.97, 1.06)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(F / "fig_fase4_D_20260926.png")
    plt.close(fig)
    rows = []
    for e in entradas:
        for a in brazos.values():
            r = s[("lammps", e, a)]
            rows.append(f"{e if a == 'base' else ''} & {NOMD[a]} & {r['n']} & {r['wall_s']:.3f} & {r['e_cpu_j']:.0f} & {r['e_gpu_j']:.0f} & "
                        f"{r['ratio_EDP']:.3f} & {r['ratio_E_gpu']:.3f} \\\\")
        rows.append("\\midrule")
    (D / "tabla_D.tex").write_text(wrap("llrrrrrr", ["Entrada", "Brazo", "$n$", "$T$ (s)", "$E_{\\mathrm{CPU}}$ (J)", "$E_{\\mathrm{GPU}}$ (J)", "EDP/REF", "$E_{\\mathrm{GPU}}$/REF"], "\n".join(rows[:-1])))


NOMD = {"base": "REF", "sombra": "Sombra", "activo_gpu": "Activo GPU", "activo_gpu_cpuobs": "Activo + CPU obs."}


def resumen_ea():
    """Confirmatorio E-A: razones por bloque frente a REF, mediana y prueba exacta bilateral de signos por bloques."""
    from itertools import product
    g = {}
    for r in load_results(D / "fase4_EA.csv"):
        g.setdefault(int(r["rep"]), {})[r["arm"]] = r | {"ok": r["state_ok"] == "1"}
    out = {"bloques": len(g), "brazos": {}}
    for arm in ("sombra", "activo_gpu", "fijo_gpu_f1"):
        val = [b for b in g if g[b][arm]["ok"] and g[b]["base"]["ok"]]
        ratios = {k: [g[b][arm][k] / g[b]["base"][k] for b in val] for k in ("wall_s", "e_gpu_j", "edp")}
        n_fav = sum(x < 1 for x in ratios["edp"])
        p = min(1.0, 2 * sum(1 for signs in product((0, 1), repeat=len(val)) if sum(signs) >= max(n_fav, len(val) - n_fav)) / 2 ** len(val)) if val else None
        out["brazos"][arm] = {"validos": len(val), "favorables_edp": n_fav, "p_exacto": p,
                              **{k: (round(float(np.median(v)), 4) if v else None) for k, v in ratios.items()}, "edp_por_bloque": [round(x, 4) for x in ratios["edp"]]}
    (D / "resumen_EA.json").write_text(json.dumps(out, indent=1))


def tablas_ce():
    nom = {"base": "REF", "base_noturbo": "REF sin turbo", "sombra": "Sombra", "activo": "Activo", "activo_f0": "Activo F0",
           "activo_nofloor": "Activo sin piso", "activo_gpu": "Activo", "activo_gpu_cpuobs": "Activo + CPU obs."}
    # escenario C
    s = summ("fase4_C")
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
    for name in ("fase4_noturbo", "fase4_E"):
        s2 = summ(name)
        for (sc, ks, a), r in sorted(s2.items()):
            if a == "base_noturbo":
                rows.append(f"{NOM[sc]} & {CONJ[ks].split()[0]} & {r['n']} & {r['ratio_T']:.3f} & {r['ratio_E_cpu']:.3f} & {r['ratio_E_gpu']:.3f} & {r['ratio_EDP']:.3f} \\\\"
                            if "ratio_T" in r else "")
    # ratios de base_noturbo frente a REF: se calculan a partir de los datos (base_noturbo es el brazo, base la referencia)
    rows = []
    for name in ("fase4_noturbo", "fase4_E"):
        g = _per_rep(name)
        for (sc, ks, a) in sorted(k for k in g if k[2] == "base_noturbo" and k[:2] != ("gpumem", "unseen")):  # E-B: REF bimodal (myocyte), la mediana no es comparable
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
    js = json.load(open(D / "clasificacion.json"))
    out = []
    for name, lab in (("C", "C"), ("E", "E")):
        for r in js[name]["score"]:
            if r["arm"] not in ("activo", "sombra", "activo_gpu") or (r["ok"] + r["wrong"] + r["abst"] == 0):
                continue
            out.append(f"{lab} & {NOM[r['scope']]} & {CONJ[r['set']].split()[0]} & {r['device'].upper()} & {nom[r['arm']]} & {r['ok']} & {r['wrong']} & {r['abst']} & "
                       f"{r['acc_decided']:.3f} & {r['covered']}/{r['phases']} \\\\")
    (D / "tabla_clasificacion_CE.tex").write_text(wrap("lllllrrrcc", ["Esc.", "Alcance", "Kernels", "Agente", "Brazo", "Correctas", "Erróneas", "Abst.", "Exactitud", "Cobertura"], "\n".join(out)))


def tablas():
    s = summ("fase4_A")
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
    sc_ = json.load(open(D / "clasificacion.json"))["A"]["score"]
    out = []
    for r in sc_:
        if r["arm"] not in ("activo", "sombra"):
            continue
        dev = r["device"].upper()
        out.append(f"{NOM[r['scope']]} & {CONJ[r['set']].split()[0]} & {dev} & {ARMS[r['arm']][0]} & {r['ok']} & {r['wrong']} & {r['abst']} & "
                   f"{r['acc_decided']:.3f} & {r['covered']}/{r['phases']} \\\\")
    (D / "tabla_clasificacion.tex").write_text(wrap("llllrrrcc", ["Alcance", "Kernels", "Agente", "Brazo", "Correctas", "Erróneas", "Abst.", "Exactitud", "Cobertura"], "\n".join(out)))

    ac = {(r["scope"], r["set"], r["arm"]): r for r in summarize(valid("fase4_A"))}
    rel = []
    for sc in ("cpu", "joint"):
        for ks in ("known", "unseen"):
            a, b = ac[(sc, ks, "activo")], ac[(sc, ks, "sombra")]
            rel.append(f"{NOM[sc]} & {CONJ[ks].split()[0]} & {b['ratio_EDP']:.3f} & {a['ratio_EDP']:.3f} & {a['ratio_EDP'] / b['ratio_EDP']:.3f} \\\\")
    (D / "tabla_activo_vs_sombra.tex").write_text(wrap("llccc", ["Alcance", "Kernels", "Sombra/REF", "Activo/REF", "Activo/sombra"], "\n".join(rel)))


if __name__ == "__main__":
    aplicaciones(); edp_alcance(); gpu_niveles(); tablas(); escenario_c(); escenario_e(); escenario_d(); tablas_ce(); resumen_ea()
    print("ok")
