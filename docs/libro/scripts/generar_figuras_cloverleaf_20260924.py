"""Genera figuras del confirmatorio CloverLeaf a partir de results.csv."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BOOK = ROOT / "docs" / "libro"
DATA = BOOK / "datos" / "fase4_20260924" / "cloverleaf_confirm_7692.csv"
FIG = BOOK / "figuras"
ORDER = ["ref", "sombra", "activo"]
LABEL = {"ref": "REF", "sombra": "Sombra", "activo": "Activo"}
COLOR = {"ref": "#a0aec0", "sombra": "#718096", "activo": "#2f855a"}

with DATA.open(newline="") as stream:
    rows = list(csv.DictReader(stream))

for row in rows:
    row["block"] = int(row["block"])
    for key in ("wall_s", "e_cpu_j", "e_gpu_j"):
        row[key] = float(row[key])
    row["edp"] = (row["e_cpu_j"] + row["e_gpu_j"]) * row["wall_s"]

ref = {row["block"]: row for row in rows if row["arm"] == "ref"}
for row in rows:
    baseline = ref[row["block"]]
    row["edp_ratio"] = row["edp"] / baseline["edp"]
    row["cpu_ratio"] = row["e_cpu_j"] / baseline["e_cpu_j"]
    row["gpu_ratio"] = row["e_gpu_j"] / baseline["e_gpu_j"]

# EDP por bloque, con la mediana de cada brazo y la referencia de paridad.
fig, ax = plt.subplots(figsize=(7.2, 3.5))
positions = np.arange(5)
for arm in ORDER:
    values = [next(r["edp_ratio"] for r in rows if r["block"] == block and r["arm"] == arm) for block in range(1, 6)]
    ax.plot(positions, values, "o-", color=COLOR[arm], label=LABEL[arm], linewidth=1.4, markersize=5)
    ax.axhline(np.median(values), color=COLOR[arm], linewidth=0.9, alpha=0.55, linestyle="--")
ax.axhline(1.0, color="#4a5568", linewidth=0.8, linestyle=":")
ax.set_xticks(positions, ["1", "2", "3", "4", "5"])
ax.set_xlabel("Bloque")
ax.set_ylabel("EDP del nodo relativo a REF")
ax.set_ylim(0.90, 1.08)
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(FIG / "fig_fase4_cloverleaf_edp_20260924.png", dpi=300)
plt.close(fig)

# Energía CPU y GPU por brazo, normalizada al REF de cada bloque.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for axis, key, title in zip(axes, ("cpu_ratio", "gpu_ratio"), ("Energía de CPU", "Energía de GPU")):
    for arm in ORDER:
        values = [next(r[key] for r in rows if r["block"] == block and r["arm"] == arm) for block in range(1, 6)]
        x = np.arange(5) + (ORDER.index(arm) - 1) * 0.24
        axis.plot(x, values, "o", color=COLOR[arm], label=LABEL[arm], markersize=5)
        axis.hlines(np.median(values), x[0] - 0.08, x[-1] + 0.08, color=COLOR[arm], linewidth=1.1)
    axis.axhline(1.0, color="#4a5568", linewidth=0.8, linestyle=":")
    axis.set_title(title)
    axis.set_xticks(np.arange(5), ["1", "2", "3", "4", "5"])
    axis.set_xlabel("Bloque")
axes[0].set_ylabel("Energía relativa a REF")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig_fase4_cloverleaf_energia_20260924.png", dpi=300)
plt.close(fig)

# Tabla LaTeX reproducible desde los mismos datos.
lines = [
    r"\begin{tabular}{@{}lrrrrrr@{}}",
    r"\toprule",
    r"\textbf{Brazo} & \textbf{$n$} & \textbf{$T$ (s)} & \textbf{$E_{\mathrm{CPU}}$ (J)} & \textbf{$E_{\mathrm{GPU}}$ (J)} & \textbf{EDP/REF} & \textbf{$E_{\mathrm{GPU}}$/REF} \\",
    r"\midrule",
]
for arm in ORDER:
    selected = [row for row in rows if row["arm"] == arm]
    median = lambda key: float(np.median([row[key] for row in selected]))
    edp_ratio = median("edp") / median("edp") if arm == "ref" else median("edp") / median("edp")
    ref_edp = float(np.median([row["edp"] for row in rows if row["arm"] == "ref"]))
    ref_gpu = float(np.median([row["e_gpu_j"] for row in rows if row["arm"] == "ref"]))
    lines.append(f"{LABEL[arm]} & 5 & {median('wall_s'):.3f} & {median('e_cpu_j'):.1f} & {median('e_gpu_j'):.1f} & {median('edp') / ref_edp:.4f} & {median('e_gpu_j') / ref_gpu:.4f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(DATA.parent / "tabla_cloverleaf.tex").write_text("\n".join(lines) + "\n")
