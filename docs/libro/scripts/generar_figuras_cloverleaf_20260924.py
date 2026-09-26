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

# Una sola figura: cada magnitud relativa al REF de su bloque, con la mediana (barra) y cada bloque (punto).
fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.1), sharey=True)
paneles = (("wall_s", "Duración"), ("cpu_ratio", "Energía de CPU"), ("gpu_ratio", "Energía de GPU"), ("edp_ratio", "EDP del nodo"))
for row in rows:
    row["wall_ratio"] = row["wall_s"] / ref[row["block"]]["wall_s"]
paneles = (("wall_ratio", "Duración"), ("cpu_ratio", "Energía de CPU"), ("gpu_ratio", "Energía de GPU"), ("edp_ratio", "EDP del nodo"))
for axis, (key, title) in zip(axes, paneles):
    for i, arm in enumerate(ORDER):
        values = [next(r[key] for r in rows if r["block"] == block and r["arm"] == arm) for block in range(1, 6)]
        axis.bar(i, np.median(values), 0.62, color=COLOR[arm])
        axis.plot([i] * len(values), values, "o", color="#1a202c", ms=3)
        axis.text(i, max(values) + 0.006, f"{np.median(values):.3f}", ha="center", fontsize=7)
    axis.axhline(1.0, color="#4a5568", linewidth=0.8, linestyle=":")
    axis.set_title(title, fontsize=9)
    axis.set_xticks(range(3), [LABEL[a] for a in ORDER], fontsize=7.5)
    axis.grid(axis="x", visible=False)
axes[0].set_ylim(0.9, 1.045)
axes[0].set_ylabel("Relativo a REF (mediana y cada bloque)", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig_fase4_cloverleaf_20260926.png", dpi=300)
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
