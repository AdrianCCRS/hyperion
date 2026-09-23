"""Construye el artefacto canónico policy_table.yaml que el daemon carga.

Reemplaza el rol de derive_policy_table.py (retirado como derivador, ver
Plan_Fase3_Daemon.md §0.3): este script NO calcula EDP ni corre pruebas de
significancia. Combina las tablas de política ya derivadas en Fase 2
--- las que realmente sustentan el libro, unidad kernel/familia, con IC95
por bootstrap --- y resuelve la frecuencia física real para las clases
donde la política es "actuar", que es lo único que
`gpu_loop/loop.py::build_controller_from_policy()` necesita para no
lanzar `ValueError`.

Entradas (comprometidas en el repo, no en tmp/):
  --cpu-policy   docs/libro/datos/cpu_calidad_30fam/politica/policy_cpu.json
                 (unidad: kernel: ya trae acción/nivel/IC95 correctos)
  --gpu-policy   docs/libro/datos/gpu_calidad_20260922/politica/policy_by_family.json
                 (unidad: familia, des-duplicada -- NO usar policy_gpu.json,
                 que es el análisis por kernel y elige F2 en vez de F1 al
                 contar dos veces dual_axpy/dual_spmv/dual_stencil)
  --gpu-dataset  tmp/historical_gpu_relaxed020_20260922.csv (para resolver
                 el reloj SM real medido en el nivel elegido)

Salida: policy_table.yaml con las 4 entradas (cpu-compute_bound,
cpu-memory_bound, gpu-compute_bound, gpu-memory_bound), mismo esquema que
ya consume build_controller_from_policy() y cpu_phase_controller.hpp
(action, resolved_freq_khz/resolved_clock_mhz cuando action=="actuar").
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

MIN_EFFECT = 0.01  # mismo criterio declarado que cpu_policy_table.py/gpu_policy_table.py
LEVELS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]

# Rejilla de CPU de la campana final (scripts/pacca/final_campaign/cpu_final.yaml):
# el nivel se declara por `fraction` sobre [800, 3200] MHz (turbo apagado, 3200 es
# la frecuencia base), y coincide con la Tabla de frecuencias del libro
# (F0=3200, F1=2900, ..., F8=800). Los IDs F* NO son iguales entre campanas
# (el cribado usa otra convencion): estos valen solo para la campana final.
CPU_RANGE_KHZ = (800_000, 3_200_000)
CPU_FINAL_FRACTIONS = {"F0": 1.0, "F1": 0.875, "F2": 0.75, "F3": 0.625, "F4": 0.5,
                       "F5": 0.375, "F6": 0.25, "F7": 0.125, "F8": 0.0}


def cpu_level_khz(level: str) -> int:
    if level not in CPU_FINAL_FRACTIONS:
        raise ValueError(f"nivel de CPU desconocido: {level!r} (validos: {sorted(CPU_FINAL_FRACTIONS)})")
    low, high = CPU_RANGE_KHZ
    return round(low + CPU_FINAL_FRACTIONS[level] * (high - low))


def cpu_entry(policy_cpu: dict, cls: str, experimental_level: str | None = None) -> dict:
    """La política de CPU ya viene en la unidad y con la prueba correctas
    (kernel, IC95 bootstrap) -- se traduce el esquema tal cual, sin
    recalcular nada. Ambas clases son hoy 'no_actuar'.

    `experimental_level` (Bloque C8) NO cambia esa conclusión medida: emite
    `action: actuar_experimental` con el nivel pedido y su frecuencia
    resuelta, y conserva `measured_action`/`measured_reason` para que nadie
    lea la entrada como una política que ganó. Solo se admite sobre una
    clase que la medición dejó en 'no_actuar'."""
    src = policy_cpu["policy"][f"cpu-{cls}"]
    if src["action"] == "actuar":
        raise NotImplementedError(
            "CPU policy pasó a 'actuar' -- este script todavía no resuelve "
            "resolved_freq_khz real (no se implementó porque hasta hoy ambas "
            "clases son no_actuar); no emitir una tabla con 'actuar' sin "
            "frecuencia resuelta en vez de fallar en silencio."
        )
    source = "docs/libro/datos/cpu_calidad_30fam/politica/policy_cpu.json"
    if experimental_level is None:
        return {"action": src["action"], "n_kernels": src["n_kernels"], "reason": src.get("reason"),
                "source": source}
    return {"action": "actuar_experimental", "chosen_level": experimental_level,
            "resolved_freq_khz": cpu_level_khz(experimental_level),
            "measured_action": src["action"], "measured_reason": src.get("reason"),
            "n_kernels": src["n_kernels"], "source": source,
            "note": "Experimento de Fase 4 (Plan_Fase3_Daemon.md, Bloque C8), no una politica que gano: "
                    "la medicion por kernel dejo esta clase en no_actuar."}


def _choose_family_level(levels: dict) -> tuple[str | None, dict | None]:
    """Mismo criterio de selección que cpu_policy_table.py/gpu_policy_table.py
    (§metodologia-politica-cpu): entre los niveles con ganancia positiva Y
    significativa (p<0.05), el de mayor ganancia; si ninguno califica,
    no_actuar. min_effect se aplica después, igual que en el resto del
    proyecto."""
    best_level, best = None, None
    for lv in LEVELS:
        if lv not in levels:
            continue
        d = levels[lv]
        if d["gain"] > 0 and d["p_wilcoxon_mejora"] < 0.05 and d["gain"] >= MIN_EFFECT:
            if best is None or d["gain"] > best["gain"]:
                best_level, best = lv, d
    return best_level, best


def gpu_entry(policy_by_family: dict, cls: str, gpu_dataset: pd.DataFrame | None) -> dict:
    levels = policy_by_family[cls]["levels"]
    chosen_level, chosen = _choose_family_level(levels)
    entry = {"n_familias": policy_by_family[cls]["n_familias"],
             "source": "docs/libro/datos/gpu_calidad_20260922/politica/policy_by_family.json",
             "lofo_ganancia_realizada": policy_by_family[cls]["lofo"]["ganancia_media_realizada"]}
    if chosen_level is None:
        entry["action"] = "no_actuar"
        entry["reason"] = "ningun_nivel_familia_mejora_edp_de_forma_significativa"
        return entry

    entry["action"] = "actuar"
    entry["chosen_level"] = chosen_level
    entry["gain"] = chosen["gain"]
    entry["gain_ci95"] = chosen["gain_ci95"]
    entry["p"] = chosen["p_wilcoxon_mejora"]
    if gpu_dataset is not None:
        sub = gpu_dataset[(gpu_dataset["gpu_freq_level_id"] == chosen_level)
                          & (gpu_dataset["phase_label_train"] == cls)]
        clocks = sub["gpu_sm_clock_mhz_median"].dropna()
        if clocks.empty:
            raise ValueError(f"gpu-{cls}: sin filas de {chosen_level} en el dataset para resolver el reloj real")
        if clocks.nunique() > 1:
            raise ValueError(
                f"gpu-{cls}: reloj SM no es constante en {chosen_level} "
                f"({clocks.nunique()} valores distintos) -- resolver manualmente, no promediar en silencio"
            )
        entry["resolved_clock_mhz"] = int(round(float(clocks.iloc[0])))
    else:
        raise ValueError(f"gpu-{cls}: action=='actuar' pero no se pasó --gpu-dataset para resolver el reloj real")
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-policy", type=Path, required=True)
    ap.add_argument("--gpu-policy", type=Path, required=True,
                    help="policy_by_family.json, NUNCA policy_gpu.json (ver docstring del modulo)")
    ap.add_argument("--gpu-dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cpu-experimental-base", metavar="NIVEL",
                    help="Bloque C8: nivel de CPU para compute_bound (estado base del daemon activo, p.ej. F0). "
                         "Debe pasarse junto con --cpu-experimental-memory.")
    ap.add_argument("--cpu-experimental-memory", metavar="NIVEL",
                    help="Bloque C8: nivel de CPU para memory_bound (p.ej. F1; F0 = variante activo-F0).")
    a = ap.parse_args()
    if bool(a.cpu_experimental_base) != bool(a.cpu_experimental_memory):
        raise SystemExit("--cpu-experimental-base y --cpu-experimental-memory van juntos: el daemon fija "
                         "siempre un nivel base, y la clase memory_bound solo decide si baja de el.")
    experimental = {"compute_bound": a.cpu_experimental_base, "memory_bound": a.cpu_experimental_memory}

    if a.gpu_policy.name == "policy_gpu.json":
        raise SystemExit(
            "--gpu-policy apunta a policy_gpu.json (analisis por kernel, sin "
            "des-duplicar dual_axpy/dual_spmv/dual_stencil) -- usar "
            "policy_by_family.json, la fuente que sustenta el libro."
        )

    policy_cpu = json.loads(a.cpu_policy.read_text())
    policy_by_family = json.loads(a.gpu_policy.read_text())
    gpu_dataset = pd.read_csv(a.gpu_dataset, low_memory=False)

    policy = {}
    for cls in ("compute_bound", "memory_bound"):
        policy[f"cpu-{cls}"] = cpu_entry(policy_cpu, cls, experimental[cls])
        policy[f"gpu-{cls}"] = gpu_entry(policy_by_family, cls, gpu_dataset)

    doc = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "fase3_daemon/policy/build_policy_table.py",
        "note": "Combina policy_cpu.json (kernel, IC95 bootstrap) y policy_by_family.json "
                "(familia, des-duplicado) de Fase 2. No recalcula EDP; ver Plan_Fase3_Daemon.md SS0.3.",
        "policy": policy,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"escrito: {a.out}")
    for k, v in policy.items():
        print(f"  {k}: {v['action']}"
              + (f" @ {v['chosen_level']} ({v.get('resolved_clock_mhz', v.get('resolved_freq_khz', '?'))})"
                 if v["action"] in ("actuar", "actuar_experimental") else ""))


if __name__ == "__main__":
    main()
