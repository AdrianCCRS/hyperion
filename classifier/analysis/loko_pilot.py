"""Primer entrenamiento y evaluacion REAL de la arquitectura reformulada
(§4 de `Estrategia_GPU_Fase2.md` / `Estrategia_CPU_Fase2.md`): regresion
por par (carga, nivel) evaluada bajo leave-one-kernel-out contra la mejor
constante honesta.

POR QUE ESTE SCRIPT EXISTE. Hasta 2026-08-25 todo el trabajo de Fase 2 fue
preparacion y diagnostico de datos; el modelo que pide el Objetivo 2 nunca
se habia entrenado ni evaluado, ni en CPU ni en GPU (riesgo 4 de
`Estrategia_GPU_Fase2.md`). Este es ese paso. Ejecuta de una vez las
pruebas V5/C6 (anti-fuga), V6/C7 (linea base honesta) y V7/C9 (latencia de
inferencia) de las listas de pruebas de ambas estrategias.

POR QUE LEE LOS DIRECTORIOS DE CAMPAÑA Y NO `local_datasets/`. Dos razones,
ambas de correctitud del dato y no de comodidad:

  1. El export `local_datasets/final_campaigns_20260821/` es ANTERIOR a la
     correccion de ARC-174 (la columna `repetition` sale constante en 1,
     confirmado 2026-08-25 sobre el .csv.gz local). Emparejar por
     repeticion sobre ese export colapsaria en silencio las 10
     repeticiones reales de cada (kernel, nivel).
  2. Su dataset de GPU es `pacca_gpu_dvfs_20260820`, la campaña de 6
     niveles ANTERIOR a la rejilla fina. La rejilla fina (job 6471,
     210/210 aceptadas) es la que contiene el margen real de 9.06 pts
     (V2, §5) y solo existe en el directorio de campaña.

MEMORIA. La agregacion es en STREAMING, fila por fila con `csv.DictReader`
sobre cada `windows.csv` por separado -- mismo patron que
`cpu_policy_headroom.read_run()`. Un intento previo que cargaba el CSV
completo del eje CPU (10.25 M de filas) en un DataFrame murio por OOM;
aqui el pico de memoria es una corrida, no la campaña.

Uso:
    python3 loko_pilot.py --axis gpu
    python3 loko_pilot.py --axis cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.eval import protocol  # noqa: E402
from classifier.features import pair_dataset  # noqa: E402

CAMPAIGNS = Path.home() / "hyperion-results/campaigns"

# --- CPU -----------------------------------------------------------------
CPU_BASE = CAMPAIGNS / "pacca_cpu_final_attempt03_20260820_arc174"
CPU_CID = "pacca_cpu_final_attempt03_20260820"
CPU_REF_LEVEL = "F0"

# --- GPU -----------------------------------------------------------------
# Rejilla fina, job 6471 -- 210/210 aceptadas, V1/V2 pasadas.
GPU_BASE = CAMPAIGNS / "pacca_gpu_fine_grid_dataset_20260823"
GPU_REF_LEVEL = "F0"

# rodinia_lud: GPU en reposo (alpha=0.030, r2 negativo, sec. 3.14 del
# libro). rodinia_backprop: energia de GPU en F0 de 8-49 J, dos ordenes de
# magnitud bajo el resto (Anexo M) -- su EDP es ruido dividido por casi
# nada. Ambos ya excluidos de `gpu_policy_headroom.py`; mismo criterio
# aqui, por las mismas razones ya documentadas.
GPU_EXCLUDE = {"rodinia_lud", "rodinia_backprop"}

# Las 4 variantes de tamaño de dwt2d: riesgo 8 de Estrategia_GPU_Fase2.md
# -- miden overhead de host, no 4 regimenes distintos. Se excluyen del
# LOKO y del margen (decision operativa ya tomada), no del dataset fisico.
GPU_EXCLUDE |= {
    "rodinia_dwt2d_s192", "rodinia_dwt2d_s2048",
    "rodinia_dwt2d_s4096", "rodinia_dwt2d_s8192",
}

_RUN_RE_GPU = re.compile(r"__(?P<kernel>.+?)__(?P<cpu>[^_]+)__gpu(?P<gpu>[^_]+)__rep(?P<rep>\d+)$")
_RUN_RE_CPU = re.compile(r"__(?P<kernel>.+?)__(?P<level>[^_]+)__rep(?P<rep>\d+)$")


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def read_cpu_run(run_dir: Path, feature_cols: list[str]) -> dict | None:
    """Una corrida de CPU -> energia RAPL (pkg+dram), duracion y medias de
    features. Streaming: nunca mantiene mas de una fila en memoria."""
    windows = run_dir / "windows.csv"
    if not windows.exists():
        return None

    energy_uj = 0.0
    n_energy = 0
    t_min = t_max = None
    feats: dict[str, list[float]] = {c: [] for c in feature_cols}

    with windows.open() as handle:
        for row in csv.DictReader(handle):
            t_start, t_end = row.get("t_start_ns"), row.get("t_end_ns")
            if t_start not in (None, "") and t_end not in (None, ""):
                t0, t1 = float(t_start), float(t_end)
                t_min = t0 if t_min is None else min(t_min, t0)
                t_max = t1 if t_max is None else max(t_max, t1)
            for col in feature_cols:
                value = row.get(col)
                if value not in (None, ""):
                    try:
                        feats[col].append(float(value))
                    except ValueError:
                        pass
            if row.get("energy_valid") != "1":
                continue
            pkg, dram = row.get("pkg_delta_uj"), row.get("dram_delta_uj")
            if pkg in (None, "") or dram in (None, ""):
                continue
            energy_uj += float(pkg) + float(dram)
            n_energy += 1

    if n_energy == 0 or t_min is None or t_max is None:
        return None
    elapsed_s = (t_max - t_min) / 1e9
    energy_j = energy_uj / 1e6
    if elapsed_s <= 0 or energy_j <= 0:
        return None

    record = {"elapsed_s": elapsed_s, "energy_j": energy_j}
    for col in feature_cols:
        mean = _mean(feats[col])
        if mean is None:
            return None  # sin feature no hay fila: no se rellena, se descarta
        record[col] = mean
    return record


def read_gpu_run(run_dir: Path, feature_cols: list[str]) -> dict | None:
    """Una corrida de GPU -> energia NVML (`gpu_energy_delta_mj` sobre filas
    `gpu_telemetry` con `gpu_energy_valid == 1`, metrica primaria tras el
    Anexo M), duracion sobre TODAS las filas, y medias de features
    restringidas a las filas de telemetria GPU (las CPU-passthrough no
    traen `gpu_util_pct`)."""
    windows = run_dir / "windows.csv"
    if not windows.exists():
        return None

    energy_mj = 0.0
    n_energy = 0
    t_min = t_max = None
    feats: dict[str, list[float]] = {c: [] for c in feature_cols}

    with windows.open() as handle:
        for row in csv.DictReader(handle):
            t_start, t_end = row.get("t_start_ns"), row.get("t_end_ns")
            # ARC-70: las filas gpu_telemetry son passthrough -- t_start_ns
            # viene VACIO y solo t_end_ns esta poblado. Tomar ambos limites
            # de lo que haya, sin exigir el par completo.
            for value in (t_start, t_end):
                if value not in (None, ""):
                    t = float(value)
                    t_min = t if t_min is None else min(t_min, t)
                    t_max = t if t_max is None else max(t_max, t)

            if row.get("quality_status") != "gpu_telemetry":
                continue
            for col in feature_cols:
                value = row.get(col)
                if value not in (None, ""):
                    try:
                        feats[col].append(float(value))
                    except ValueError:
                        pass
            if row.get("gpu_energy_valid") != "1":
                continue
            delta = row.get("gpu_energy_delta_mj")
            if delta in (None, ""):
                continue
            energy_mj += float(delta)
            n_energy += 1

    if n_energy == 0 or t_min is None or t_max is None:
        return None
    elapsed_s = (t_max - t_min) / 1e9
    energy_j = energy_mj / 1e3
    if elapsed_s <= 0 or energy_j <= 0:
        return None

    record = {"elapsed_s": elapsed_s, "energy_j": energy_j}
    for col in feature_cols:
        mean = _mean(feats[col])
        if mean is None:
            return None
        record[col] = mean
    return record


def calibration_kernel_ids() -> set[str]:
    """Ids del catalogo con `role: calibration`.

    Se derivan del catalogo y no se listan a mano: la primera version de
    este script no los excluia y metio `ert_probe` y `stream_official`
    como pliegues del LOKO de CPU. No son sujetos del dataset -- son los
    probes que fijan el ridge de Roofline (P_pico y BW_pico), y ademas
    MAN-07 obliga a declararlos en toda campaña, asi que aparecen como
    directorios de corrida en todas ellas. Incluirlos anade dos pliegues
    que el modelo nunca vera en produccion y, peor, mete a
    `stream_official` -- la unica carga del proyecto con alpha bajo el
    umbral (0.154) -- como si fuera una carga real del catalogo.
    """
    from orchestrator.catalog import load_catalog

    catalog_path = (
        Path(__file__).resolve().parents[2]
        / "orchestrator/schemas/kernels/catalog.yaml"
    )
    catalog = load_catalog(catalog_path)
    return {
        entry_id for entry_id, entry in catalog.items()
        if getattr(entry, "role", None) == "calibration"
    }


def collect_runs(base: Path, axis: str, feature_cols: list[str]) -> pd.DataFrame:
    """Recorre los directorios de corrida y arma una fila por
    (kernel, repeticion, nivel). Los `__baseline` se saltan: son el par de
    medicion de overhead (CAM-04), no corridas del dataset."""
    reader = read_gpu_run if axis == "gpu" else read_cpu_run
    pattern = _RUN_RE_GPU if axis == "gpu" else _RUN_RE_CPU

    rows = []
    skipped_no_match = skipped_unreadable = 0
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir() or run_dir.name.endswith("__baseline"):
            continue
        match = pattern.search(run_dir.name)
        if match is None:
            skipped_no_match += 1
            continue
        kernel = match.group("kernel")
        level = match.group("gpu") if axis == "gpu" else match.group("level")
        record = reader(run_dir, feature_cols)
        if record is None:
            skipped_unreadable += 1
            continue
        record.update(
            kernel_ref=kernel,
            repetition=int(match.group("rep")),
            freq_level_id=level,
        )
        rows.append(record)

    out = pd.DataFrame(rows)
    out.attrs["skipped_no_match"] = skipped_no_match
    out.attrs["skipped_unreadable"] = skipped_unreadable
    return out


def edp_by_level_table(pairs: pd.DataFrame, ref_level: str) -> pd.DataFrame:
    """EDP RELATIVO a la referencia (energy_ratio * time_ratio), promediado
    sobre repeticiones. Una fila por kernel, una columna por nivel; la
    columna del propio nivel de referencia vale 1.0 por construccion."""
    pairs = pairs.copy()
    pairs["edp_ratio"] = pairs["energy_ratio"] * pairs["time_ratio"]
    table = pairs.pivot_table(
        index="kernel_ref", columns="freq_level_id",
        values="edp_ratio", aggfunc="mean",
    )
    table[ref_level] = 1.0
    return table


def check_v5_anti_leak() -> str:
    """V5/C6: el guardarrail anti-fuga debe fallar RUIDOSAMENTE si se le
    fuerza una columna que es el objetivo. Se prueba de verdad, inyectando
    la columna a proposito -- no se asume que el codigo funciona."""
    try:
        pair_dataset.assert_no_target_leak(["ipc", "energy_ratio"])
    except AssertionError as error:
        return f"PASA -- el guardarrail disparo: {error}"
    return "FALLA -- el guardarrail NO disparo con una columna de objetivo inyectada"


def run_loko(
    pairs: pd.DataFrame,
    feature_cols: list[str],
    ref_level: str,
) -> dict:
    """Entrena las dos regresiones de §4 (energy_ratio, time_ratio) bajo
    LOKO y compara la politica resultante contra la mejor constante
    honesta y contra el oraculo.

    Anti-fuga POR CONSTRUCCION: las features son unicamente las observadas
    en la corrida de REFERENCIA (prefijo `ref_`) mas el nivel candidato
    codificado como su fraccion nominal de reloj -- nada de la corrida
    candidata entra al modelo.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    model_cols = [f"ref_{c}" for c in feature_cols] + ["level_code"]
    levels = sorted(pairs["freq_level_id"].unique())
    level_index = {lv: i for i, lv in enumerate(levels)}
    pairs = pairs.copy()
    pairs["level_code"] = pairs["freq_level_id"].map(level_index)

    edp_table = edp_by_level_table(pairs, ref_level)
    oracle = edp_table.min(axis=1)

    chosen_by_model: dict[str, float] = {}
    chosen_level_by_model: dict[str, str] = {}
    latencies_us: list[float] = []

    for idx_train, idx_test, held_out in protocol.leave_one_kernel_out(pairs):
        protocol.assert_no_kernel_leak(pairs, idx_train, idx_test)
        train = pairs.iloc[idx_train]
        test = pairs.iloc[idx_test]

        x_train = train[model_cols].to_numpy(dtype=float)
        energy_model = GradientBoostingRegressor(random_state=20260806)
        time_model = GradientBoostingRegressor(random_state=20260806)
        energy_model.fit(x_train, train["energy_ratio"].to_numpy(dtype=float))
        time_model.fit(x_train, train["time_ratio"].to_numpy(dtype=float))

        # Politica: para el kernel excluido, predecir EDP en cada nivel
        # candidato y quedarse con el minimo. Se usa la PRIMERA repeticion
        # como observacion de referencia -- en produccion el daemon vera
        # una sola corrida, no un promedio.
        per_level_pred: dict[str, float] = {}
        for level in levels:
            subset = test[test["freq_level_id"] == level]
            if subset.empty:
                continue
            x_test = subset[model_cols].to_numpy(dtype=float)[:1]
            t0 = time.perf_counter()
            e_hat = float(energy_model.predict(x_test)[0])
            t_hat = float(time_model.predict(x_test)[0])
            latencies_us.append((time.perf_counter() - t0) * 1e6)
            per_level_pred[level] = e_hat * t_hat

        if not per_level_pred:
            continue
        # El nivel de referencia siempre es una opcion valida (EDP=1).
        per_level_pred[ref_level] = 1.0
        best_level = min(per_level_pred, key=per_level_pred.get)
        if held_out not in edp_table.index or best_level not in edp_table.columns:
            continue
        real_edp = edp_table.loc[held_out, best_level]
        if pd.isna(real_edp):
            continue
        chosen_by_model[held_out] = float(real_edp)
        chosen_level_by_model[held_out] = best_level

    common = [k for k in chosen_by_model if k in oracle.index]
    model_loss = protocol.edp_loss(
        np.array([chosen_by_model[k] for k in common]),
        oracle.loc[common].to_numpy(),
    )
    honest = protocol.honest_constant_baseline(edp_table)

    return {
        "edp_table": edp_table,
        "model_edp_loss": model_loss,
        "model_chosen_level": chosen_level_by_model,
        "honest_constant": honest,
        "oracle_edp_loss": 1.0,
        "always_ref_edp_loss": protocol.edp_loss(
            np.ones(len(oracle)), oracle.to_numpy()
        ),
        "latency_p50_us": float(np.percentile(latencies_us, 50)) if latencies_us else float("nan"),
        "latency_p99_us": float(np.percentile(latencies_us, 99)) if latencies_us else float("nan"),
        "n_folds": len(chosen_by_model),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if args.axis == "gpu":
        base, ref_level = GPU_BASE, GPU_REF_LEVEL
        feature_cols = list(pair_dataset.GPU_FEATURES)
        exclude = set(GPU_EXCLUDE)
    else:
        base, ref_level = CPU_BASE, CPU_REF_LEVEL
        feature_cols = list(pair_dataset.CPU_FEATURES)
        exclude = set()
    # Los kernels de calibracion nunca son pliegues del LOKO -- ver
    # calibration_kernel_ids() para por que aparecen en el directorio.
    exclude |= calibration_kernel_ids()

    print(f"=== V5/C6 anti-fuga de objetivo ===")
    print(check_v5_anti_leak())
    print()

    print(f"=== Agregando corridas de {base.name} (streaming) ===")
    runs = collect_runs(base, args.axis, feature_cols)
    print(f"corridas leidas: {len(runs)}  "
          f"(saltadas sin match de nombre: {runs.attrs['skipped_no_match']}, "
          f"ilegibles/incompletas: {runs.attrs['skipped_unreadable']})")
    if exclude:
        before = len(runs)
        runs = runs[~runs["kernel_ref"].isin(exclude)].reset_index(drop=True)
        print(f"excluidos por criterio ya documentado ({sorted(exclude)}): "
              f"{before - len(runs)} corridas")
    print(f"kernels: {sorted(runs['kernel_ref'].unique())}")
    print(f"niveles: {sorted(runs['freq_level_id'].unique())}")
    print(f"repeticiones: {sorted(runs['repetition'].unique())}")
    print()

    pairs = pair_dataset.build_pair_dataset(
        runs, feature_cols, ref_level=ref_level,
    )
    print(f"=== Dataset por par ===")
    print(f"filas: {len(pairs)}  descartadas sin referencia emparejable: "
          f"{pairs.attrs.get('dropped_no_ref')}")
    print()

    result = run_loko(pairs, feature_cols, ref_level)

    print("=== EDP relativo por kernel y nivel (1.0 = igual que la referencia) ===")
    print(result["edp_table"].round(4).to_string())
    print()

    honest = result["honest_constant"]
    trivial = result["always_ref_edp_loss"]
    print("=== VEREDICTO (EDP loss: 1.0 = tan bueno como el oraculo) ===")
    print(f"  oraculo (techo inalcanzable)          : 1.0000")
    print(f"  MODELO aprendido (LOKO)               : {result['model_edp_loss']:.4f}")
    print(f"  mejor constante honesta (V6/C7)       : {honest['edp_loss']:.4f}")
    print(f"  TRIVIAL: siempre a {ref_level} (max reloj){'':<3}: {trivial:.4f}")
    print()
    print(f"  niveles distintos que elige la constante honesta por pliegue: "
          f"{honest['n_distinct_levels']}")
    print(f"  nivel elegido por el modelo, por kernel: {result['model_chosen_level']}")
    print()
    margen = honest["edp_loss"] - result["model_edp_loss"]
    print(f"  margen del modelo sobre la constante honesta: {margen:+.4f}")
    print()
    # El rival que de verdad hay que vencer no es la constante honesta sino
    # el TRIVIAL de no hacer nada: la constante honesta se elige por
    # pliegue y puede salir PEOR que quedarse quieto (pasa cuando la media
    # de entrenamiento favorece un nivel que el kernel excluido detesta).
    # Un modelo que le gana a la constante honesta pero pierde contra "no
    # hacer nada" no justifica existir, y reportar solo la primera
    # comparacion haria pasar por logro justo lo contrario.
    margen_trivial = trivial - result["model_edp_loss"]
    print(f"  MARGEN DEL MODELO SOBRE EL TRIVIAL (el que decide): {margen_trivial:+.4f}")
    if margen_trivial > 0:
        print(f"  -> el modelo GANA a no hacer nada: justifica existir")
    else:
        print(f"  -> el modelo PIERDE contra no hacer nada: NO justifica existir todavia")
    print()
    print(f"=== V7/C9 latencia de inferencia ===")
    print(f"  p50: {result['latency_p50_us']:.1f} us   p99: {result['latency_p99_us']:.1f} us")
    print(f"  (dos regresiones por decision; el daemon decide por corrida, no por ventana)")

    if args.out_json:
        payload = {k: v for k, v in result.items() if k != "edp_table"}
        payload["edp_table"] = result["edp_table"].to_dict()
        args.out_json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nresultado guardado en {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
