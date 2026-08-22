#!/usr/bin/env python3
"""Ola 0 del plan maestro: las cuatro comprobaciones que no gastan nodo.

Ver docs/general/PLAN_MAESTRO_FASE2.md, Parte IV.

  T0.1  Autocorrelacion de `b` dentro de la corrida contra un control nulo
        permutado. Separa estructura de fase de ruido de muestreo, que es
        lo unico que C1 no pudo distinguir.

  T0.2  Por que alpha > 1 en cuatro kernels. Bajo la ley de Amdahl alpha
        es una fraccion y no puede exceder 1; un ajuste con r2 de 0.999 y
        alpha = 1.16 significa modelo mal especificado o dato mal medido.
        Se prueban tres causas candidatas, en orden de plausibilidad:

          (a) DURACION DE REFERENCIA SUBESTIMADA. Las duraciones se suman
              sobre ventanas que SOBREVIVEN al filtro de calidad de
              frecuencia. Si en F0 se rechazan mas ventanas que en F4,
              T(F0) queda corto y el cociente T(F4)/T(F0) se infla por
              encima de f_ref/f -- que es exactamente la firma de
              alpha > 1. Es la hipotesis mas simple y la primera que hay
              que descartar.
          (b) FRECUENCIA OBSERVADA distinta de la nominal por kernel. El
              ajuste anterior uso la mediana GLOBAL por nivel; si un
              kernel concreto no alcanza su nivel, su alpha se infla.
          (c) Acoplamiento residual del camino a memoria. STREAM conserva
              78.4 % del ancho de banda al 25 % del reloj: no es cero
              acoplamiento, es 21.6 % de perdida. Se reporta el alpha que
              predeciria un modelo de dos terminos.

  T0.3  LOKO por SUITE y no por kernel: seis de los nueve kernels de CPU
        son NPB clase B, asi que nueve pliegues "independientes" son en
        realidad cuatro. Se reporta ademas el techo intra-kernel, que
        distingue "el catalogo no generaliza" de "los rasgos no tienen la
        informacion".

  T0.4  Mapa de regimen: alpha contra fraccion del ancho de banda de
        STREAM, para fijar el objetivo cuantitativo de los kernels nuevos.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier.analysis.gates_c1_c2_c3 import (  # noqa: E402
    FEATURES, USECOLS, discover_runs, load_kernel,
)
from classifier.features import align, targets  # noqa: E402

# Suites reales. Seis kernels comparten NPB-OMP: tratarlos como pliegues
# independientes sobrestima el numero de muestras efectivas.
SUITE_OF = {
    "npb_bt": "NPB", "npb_mg": "NPB", "npb_cg": "NPB",
    "npb_sp": "NPB", "npb_ft": "NPB", "npb_lu": "NPB",
    "dgemm_n2048": "DGEMM",
    "rodinia_lavamd_omp": "Rodinia",
    "rajaperf_polybench_3mm_omp": "RAJAPerf",
}

NOMINAL_MHZ = {"F0": 3200.0, "F1": 2600.0, "F2": 2000.0, "F3": 1400.0, "F4": 800.0}
BREAK_EVEN = 0.226
STREAM_BW_F0_GBS = 76.80


# --------------------------------------------------------------------- T0.1

def t01_autocorrelation(per_kernel: dict[str, pd.DataFrame], max_lag: int,
                        seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for kernel, df in per_kernel.items():
        acf_real: list[float] = []
        acf_null: list[float] = []
        for _, run in df.groupby(["freq_level_id", "rep_idx"], observed=True):
            b = run.sort_values("window_index")["b"].to_numpy(dtype=float)
            b = b[np.isfinite(b)]
            if b.size < 50 or b.std() == 0:
                continue
            centered = b - b.mean()
            denom = float((centered ** 2).sum())
            acf_real.append(float((centered[:-1] * centered[1:]).sum() / denom))
            # Control nulo: la MISMA serie permutada. Conserva la
            # distribucion marginal exacta y destruye solo el orden
            # temporal, asi que cualquier diferencia es estructura y no
            # una propiedad de los valores.
            shuffled = rng.permutation(b)
            cs = shuffled - shuffled.mean()
            acf_null.append(float((cs[:-1] * cs[1:]).sum() / float((cs ** 2).sum())))

        if not acf_real:
            continue

        # Longitud de persistencia: hasta que rezago la ACF sigue sobre
        # 1/e. Traduce la autocorrelacion a "cuantas ventanas dura una
        # fase", que es lo que importa para decidir si vale conmutar.
        sample = df.sort_values(["freq_level_id", "rep_idx", "window_index"])
        one_run = next(iter(sample.groupby(["freq_level_id", "rep_idx"], observed=True)))[1]
        series = one_run["b"].to_numpy(dtype=float)
        series = series[np.isfinite(series)]
        persistence = 0
        if series.size > max_lag and series.std() > 0:
            c = series - series.mean()
            d = float((c ** 2).sum())
            for lag in range(1, max_lag + 1):
                if float((c[:-lag] * c[lag:]).sum() / d) < np.exp(-1.0):
                    break
                persistence = lag

        rows.append({
            "kernel": kernel,
            "acf1_median": round(float(np.median(acf_real)), 4),
            "acf1_null_median": round(float(np.median(acf_null)), 4),
            "persistence_windows_1e": persistence,
            "n_runs": len(acf_real),
        })
    return {"per_kernel": rows,
            "criterio": "acf1 >> acf1_null => estructura de fase; acf1 ~ acf1_null => ruido"}


# --------------------------------------------------------------------- T0.2

def t02_alpha_anomaly(per_kernel: dict[str, pd.DataFrame],
                      index: pd.DataFrame) -> dict:
    """Reajusta alpha por kernel probando las tres causas candidatas."""
    rows = []
    for kernel, df in per_kernel.items():
        # (a) Retencion de ventanas por nivel: cuantas sobreviven al filtro
        # de calidad frente a cuantas se escribieron. Se recuenta desde el
        # CSV crudo porque `df` ya viene filtrado.
        retention = {}
        raw_duration = {}
        for row in index[index["kernel_ref"] == kernel].itertuples():
            raw = pd.read_csv(row.windows_path, usecols=USECOLS, low_memory=False)
            kept = raw[
                (raw["quality_status"] == "ok")
                & raw["frequency_quality_status"].isin(["valid", "not_applicable_native"])
            ]
            lvl = row.freq_level_id
            r = retention.setdefault(lvl, [0, 0])
            r[0] += len(kept)
            r[1] += len(raw)
            d = raw_duration.setdefault(lvl, [0.0, 0.0])
            d[0] += float(pd.to_numeric(kept["delta_t_ns"], errors="coerce").sum())
            d[1] += float(pd.to_numeric(raw["delta_t_ns"], errors="coerce").sum())

        levels = [lv for lv in NOMINAL_MHZ if lv in raw_duration]
        if "F0" not in levels or len(levels) < 3:
            continue

        # alpha con duraciones FILTRADAS (lo que se hizo antes) y con
        # duraciones CRUDAS (todas las ventanas escritas).
        def fit(which: int, mhz_map: dict[str, float]) -> tuple[float, float] | None:
            durations = {mhz_map[lv]: raw_duration[lv][which]
                         for lv in levels if mhz_map.get(lv)}
            if len(durations) < 3 or mhz_map["F0"] not in durations:
                return None
            try:
                return align.fit_alpha(durations, mhz_map["F0"])
            except ValueError:
                return None

        # (b) frecuencia OBSERVADA por kernel y nivel, no la mediana global
        observed = (df.groupby("freq_level_id")["freq_khz_observed"].median() / 1000.0).to_dict()
        obs_map = {lv: float(observed[lv]) for lv in levels if lv in observed}

        a_filtered = fit(0, NOMINAL_MHZ)
        a_raw = fit(1, NOMINAL_MHZ)
        a_observed = fit(0, obs_map) if len(obs_map) >= 3 and "F0" in obs_map else None

        rows.append({
            "kernel": kernel,
            "alpha_filtrado_nominal": round(a_filtered[0], 4) if a_filtered else None,
            "alpha_crudo_nominal": round(a_raw[0], 4) if a_raw else None,
            "alpha_filtrado_observado": round(a_observed[0], 4) if a_observed else None,
            "retencion_F0": round(retention["F0"][0] / retention["F0"][1], 4) if retention.get("F0") else None,
            "retencion_F4": round(retention["F4"][0] / retention["F4"][1], 4) if retention.get("F4") else None,
            "mhz_observado_F0": round(obs_map.get("F0", float("nan")), 1),
            "mhz_observado_F4": round(obs_map.get("F4", float("nan")), 1),
        })
    return {
        "per_kernel": rows,
        "criterio": (
            "si alpha_crudo < alpha_filtrado y retencion_F0 < retencion_F4, la causa "
            "es (a) duracion de referencia subestimada por el filtro. Si alpha_observado "
            "< alpha_nominal, la causa es (b). Si ninguno baja de 1, queda (c) u otra."
        ),
    }


# --------------------------------------------------------------------- T0.3

def t03_loko_by_suite(per_kernel: dict[str, pd.DataFrame], max_rows_per_run: int,
                      seed: int) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    samples = []
    for kernel, df in per_kernel.items():
        for _, run in df.groupby(["freq_level_id", "rep_idx"], observed=True):
            run = run.dropna(subset=[*FEATURES, "b"])
            if run.empty:
                continue
            if len(run) > max_rows_per_run:
                run = run.iloc[rng.choice(len(run), size=max_rows_per_run, replace=False)]
            samples.append(run[[*FEATURES, "b", "kernel_ref"]])
    data = pd.concat(samples, ignore_index=True)
    data["suite"] = data["kernel_ref"].map(SUITE_OF)

    def rf() -> "RandomForestRegressor":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1)

    # Pliegues por SUITE
    folds = []
    for suite in sorted(data["suite"].dropna().unique()):
        test = data[data["suite"] == suite]
        train = data[data["suite"] != suite]
        if train.empty or test.empty:
            continue
        model = rf()
        model.fit(train[FEATURES], train["b"])
        pred = model.predict(test[FEATURES])
        trivial = np.full(len(test), float(train["b"].mean()))
        folds.append({
            "held_out_suite": suite,
            "n_kernels": int(test["kernel_ref"].nunique()),
            "mae_model": round(float(mean_absolute_error(test["b"], pred)), 4),
            "mae_trivial": round(float(mean_absolute_error(test["b"], trivial)), 4),
            "r2_model": round(float(r2_score(test["b"], pred)), 3),
        })

    # TECHO intra-kernel: entrenar y probar dentro del mismo kernel. No es
    # un resultado valido de generalizacion --por eso es un techo-- pero
    # distingue "el catalogo no generaliza" de "los rasgos no contienen la
    # informacion". Si ni siquiera el techo es alto, ningun catalogo lo
    # arregla.
    ceiling = []
    for kernel in sorted(data["kernel_ref"].unique()):
        sub = data[data["kernel_ref"] == kernel]
        if len(sub) < 200:
            continue
        tr, te = train_test_split(sub, test_size=0.3, random_state=seed, shuffle=True)
        model = rf()
        model.fit(tr[FEATURES], tr["b"])
        ceiling.append({
            "kernel": kernel,
            "r2_intra_kernel": round(float(r2_score(te["b"], model.predict(te[FEATURES]))), 3),
        })

    return {
        "folds_por_suite": folds,
        "techo_intra_kernel": ceiling,
        "criterio": (
            "techo alto + LOKO negativo => falta cobertura de regimen (arreglable con "
            "catalogo). techo tambien bajo => los rasgos no tienen la informacion "
            "(hay que revisar el vector de entrada antes que el catalogo)."
        ),
    }


# --------------------------------------------------------------------- T0.4

def t04_regime_map(alpha_rows: list[dict], bw_json: Path | None) -> dict:
    bw = {}
    if bw_json and bw_json.exists():
        payload = json.loads(bw_json.read_text())
        for kernel, levels in payload.get("bandwidth_corrected", {}).get("F0", {}).items() \
                if isinstance(payload.get("bandwidth_corrected", {}).get("F0"), dict) else []:
            bw[kernel] = levels
        if not bw:
            corrected = payload.get("bandwidth_corrected", {})
            f0 = corrected.get("F0", {})
            if isinstance(f0, dict):
                bw = {k: v for k, v in f0.items()}

    rows = []
    for entry in alpha_rows:
        kernel = entry["kernel"]
        a = entry.get("alpha_crudo_nominal") or entry.get("alpha_filtrado_nominal")
        b_gbs = bw.get(kernel)
        rows.append({
            "kernel": kernel,
            "alpha": a,
            "bw_f0_gbs": round(float(b_gbs), 2) if b_gbs else None,
            "pct_saturacion_stream": round(100.0 * float(b_gbs) / STREAM_BW_F0_GBS, 1) if b_gbs else None,
            "bajo_umbral": (a is not None and a <= BREAK_EVEN),
        })
    rows.sort(key=lambda r: (r["alpha"] is None, r["alpha"]))
    return {
        "break_even": BREAK_EVEN,
        "stream_alpha": 0.1538,
        "stream_bw_f0_gbs": STREAM_BW_F0_GBS,
        "per_kernel": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--bw-json", default=None)
    parser.add_argument("--max-lag", type=int, default=200)
    parser.add_argument("--max-rows-per-run", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir)
    index = discover_runs(campaign_dir)
    kernels = sorted(index["kernel_ref"].unique())
    print(f"nodo de análisis: {platform.node()}", flush=True)
    print(f"corridas: {len(index)}  kernels: {len(kernels)}", flush=True)

    per_kernel: dict[str, pd.DataFrame] = {}
    for kernel in kernels:
        df = load_kernel(index, kernel)
        if not df.empty:
            per_kernel[kernel] = df
            print(f"  {kernel}: {len(df)} ventanas", flush=True)

    oi = pd.concat([d["operational_intensity_uncore_real"] for d in per_kernel.values()])
    ridge = pd.concat([d["i_ridge_used"] for d in per_kernel.values()])
    k = targets.calibrate_k(oi, ridge)
    for df in per_kernel.values():
        df["b"] = targets.boundedness_score(
            df["operational_intensity_uncore_real"], df["i_ridge_used"], k=k)

    print("\n== T0.1  autocorrelación de b ==", flush=True)
    t01 = t01_autocorrelation(per_kernel, args.max_lag, args.seed)
    print(pd.DataFrame(t01["per_kernel"]).to_string(index=False), flush=True)

    print("\n== T0.2  anomalía alpha > 1 ==", flush=True)
    t02 = t02_alpha_anomaly(per_kernel, index)
    print(pd.DataFrame(t02["per_kernel"]).to_string(index=False), flush=True)
    print(t02["criterio"], flush=True)

    print("\n== T0.3  LOKO por suite + techo intra-kernel ==", flush=True)
    t03 = t03_loko_by_suite(per_kernel, args.max_rows_per_run, args.seed)
    print(pd.DataFrame(t03["folds_por_suite"]).to_string(index=False), flush=True)
    print("\ntecho intra-kernel:", flush=True)
    print(pd.DataFrame(t03["techo_intra_kernel"]).to_string(index=False), flush=True)

    print("\n== T0.4  mapa de régimen ==", flush=True)
    t04 = t04_regime_map(t02["per_kernel"],
                         Path(args.bw_json) if args.bw_json else None)
    print(pd.DataFrame(t04["per_kernel"]).to_string(index=False), flush=True)

    Path(args.out).write_text(json.dumps({
        "analysis_node": platform.node(),
        "t01_autocorrelacion": t01,
        "t02_alpha_anomalia": t02,
        "t03_loko_por_suite": t03,
        "t04_mapa_regimen": t04,
    }, indent=2))
    print(f"\nreporte -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
