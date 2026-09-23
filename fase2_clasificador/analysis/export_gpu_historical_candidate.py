"""Congela, mide latencia y exporta el candidato final del clasificador GPU
historico (dataset por corrida, +A1 relajado +A2 rajaperf_cuda separado +A6
gpu_mem_util_pct_std -- cierre de la ronda 2026-09-22).

Empate estadistico confirmado por bootstrap pareado entre regresion_log,
random_forest y extra_trees (P entre 0.53 y 0.60, ninguno domina). Mismo
criterio de desempate que CPU (train_phase.py/train_phase_gpu.py
select_best_model): score = (1 - cell_balanced_acc) + latency_weight *
(p99/max_p99), latency_weight=0.2. cell_balanced_acc de cada modelo viene
del matrix.json ya calculado (5 semillas, LOFO por familia) -- este script
NO reentrena por familia, mide latencia y ajusta el modelo final sobre TODO
el dataset (igual que CPU: el modelo de produccion no deja ninguna familia
fuera).

Correr siempre en el nodo de destino real (paccaA100, donde el daemon GPU
leeria NVML en produccion) -- nunca en la laptop local.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from fase2_clasificador.analysis.gpu_quality_report import FAMILY_COL, FEATURES, LABEL, load, make_model


def measure_latency(model, sample: np.ndarray, repeats: int = 200) -> tuple[float, float, float]:
    """Identico a train_phase_gpu.py::measure_latency (fila a fila, no lote)."""
    one = sample[:1]
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(one)
        timings.append((time.perf_counter() - start) * 1e6)
    return (float(np.percentile(timings, 50)), float(np.percentile(timings, 95)), float(np.percentile(timings, 99)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--latency-weight", type=float, default=0.2)
    ap.add_argument("--split-rajaperf-cuda", action="store_true")
    ap.add_argument("--extra-features", nargs="*", default=None)
    ap.add_argument("--drop-features", nargs="*", default=[],
                    help="variables a quitar del conjunto (p.ej. gpu_sm_clock_mhz_median gpu_power_mw_median: el daemon "
                         "cambia el reloj, y la potencia depende de el; ver gpu_clock_feature_audit.py)")
    ap.add_argument("--tag", default="20260922", help="sufijo del nombre de archivo del candidato exportado")
    ap.add_argument("--rationale", default=None, help="texto libre que se guarda en el metadata")
    # cell_balanced_acc de cada candidato, tomado del matrix.json de la ronda
    # de hoy (gpu_quality_FINAL_20260922, 5 semillas, LOFO por familia) --
    # pasado explicito en vez de recalculado para no repetir 5x16x7 ajustes
    # solo para desempatar por latencia.
    ap.add_argument("--cell-balanced-acc", nargs="*", default=[
        "regresion_log=0.7924", "random_forest=0.8074", "extra_trees=0.8064",
    ], help="nombre=valor, candidatos ya empatados por bootstrap a comparar")
    args = ap.parse_args()

    scores = dict(kv.split("=") for kv in args.cell_balanced_acc)
    scores = {k: float(v) for k, v in scores.items()}

    frame = load(args.source, split_rajaperf_cuda=args.split_rajaperf_cuda, extra_features=args.extra_features)
    for f in args.drop_features:
        FEATURES.remove(f)
    frame = frame.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"features: {FEATURES}", flush=True)
    X = frame[FEATURES].to_numpy(dtype=np.float32)
    y = frame["y"].to_numpy()

    latencies = {}
    for name in scores:
        model = make_model(name, args.seed)
        model.fit(X, y)
        latencies[name] = measure_latency(model, X)
        print(f"{name}: p50={latencies[name][0]:.1f}us p95={latencies[name][1]:.1f}us p99={latencies[name][2]:.1f}us "
              f"cell_balanced_acc={scores[name]:.4f}", flush=True)

    max_p99 = max(lat[2] for lat in latencies.values()) or 1.0

    def score(name: str) -> float:
        return (1.0 - scores[name]) + args.latency_weight * (latencies[name][2] / max_p99)

    winner = min(scores, key=score)
    print(f"\nGANADOR: {winner} (score={score(winner):.4f})", flush=True)
    for name in scores:
        print(f"  {name}: score={score(name):.4f}")

    final_model = make_model(winner, args.seed)
    final_model.fit(X, y)

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / f"gpu_{winner}_historical_{args.tag}.joblib"
    metadata_path = args.out / f"gpu_{winner}_historical_{args.tag}.metadata.json"
    joblib.dump(final_model, model_path)
    metadata_path.write_text(json.dumps({
        "schema": "gpu_historical_candidate/1",
        "model": winner,
        "features": FEATURES,
        "dropped_features": args.drop_features,
        "rationale": args.rationale,
        "label": LABEL,
        "candidates_compared": list(scores.keys()),
        "candidates_cell_balanced_acc": scores,
        "candidates_latency_us": {k: {"p50": v[0], "p95": v[1], "p99": v[2]} for k, v in latencies.items()},
        "selection_criterion": "empate estadistico por bootstrap pareado de familias (P entre 0.53 y 0.60 entre los 3); "
                                "desempate por score=(1-cell_balanced_acc)+latency_weight*(p99/max_p99), "
                                "mismo criterio que train_phase.py/train_phase_gpu.py::select_best_model",
        "latency_weight": args.latency_weight,
        "n_rows": len(frame), "n_families": int(frame[FAMILY_COL].nunique()),
        "source": str(args.source), "seed": args.seed,
        "dataset_contract": "granularity=run, etiqueta ncu constante -- NO el dataset por ventana CUPTI",
        "fitted_on": "todas las filas disponibles (sin retener ninguna familia), igual que el modelo final de CPU",
        "measured_on_node": "paccaA100 (nodo de destino real para el daemon GPU)",
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"\nmodelo={model_path}")


if __name__ == "__main__":
    main()
