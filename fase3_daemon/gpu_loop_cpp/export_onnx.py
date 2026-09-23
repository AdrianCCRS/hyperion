"""Exporta el candidato de GPU (random forest sin reloj ni potencia,
`fase2_clasificador/models/gpu_random_forest_historical_20260923_sin_reloj`) a
ONNX para el daemon de GPU en C++ (`gpu_loop_main`), igual que el candidato de
CPU (`fase3_daemon/cpu_loop/export_onnx.py`).

Salidas junto al script: `<nombre>.onnx`, `<nombre>.features.txt` (una variable
por linea, EN EL ORDEN del modelo: el daemon en C++ no hardcodea nombres) y
`<nombre>.onnx.metadata.json` (sha256 del .joblib, verificacion, criterio).

Verificacion obligatoria: `predict_proba` de scikit-learn contra la sesion ONNX
real, FILA A FILA, sobre las filas de entrenamiento; una diferencia mayor a
`--tol` o una etiqueta distinta en cualquier fila hace fallar el script (codigo
de salida != 0) en vez de exportar un artefacto silenciosamente incorrecto.

Correr en pacca con el venv `~/hyperion-onnx-venv` (scikit-learn 1.9.0 = el de
entrenamiento, skl2onnx, onnx, onnxruntime); nunca en local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_DEFAULT_NAME = "gpu_random_forest_historical_20260923_sin_reloj"


def convert_rf_to_onnx(model, feature_count: int, model_name: str = "gpu_rf_sin_reloj"):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    return convert_sklearn(
        model, model_name, initial_types=[("input", FloatTensorType([None, feature_count]))],
        options={id(model): {"zipmap": False}}, target_opset={"": 18, "ai.onnx.ml": 3},
    )


def verify_row_by_row(model, onnx_path: Path, X: np.ndarray, tol: float) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_label, onnx_proba = sess.run(None, {input_name: X.astype(np.float32)})
    onnx_proba = np.asarray(onnx_proba)
    sk_proba = model.predict_proba(X.astype(np.float32))
    sk_label = model.predict(X.astype(np.float32))
    diff = np.abs(onnx_proba - sk_proba)
    max_diff = float(diff.max())
    worst = int(diff.max(axis=1).argmax())
    # el daemon decide con P(memory) > 0.5; debe coincidir con predict() en TODAS las filas
    daemon_label = (onnx_proba[:, 1] > 0.5)
    label_mismatches = int((daemon_label != sk_label.astype(bool)).sum())
    onnx_label_mismatches = int((np.asarray(onnx_label).astype(bool) != sk_label.astype(bool)).sum())
    if max_diff > tol or label_mismatches or onnx_label_mismatches:
        raise AssertionError(
            f"verificacion ONNX fallo: max_diff_proba={max_diff:.2e} (tol={tol:.0e}), peor fila={worst}, "
            f"{label_mismatches} discrepancias con la regla del daemon (P>0.5), {onnx_label_mismatches} con la etiqueta ONNX "
            f"de {len(X)} filas"
        )
    return {"n_rows": len(X), "max_diff_proba": max_diff, "label_mismatches_daemon_rule": label_mismatches,
            "label_mismatches_onnx_label": onnx_label_mismatches}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=_REPO_ROOT / "fase2_clasificador" / "models")
    ap.add_argument("--name", default=_DEFAULT_NAME)
    ap.add_argument("--verify-source", type=Path, required=True,
                    help="CSV de entrenamiento con las columnas de las variables del modelo")
    ap.add_argument("--out-dir", type=Path, default=_HERE)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    joblib_path = args.models_dir / f"{args.name}.joblib"
    meta = json.loads((args.models_dir / f"{args.name}.metadata.json").read_text())
    features = list(meta["features"])
    model = joblib.load(joblib_path)

    out_base = args.out_dir / "gpu_rf_sin_reloj"
    onnx_path = out_base.with_suffix(".onnx")
    onnx_model = convert_rf_to_onnx(model, len(features))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model.SerializeToString())
    print(f"exportado: {onnx_path} ({onnx_path.stat().st_size} bytes)")

    df = pd.read_csv(args.verify_source, low_memory=False)
    missing = set(features) - set(df.columns)
    if missing:
        raise SystemExit(f"--verify-source no tiene las columnas: {sorted(missing)}")
    df = df.dropna(subset=features)
    X = df[features].to_numpy(dtype=np.float32)
    summary = verify_row_by_row(model, onnx_path, X, args.tol)
    print(f"verificacion OK: {summary}")

    features_path = out_base.with_suffix(".features.txt")
    features_path.write_text("\n".join(features) + "\n")
    out_base.with_suffix(".onnx.metadata.json").write_text(json.dumps({
        "schema": "gpu_onnx_export/1",
        "source_joblib": f"fase2_clasificador/models/{args.name}.joblib",
        "source_joblib_sha256": hashlib.sha256(joblib_path.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "features_in_order": features,
        "decision_rule": "clase = memory_bound si P(memory_bound) > 0.5 (equivale a RandomForestClassifier.predict)",
        "verification": {**summary, "tol": args.tol, "verify_source": str(args.verify_source)},
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"variables: {features_path}")


if __name__ == "__main__":
    main()
