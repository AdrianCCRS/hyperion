"""Exporta el candidato CPU (`fase2_clasificador/models/xgboost_cpu.joblib`)
a ONNX (§4.3 punto 3 del plan de realineación): "exportar los modelos
entrenados a un formato de inferencia liviano invocable desde C++".

Por qué hace falta esto y no basta con el .joblib: el loop de CPU corre en
C++ dentro de `telemetry/` (§4.3 punto 2), en el camino caliente de ~1ms,
y no puede pagar el costo de un intérprete Python por tick. ONNX Runtime
tiene SDK C++ (confirmado disponible en conda-forge,
`environment-hyperion-verify.yml`) que sí puede invocarse desde ahí.

No hay conversión directa de XGBClassifier con skl2onnx solo -- requiere
el convertidor de `onnxmltools` registrado en skl2onnx (patrón estándar
para modelos de árboles de xgboost/lightgbm, no un caso especial de este
proyecto).

Verificación (obligatoria, no opcional): compara predict_proba de
sklearn/xgboost contra la sesión ONNX real, FILA A FILA, sobre el propio
conjunto de entrenamiento -- no basta con que la conversión "no truene".
Una diferencia mayor a `--tol` en cualquier fila hace fallar el script con
código de salida distinto de 0 en vez de exportar un artefacto silenciosamente
incorrecto.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def convert_xgb_classifier_to_onnx(model, feature_count: int, model_name: str = "xgboost_cpu"):
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
    from xgboost import XGBClassifier

    update_registered_converter(
        XGBClassifier, "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes, convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )
    initial_types = [("input", FloatTensorType([None, feature_count]))]
    # target_opset explicito: la version de ai.onnx.ml que emite el
    # convertidor de xgboost (onnxmltools) por defecto va por delante de lo
    # que esta version de skl2onnx sabe leer sin que se lo digan.
    return convert_sklearn(model, model_name, initial_types=initial_types,
                           options={type(model): {"zipmap": False}},
                           target_opset={"": 18, "ai.onnx.ml": 3})


def verify_row_by_row(model, onnx_path: Path, X: np.ndarray, tol: float) -> dict:
    """Compara predict_proba(sklearn) contra la sesión ONNX real, fila a
    fila. Devuelve un resumen; lanza AssertionError si alguna fila excede
    `tol` -- nunca un resumen agregado que pueda esconder un outlier."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_label, onnx_proba = sess.run(None, {input_name: X.astype(np.float32)})
    onnx_proba = np.asarray(onnx_proba)

    sk_proba = model.predict_proba(X)
    sk_label = model.predict(X)

    diff = np.abs(onnx_proba - sk_proba)
    max_diff = float(diff.max())
    worst_row = int(diff.max(axis=1).argmax())
    label_mismatches = int((np.asarray(onnx_label) != sk_label).sum())

    if max_diff > tol or label_mismatches > 0:
        raise AssertionError(
            f"verificación ONNX falló: max_diff_proba={max_diff:.2e} (tol={tol:.0e}), "
            f"peor fila={worst_row}, {label_mismatches} discrepancias de etiqueta de {len(X)} filas -- "
            f"sklearn[{worst_row}]={sk_proba[worst_row]}, onnx[{worst_row}]={onnx_proba[worst_row]}"
        )
    return {"n_rows": len(X), "max_diff_proba": max_diff, "worst_row": worst_row, "label_mismatches": label_mismatches}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=_REPO_ROOT / "fase2_clasificador/models/xgboost_cpu.joblib")
    ap.add_argument("--metadata", type=Path, default=_REPO_ROOT / "fase2_clasificador/models/xgboost_cpu.metadata.json")
    ap.add_argument("--verify-source", type=Path, required=True,
                    help="CSV con las columnas de --metadata features, para la verificación fila a fila "
                         "(recomendado: el mismo source del entrenamiento, o una muestra representativa)")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "fase3_daemon/cpu_loop/xgboost_cpu.onnx")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="Tolerancia máxima de diferencia absoluta en predict_proba entre sklearn y ONNX")
    ap.add_argument("--max-verify-rows", type=int, default=50_000,
                    help="Límite de filas a verificar (la verificación completa es barata, pero se acota "
                         "para no depender de tener el CSV de 42458 filas completo a mano)")
    args = ap.parse_args()

    metadata = json.loads(args.metadata.read_text())
    features = metadata["features"]
    model = joblib.load(args.model)

    onnx_model = convert_xgb_classifier_to_onnx(model, feature_count=len(features))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(onnx_model.SerializeToString())
    print(f"exportado: {args.out} ({args.out.stat().st_size} bytes)")

    df = pd.read_csv(args.verify_source, low_memory=False)
    missing = set(features) - set(df.columns)
    if missing:
        raise SystemExit(f"--verify-source no tiene las columnas requeridas: {sorted(missing)}")
    df = df.dropna(subset=features)
    if len(df) > args.max_verify_rows:
        df = df.sample(n=args.max_verify_rows, random_state=20260918)
    X = df[features].to_numpy(dtype=np.float32)

    summary = verify_row_by_row(model, args.out, X, tol=args.tol)
    print(f"verificación OK: {summary['n_rows']} filas, max_diff_proba={summary['max_diff_proba']:.2e}, "
          f"0 discrepancias de etiqueta (tol={args.tol:.0e})")

    onnx_metadata_path = args.out.with_suffix(".onnx.metadata.json")
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(_REPO_ROOT))
        except ValueError:
            return str(p)

    onnx_metadata_path.write_text(json.dumps({
        "schema": "cpu_onnx_export/1",
        "source_joblib": _rel(args.model),
        "source_joblib_sha256": __import__("hashlib").sha256(args.model.read_bytes()).hexdigest(),
        "features_in_order": features,
        "threshold": metadata["decision"]["threshold"],
        "verification": {**summary, "tol": args.tol, "verify_source": _rel(args.verify_source)},
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"metadata: {onnx_metadata_path}")


if __name__ == "__main__":
    main()
