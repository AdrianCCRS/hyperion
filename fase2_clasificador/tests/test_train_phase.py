"""Tests de train_phase.py -- añadidos durante la reconstrucción en 4
fases (no existían en origin/fase-02). Cubren específicamente los 4 fixes
de esta reconstrucción: XGBoost en build_models(), p95 en measure_latency,
select_best_model() (error + latencia), y la serialización real del
modelo elegido -- antes de esto, train_phase.py solo imprimía tablas.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase2_clasificador.training import train_phase


def test_build_models_incluye_xgboost():
    modelos = train_phase.build_models(seed=0)
    assert "xgboost" in modelos
    # Resto de la comparación (§3.2 del plan) también presente.
    for esperado in ("mayoritaria", "arbol_prof1", "regresion_log", "arbol_prof6", "random_forest"):
        assert esperado in modelos


def test_measure_latency_devuelve_p50_p95_p99_ordenados():
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)

    p50, p95, p99 = train_phase.measure_latency(model, X, repeats=20)
    assert p50 <= p95 <= p99
    assert p50 >= 0.0


def test_select_best_model_prefiere_f1_alto_si_latency_weight_es_cero():
    results = {
        "lento_mejor": {"mean": 0.95},
        "rapido_peor": {"mean": 0.80},
    }
    latencies = {
        "lento_mejor": (100.0, 150.0, 200.0),
        "rapido_peor": (1.0, 1.5, 2.0),
    }
    assert train_phase.select_best_model(results, latencies, latency_weight=0.0) == "lento_mejor"


def test_select_best_model_penaliza_latencia_cuando_el_peso_es_alto():
    # F1 casi empatado, pero uno es 100x más lento -- con latency_weight
    # alto, el más rápido debe ganar aunque su F1 sea marginalmente peor.
    results = {
        "lento": {"mean": 0.901},
        "rapido": {"mean": 0.900},
    }
    latencies = {
        "lento": (100.0, 150.0, 200.0),
        "rapido": (1.0, 1.5, 2.0),
    }
    assert train_phase.select_best_model(results, latencies, latency_weight=5.0) == "rapido"


def test_select_best_model_nunca_elige_la_linea_base_mayoritaria():
    # 'mayoritaria' es la línea base obligatoria (DummyClassifier) -- nunca
    # debe ser el modelo serializado para producción, sin importar su score.
    results = {
        "mayoritaria": {"mean": 0.99},
        "arbol_prof6": {"mean": 0.80},
    }
    latencies = {
        "mayoritaria": (0.1, 0.1, 0.1),
        "arbol_prof6": (5.0, 6.0, 7.0),
    }
    assert train_phase.select_best_model(results, latencies, latency_weight=0.2) == "arbol_prof6"


def _write_fake_training_intervals_csv(path: Path, *, n_rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    ipc = rng.uniform(0.2, 3.0, size=n_rows)
    label = np.where(ipc < 1.2, "memory_bound", "compute_bound")
    frame = pd.DataFrame({
        "ipc": ipc,
        "mpki": rng.uniform(0, 50, size=n_rows),
        "cache_miss_rate": rng.uniform(0, 1, size=n_rows),  # F1-CPU-003: antes llc_miss_rate
        "stall_mem_ratio": rng.uniform(0, 1, size=n_rows),
        "ips": rng.uniform(1e8, 1e10, size=n_rows),
        "running_ratio": rng.uniform(0.5, 1.0, size=n_rows),
        "freq_khz_observed": rng.choice([800000, 3600000], size=n_rows),
        "phase_label_train": label,
        "kernel_ref": "fake_kernel",
        "freq_level_id": "REF",
        "training_quality_status": "ok",
        "frequency_quality_status": "valid",
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture
def fake_campaign(tmp_path) -> tuple[Path, str, list[str]]:
    campaign_id = "fake_campaign"
    kernels = ["kernel_a", "kernel_b", "kernel_c"]
    for i, kernel in enumerate(kernels):
        run_dir = tmp_path / f"{campaign_id}__{kernel}__REF__rep01"
        _write_fake_training_intervals_csv(
            run_dir / train_phase.TRAINING_INPUT_FILENAME, n_rows=200, seed=i
        )
    return tmp_path, campaign_id, kernels


def test_load_lee_las_corridas_sinteticas(fake_campaign):
    campaign_dir, campaign_id, kernels = fake_campaign
    df = train_phase.load(
        per_run_sample=200, seed=0, kernels=kernels,
        campaign_dir=campaign_dir, campaign_id=campaign_id, levels=["REF"],
    )
    assert set(df["kernel_ref"].unique()) == {"fake_kernel"}
    assert len(df) > 0
    assert set(df["phase_label_train"].unique()) <= {"memory_bound", "compute_bound"}


def test_load_acepta_csv_historico_con_llc_miss_rate(tmp_path):
    """F1-CPU-003: un training_cpu_intervals.csv grabado antes del rename
    trae `llc_miss_rate`; load() lo renombra a `cache_miss_rate` y no falla
    por esquema. No se producen las dos columnas a la vez."""
    campaign_id = "historico"
    run_dir = tmp_path / f"{campaign_id}__kernel_a__REF__rep01"
    _write_fake_training_intervals_csv(
        run_dir / train_phase.TRAINING_INPUT_FILENAME, n_rows=120, seed=1
    )
    p = run_dir / train_phase.TRAINING_INPUT_FILENAME
    frame = pd.read_csv(p)
    frame = frame.rename(columns={"cache_miss_rate": "llc_miss_rate"})  # simula CSV viejo
    assert "cache_miss_rate" not in frame.columns
    frame.to_csv(p, index=False)

    df = train_phase.load(
        per_run_sample=120, seed=0, kernels=["kernel_a"],
        campaign_dir=tmp_path, campaign_id=campaign_id, levels=["REF"],
    )
    assert "cache_miss_rate" in df.columns
    assert "llc_miss_rate" not in df.columns
    assert df["cache_miss_rate"].notna().all()


def test_load_falla_con_mensaje_claro_si_no_hay_datos(tmp_path):
    with pytest.raises(FileNotFoundError, match="ningún training_cpu_intervals.csv"):
        train_phase.load(
            per_run_sample=10, seed=0, kernels=["no_existe"],
            campaign_dir=tmp_path, campaign_id="vacio", levels=["REF"],
        )


def test_load_rechaza_csv_agregado_con_esquema_incompleto(tmp_path):
    campaign_id = "incompleto"
    run_dir = tmp_path / f"{campaign_id}__kernel__REF__rep01"
    run_dir.mkdir()
    pd.DataFrame({"ipc": [1.0]}).to_csv(
        run_dir / train_phase.TRAINING_INPUT_FILENAME, index=False
    )
    with pytest.raises(ValueError, match="esquema F1-CPU-002"):
        train_phase.load(
            per_run_sample=10, seed=0, kernels=["kernel"],
            campaign_dir=tmp_path, campaign_id=campaign_id, levels=["REF"],
        )


def test_main_serializa_modelo_y_metadata_reales(fake_campaign, monkeypatch, tmp_path):
    campaign_dir, campaign_id, kernels = fake_campaign
    output_dir = tmp_path / "models_out"
    # Todas las corridas sintéticas usan el mismo kernel_ref/familia
    # ("fake_kernel") a propósito -- leave_one_familia_out con una sola
    # familia no puede separar train/test, así que se generan 3 corridas
    # con distinto kernel_ref real para tener >=2 familias.
    for i, kernel in enumerate(kernels):
        run_dir = campaign_dir / f"{campaign_id}__{kernel}__REF__rep01"
        interval_path = run_dir / train_phase.TRAINING_INPUT_FILENAME
        frame = pd.read_csv(interval_path)
        frame["kernel_ref"] = kernel  # familia distinta por archivo
        frame.to_csv(interval_path, index=False)

    argv = [
        "train_phase.py",
        "--campaign-dir", str(campaign_dir),
        "--campaign-id", campaign_id,
        "--kernels", ",".join(kernels),
        "--levels", "REF",
        "--per-run-sample", "200",
        "--seed", "0",
        "--output-dir", str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_phase.main()

    joblib_files = list(output_dir.glob("*.joblib"))
    metadata_files = list(output_dir.glob("*.metadata.json"))
    assert len(joblib_files) == 1, "main() debe serializar exactamente un modelo elegido"
    assert len(metadata_files) == 1

    import joblib
    model = joblib.load(joblib_files[0])
    assert hasattr(model, "predict")

    import json
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["model_name"] != "mayoritaria"
    assert metadata["n_familias"] == 3
    assert set(metadata["features"]) == set(train_phase.FEATURES)
    assert metadata["training_granularity"] == "uncore_interval"
    assert metadata["training_input_filename"] == train_phase.TRAINING_INPUT_FILENAME
    assert "all_models_compared" in metadata and "xgboost" in metadata["all_models_compared"]
