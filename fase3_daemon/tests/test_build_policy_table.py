import json

import pandas as pd
import pytest

from fase3_daemon.policy import build_policy_table as bpt


def _policy_cpu(action="no_actuar"):
    entry = {"action": action, "n_kernels": 19, "reason": "ningun_nivel_mejora_edp"}
    if action == "actuar":
        entry |= {"chosen_level": "F0", "gain": 0.02, "gain_ci95": [0.01, 0.03], "p": 0.01}
    return {"policy": {"cpu-compute_bound": entry, "cpu-memory_bound": entry}}


def _policy_by_family(f1_significant=True, f1_gain=0.089, f1_p=0.0195, f2_gain=0.074, f2_p=0.156):
    def levels():
        return {
            "F0": {"gain": -0.006, "gain_ci95": [-0.02, 0.01], "p_wilcoxon_mejora": 0.77},
            "F1": {"gain": f1_gain, "gain_ci95": [0.03, 0.13], "p_wilcoxon_mejora": f1_p if f1_significant else 0.5},
            "F2": {"gain": f2_gain, "gain_ci95": [-0.04, 0.16], "p_wilcoxon_mejora": f2_p},
        }
    return {
        "compute_bound": {"n_familias": 5, "levels": {"F0": {"gain": -0.007, "gain_ci95": [0, 0],
                          "p_wilcoxon_mejora": 0.9}}, "lofo": {"ganancia_media_realizada": -0.048}},
        "memory_bound": {"n_familias": 8, "levels": levels(), "lofo": {"ganancia_media_realizada": 0.065}},
    }


def _gpu_dataset(clock_f1=1260.0, constant=True):
    rows = []
    clocks = [clock_f1, clock_f1] if constant else [clock_f1, clock_f1 + 10]
    for c in clocks:
        rows.append({"gpu_freq_level_id": "F1", "phase_label_train": "memory_bound", "gpu_sm_clock_mhz_median": c})
    return pd.DataFrame(rows)


# --- _choose_family_level ---

def test_choose_family_level_picks_significant_positive_max_gain():
    levels = _policy_by_family()["memory_bound"]["levels"]
    lv, entry = bpt._choose_family_level(levels)
    assert lv == "F1"  # F1 es significativo (p<0.05); F2 no lo es en este fixture
    assert entry["gain"] == pytest.approx(0.089)


def test_choose_family_level_none_when_nothing_significant():
    levels = _policy_by_family(f1_significant=False)["memory_bound"]["levels"]
    lv, entry = bpt._choose_family_level(levels)
    assert lv is None
    assert entry is None


def test_choose_family_level_ignores_negative_gain_even_if_significant():
    levels = {"F0": {"gain": -0.05, "gain_ci95": [0, 0], "p_wilcoxon_mejora": 0.001}}
    lv, _ = bpt._choose_family_level(levels)
    assert lv is None


# --- cpu_entry ---

def test_cpu_entry_no_actuar_passthrough():
    entry = bpt.cpu_entry(_policy_cpu(), "compute_bound")
    assert entry["action"] == "no_actuar"
    assert entry["n_kernels"] == 19


def test_cpu_entry_actuar_not_implemented():
    with pytest.raises(NotImplementedError):
        bpt.cpu_entry(_policy_cpu(action="actuar"), "compute_bound")


# --- gpu_entry ---

def test_gpu_entry_actuar_resolves_real_clock():
    entry = bpt.gpu_entry(_policy_by_family(), "memory_bound", _gpu_dataset())
    assert entry["action"] == "actuar"
    assert entry["chosen_level"] == "F1"
    assert entry["resolved_clock_mhz"] == 1260


def test_gpu_entry_no_actuar_when_nothing_significant():
    pbf = _policy_by_family(f1_significant=False)
    pbf["memory_bound"]["levels"]["F2"]["p_wilcoxon_mejora"] = 0.9
    entry = bpt.gpu_entry(pbf, "memory_bound", _gpu_dataset())
    assert entry["action"] == "no_actuar"
    assert "resolved_clock_mhz" not in entry


def test_gpu_entry_raises_on_non_constant_clock():
    with pytest.raises(ValueError, match="no es constante"):
        bpt.gpu_entry(_policy_by_family(), "memory_bound", _gpu_dataset(constant=False))


def test_gpu_entry_raises_without_dataset_when_actuar():
    with pytest.raises(ValueError, match="no se pasó --gpu-dataset"):
        bpt.gpu_entry(_policy_by_family(), "memory_bound", None)


def test_gpu_entry_no_actuar_does_not_need_dataset():
    entry = bpt.gpu_entry(_policy_by_family(), "compute_bound", None)
    assert entry["action"] == "no_actuar"


# --- main(): guarda contra el artefacto equivocado ---

def test_main_refuses_policy_gpu_json(tmp_path, capsys):
    cpu_path = tmp_path / "policy_cpu.json"
    cpu_path.write_text(json.dumps(_policy_cpu()))
    wrong_path = tmp_path / "policy_gpu.json"  # nombre exacto del artefacto por-kernel a rechazar
    wrong_path.write_text(json.dumps(_policy_by_family()))
    dataset_path = tmp_path / "dataset.csv"
    _gpu_dataset().to_csv(dataset_path, index=False)

    import sys
    argv = ["build_policy_table.py", "--cpu-policy", str(cpu_path), "--gpu-policy", str(wrong_path),
            "--gpu-dataset", str(dataset_path), "--out", str(tmp_path / "out.yaml")]
    monkeypatch_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit):
            bpt.main()
    finally:
        sys.argv = monkeypatch_argv


def test_main_end_to_end(tmp_path):
    cpu_path = tmp_path / "policy_cpu.json"
    cpu_path.write_text(json.dumps(_policy_cpu()))
    gpu_path = tmp_path / "policy_by_family.json"
    gpu_path.write_text(json.dumps(_policy_by_family()))
    dataset_path = tmp_path / "dataset.csv"
    _gpu_dataset().to_csv(dataset_path, index=False)
    out_path = tmp_path / "policy_table.yaml"

    import sys
    import yaml
    argv = ["build_policy_table.py", "--cpu-policy", str(cpu_path), "--gpu-policy", str(gpu_path),
            "--gpu-dataset", str(dataset_path), "--out", str(out_path)]
    original_argv = sys.argv
    sys.argv = argv
    try:
        bpt.main()
    finally:
        sys.argv = original_argv

    doc = yaml.safe_load(out_path.read_text())
    assert doc["policy"]["gpu-memory_bound"]["action"] == "actuar"
    assert doc["policy"]["gpu-memory_bound"]["chosen_level"] == "F1"
    assert doc["policy"]["gpu-memory_bound"]["resolved_clock_mhz"] == 1260
    assert doc["policy"]["gpu-compute_bound"]["action"] == "no_actuar"
    assert doc["policy"]["cpu-compute_bound"]["action"] == "no_actuar"

    # el resultado debe cargar sin error en el consumidor real del daemon
    from fase3_daemon.gpu_loop.loop import build_controller_from_policy
    controller = build_controller_from_policy(doc["policy"], min_dwell_ns=1, set_clock=lambda mhz: True)
    assert controller is not None
