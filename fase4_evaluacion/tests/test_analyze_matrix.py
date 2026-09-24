from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase4_evaluacion.analyze_matrix import is_gpu_kernel, score_cell, summarize


def _row(arm, wall, ecpu, egpu, rep=1):
    return {"scope": "gpu", "set": "known", "arm": arm, "wall_s": wall, "e_cpu_j": ecpu, "e_gpu_j": egpu,
            "e_total_j": ecpu + egpu, "edp": (ecpu + egpu) * wall, "rep": rep}


def test_razones_frente_a_base():
    s = {r["arm"]: r for r in summarize([_row("base", 10, 100, 100), _row("activo", 11, 90, 100)])}
    assert s["activo"]["ratio_T"] == 1.1
    assert s["activo"]["ratio_E_cpu"] == 0.9
    assert s["activo"]["ratio_EDP"] == round((190 * 11) / (200 * 10), 3)
    assert s["base"]["ratio_EDP"] == 1.0


def test_puntuacion_atribuye_por_ts_y_cuenta_abstenciones():
    phases = [{"begin_ns": 0, "end_ns": 100, "phase_label_hint": "memory_bound", "kernel_id": "a"},
              {"begin_ns": 200, "end_ns": 300, "phase_label_hint": "compute_bound", "kernel_id": "b"}]
    decisions = [{"ts_ns": 50, "label": "memory_bound"}, {"ts_ns": 60, "label": "compute_bound"},
                 {"ts_ns": 250, "label": "revisar"}, {"ts_ns": 150, "label": "memory_bound"}, {"ts_ns": 10, "label": None}]
    s = score_cell(phases, decisions)
    assert (s["ok"], s["wrong"], s["abst"], s["covered"]) == (1, 1, 1, 2)


def test_razones_frente_a_ref_sin_turbo_y_edp_de_gpu_sola():
    rows = [_row("base", 10, 100, 100), _row("base_noturbo", 10.5, 90, 100), _row("activo_gpu", 10, 100, 90)]
    for r in rows:
        r["edp_gpu"] = r["e_gpu_j"] * r["wall_s"]
    s = {r["arm"]: r for r in summarize(rows)}
    assert s["activo_gpu"]["ratio_EDP"] == round((190 * 10) / (200 * 10), 3)
    assert s["activo_gpu"]["ratio_EDPgpu"] == 0.9
    assert s["activo_gpu"]["ratio_T_nt"] == round(10 / 10.5, 3)
    assert s["activo_gpu"]["ratio_EDP_nt"] == round((190 * 10) / (190 * 10.5), 3)
    assert "ratio_EDP_nt" not in s["base"] or s["base"]["ratio_EDP_nt"] == round(2000 / (190 * 10.5), 3)


def test_los_kernels_dual_de_gpu_cuentan_como_gpu_y_las_variantes_omp_como_cpu():
    assert is_gpu_kernel("dual_spmv_gpu_N200000000") and is_gpu_kernel("gpu_stream_bw") and is_gpu_kernel("rodinia_backprop")
    assert not is_gpu_kernel("dgemm_n2048") and not is_gpu_kernel("npb_cg") and not is_gpu_kernel("cpu_xsbench_omp")
    assert not is_gpu_kernel("rodinia_lavamd_omp")
