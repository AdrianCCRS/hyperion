"""F1-GPU-004: pruebas del parser de `ncu` y de la lógica de convergencia.

No ejecutan `ncu` (no está en el entorno). Usan salida CSV de `ncu` fabricada
a mano con las columnas reales de métricas.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria import ncu_convergence as nc


_HEADER = (
    '"Kernel Name",'
    '"sm__sass_thread_inst_executed_op_fadd_pred_on.sum",'
    '"sm__sass_thread_inst_executed_op_fmul_pred_on.sum",'
    '"sm__sass_thread_inst_executed_op_ffma_pred_on.sum",'
    '"sm__sass_thread_inst_executed_op_dadd_pred_on.sum",'
    '"sm__sass_thread_inst_executed_op_dmul_pred_on.sum",'
    '"sm__sass_thread_inst_executed_op_dfma_pred_on.sum",'
    '"sm__sass_thread_inst_executed_op_iadd_pred_on.sum",'
    '"sm__sass_thread_inst_executed_op_imad_pred_on.sum",'
    '"dram__bytes.sum"'
)


def _csv(*rows: str) -> str:
    return _HEADER + "\n" + "\n".join(rows) + "\n"


def test_parser_suma_por_bucket_y_cuenta_launches():
    text = _csv(
        '"k",100,100,50,0,0,0,10,10,8000',
        '"k",100,100,50,0,0,0,10,10,8000',
    )
    p = nc.parse_ncu_csv(text)
    assert p["launches_observed"] == 2
    assert p["fp32_inst"] == 500      # (100+100+50)*2
    assert p["fp64_inst"] == 0
    assert p["fma_inst"] == 100       # 50*2
    assert p["dram_bytes"] == 16000


def test_flops_fp64_y_precision():
    p = nc.parse_ncu_csv(_csv('"k",0,0,0,1000,1000,500,0,0,20000'))
    flops, prec = nc.flops_and_precision(p)
    assert prec == "fp64"
    # fp64_inst = 2500, fma_inst = 500 -> flops = 3000
    assert flops == 3000
    assert nc.operational_intensity(flops, p["dram_bytes"]) == 3000 / 20000


def test_precision_mixta():
    p = nc.parse_ncu_csv(_csv('"k",500,500,0,500,500,0,0,0,10000'))
    _, prec = nc.flops_and_precision(p)
    assert prec == "mixed"


def test_kernel_entero_sin_flops_es_no_apto():
    p = nc.parse_ncu_csv(_csv('"k",0,0,0,0,0,0,9000,9000,50000'))
    flops, prec = nc.flops_and_precision(p)
    assert flops == 0.0
    assert prec == "integer_no_flops"
    rep = nc.build_kernel_report("gap_bfs", [
        nc.NcuPoint(10, 10, 0.0, 50000, None, "integer_no_flops"),
        nc.NcuPoint(50, 50, 0.0, 50000, None, "integer_no_flops"),
    ])
    assert rep.status == "not_suitable_for_roofline_truth"
    assert rep.roofline_label_eligible is False
    assert "FLOPs" in rep.reason


def test_convergencia_cuando_la_oi_se_estabiliza():
    pts = [
        nc.NcuPoint(10, 10, 1000, 1000, 1.0, "fp32"),
        nc.NcuPoint(50, 50, 6000, 3000, 2.0, "fp32"),
        nc.NcuPoint(100, 100, 12100, 6000, 2.0167, "fp32"),
        nc.NcuPoint(500, 500, 60500, 30000, 2.0167, "fp32"),  # cambio ~0
    ]
    r = nc.assess_convergence(pts, rel_tol=0.01)
    assert r["converged"] is True
    assert r["converged_at_launch_count"] == 500
    rep = nc.build_kernel_report("rodinia_cfd", pts)
    assert rep.status == "converged"
    assert rep.roofline_label_eligible is True
    assert rep.final_operational_intensity == 2.0167


def test_no_convergencia_si_la_oi_sigue_moviendose():
    pts = [
        nc.NcuPoint(10, 10, 1000, 1000, 1.0, "fp64"),
        nc.NcuPoint(50, 50, 3000, 1000, 3.0, "fp64"),
        nc.NcuPoint(100, 100, 8000, 1000, 8.0, "fp64"),   # sigue creciendo
    ]
    r = nc.assess_convergence(pts)
    assert r["converged"] is False
    rep = nc.build_kernel_report("k", pts)
    assert rep.status == "not_converged"
    assert rep.roofline_label_eligible is False


def test_no_convergencia_si_launches_observados_no_coinciden():
    pts = [
        nc.NcuPoint(100, 100, 10000, 5000, 2.0, "fp32"),
        nc.NcuPoint(500, 480, 50000, 25000, 2.0, "fp32"),  # 480 != 500
    ]
    r = nc.assess_convergence(pts)
    assert r["converged"] is False
    assert "observados" in r["reason"]


def test_menos_de_dos_puntos_no_converge():
    r = nc.assess_convergence([nc.NcuPoint(10, 10, 100, 100, 1.0, "fp32")])
    assert r["converged"] is False


def test_runbook_se_genera_cuando_no_hay_ncu(tmp_path):
    rb = nc.runbook("rodinia_cfd", "cfd_bin -launches {launches}", [10, 100], tmp_path)
    assert rb.exists()
    txt = rb.read_text()
    assert "ncu --csv --metrics" in txt
    assert "lc10" in txt and "lc100" in txt


def test_main_from_csv_produce_reporte_para_el_gate_h(tmp_path):
    for lc, oi_bytes in ((10, (1000, 1000)), (100, (20167, 10000)), (500, (100835, 50000))):
        (tmp_path / f"rodinia_cfd__lc{lc}.csv").write_text(
            _csv(f'"cfd",{oi_bytes[0] // 2},{oi_bytes[0] // 2},0,0,0,0,0,0,{oi_bytes[1]}')
        )
    rc = nc.main(["--kernel", "rodinia_cfd", "--from-csv",
                  str(tmp_path / "rodinia_cfd__lc*.csv"), "--out-dir", str(tmp_path)])
    report = tmp_path / "rodinia_cfd.json"
    assert report.exists()
    import json
    j = json.loads(report.read_text())
    assert "converged" in j and "roofline_label_eligible" in j
    assert j["precision"] == "fp32"
    assert rc in (0, 1)
