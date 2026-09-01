"""Tests para docs/justifications/scripts/ncu_gpu_precision.py (ARC-110):
el modulo compartido que reemplaza la logica de seleccion de precision
unica (FP32 xor FP64 segun gpu_precision) por recoleccion simultanea de
ambas, encontrada en sweep_ncu_launch_count.py/reparse_ncu_launch_count.py/
extend_ncu_lud_convergence.py.
"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "justifications" / "scripts"))

from ncu_gpu_precision import (  # noqa: E402
    compute_gpu_precision_result,
    is_mixed_precision,
    parse_ncu_csv_totals,
)


HEADER = '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"'


def _row(launch_id: str, metric: str, value: str) -> str:
    return (
        f'"{launch_id}","123","k","127.0.0.1","Kernel","1","7","(1,1,1)","(1,1,1)","0","8.0",'
        f'"Command line profiler metrics","{metric}","unit","{value}"'
    )


def _csv(*rows: str) -> str:
    return "\n".join([HEADER, *rows])


def test_arc110_kernel_fp32_puro():
    # Solo ADD/MUL FP32, nada de FP64 -- fraction_fp32 debe ser 1.0.
    text = _csv(
        _row("0", "dram__bytes.sum", "1000"),
        _row("0", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "100"),
        _row("0", "sm__sass_thread_inst_executed_op_fmul_pred_on.sum", "50"),
    )
    totals, n = parse_ncu_csv_totals(text)
    result = compute_gpu_precision_result(totals, n)

    assert result.flops_fp32 == 150.0
    assert result.flops_fp64 == 0.0
    assert result.flops_total == 150.0
    assert result.fraction_fp32 == 1.0
    assert result.fraction_fp64 == 0.0
    assert is_mixed_precision(result) is False


def test_arc110_kernel_fp64_puro():
    text = _csv(
        _row("0", "dram__bytes.sum", "2000"),
        _row("0", "sm__sass_thread_inst_executed_op_dadd_pred_on.sum", "200"),
        _row("0", "sm__sass_thread_inst_executed_op_dmul_pred_on.sum", "80"),
    )
    totals, n = parse_ncu_csv_totals(text)
    result = compute_gpu_precision_result(totals, n)

    assert result.flops_fp64 == 280.0
    assert result.flops_fp32 == 0.0
    assert result.fraction_fp64 == 1.0
    assert is_mixed_precision(result) is False


def test_arc110_kernel_mixto_fp32_y_fp64():
    # El caso que las 3 herramientas anteriores no podian detectar --
    # ambas precisiones con FLOPs no triviales.
    text = _csv(
        _row("0", "dram__bytes.sum", "5000"),
        _row("0", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "10000"),
        _row("0", "sm__sass_thread_inst_executed_op_dfma_pred_on.sum", "5000"),
    )
    totals, n = parse_ncu_csv_totals(text)
    result = compute_gpu_precision_result(totals, n)

    assert result.flops_fp32 == 10000.0
    assert result.flops_fp64 == 10000.0  # dfma pesa 2x
    assert result.flops_total == 20000.0
    assert result.fraction_fp32 == 0.5
    assert result.fraction_fp64 == 0.5
    assert is_mixed_precision(result) is True


def test_arc110_fma_pondera_como_2_flops_add_mul_como_1():
    text = _csv(
        _row("0", "dram__bytes.sum", "100"),
        _row("0", "sm__sass_thread_inst_executed_op_ffma_pred_on.sum", "10"),
        _row("0", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "10"),
        _row("0", "sm__sass_thread_inst_executed_op_fmul_pred_on.sum", "10"),
    )
    totals, n = parse_ncu_csv_totals(text)
    result = compute_gpu_precision_result(totals, n)

    # 2*10 (fma) + 10 (add) + 10 (mul) = 40
    assert result.flops_fp32 == 40.0


def test_arc110_dram_bytes_ausente_operational_intensity_none():
    text = _csv(
        _row("0", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "10"),
    )
    totals, n = parse_ncu_csv_totals(text)
    result = compute_gpu_precision_result(totals, n)

    assert result.dram_bytes == 0.0
    assert result.operational_intensity is None


def test_arc110_dram_bytes_cero_explicito_operational_intensity_none():
    text = _csv(
        _row("0", "dram__bytes.sum", "0"),
        _row("0", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "10"),
    )
    totals, n = parse_ncu_csv_totals(text)
    result = compute_gpu_precision_result(totals, n)

    assert result.operational_intensity is None


def test_arc110_metrica_na_se_trata_como_cero():
    text = _csv(
        _row("0", "dram__bytes.sum", "N/A"),
        _row("0", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "N/A"),
    )
    totals, n = parse_ncu_csv_totals(text)
    result = compute_gpu_precision_result(totals, n)

    assert result.dram_bytes == 0.0
    assert result.flops_fp32 == 0.0
    assert result.operational_intensity is None


def test_arc110_flops_total_cero_fraccion_none_no_zerodivisionerror():
    text = _csv(_row("0", "dram__bytes.sum", "500"))
    totals, n = parse_ncu_csv_totals(text)
    result = compute_gpu_precision_result(totals, n)

    assert result.flops_total == 0.0
    assert result.fraction_fp32 is None
    assert result.fraction_fp64 is None


def test_arc110_varios_lanzamientos_se_agregan_y_se_cuentan():
    # 3 lanzamientos (ID 0,1,2), cada uno con su propia fila -- los
    # totales deben sumarse y n_launches debe reflejar los 3 IDs
    # distintos, no el numero de filas.
    text = _csv(
        _row("0", "dram__bytes.sum", "1000"),
        _row("0", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "100"),
        _row("1", "dram__bytes.sum", "1000"),
        _row("1", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "100"),
        _row("2", "dram__bytes.sum", "1000"),
        _row("2", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "100"),
    )
    totals, n = parse_ncu_csv_totals(text)
    result = compute_gpu_precision_result(totals, n)

    assert n == 3
    assert result.dram_bytes == 3000.0
    assert result.flops_fp32 == 300.0


def test_arc110_csv_sin_encabezado_devuelve_vacio():
    totals, n = parse_ncu_csv_totals("basura sin encabezado ID")
    assert totals == {}
    assert n == 0


def test_arc110_fraccion_trivial_no_se_marca_como_mixta():
    # Caso real encontrado en la auditoria: rodinia_heartwall midio
    # fp64=1.1e6 frente a fp32=9.2e10 (fraccion ~0.00001) -- ruido de
    # arranque, no mezcla real. Un piso absoluto de FLOPs (version anterior
    # de este modulo) marcaba esto como mixto por error.
    text = _csv(
        _row("0", "dram__bytes.sum", "1000"),
        _row("0", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "92000000000"),
        _row("0", "sm__sass_thread_inst_executed_op_dadd_pred_on.sum", "1100000"),
    )
    totals, n = parse_ncu_csv_totals(text)
    result = compute_gpu_precision_result(totals, n)

    assert result.flops_fp64 == 1100000.0  # no trivial en terminos absolutos
    assert is_mixed_precision(result) is False  # pero si en terminos relativos


def test_arc110_fraccion_justo_en_el_umbral_si_se_marca_mixta():
    # 0.1% exacto en ambos lados -- el umbral es inclusivo (>=).
    text = _csv(
        _row("0", "dram__bytes.sum", "1000"),
        _row("0", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "999"),
        _row("0", "sm__sass_thread_inst_executed_op_dadd_pred_on.sum", "1"),
    )
    totals, n = parse_ncu_csv_totals(text)
    result = compute_gpu_precision_result(totals, n)

    assert result.fraction_fp64 == pytest.approx(0.001)
    assert is_mixed_precision(result) is True
