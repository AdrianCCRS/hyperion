from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase1_telemetria.gpu_window_oi import (  # noqa: E402
    LaunchWork,
    aggregate_launches_in_window,
    build_window_truth,
    load_cupti_launch_work,
)


def test_lanzamiento_que_cruza_limite_se_prorratea_sin_duplicar_trabajo():
    launch = LaunchWork("kernel", 50, 150, flops=1_000, bytes_moved=100)

    windows = build_window_truth([launch], start_ns=0, end_ns=200, window_ns=100)

    assert len(windows) == 2
    assert [w.flops for w in windows] == pytest.approx([500, 500])
    assert [w.bytes_moved for w in windows] == pytest.approx([50, 50])
    assert [w.operational_intensity for w in windows] == pytest.approx([10, 10])


def test_varios_lanzamientos_producen_oi_ponderada_por_bytes_no_media_simple():
    launches = [
        LaunchWork("compute", 0, 50, flops=1_000, bytes_moved=10),
        LaunchWork("memory", 50, 100, flops=100, bytes_moved=100),
    ]

    truth = aggregate_launches_in_window(launches, start_ns=0, end_ns=100)

    assert truth.operational_intensity == pytest.approx(1_100 / 110)
    assert truth.active_fraction == 1.0
    assert truth.overlapping_launches == 2


def test_actividad_concurrente_no_produce_duty_cycle_mayor_que_uno():
    launches = [
        LaunchWork("a", 0, 100, flops=100, bytes_moved=10),
        LaunchWork("b", 20, 80, flops=60, bytes_moved=10),
    ]

    truth = aggregate_launches_in_window(launches, start_ns=0, end_ns=100)

    assert truth.active_ns == 100
    assert truth.active_fraction == 1.0


def test_ventana_sin_actividad_queda_sin_oi():
    truth = aggregate_launches_in_window([], start_ns=0, end_ns=120_000_000)

    assert truth.operational_intensity is None
    assert truth.active_fraction == 0.0
    assert truth.overlapping_launches == 0


def test_rechaza_trabajo_o_intervalos_invalidos():
    with pytest.raises(ValueError):
        LaunchWork("bad", 10, 10, flops=1, bytes_moved=1)
    with pytest.raises(ValueError):
        LaunchWork("bad", 0, 10, flops=1, bytes_moved=0)
    with pytest.raises(ValueError):
        build_window_truth([], start_ns=0, end_ns=10, window_ns=0)


def test_carga_traza_cupti_y_adjunta_modelo_analitico(tmp_path):
    trace = tmp_path / "cupti.csv"
    trace.write_text(
        "launch_index,kernel_name,start_ns,end_ns\n"
        "0,_Z4Fan1PfS_ii,100,200\n"
        "1,_Z4Fan2PfS_S_iii,220,400\n",
        encoding="utf-8",
    )

    launches = load_cupti_launch_work(
        trace, kernel_ref="rodinia_gaussian", parameters={"size": 2}
    )

    assert [launch.launch_index for launch in launches] == [0, 1]
    assert launches[0].flops == 1
    assert launches[1].flops == 6


def test_carga_spmv_por_operacion_y_prorratea_su_trabajo_una_sola_vez(tmp_path):
    trace = tmp_path / "cupti_spmv.csv"
    trace.write_text(
        "launch_index,kernel_name,start_ns,end_ns,activity_kind,transfer_bytes\n"
        "0,[memcpy],0,10,memcpy,20\n"
        "1,[memcpy],10,20,memcpy,112\n"
        "2,[memcpy],20,30,memcpy,224\n"
        "3,[memcpy],30,40,memcpy,32\n"
        "4,csr_partition_kernel,40,50,kernel,0\n"
        "5,csrmv_v3_kernel,50,60,kernel,0\n"
        "6,[memcpy],60,70,memcpy,32\n",
        encoding="utf-8",
    )
    events = load_cupti_launch_work(
        trace, kernel_ref="dual_spmv_gpu_N1000000", parameters={"size": 4}
    )
    assert len(events) == 7
    assert sum(event.flops for event in events) == pytest.approx(56)
    assert sum(event.bytes_moved for event in events) == pytest.approx(420)
    assert {event.activity_kind for event in events} == {"kernel", "memcpy"}


def test_region_medida_descarta_eventos_de_setup_y_verificacion(tmp_path):
    trace = tmp_path / "cupti.csv"
    trace.write_text(
        "launch_index,kernel_name,start_ns,end_ns\n"
        "0,Fan1,0,10\n"
        "1,Fan2,10,20\n"
        "2,Fan1,20,30\n"
        "3,Fan2,30,40\n",
        encoding="utf-8",
    )
    launches = load_cupti_launch_work(
        trace, kernel_ref="rodinia_gaussian", parameters={"size": 3},
        measured_start_ns=20, measured_end_ns=40,
    )
    assert [launch.launch_index for launch in launches] == [2, 3]


def test_kmeans_descarta_prefijo_de_inicializacion_y_agrupa_iteraciones(tmp_path):
    trace = tmp_path / "cupti_kmeans.csv"
    trace.write_text(
        "launch_index,kernel_name,start_ns,end_ns,activity_kind,transfer_bytes\n"
        "0,[memcpy],0,10,memcpy,120\n"
        "1,invert_mapping,10,20,kernel,0\n"
        "2,[memcpy],20,30,memcpy,40\n"
        "3,[memcpy],30,40,memcpy,24\n"
        "4,kmeansPoint,40,50,kernel,0\n"
        "5,[memcpy],50,60,memcpy,40\n"
        "6,[memcpy],60,70,memcpy,40\n"
        "7,[memcpy],70,80,memcpy,24\n"
        "8,kmeansPoint,80,90,kernel,0\n"
        "9,[memcpy],90,100,memcpy,40\n",
        encoding="utf-8",
    )
    events = load_cupti_launch_work(
        trace, kernel_ref="rodinia_kmeans_gpu_N350000_D34_K200",
        parameters={"n_points": 10, "n_features": 3, "n_clusters": 2},
    )
    assert [event.launch_index for event in events] == list(range(2, 10))
    assert sum(event.flops for event in events) == pytest.approx(360)
