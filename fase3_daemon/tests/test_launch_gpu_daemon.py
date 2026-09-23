import pytest

from fase3_daemon.gpu_loop_cpp import launch_gpu_daemon as lgd


def _policy(compute=("no_actuar", None), memory=("actuar", 1260.0)):
    def entry(action, mhz):
        e = {"action": action}
        if mhz is not None:
            e["resolved_clock_mhz"] = mhz
        return e
    return {"policy": {"gpu-compute_bound": entry(*compute), "gpu-memory_bound": entry(*memory)}}


def test_politica_vigente_memory_1260_compute_sin_actuar():
    flags = lgd.gpu_flags(_policy()["policy"])
    assert flags == ["--compute-clock-mhz", "0", "--memory-clock-mhz", "1260"]


def test_actuar_sin_reloj_resuelto_falla():
    with pytest.raises(ValueError, match="resolved_clock_mhz"):
        lgd.gpu_flags(_policy(memory=("actuar", None))["policy"])


def test_accion_desconocida_y_entrada_faltante_fallan():
    with pytest.raises(ValueError, match="desconocida"):
        lgd.gpu_flags(_policy(memory=("quizas", None))["policy"])
    with pytest.raises(ValueError, match="gpu-memory_bound"):
        lgd.gpu_flags({"gpu-compute_bound": {"action": "no_actuar"}})


def test_comando_completo_y_flags_prohibidos_tras_guiones():
    cmd = lgd.build_command(_policy(), "/x/gpu_loop_main", "activo", ["--model", "m.onnx"])
    assert cmd[:3] == ["/x/gpu_loop_main", "--arm", "activo"] and cmd[-2:] == ["--model", "m.onnx"]
    with pytest.raises(ValueError, match="no puede repetirse"):
        lgd.build_command(_policy(), "/x", "sombra", ["--memory-clock-mhz", "810"])
    with pytest.raises(ValueError, match="--arm"):
        lgd.build_command(_policy(), "/x", "base", [])
