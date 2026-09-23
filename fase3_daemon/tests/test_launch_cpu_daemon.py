import pytest

from fase3_daemon.cpu_loop import launch_cpu_daemon as lcd


def _policy(compute="no_actuar", memory="no_actuar", c_khz=3_200_000, m_khz=2_900_000):
    def entry(action, khz):
        e = {"action": action}
        if action != "no_actuar":
            e["resolved_freq_khz"] = khz
        return e
    return {"policy": {"cpu-compute_bound": entry(compute, c_khz), "cpu-memory_bound": entry(memory, m_khz)}}


def test_no_actuar_no_genera_flags():
    assert lcd.cpu_flags(_policy()["policy"]) == []


def test_f0_base_y_f1_en_memory():
    flags = lcd.cpu_flags(_policy("actuar_experimental", "actuar_experimental")["policy"])
    assert flags == ["--compute-actuar", "--compute-freq-khz", "3200000",
                     "--memory-actuar", "--memory-freq-khz", "2900000"]


def test_solo_base_sin_memory():
    flags = lcd.cpu_flags(_policy("actuar_experimental", "no_actuar")["policy"])
    assert flags == ["--compute-actuar", "--compute-freq-khz", "3200000"]


def test_memory_sin_base_es_error():
    with pytest.raises(ValueError, match="nivel base"):
        lcd.cpu_flags(_policy("no_actuar", "actuar_experimental")["policy"])


def test_actuar_sin_frecuencia_resuelta_es_error():
    doc = _policy("actuar_experimental")
    del doc["policy"]["cpu-compute_bound"]["resolved_freq_khz"]
    with pytest.raises(ValueError, match="resolved_freq_khz"):
        lcd.cpu_flags(doc["policy"])


def test_accion_desconocida_es_error():
    doc = _policy()
    doc["policy"]["cpu-compute_bound"]["action"] = "quizas"
    with pytest.raises(ValueError, match="desconocida"):
        lcd.cpu_flags(doc["policy"])


def test_entrada_faltante_es_error():
    with pytest.raises(ValueError, match="cpu-memory_bound"):
        lcd.cpu_flags({"cpu-compute_bound": {"action": "no_actuar"}})


def test_comando_completo_y_brazo_invalido():
    cmd = lcd.build_command(_policy("actuar_experimental", "actuar_experimental"), "/x/cpu_loop_main", "activo",
                            ["--perf-cpus", "0,1"])
    assert cmd[:3] == ["/x/cpu_loop_main", "--arm", "activo"]
    assert cmd[-2:] == ["--perf-cpus", "0,1"]
    with pytest.raises(ValueError, match="--arm"):
        lcd.build_command(_policy(), "/x", "base", [])
