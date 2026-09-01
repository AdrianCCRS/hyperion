from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase4_evaluacion import governors


def _write_cpu(tmp_path: Path, cpu: int, *, governor: str = "performance",
               available: str = "performance powersave ondemand schedutil") -> dict[str, str]:
    cpufreq = tmp_path / f"cpu{cpu}" / "cpufreq"
    cpufreq.mkdir(parents=True)
    (cpufreq / "scaling_governor").write_text(governor)
    (cpufreq / "scaling_available_governors").write_text(available)
    return {"scaling_governor": str(cpufreq / "scaling_governor")}


def _env(control_paths: dict, *, write_capable: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        frequency_write_capable=write_capable,
        frequency_control_paths=control_paths,
    )


def test_available_governors_interseccion_entre_cpus(tmp_path):
    paths = {
        0: _write_cpu(tmp_path, 0, available="performance ondemand schedutil"),
        1: _write_cpu(tmp_path, 1, available="performance ondemand"),  # sin schedutil
    }
    env = _env(paths)
    assert governors.available_governors((0, 1), env) == {"performance", "ondemand"}


def test_available_governors_cpu_sin_path_da_conjunto_vacio_en_la_interseccion(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0)}
    env = _env(paths)
    assert governors.available_governors((0, 1), env) == set()  # cpu 1 no declarado -> interseccion vacia


def test_governor_scenario_conmuta_y_restaura(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0, governor="performance")}
    env = _env(paths)

    with governors.governor_scenario((0,), "ondemand", env):
        assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == "ondemand"

    assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == "performance"


def test_governor_scenario_restaura_incluso_si_el_bloque_lanza(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0, governor="performance")}
    env = _env(paths)

    with pytest.raises(ValueError, match="boom"):
        with governors.governor_scenario((0,), "schedutil", env):
            assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == "schedutil"
            raise ValueError("boom")

    assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == "performance"


def test_governor_scenario_gobernador_no_disponible_lanza_sin_tocar_nada(tmp_path):
    paths = {0: _write_cpu(tmp_path, 0, governor="performance", available="performance powersave")}
    env = _env(paths)

    with pytest.raises(governors.GovernorNotAvailableError, match="schedutil"):
        with governors.governor_scenario((0,), "schedutil", env):
            pytest.fail("no debería entrar al bloque si el gobernador no está disponible")

    assert (tmp_path / "cpu0/cpufreq/scaling_governor").read_text() == "performance"


def test_los_3_escenarios_de_5_1_son_ondemand_schedutil_performance():
    assert governors.SCENARIO_GOVERNORS == ("ondemand", "schedutil", "performance")
