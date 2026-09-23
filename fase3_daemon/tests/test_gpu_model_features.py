"""Guarda del candidato de GPU que usa `gpu_loop_main` (Bloque D): sus variables no pueden incluir el reloj SM ni la
potencia. El daemon cambia el reloj y la potencia depende de el, asi que el modelo entrenaria y decidiria sobre una
variable que la propia actuacion altera (auditoria job 7601: la regresion logistica cambiaba 26.7% de sus
predicciones al forzar el reloj). Sustituye a la prueba equivalente del clasificador en Python, retirado."""
from pathlib import Path

_DIR = Path(__file__).resolve().parents[1] / "gpu_loop_cpp"
_FORBIDDEN = {"gpu_sm_clock_mhz_median", "gpu_sm_clock_mhz_std", "gpu_power_mw_median", "gpu_power_mw_std"}


def test_variables_del_modelo_de_gpu_no_incluyen_reloj_ni_potencia():
    names = [ln.strip() for ln in (_DIR / "gpu_rf_sin_reloj.features.txt").read_text().splitlines() if ln.strip()]
    assert names, "archivo de variables vacio"
    assert not (set(names) & _FORBIDDEN), sorted(set(names) & _FORBIDDEN)
