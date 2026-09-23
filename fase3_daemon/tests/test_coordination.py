"""Pruebas de `fase3_daemon/gpu_loop/coordination.py` (Bloque C, ítem C4).
`fase3_daemon/cpu_loop/tests/test_gpu_active_reader.cpp` es el espejo en
C++ del lector; aquí solo se prueba el escritor Python."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fase3_daemon.gpu_loop.coordination import GpuActiveSignalWriter


def test_write_true_escribe_1(tmp_path):
    path = tmp_path / "gpu_active"
    GpuActiveSignalWriter(path).write(True)
    assert path.read_bytes() == b"1"


def test_write_false_escribe_0(tmp_path):
    path = tmp_path / "gpu_active"
    GpuActiveSignalWriter(path).write(False)
    assert path.read_bytes() == b"0"


def test_write_no_deja_archivo_temporal(tmp_path):
    path = tmp_path / "gpu_active"
    writer = GpuActiveSignalWriter(path)
    writer.write(True)
    assert not writer._tmp_path.exists()  # os.replace() se lo llevo consigo
    assert set(p.name for p in tmp_path.iterdir()) == {"gpu_active"}


def test_write_sobrescribe_transiciones_sucesivas(tmp_path):
    path = tmp_path / "gpu_active"
    writer = GpuActiveSignalWriter(path)
    writer.write(True)
    assert path.read_bytes() == b"1"
    writer.write(False)
    assert path.read_bytes() == b"0"
    writer.write(True)
    assert path.read_bytes() == b"1"


def test_crea_el_directorio_padre_si_no_existe(tmp_path):
    path = tmp_path / "sub" / "dir" / "gpu_active"
    GpuActiveSignalWriter(path).write(True)
    assert path.read_bytes() == b"1"


def test_path_property_devuelve_la_ruta_original(tmp_path):
    path = tmp_path / "gpu_active"
    writer = GpuActiveSignalWriter(path)
    assert writer.path == path
