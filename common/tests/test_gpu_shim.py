from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.hpc import gpu_shim


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_arc74_find_cuda_root_usa_directorio_de_nvcc_si_tiene_cuda_runtime(tmp_path, monkeypatch):
    nvcc = tmp_path / "compilers" / "bin" / "nvcc"
    _touch(nvcc)
    _touch(tmp_path / "compilers" / "include" / "cuda_runtime.h")
    monkeypatch.setattr(gpu_shim.shutil, "which", lambda name: str(nvcc))

    root = gpu_shim._find_cuda_root()
    assert root == tmp_path / "compilers"


def test_arc74_find_cuda_root_busca_hermano_cuda_si_nvcc_no_lo_trae(tmp_path, monkeypatch):
    # ARC-74: patrón real encontrado en paccaA100 -- nvcc resuelve al del
    # NVIDIA HPC SDK bajo compilers/bin, sin cuda_runtime.h; el toolkit real
    # vive en un hermano cuda/<version>/.
    hpc_sdk_root = tmp_path / "Linux_x86_64" / "23.1"
    nvcc = hpc_sdk_root / "compilers" / "bin" / "nvcc"
    _touch(nvcc)
    _touch(hpc_sdk_root / "cuda" / "12.0" / "include" / "cuda_runtime.h")
    monkeypatch.setattr(gpu_shim.shutil, "which", lambda name: str(nvcc))

    root = gpu_shim._find_cuda_root()
    assert root == hpc_sdk_root / "cuda" / "12.0"


def test_arc74_find_cuda_root_sin_nvcc_en_path(monkeypatch):
    monkeypatch.setattr(gpu_shim.shutil, "which", lambda name: None)
    assert gpu_shim._find_cuda_root() is None


def test_arc74_find_cuda_root_sin_cuda_runtime_en_ningun_lado(tmp_path, monkeypatch):
    nvcc = tmp_path / "compilers" / "bin" / "nvcc"
    _touch(nvcc)
    monkeypatch.setattr(gpu_shim.shutil, "which", lambda name: str(nvcc))
    assert gpu_shim._find_cuda_root() is None


def test_arc74_find_cublas_lib_dir_en_cuda_lib64(tmp_path):
    cuda_root = tmp_path / "cuda" / "12.0"
    _touch(cuda_root / "lib64" / "libcublas.so.12")
    assert gpu_shim._find_cublas_lib_dir(cuda_root) == cuda_root / "lib64"


def test_arc74_find_cublas_lib_dir_en_math_libs_hermano(tmp_path):
    # ARC-74: patrón real en paccaA100 -- libcublas NO está en cuda/lib64,
    # vive en math_libs/<version>/targets/<arch>/lib.
    hpc_sdk_root = tmp_path / "Linux_x86_64" / "23.1"
    cuda_root = hpc_sdk_root / "cuda" / "12.0"
    (cuda_root / "lib64").mkdir(parents=True)
    cublas_dir = hpc_sdk_root / "math_libs" / "12.0" / "targets" / "x86_64-linux" / "lib"
    _touch(cublas_dir / "libcublas.so.12")

    assert gpu_shim._find_cublas_lib_dir(cuda_root) == cublas_dir


def test_arc74_find_cublas_lib_dir_ausente(tmp_path):
    cuda_root = tmp_path / "cuda" / "12.0"
    (cuda_root / "lib64").mkdir(parents=True)
    assert gpu_shim._find_cublas_lib_dir(cuda_root) is None


def test_arc74_cuda_lib_dirs_incluye_cudart_y_cublas(tmp_path, monkeypatch):
    hpc_sdk_root = tmp_path / "Linux_x86_64" / "23.1"
    nvcc = hpc_sdk_root / "compilers" / "bin" / "nvcc"
    _touch(nvcc)
    cuda_root = hpc_sdk_root / "cuda" / "12.0"
    _touch(cuda_root / "include" / "cuda_runtime.h")
    (cuda_root / "lib64").mkdir(parents=True, exist_ok=True)
    cublas_dir = hpc_sdk_root / "math_libs" / "12.0" / "targets" / "x86_64-linux" / "lib"
    _touch(cublas_dir / "libcublas.so.12")
    monkeypatch.setattr(gpu_shim.shutil, "which", lambda name: str(nvcc))

    dirs = gpu_shim.cuda_lib_dirs()
    assert cuda_root / "lib64" in dirs
    assert cublas_dir in dirs


def test_arc74_cuda_lib_dirs_vacio_sin_cuda(monkeypatch):
    monkeypatch.setattr(gpu_shim.shutil, "which", lambda name: None)
    assert gpu_shim.cuda_lib_dirs() == []
