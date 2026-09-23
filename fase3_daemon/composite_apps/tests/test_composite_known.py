"""Pruebas de `fase3_daemon/composite_apps/composite_known.py` (Bloque C,
ítem C2, Aplicación A). `run_fn`/`now_fn` inyectados -- sin lanzar
binarios reales ni depender del catálogo real de pacca, corrible en
cualquier máquina."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.hpc.catalog import KernelEntry
from fase3_daemon.composite_apps.composite_known import (
    PhaseRecord, resolve_entries, run_composite,
)


def _entry(id_: str, *, label: str = "compute_bound", checksum: str = "sha256:abc") -> KernelEntry:
    return KernelEntry(
        id=id_, suite="TEST", role="dataset", exec_path=f"bin/{id_}",
        binary_checksum={"pacca-a100": checksum}, phase_label_hint=label,
        size_variant="TEST", expected_runtime_seconds=1, warmup_seconds=0.0,
        estimated_memory_bytes=1024,
        success_check={"type": "exit_code", "expected": 0}, exec_args="",
    )


def test_resolve_entries_orden_y_contenido():
    catalog = {"a": _entry("a"), "b": _entry("b", label="memory_bound")}
    entries = resolve_entries(catalog, ("b", "a"))
    assert [e.id for e in entries] == ["b", "a"]


def test_resolve_entries_falla_si_falta_un_kernel():
    catalog = {"a": _entry("a")}
    with pytest.raises(ValueError, match="no encontrados"):
        resolve_entries(catalog, ("a", "z"))


def test_resolve_entries_falla_sin_phase_label_hint():
    catalog = {"a": _entry("a")}
    catalog["a"].phase_label_hint = None
    with pytest.raises(ValueError, match="phase_label_hint"):
        resolve_entries(catalog, ("a",))


class _FakeRunFn:
    """Reemplaza subprocess.run -- registra los argv recibidos y devuelve
    un resultado configurable por kernel_id (via el ultimo segmento de
    argv[0])."""

    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stdout = stdout

    def __call__(self, argv, **_kwargs):
        self.calls.append(argv)
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout)


def _verified_entries(monkeypatch, ids):
    entries = [_entry(i) for i in ids]
    monkeypatch.setattr(
        "fase3_daemon.composite_apps.composite_known.verify_binary", lambda *_a, **_kw: True,
    )
    return entries


def test_run_composite_respeta_orden_y_ciclos(monkeypatch):
    entries = _verified_entries(monkeypatch, ["a", "b"])
    run_fn = _FakeRunFn()
    now_values = iter(range(100))

    records = run_composite(
        entries, cycles=2, node_id="pacca-a100", kernels_root=Path("/kroot"),
        run_fn=run_fn, now_fn=lambda: next(now_values),
    )

    assert [(r.cycle, r.kernel_id) for r in records] == [(0, "a"), (0, "b"), (1, "a"), (1, "b")]
    assert run_fn.calls[0][0] == "/kroot/bin/a"


def test_run_composite_marca_exito_segun_success_check(monkeypatch):
    entries = _verified_entries(monkeypatch, ["a"])
    run_fn = _FakeRunFn(returncode=1)  # exit_code esperado es 0 -> success=False

    records = run_composite(
        entries, cycles=1, node_id="pacca-a100", kernels_root=Path("/kroot"),
        run_fn=run_fn, now_fn=lambda: 0,
    )
    assert records[0].success is False
    assert records[0].exit_code == 1


def test_run_composite_registra_fronteras_begin_end(monkeypatch):
    entries = _verified_entries(monkeypatch, ["a"])
    run_fn = _FakeRunFn()
    now_values = iter([100, 250])  # begin, end

    records = run_composite(
        entries, cycles=1, node_id="pacca-a100", kernels_root=Path("/kroot"),
        run_fn=run_fn, now_fn=lambda: next(now_values),
    )
    assert records[0].begin_ns == 100
    assert records[0].end_ns == 250


def test_run_composite_on_phase_se_llama_por_cada_fase(monkeypatch):
    entries = _verified_entries(monkeypatch, ["a", "b"])
    run_fn = _FakeRunFn()
    seen: list[PhaseRecord] = []

    run_composite(
        entries, cycles=1, node_id="pacca-a100", kernels_root=Path("/kroot"),
        run_fn=run_fn, now_fn=lambda: 0, on_phase=seen.append,
    )
    assert [r.kernel_id for r in seen] == ["a", "b"]


def test_run_composite_verifica_checksum_contra_ruta_resuelta_no_cwd(monkeypatch):
    # Bug real (job 7568): verify_binary() resuelve exec_path relativo al
    # cwd del PROCESO, no a kernels_root -- pasarle entry.exec_path tal
    # cual ("bin/a") falla siempre que el proceso no esté parado en
    # kernels_root, incluso con el binario real presente y correcto.
    entries = [_entry("a")]
    seen_paths = []

    def fake_verify_binary(entry, node_id=None):
        seen_paths.append(entry.exec_path)
        return True

    monkeypatch.setattr(
        "fase3_daemon.composite_apps.composite_known.verify_binary", fake_verify_binary,
    )
    run_fn = _FakeRunFn()

    run_composite(
        entries, cycles=1, node_id="pacca-a100", kernels_root=Path("/kroot"),
        run_fn=run_fn, now_fn=lambda: 0,
    )
    assert seen_paths == ["/kroot/bin/a"]


def test_run_composite_nunca_corre_sin_checksum_verificado(monkeypatch):
    entries = [_entry("a")]
    monkeypatch.setattr(
        "fase3_daemon.composite_apps.composite_known.verify_binary", lambda *_a, **_kw: False,
    )
    run_fn = _FakeRunFn()

    with pytest.raises(RuntimeError, match="C02"):
        run_composite(
            entries, cycles=1, node_id="pacca-a100", kernels_root=Path("/kroot"),
            run_fn=run_fn, now_fn=lambda: 0,
        )
    assert run_fn.calls == []  # nunca se llegó a invocar el binario
