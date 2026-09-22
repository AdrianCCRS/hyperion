from types import SimpleNamespace

import pytest

from fase1_telemetria.audit_gpu_oi_coverage import audit, require_complete


def _manifest(kernels, enabled=True):
    return SimpleNamespace(
        kernels=tuple(kernels),
        gpu={"activity_trace": {
            "enabled": enabled,
            "parameters_by_kernel": {"rodinia_gaussian": {"size": 4096}},
        }},
    )


def _entry(device="gpu", direct=True):
    return SimpleNamespace(
        device=device,
        cupti_activity_exec_path="/kernels/real" if direct else None,
        cupti_activity_binary_checksum="sha256:real" if direct else None,
    )


def test_auditoria_exige_modelo_y_parametros_para_cada_kernel_gpu():
    manifest = _manifest(["rodinia_gaussian", "otro_gpu"])
    errors = audit(manifest, {
        "rodinia_gaussian": _entry(), "otro_gpu": _entry(),
    })
    assert errors == ["otro_gpu: sin modelo analítico registrado"]
    with pytest.raises(ValueError, match="Cobertura OI analítica incompleta"):
        require_complete(manifest, {"rodinia_gaussian": _entry(), "otro_gpu": _entry()})


def test_auditoria_no_afecta_campanas_historicas_sin_activity_trace():
    assert audit(_manifest(["otro_gpu"], enabled=False), {"otro_gpu": _entry()}) == []
