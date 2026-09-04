from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from fase1_telemetria import screening_workflow as sw


def test_prepare_crea_copia_de_trabajo_y_manifiestos_validos(tmp_path):
    kernel_root = tmp_path / "kernels"
    kernel_root.mkdir()
    wf = sw.prepare(tmp_path / "results", "t1", "pacca-a100", kernel_root)
    assert Path(wf["catalog"]).is_file()
    assert Path(wf["cpu_manifest"]).is_file()
    assert Path(wf["gpu_candidates_manifest"]).is_file()
    assert wf["cpu_campaign_id"] == "t1_cpu_screen"
    assert json.loads((tmp_path / "results/t1/workflow.json").read_text())["node_id"] == "pacca-a100"


def _long_csv(n_launches: int, *, floating: bool) -> str:
    header = ('"ID","Process ID","Process Name","Host Name","Kernel Name",'
              '"Context","Stream","Block Size","Grid Size","Device","CC",'
              '"Section Name","Metric Name","Metric Unit","Metric Value"')
    rows = [header]
    for launch in range(n_launches):
        base = (f'"{launch}","9","app","host","k","1","7","b","g",'
                '"0","8.0","Raw"')
        rows.append(base + ',"dram__bytes.sum","byte","100"')
        metric = ("sm__sass_thread_inst_executed_op_fadd_pred_on.sum"
                  if floating else "sm__sass_thread_inst_executed_op_iadd_pred_on.sum")
        rows.append(base + f',"{metric}","inst","{200 if floating else 100}"')
        rows.append(base + ',"sm__inst_executed_pipe_tensor.sum","inst","0"')
    return "\n".join(rows)


def test_ncu_batch_filtra_y_actualiza_solo_copia_de_catalogo(tmp_path, monkeypatch):
    kernel_root = tmp_path / "kernels"
    (kernel_root / "bin").mkdir(parents=True)
    wf = sw.prepare(tmp_path / "results", "t2", "pacca-a100", kernel_root)
    catalog_path = Path(wf["catalog"])
    catalog = yaml.safe_load(catalog_path.read_text())
    entries = {entry["id"]: entry for entry in catalog["kernels"]}
    refs = ["rodinia_gaussian", "gpu_rajaperf_reduce3_int"]
    for ref in refs:
        binary = kernel_root / entries[ref]["exec_path"]
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(ref.encode())
        checksum = "sha256:" + hashlib.sha256(ref.encode()).hexdigest()
        entries[ref]["binary_checksum"]["pacca-a100"] = checksum
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False))
    gpu_manifest = Path(wf["gpu_candidates_manifest"])
    document = yaml.safe_load(gpu_manifest.read_text())
    document["kernels"] = [{"kernel_ref": ref} for ref in refs]
    gpu_manifest.write_text(yaml.safe_dump(document, sort_keys=False))

    monkeypatch.setattr(sw.shutil, "which", lambda _: "/usr/bin/ncu")
    monkeypatch.setattr(sw.ncu, "_ncu_versions", lambda _: {
        "ncu_version": "test", "driver_version": "test", "cuda_version": "test",
    })

    def fake_run(command, **kwargs):
        count = int(command[command.index("--launch-count") + 1])
        floating = "rodinia_gaussian" in " ".join(command)
        return subprocess.CompletedProcess(command, 0, _long_csv(count, floating=floating), "")

    monkeypatch.setattr(sw.subprocess, "run", fake_run)
    summary = sw.profile_ncu_batch(Path(wf["root"]) / "workflow.json", [5, 20, 50])
    assert summary["eligible"] == ["rodinia_gaussian"]
    assert "gpu_rajaperf_reduce3_int" in summary["excluded"]
    eligible = yaml.safe_load(Path(wf["gpu_eligible_manifest"]).read_text())
    assert eligible["kernels"] == [{"kernel_ref": "rodinia_gaussian"}]
    updated = {entry["id"]: entry for entry in yaml.safe_load(catalog_path.read_text())["kernels"]}
    assert updated["rodinia_gaussian"]["operational_intensity_flops_per_byte"] == 2.0
