"""Preparación y gates estructurados del cribado Fase 1.

Este módulo es el soporte de ``run_screening_to_report.sh``. Mantiene YAML y
JSON fuera de Bash para evitar editar manifiestos con ``sed`` y, sobre todo,
impide que el cribado GPU use la OI histórica del catálogo: primero perfila
cada candidato con ncu, actualiza una copia de trabajo del catálogo y genera
un manifiesto que contiene únicamente candidatos con verdad convergente.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any

import pandas as pd
import yaml

from common.hpc import manifest as manifest_module
from fase1_telemetria import ncu_convergence as ncu
from fase2_clasificador.eval.protocol import derive_kernel_family

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_SOURCE = REPO_ROOT / "fase1_telemetria/catalog/catalog.yaml"
CPU_TEMPLATE = REPO_ROOT / "fase1_telemetria/catalog/campaigns/campaign_pacca_phase_coverage_cpu_screen.yaml"
GPU_TEMPLATE = REPO_ROOT / "fase1_telemetria/catalog/campaigns/campaign_pacca_phase_coverage_gpu_screen.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"YAML inválido: {path}")
    return doc


def _write_yaml(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _kernel_map(catalog_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    kernels = catalog_doc.get("kernels")
    if not isinstance(kernels, list):
        raise ValueError("catalog.yaml no contiene una lista kernels")
    return {str(entry["id"]): entry for entry in kernels}


def prepare(results_root: Path, tag: str, node_id: str, kernel_root: Path) -> dict[str, Any]:
    root = results_root.resolve() / tag
    config_dir = root / "config"
    manifests_dir = root / "manifests"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = config_dir / "catalog.screening.yaml"
    if not catalog_path.exists():
        shutil.copy2(CATALOG_SOURCE, catalog_path)

    def materialize(template: Path, device: str) -> tuple[Path, str, Path]:
        doc = _read_yaml(template)
        campaign_id = f"{tag}_{device}_screen"
        campaign_dir = root / "campaigns" / device
        doc["campaign_id"] = campaign_id
        doc["output_dir"] = str(campaign_dir)
        doc["catalog_path"] = str(catalog_path)
        # Identificador único + reanudación controlada por el orquestador.
        # Nunca borrar resultados previos desde este preparador.
        doc["overwrite"] = True
        path = manifests_dir / f"{device}_{'candidates' if device == 'gpu' else 'screen'}.yaml"
        _write_yaml(path, doc)
        return path, campaign_id, campaign_dir

    cpu_manifest, cpu_id, cpu_dir = materialize(CPU_TEMPLATE, "cpu")
    gpu_candidates, gpu_id, gpu_dir = materialize(GPU_TEMPLATE, "gpu")
    workflow = {
        "schema": "f1/screening_workflow/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "kernel_root": str(kernel_root.resolve()),
        "node_id": node_id,
        "tag": tag,
        "root": str(root),
        "catalog": str(catalog_path),
        "cpu_manifest": str(cpu_manifest),
        "gpu_candidates_manifest": str(gpu_candidates),
        "gpu_eligible_manifest": str(manifests_dir / "gpu_eligible.yaml"),
        "cpu_campaign_id": cpu_id,
        "gpu_campaign_id": gpu_id,
        "cpu_campaign_dir": str(cpu_dir),
        "gpu_campaign_dir": str(gpu_dir),
        "ncu_reports_dir": str(root / "ncu"),
        "transition_dir": str(root / "gpu_transition"),
        "warmup_cpu_dir": str(root / "warmup" / "cpu"),
        "warmup_gpu_dir": str(root / "warmup" / "gpu"),
        "coverage_cpu_dir": str(root / "coverage" / "cpu"),
        "coverage_gpu_dir": str(root / "coverage" / "gpu"),
        "utility_dir": str(root / "kernel_utility"),
    }
    workflow_path = root / "workflow.json"
    workflow_path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False) + "\n")
    # Validar ahora el CPU y el manifiesto candidato GPU contra el parser real.
    manifest_module.load(cpu_manifest)
    manifest_module.load(gpu_candidates)
    return workflow


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _expected_checksum(entry: dict[str, Any], node_id: str) -> str | None:
    declared = entry.get("binary_checksum")
    if isinstance(declared, str):
        return declared
    if isinstance(declared, dict):
        return declared.get(node_id)
    return None


def _error_report(kernel_ref: str, status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "kernel_ref": kernel_ref, "status": status, "reason": reason,
        "converged": False, "roofline_label_eligible": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), **extra,
    }


def check_binaries(workflow_path: Path) -> dict[str, Any]:
    wf = json.loads(workflow_path.read_text())
    catalog = _kernel_map(_read_yaml(Path(wf["catalog"])))
    refs: set[str] = set()
    def reference(item: Any) -> str:
        return item["kernel_ref"] if isinstance(item, dict) else str(item)
    for key in ("cpu_manifest", "gpu_candidates_manifest"):
        doc = _read_yaml(Path(wf[key]))
        refs.update(reference(item) for item in doc.get("kernels", []))
        refs.update(reference(item) for item in doc.get("calibration", []))
        refs.update(reference(item) for item in doc.get("gpu", {}).get("calibration", []))
    rows = []
    kernel_root = Path(wf["kernel_root"])
    for ref in sorted(refs):
        entry = catalog[ref]
        path = kernel_root / entry["exec_path"]
        expected = _expected_checksum(entry, wf["node_id"])
        actual = _sha256(path) if path.is_file() else None
        status = "ok" if actual is not None and expected is not None and actual == expected else (
            "missing" if actual is None else "checksum_mismatch" if expected is not None else "checksum_not_declared"
        )
        rows.append({
            "kernel_ref": ref, "path": str(path), "expected_checksum": expected,
            "actual_checksum": actual, "status": status,
        })
    report = {
        "schema": "f1/screening_binary_preflight/1", "node_id": wf["node_id"],
        "ok": all(row["status"] == "ok" for row in rows), "binaries": rows,
    }
    out = Path(wf["root"]) / "binary_preflight.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def profile_ncu_batch(workflow_path: Path, launch_counts: list[int], *,
                      ncu_binary: str = "ncu", force: bool = False,
                      only_kernels: set[str] | None = None) -> dict[str, Any]:
    wf = json.loads(workflow_path.read_text())
    kernel_root = Path(wf["kernel_root"])
    reports_dir = Path(wf["ncu_reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    catalog_doc = _read_yaml(Path(wf["catalog"]))
    entries = _kernel_map(catalog_doc)
    gpu_doc = _read_yaml(Path(wf["gpu_candidates_manifest"]))
    refs = [item["kernel_ref"] for item in gpu_doc["kernels"]]
    versions = ncu._ncu_versions(ncu_binary)
    if shutil.which(ncu_binary) is None:
        raise RuntimeError(f"no se encontró ncu: {ncu_binary}")

    reports: list[dict[str, Any]] = []
    for ref in refs:
        report_path = reports_dir / f"{ref}.json"
        selected = only_kernels is None or ref in only_kernels
        if report_path.exists():
            existing = json.loads(report_path.read_text())
            should_retry_placeholder = selected and existing.get("status") == "not_profiled"
            if (not force or not selected) and not should_retry_placeholder:
                reports.append(existing)
                continue
        if not selected:
            report = _error_report(ref, "not_profiled", "sin reporte ncu; no fue seleccionado en esta invocación")
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            reports.append(report)
            continue
        entry = entries[ref]
        executable = kernel_root / entry["exec_path"]
        expected = _expected_checksum(entry, wf["node_id"])
        if not executable.is_file():
            report = _error_report(ref, "binary_missing", f"no existe {executable}")
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            reports.append(report)
            continue
        actual = _sha256(executable)
        if expected is None or actual != expected:
            report = _error_report(
                ref, "checksum_mismatch",
                f"checksum real {actual}; esperado para {wf['node_id']}: {expected}",
                exec_path=entry["exec_path"], binary_checksum=actual,
            )
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            reports.append(report)
            continue

        command = [entry["exec_path"], *shlex.split(entry.get("exec_args") or "")]
        points: list[ncu.NcuPoint] = []
        tensor_total = 0.0
        failure: str | None = None
        for count in launch_counts:
            csv_path = reports_dir / f"{ref}__lc{count}.csv"
            log_path = reports_dir / f"{ref}__lc{count}.log"
            cmd = [
                ncu_binary, "--csv", "--page", "raw", "--print-units", "base",
                "--metrics", ncu._METRICS_ARG, "--target-processes", "all",
                "--launch-count", str(count), *command,
            ]
            try:
                result = subprocess.run(
                    cmd, cwd=kernel_root, capture_output=True, text=True,
                    timeout=1200, check=False,
                )
            except (OSError, subprocess.SubprocessError) as error:
                failure = f"no se pudo ejecutar ncu para launch-count={count}: {error}"
                break
            csv_path.write_text(result.stdout, encoding="utf-8")
            log_path.write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0:
                failure = f"ncu rc={result.returncode} para launch-count={count}; ver {log_path}"
                break
            parsed = ncu.parse_ncu_csv(result.stdout)
            if "dram__bytes.sum" not in parsed.get("metric_names_present", []):
                failure = f"CSV ncu sin dram__bytes.sum para launch-count={count}"
                break
            flops, precision = ncu.flops_and_precision(parsed)
            tensor_total += float(parsed.get("tensor_inst", 0.0))
            points.append(ncu.NcuPoint(
                count, int(parsed.get("launches_observed", 0)), flops,
                float(parsed.get("dram_bytes", 0.0)),
                ncu.operational_intensity(flops, parsed.get("dram_bytes", 0.0)),
                precision, parsed.get("by_metric", {}),
            ))

        if failure:
            report = _error_report(
                ref, "profiling_error", failure, exec_path=entry["exec_path"],
                binary_checksum=actual, points=[asdict(point) for point in points], **versions,
            )
        else:
            built = ncu.build_kernel_report(
                ref, points, exec_path=entry["exec_path"], binary_checksum=actual,
                kernel_args=command[1:], tensor_instructions_observed=tensor_total,
                catalog_declared_operational_intensity=entry.get("operational_intensity_flops_per_byte"),
                catalog_declared_precision=entry.get("gpu_precision"), **versions,
            )
            report = asdict(built)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        reports.append(report)

    eligible = [report["kernel_ref"] for report in reports if report.get("roofline_label_eligible") is True]
    blocked = [report["kernel_ref"] for report in reports if report.get("status") in {
        "binary_missing", "checksum_mismatch", "profiling_error", "not_profiled",
    }]
    excluded = [report["kernel_ref"] for report in reports if report["kernel_ref"] not in eligible]

    # Actualizar SOLO la copia de trabajo; el catálogo versionado permanece
    # intacto hasta que el investigador revise el informe.
    for report in reports:
        if report.get("roofline_label_eligible") is not True:
            continue
        entry = entries[report["kernel_ref"]]
        entry["operational_intensity_flops_per_byte"] = report["final_operational_intensity"]
        entry["gpu_precision"] = report["precision"]
    _write_yaml(Path(wf["catalog"]), catalog_doc)

    eligible_doc = gpu_doc.copy()
    eligible_doc["kernels"] = [{"kernel_ref": ref} for ref in eligible]
    eligible_doc.setdefault("workflow_notes", {})["ncu_gate"] = {
        "reports_dir": str(reports_dir), "eligible": eligible, "excluded": excluded,
    }
    _write_yaml(Path(wf["gpu_eligible_manifest"]), eligible_doc)
    if eligible:
        manifest_module.load(Path(wf["gpu_eligible_manifest"]))

    summary = {
        "schema": "f1-gpu-004/ncu_batch_summary/1",
        "launch_counts": launch_counts, "candidate_count": len(refs),
        "eligible_count": len(eligible), "eligible": eligible,
        "excluded": excluded, "blocking_errors": blocked,
        "reports": reports,
    }
    (reports_dir / "ncu_batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    pd.DataFrame([{
        "kernel_ref": report.get("kernel_ref"), "status": report.get("status"),
        "roofline_label_eligible": report.get("roofline_label_eligible"),
        "precision": report.get("precision"),
        "operational_intensity": report.get("final_operational_intensity"),
        "catalog_oi": report.get("catalog_declared_operational_intensity"),
        "relative_difference": report.get("operational_intensity_relative_difference"),
        "reason": report.get("reason"),
    } for report in reports]).to_csv(reports_dir / "ncu_batch_summary.csv", index=False)
    return summary


def apply_cadence(workflow_path: Path, cadence_report: Path) -> int:
    wf = json.loads(workflow_path.read_text())
    cadence = json.loads(cadence_report.read_text())
    interval = int(cadence["q_produccion_ns"])
    for key in ("gpu_candidates_manifest", "gpu_eligible_manifest"):
        path = Path(wf[key])
        if not path.exists():
            continue
        doc = _read_yaml(path)
        doc["gpu_interval_ns"] = interval
        doc.setdefault("workflow_notes", {})["cadence_report"] = str(cadence_report.resolve())
        _write_yaml(path, doc)
    return interval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--results-root", type=Path, required=True)
    prep.add_argument("--tag", required=True)
    prep.add_argument("--node-id", default="pacca-a100")
    prep.add_argument("--kernel-root", type=Path, required=True)
    batch = sub.add_parser("ncu")
    batch.add_argument("--workflow", type=Path, required=True)
    batch.add_argument("--launch-counts", default="5,20,50")
    batch.add_argument("--ncu", default="ncu")
    batch.add_argument("--force", action="store_true")
    batch.add_argument("--kernel", action="append", dest="kernels",
                       help="Reperfilar solo este kernel (repetible); los demás reportes se conservan.")
    cadence = sub.add_parser("apply-cadence")
    cadence.add_argument("--workflow", type=Path, required=True)
    cadence.add_argument("--cadence-report", type=Path, required=True)
    binaries = sub.add_parser("check-binaries")
    binaries.add_argument("--workflow", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        workflow = prepare(args.results_root, args.tag, args.node_id, args.kernel_root)
        print(json.dumps(workflow, indent=2, ensure_ascii=False))
        return 0
    if args.command == "check-binaries":
        report = check_binaries(args.workflow)
        for row in report["binaries"]:
            print(f"  {row['kernel_ref']:<36} {row['status']}")
        return 0 if report["ok"] else 2
    if args.command == "ncu":
        counts = [int(value) for value in args.launch_counts.split(",")]
        summary = profile_ncu_batch(
            args.workflow, counts, ncu_binary=args.ncu, force=args.force,
            only_kernels=set(args.kernels) if args.kernels else None,
        )
        print(f"ncu: {summary['eligible_count']}/{summary['candidate_count']} elegibles")
        for report in summary["reports"]:
            print(f"  {report['kernel_ref']:<36} {report.get('status')}: {report.get('reason', '')}")
        return 2 if summary["blocking_errors"] or not summary["eligible"] else 0
    interval = apply_cadence(args.workflow, args.cadence_report)
    print(f"gpu_interval_ns actualizado a {interval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
