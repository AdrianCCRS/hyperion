#!/usr/bin/env python3
"""ARC-174: reprocesa offline las corridas de CPU ya ejecutadas de una
campaña, aplicando la clasificación de frecuencia POR VENTANA (ver
orchestrator/validation.py::classify_frequency_window()) en vez del gate
agregado de tolerancia que existía antes. No vuelve a correr NADA en el
cluster -- lee samples.csv/metadata.json ya en disco, reconstruye
windows.csv con la nueva lógica y recalcula el veredicto final.

Alcance limitado a CPU (decisión explícita, ver
docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md ARC-174):
un kernel con device=="gpu" en el catálogo se salta, nunca se reprocesa
aquí -- GPU tiene otra cadencia/señal (gpu_sm_clock_mhz) y sus propios
factores G.

La salida SIEMPRE va a un directorio derivado separado
(--derived-dir), nunca sobrescribe windows.csv/verdict.json de la
corrida original -- cada corrida reprocesada escribe:
  <derived-dir>/<run_id>/windows.csv
  <derived-dir>/<run_id>/verdict.json
  <derived-dir>/<run_id>/frequency_quality_summary.json   (si se llegó a
                                                             construir windows.csv)
  <derived-dir>/<run_id>/reprocess_provenance.json         (fingerprint del
                                                             run_dir original,
                                                             checksum de este
                                                             script, versión
                                                             de esquema)

Uso:
    python3 -m orchestrator.schemas.reprocess_frequency_quality_v2 \\
        --manifest orchestrator/schemas/campaigns/campaign_pacca_dvfs.yaml \\
        --campaign-dir /home/latorresn/hyperion-results/campaigns/pacca_cpu_final_attempt03_20260820 \\
        --derived-dir /home/latorresn/hyperion-results/campaigns/pacca_cpu_final_attempt03_20260820_arc174 \\
        [--run-id run_id_especifico ...]   # opcional, si se omite reprocesa todas

Ninguna corrida ya aceptada bajo el criterio viejo puede volverse
"rechazada" bajo el nuevo (el nuevo criterio es estrictamente más permisivo
por muestra), pero SÍ puede cambiar el conteo de ventanas usables -- por
eso cada corrida se recalcula de punta a punta en vez de solo revisar las
que estaban rechazadas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import catalog as catalog_module
from orchestrator import calibration as calibration_module
from orchestrator import manifest as manifest_module
from orchestrator import node_profile as node_profile_module
from orchestrator import postprocess as postprocess_module
from orchestrator import runner as runner_module
from orchestrator import validation as validation_module

SCHEMA_VERSION = "arc174-v1"


def _script_checksum() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_metadata(run_dir: Path) -> dict | None:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _first_observed_freq_khz(samples_path: Path) -> int | None:
    """``freq_khz_observed`` (WindowContext) solo alimenta el flag
    ``no_freq_reading`` de ``quality_status`` -- nunca se persistió en
    ``metadata.json`` (siempre fue una lectura EN VIVO tomada por
    ``read_observed_frequency_khz()`` justo después de que la corrida
    terminaba, campaign.py). Para el reprocesamiento offline, cualquier
    lectura real ya presente en ``samples.csv`` sirve para el mismo
    propósito -- confirmar que el colector sí leyó frecuencia durante la
    corrida -- sin inventar un valor. None si ninguna fila CPU tiene una
    lectura utilizable."""
    import csv as _csv
    with samples_path.open(newline="", encoding="utf-8") as handle:
        for row in _csv.DictReader(handle):
            if row.get("tag") != "CPU":
                continue
            raw = row.get("scaling_cur_freq_khz")
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def reprocess_run(run_dir: Path, *, manifest, catalog, derived_root: Path) -> dict:
    """Reprocesa UNA corrida ya ejecutada. Nunca lanza -- cualquier corrida
    que no se pueda reprocesar (kernel GPU, metadata/samples ausentes,
    freq_level_id desconocido) se reporta como saltada, no como error."""
    run_id = run_dir.name
    metadata = _load_metadata(run_dir)
    if metadata is None:
        return {"run_id": run_id, "skipped": True, "reason": "sin metadata.json legible"}

    kernel_ref = metadata.get("kernel_ref")
    entry = catalog.get(kernel_ref) if kernel_ref else None
    if entry is None:
        return {"run_id": run_id, "skipped": True, "reason": f"kernel_ref={kernel_ref!r} no está en el catálogo"}
    if getattr(entry, "device", "cpu") == "gpu":
        return {"run_id": run_id, "skipped": True, "reason": "device=gpu, fuera de alcance (ARC-174)"}

    samples_path = run_dir / "samples.csv"
    if not samples_path.exists():
        return {"run_id": run_id, "skipped": True, "reason": "sin samples.csv"}

    freq_level_id = metadata.get("freq_level_id")
    if not freq_level_id:
        return {"run_id": run_id, "skipped": True, "reason": "metadata.json sin freq_level_id"}
    try:
        frequency_level = runner_module._resolve_frequency_level(manifest, freq_level_id)
    except ValueError as exc:
        return {"run_id": run_id, "skipped": True, "reason": str(exc)}

    frequency_validation = getattr(manifest, "frequency_validation", None) or {}
    tolerance_fraction = frequency_validation.get("tolerance_fraction")
    grace_seconds = float(frequency_validation.get("grace_seconds", 0.0))
    tail_grace_seconds = float(frequency_validation.get("tail_grace_seconds", 0.0))
    expected_cpu_count = len(manifest.cores.delegated_cpus)
    freq_khz_applied = metadata.get("freq_khz_applied")
    is_native_governor = frequency_level.mode == "native_governor"

    # ARC-174: mismos parámetros que el manifiesto original -- el chequeo
    # estructural (traza vacía/incompleta/incoherente) es idéntico al de
    # producción, solo que ahora (bajo el código ya corregido) el
    # resultado de TOLERANCIA nunca rechaza la corrida completa, ver
    # validation.validate_cpu_frequency_trace().
    structural_verdict, structural_summary = validation_module.validate_cpu_frequency_trace(
        samples_path,
        require_per_window=True,
        expected_khz=(freq_khz_applied if not is_native_governor else None),
        tolerance_fraction=tolerance_fraction,
        expected_cpu_count=expected_cpu_count,
        grace_seconds=grace_seconds,
        tail_grace_seconds=tail_grace_seconds,
    )

    derived_dir = derived_root / run_id
    derived_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "run_id": run_id, "skipped": False, "kernel_ref": kernel_ref, "freq_level_id": freq_level_id,
        "structural_valid": structural_summary.get("structural_valid"),
    }

    if not structural_verdict.accepted:
        verdict = structural_verdict
    else:
        roofline = calibration_module.load_calibration(manifest.output_dir, freq_level_id)
        profile = node_profile_module.load_node_profile(manifest.output_dir)
        context = postprocess_module.WindowContext(
            run_id=run_id, repetition=metadata.get("repetition", 1), kernel_ref=kernel_ref,
            node_id=metadata.get("node_id", ""), phase_label_hint=getattr(entry, "phase_label_hint", None),
            freq_level_id=freq_level_id,
            freq_khz_requested=metadata.get("freq_khz_requested"),
            freq_khz_applied=freq_khz_applied,
            freq_khz_observed=metadata.get("freq_khz_observed") or _first_observed_freq_khz(samples_path),
            binary_checksum=entry.binary_checksum,
            roofline_calibration_ref=str(
                Path(manifest.output_dir) / calibration_module.calibration_filename(freq_level_id)
            ),
            node_profile_ref=str(Path(manifest.output_dir) / "node_profile.json"),
            calibration_ref=str(Path(manifest.output_dir) / "calibration_references.json"),
            i_ridge_flops_per_byte=roofline.i_ridge_flops_per_byte,
            llc_line_size_bytes=profile.cache_line_size_bytes,
            warmup_seconds=entry.warmup_seconds or 0.0,
            running_ratio_min=manifest.running_ratio_min,
            rapl_enabled=bool(manifest.rapl.get("enabled", False)),
            freq_tolerance_fraction=tolerance_fraction,
            freq_expected_cpu_count=expected_cpu_count,
            freq_grace_seconds=grace_seconds,
            freq_tail_grace_seconds=tail_grace_seconds,
            freq_is_native_governor=is_native_governor,
        )
        windows = postprocess_module.build_windows(samples_path, context)
        windows_path = postprocess_module.write_windows_csv(windows, derived_dir / "windows.csv")
        verdict = validation_module.validate_windows(
            windows_path, target_windows_per_repetition=manifest.target_windows_per_repetition, device="cpu",
        )
        freq_summary = validation_module.summarize_frequency_quality(windows_path)
        with (derived_dir / "frequency_quality_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(freq_summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        result["frequency_quality_summary"] = freq_summary

    validation_module.write_verdict(verdict, derived_dir)

    provenance = {
        "run_id": run_id,
        "original_run_dir": str(run_dir),
        "schema_version": SCHEMA_VERSION,
        "transformer_script": Path(__file__).name,
        "transformer_checksum_sha256": _script_checksum(),
        "structural_summary": structural_summary,
    }
    with (derived_dir / "reprocess_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")

    result["accepted"] = verdict.accepted
    result["factor_id"] = verdict.factor_id
    return result


def _reprocess_run_worker(args: tuple) -> dict:
    # ProcessPoolExecutor.map() solo acepta una función de un argumento por
    # iterable -- reprocess_run() se queda con kwargs porque también se usa
    # directamente (smoke test, cli); este wrapper es solo el adaptador de
    # picklability para el pool de procesos.
    run_dir, manifest, catalog, derived_root = args
    return reprocess_run(run_dir, manifest=manifest, catalog=catalog, derived_root=derived_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--derived-dir", required=True)
    parser.add_argument("--run-id", action="append", default=None, help="reprocesar solo estos run_ids (repetible)")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="corridas independientes en paralelo (multiprocessing) -- cada una lee/escribe su propio "
             "run_dir/derived_dir, sin estado compartido. Default 1 (secuencial, como antes).",
    )
    args = parser.parse_args(argv)

    manifest = manifest_module.load(args.manifest)
    catalog = catalog_module.load_catalog(str(manifest.catalog_path))
    campaign_dir = Path(args.campaign_dir)
    derived_root = Path(args.derived_dir)

    if args.run_id:
        run_dirs = [campaign_dir / run_id for run_id in args.run_id]
    else:
        run_dirs = sorted(
            p for p in campaign_dir.iterdir()
            if p.is_dir() and (p / "metadata.json").exists()
        )

    if args.workers > 1:
        import concurrent.futures
        tasks = [(run_dir, manifest, catalog, derived_root) for run_dir in run_dirs]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_reprocess_run_worker, tasks))
    else:
        results = [reprocess_run(run_dir, manifest=manifest, catalog=catalog, derived_root=derived_root) for run_dir in run_dirs]

    processed = [r for r in results if not r["skipped"]]
    accepted = [r for r in processed if r["accepted"]]
    skipped = [r for r in results if r["skipped"]]

    summary = {
        "total_run_dirs": len(results),
        "processed": len(processed),
        "accepted": len(accepted),
        "rejected": len(processed) - len(accepted),
        "skipped": len(skipped),
        "skipped_reasons": sorted({r["reason"] for r in skipped}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    derived_root.mkdir(parents=True, exist_ok=True)
    with (derived_root / "reprocess_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
