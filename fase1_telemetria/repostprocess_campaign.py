#!/usr/bin/env python3
"""F1-XDEV-002 -- re-postprocesa una campaña ya recolectada, sin relanzar
ningún kernel, después de corregir `warmup_seconds` en el catálogo.

Contexto: `warmup_seconds` solo se usa en `postprocess.py` (marca ventanas
como `warmup_excluded`); `runner.py` no lo lee en absoluto, así que
`samples.csv` siempre contiene la traza completa desde el inicio,
independientemente de lo que declare el catálogo o `manifest.warmup_seconds_
override` en el momento de la recolección. Eso permite plegar la calibración
de warmup (F1-XDEV-002) dentro de la campaña real en un solo flujo, sin una
mini-campaña previa:

1. Correr la campaña real con `warmup_seconds_override: 0.0` en el
   manifiesto (ver `common/hpc/manifest.py`) -- nada se excluye al
   postprocesar, se conserva el transitorio completo de cada corrida.
2. Calibrar con `fase1_telemetria/warmup_calibration.py` sobre los
   `windows.csv` ya producidos por esa misma campaña (mismo criterio de
   detección: CV de dos ventanas + segmentación por puntos de cambio,
   máximo entre >=3 repeticiones).
3. Aplicar los valores calibrados al catálogo REAL con
   `warmup_calibration.apply_proposals_to_catalog(..., apply=True)`
   (backup `.bak` + verificación de checksum, sin reemplazo silencioso).
4. Re-postprocesar la MISMA campaña con este módulo -- reutiliza los
   `samples.csv`/`metadata.json` ya escritos en disco, nunca vuelve a
   ejecutar el binario del kernel.

Este módulo IGNORA `manifest.warmup_seconds_override` a propósito (parámetro
`ignore_manifest_override`, default `True`): el paso 4 debe reflejar siempre
el catálogo ya corregido, no repetir el forzado a 0 del paso 1. Si de verdad
se quiere reprocesar con el override, pasar `ignore_manifest_override=False`
explícitamente.

Re-validación del veredicto (accepted/rejected). El accept/reject de cada
corrida se decide, en la campaña en vivo, con el `windows.csv` PROVISIONAL
(warmup=0, nada excluido) -- es el único que existe en ese momento, antes de
calibrar. Una corrida al límite de `target_windows_per_repetition` podría
tener MENOS ventanas usables una vez excluido el warmup real, y seguir
figurando como `accepted` si nadie vuelve a evaluarla. Por eso este módulo, al
reprocesar con éxito, también vuelve a correr
`validation.validate_windows()` sobre el `windows.csv` corregido y sobrescribe
`verdict.json` (`validation.write_verdict()`) -- nunca borra ni mueve la
corrida (VAL-06), solo dice honestamente si sigue aceptada. `verdict_changed`
en el resultado marca los casos donde el veredicto cambió de aceptado a
rechazado (o viceversa) al recalcular con el warmup correcto -- revisar esos
casos a mano antes de dar la campaña por cerrada.

Uso:
    python3 -m fase1_telemetria.repostprocess_campaign \\
        --manifest campaign.yaml --node-id paccaA100 \\
        [--output-dir DIR] [--catalog-path catalog_corregido.yaml] \\
        [--kernel k1 --kernel k2] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.hpc import catalog as catalog_module
from common.hpc import manifest as manifest_module
from . import campaign as campaign_module
from . import postprocess as postprocess_module
from . import runner as runner_module
from . import validation as validation_module


def repostprocess_campaign(
    manifest: Any,
    catalog: Mapping[str, Any],
    *,
    node_id: str,
    output_dir: str | Path | None = None,
    only_kernels: list[str] | None = None,
    ignore_manifest_override: bool = True,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Re-postprocesa cada corrida ya recolectada de una campaña.

    `manifest`/`catalog` ya cargados (facilita las pruebas sin escribir YAML a
    disco); `main()` los carga desde archivo. No relanza ningún kernel: solo
    busca `<output_dir>/<run_id>/samples.csv` (el `run_id` exacto que
    produciría `build_matrix()` + `build_run_id()` para esta combinación,
    nunca una heurística de nombre de directorio) y, si existe, llama a
    `postprocess.run_postprocess()` de nuevo con `catalog` (que se espera ya
    corregido -- este módulo nunca lo modifica).

    Devuelve una lista de resultados por combinación esperada: `reprocessed`,
    `skipped` (sin `samples.csv`, p. ej. una corrida rechazada) o `error`
    (fallo real de `run_postprocess`, nunca se oculta). Una corrida
    `reprocessed` también trae `verdict_accepted` (el veredicto recalculado
    tras corregir el warmup) y `verdict_changed` (si difiere del
    `verdict.json` que ya estaba en disco) -- nunca se borra ni se mueve una
    corrida, solo se sobrescribe `verdict.json` con el resultado honesto.
    """
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(manifest.output_dir)

    if ignore_manifest_override:
        # A propósito: el override de recolección (warmup_seconds_override)
        # NUNCA debe reaplicarse aquí -- este paso existe precisamente para
        # sustituirlo por el valor ya corregido del catálogo.
        effective_manifest = replace(manifest, warmup_seconds_override=None)
    else:
        effective_manifest = manifest

    combinations = campaign_module.build_matrix(manifest, catalog)
    frequency_validation = dict(getattr(manifest, "frequency_validation", None) or {})

    results: list[dict[str, Any]] = []
    for combo in combinations:
        if only_kernels and combo.kernel_ref not in only_kernels:
            continue
        gpu_level_id = (
            combo.gpu_frequency_level.id if combo.gpu_frequency_level is not None else None
        )
        run_id = runner_module.build_run_id(
            manifest.campaign_id, combo.kernel_ref, combo.frequency_level.id,
            combo.repetition_index, gpu_level_id,
        )
        run_dir = resolved_output_dir / run_id
        samples_path = run_dir / "samples.csv"
        if not samples_path.exists():
            results.append({"run_id": run_id, "status": "skipped", "reason": "sin samples.csv"})
            continue
        entry = catalog.get(combo.kernel_ref)
        if entry is None:
            results.append({
                "run_id": run_id, "status": "skipped",
                "reason": f"kernel_ref {combo.kernel_ref!r} no está en el catálogo dado",
            })
            continue

        warmup_seconds = (
            effective_manifest.warmup_seconds_override
            if effective_manifest.warmup_seconds_override is not None
            else (entry.warmup_seconds or 0.0)
        )

        if dry_run:
            results.append({
                "run_id": run_id, "status": "would_reprocess",
                "warmup_seconds_would_use": warmup_seconds,
            })
            continue

        freq_khz_applied = None
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            try:
                freq_khz_applied = json.loads(metadata_path.read_text(encoding="utf-8")).get("freq_khz_applied")
            except (json.JSONDecodeError, OSError):
                freq_khz_applied = None

        # El veredicto que YA está en disco (decidido en vivo, sobre el
        # windows.csv provisional) -- se lee ANTES de sobrescribirlo, para
        # poder reportar si la corrección del warmup lo cambió. Ausente (p.
        # ej. una corrida nunca validada así) no bloquea el reproceso.
        previous_accepted: bool | None = None
        try:
            previous_accepted = validation_module.load_verdict(run_dir).accepted
        except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError):
            previous_accepted = None

        try:
            windows_path = postprocess_module.run_postprocess(
                run_dir, run_id=run_id, repetition=combo.repetition_index,
                kernel_ref=combo.kernel_ref, kernel_entry=entry, node_id=node_id,
                freq_level_id=combo.frequency_level.id,
                calibration_dir=resolved_output_dir,
                freq_khz_applied=freq_khz_applied,
                warmup_seconds=warmup_seconds,
                running_ratio_min=manifest.running_ratio_min,
                rapl_enabled=bool((manifest.rapl or {}).get("enabled", False)),
                gpu_freq_level_id=gpu_level_id,
                freq_tolerance_fraction=frequency_validation.get("tolerance_fraction"),
                freq_expected_cpu_count=len(manifest.cores.delegated_cpus),
                freq_grace_seconds=float(frequency_validation.get("grace_seconds", 0.0)),
                freq_tail_grace_seconds=float(frequency_validation.get("tail_grace_seconds", 0.0)),
                freq_is_native_governor=combo.frequency_level.mode == "native_governor",
            )
            # Re-validación (ver docstring del módulo): el accept/reject debe
            # reflejar el windows.csv YA corregido, no el provisional con el
            # que se decidió en vivo. VAL-06: esto nunca borra ni mueve la
            # corrida, solo sobrescribe verdict.json.
            verdict = validation_module.validate_windows(
                windows_path,
                target_windows_per_repetition=manifest.target_windows_per_repetition,
                device=getattr(entry, "device", "cpu"),
            )
            validation_module.write_verdict(verdict, run_dir)
            results.append({
                "run_id": run_id, "status": "reprocessed",
                "windows_path": str(windows_path), "warmup_seconds_used": warmup_seconds,
                "verdict_accepted": verdict.accepted,
                "verdict_factor_id": verdict.factor_id,
                "verdict_message": verdict.message,
                "verdict_changed": (
                    previous_accepted is not None and previous_accepted != verdict.accepted
                ),
            })
        except Exception as exc:  # nunca oculta un fallo real de una corrida
            results.append({"run_id": run_id, "status": "error", "reason": str(exc)})

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--catalog-path", type=Path, default=None,
                        help="Catálogo a usar para el re-postproceso (default: manifest.catalog_path, "
                             "ya debería estar corregido con los valores calibrados).")
    parser.add_argument("--kernel", action="append", dest="kernels", default=None,
                        help="Restringe a estos kernel_ref (repetible). Default: todos los del manifiesto.")
    parser.add_argument("--use-manifest-warmup-override", action="store_true",
                        help="NO recomendado: reaplica manifest.warmup_seconds_override en vez de "
                             "ignorarlo. Solo para depuración.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo reporta qué se reprocesaría y con qué warmup_seconds, sin escribir nada.")
    args = parser.parse_args(argv)

    manifest = manifest_module.load(args.manifest)
    catalog_path = args.catalog_path if args.catalog_path is not None else manifest.catalog_path
    catalog = catalog_module.load_catalog(str(catalog_path))

    results = repostprocess_campaign(
        manifest, catalog, node_id=args.node_id, output_dir=args.output_dir,
        only_kernels=args.kernels,
        ignore_manifest_override=not args.use_manifest_warmup_override,
        dry_run=args.dry_run,
    )
    by_status: dict[str, int] = {}
    changed_verdicts = []
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        if r["status"] == "error":
            print(f"ERROR  {r['run_id']}: {r['reason']}")
        if r.get("verdict_changed"):
            changed_verdicts.append(r)
            print(f"VEREDICTO CAMBIÓ  {r['run_id']}: ahora accepted={r['verdict_accepted']} "
                  f"({r['verdict_factor_id']}: {r['verdict_message']})")
    print(f"\nresumen: {by_status}")
    if changed_verdicts:
        print(f"{len(changed_verdicts)} corrida(s) cambiaron de veredicto al corregir el warmup "
              "-- revisar a mano antes de dar la campaña por cerrada.")
    return 1 if by_status.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
