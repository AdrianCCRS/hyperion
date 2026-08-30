"""CLI del pipeline de Fase 2."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .dataset import BuildConfig, build_selector_datasets
from .eda import generate_eda
from .search import FAMILIES, evaluate_existing, run_nested_tuning


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    # --cpu-campaign/--gpu-campaign/--cpu-manifest/--gpu-manifest admiten
    # repetirse (action="append") para combinar varios directorios de
    # campana del mismo eje en un solo dataset -- ej. campana base (68
    # config_id) + campana suplementaria "big" (9 config_id):
    #   --cpu-campaign .../pacca_dual_cpu_full_20260828 \
    #   --cpu-campaign .../pacca_dual_cpu_big_20260830 \
    #   --cpu-manifest .../campaign_pacca_dual_cpu_full.yaml \
    #   --cpu-manifest .../campaign_pacca_dual_cpu_big.yaml
    # El conteo esperado de config_id se deriva de la union de kernel_ref de
    # TODOS los --cpu-manifest pasados (ver dataset.expected_config_ids), no
    # de un numero fijo.
    parser.add_argument("--cpu-campaign", required=True, action="append", type=Path)
    parser.add_argument("--gpu-campaign", action="append", type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--cpu-manifest", action="append", type=Path)
    parser.add_argument("--gpu-manifest", action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["cpu-provisional", "final"], default="cpu-provisional")
    parser.add_argument("--idle-gpu-power-w", type=float, default=34.8379)
    parser.add_argument("--idle-gpu-power-source-job", default="6714")
    parser.add_argument("--expected-repetitions", type=int, default=3)


def _build_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return build_selector_datasets(BuildConfig(
        cpu_campaign_dir=args.cpu_campaign,
        gpu_campaign_dir=args.gpu_campaign,
        catalog_path=args.catalog,
        cpu_manifest_path=args.cpu_manifest,
        gpu_manifest_path=args.gpu_manifest,
        output_dir=args.output_dir,
        mode=args.mode,
        idle_gpu_power_w=args.idle_gpu_power_w,
        idle_gpu_power_source_job=args.idle_gpu_power_source_job,
        expected_repetitions=args.expected_repetitions,
    ))


def _families(value: str) -> tuple[str, ...]:
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(requested) - set(FAMILIES)
    if unknown:
        raise argparse.ArgumentTypeError(f"familias desconocidas: {sorted(unknown)}")
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline del selector CPU/GPU Hyperion")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="construye dataset nivel 2")
    _add_build_arguments(build)

    eda = sub.add_parser("eda", help="genera correlaciones y EDA")
    eda.add_argument("--dataset-dir", required=True, type=Path)
    eda.add_argument("--output-dir", type=Path)

    tune = sub.add_parser("tune", help="Optuna anidado + evaluacion externa")
    tune.add_argument("--dataset", required=True, type=Path)
    tune.add_argument("--output-dir", required=True, type=Path)
    tune.add_argument("--families", type=_families, default=FAMILIES)
    tune.add_argument("--trials", type=int, default=100)
    tune.add_argument("--seed", type=int, default=20260828)
    tune.add_argument("--latency-warmups", type=int, default=50)
    tune.add_argument("--latency-repeats", type=int, default=200)

    evaluate = sub.add_parser("evaluate", help="resume una evaluacion ya ejecutada")
    evaluate.add_argument("--output-dir", required=True, type=Path)

    all_parser = sub.add_parser("all", help="build + eda + tune A/C")
    _add_build_arguments(all_parser)
    all_parser.add_argument("--families", type=_families, default=FAMILIES)
    all_parser.add_argument("--trials", type=int, default=100)
    all_parser.add_argument("--seed", type=int, default=20260828)
    all_parser.add_argument("--latency-warmups", type=int, default=50)
    all_parser.add_argument("--latency-repeats", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        paths = _build_from_args(args)
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "eda":
        for name, path in generate_eda(args.dataset_dir, args.output_dir).items():
            print(f"{name}: {path}")
    elif args.command == "tune":
        paths = run_nested_tuning(
            args.dataset, args.output_dir, families=args.families,
            trials=args.trials, seed=args.seed,
            latency_warmups=args.latency_warmups,
            latency_repeats=args.latency_repeats,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "evaluate":
        print(evaluate_existing(args.output_dir))
    elif args.command == "all":
        paths = _build_from_args(args)
        generate_eda(args.output_dir)
        comparisons = []
        for key in ("strategy_a", "strategy_c"):
            path = paths[key]
            if path.exists() and path.stat().st_size > 1:
                model_paths = run_nested_tuning(
                    path, args.output_dir / "models" / key,
                    families=args.families, trials=args.trials, seed=args.seed,
                    latency_warmups=args.latency_warmups,
                    latency_repeats=args.latency_repeats,
                )
                comparison = pd.read_csv(model_paths["comparison"])
                comparison.insert(0, "strategy_dataset", key)
                comparisons.append(comparison)
        if comparisons:
            pd.concat(comparisons, ignore_index=True).to_csv(
                args.output_dir / "model_comparison_all.csv", index=False,
            )
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
