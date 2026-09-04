"""F1-XDEV-004 -- análisis de correlación/VIF y contrato final de features.

El plan (§2.5) exige Pearson, Spearman y VIF sobre las columnas candidatas
del dataset real ANTES de fijar el conjunto de features, y documentar qué se
descarta y por qué. Hoy el entrenador CPU lleva a la vez `mpki` y
`cache_miss_rate`, que miden lo mismo por dos caminos.

Este módulo:

- **no entrena nada**;
- consume el CSV intermedio final por dispositivo (`training_cpu_intervals.csv`
  para CPU; el `training_gpu_phases.csv` de F1-GPU-003 para GPU);
- opera SOLO sobre filas elegibles (calidad ok, etiqueta válida);
- calcula Pearson y Spearman, reporta pares con `|rho| > umbral`;
- calcula VIF tras un primer filtrado de pares muy correlados;
- trata ausencias, constantes, infinitos y escalas de forma explícita;
- recomienda descartes priorizando la medición física más directa;
- **nunca** propone una columna de verdad Roofline como feature (fuga);
- produce CSV + JSON;
- permite CONGELAR un contrato versionado de features por dispositivo, como
  artefacto revisable, sin modificar `train_phase.py` automáticamente.

La implementación está probada con fixtures; la selección definitiva queda
pendiente de ejecutarla sobre el dataset real de la mini campaña.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Verdad Roofline / insumos de la etiqueta: PROHIBIDOS como features. Se
# mantiene alineado con fase2_clasificador/training/train_phase.py::FORBIDDEN
# y con §2.5 del plan (fuga de información).
ROOFLINE_TRUTH_COLUMNS = frozenset({
    "operational_intensity", "operational_intensity_uncore_real",
    "i_ridge_used", "i_ridge_gpu", "gpu_i_ridge_flops_per_byte",
    "flops_measured_window", "flops_measured_interval",
    "bytes_moved_window", "bytes_moved_uncore_real",
    "uncore_cas_count_read_interval", "uncore_cas_count_write_interval",
    "phase_label_uncore_real", "phase_label_hint", "phase_label_train",
    # métricas GPU offline que sólo existen para construir la verdad:
    "gpu_operational_intensity",
})

# Prioridad de conservación cuando dos features están muy correladas: número
# menor = medición más directa/barata, se conserva frente a un proxy derivado.
# (§2.5: "priorizando la medición física más directa sobre el proxy derivado".)
_DIRECTNESS_RANK = {
    # CPU
    "ipc": 0, "stall_mem_ratio": 0, "power_w": 0, "freq_khz_observed": 0,
    "running_ratio": 0, "ips": 1,
    "cache_miss_rate": 1, "mpki": 2,            # ambos derivan de cache_misses; mpki es el más indirecto
    "ipc_relative": 3, "mpki_relative": 3, "cache_miss_rate_relative": 3,
    # GPU (features NVML online, F1-GPU-003)
    "gpu_util_pct": 0, "gpu_mem_util_pct": 0, "gpu_power_mw": 0,
    "gpu_sm_clock_mhz": 0, "gpu_temperature_c": 0,
    "gpu_util_pct_median": 0, "gpu_mem_util_pct_median": 0, "gpu_power_mw_median": 0,
    "gpu_sm_clock_mhz_median": 0, "gpu_temperature_c_median": 0,
    "gpu_util_pct_iqr": 1, "gpu_power_mw_iqr": 1,
}


@dataclass
class ColumnDiagnosis:
    name: str
    n_total: int
    n_missing: int
    n_inf: int
    is_constant: bool
    std: float
    mean: float
    eligible_as_feature: bool
    reason_excluded: str | None = None


@dataclass
class FeatureContractReport:
    device: str
    schema_version: int = 1
    generated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    n_rows_total: int = 0
    n_rows_eligible: int = 0
    candidate_columns: list[str] = field(default_factory=list)
    column_diagnosis: list[ColumnDiagnosis] = field(default_factory=list)
    high_corr_pairs: list[dict] = field(default_factory=list)
    vif: dict[str, float] = field(default_factory=dict)
    recommended_drops: list[dict] = field(default_factory=list)
    recommended_feature_set: list[str] = field(default_factory=list)
    roofline_truth_columns_seen: list[str] = field(default_factory=list)
    frozen: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "schema": "f1-xdev-004/feature_contract/1",
            "device": self.device,
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "n_rows_total": self.n_rows_total,
            "n_rows_eligible": self.n_rows_eligible,
            "candidate_columns": self.candidate_columns,
            "column_diagnosis": [vars(c) for c in self.column_diagnosis],
            "high_corr_pairs": self.high_corr_pairs,
            "vif": self.vif,
            "recommended_drops": self.recommended_drops,
            "recommended_feature_set": self.recommended_feature_set,
            "roofline_truth_columns_seen": self.roofline_truth_columns_seen,
            "frozen": self.frozen,
            "notes": self.notes,
        }
        return d


# ------------------------------------------------------------------ helpers

def _eligible_rows(df: pd.DataFrame, device: str) -> pd.DataFrame:
    """Filas de calidad ok con etiqueta válida. CPU y GPU tienen columnas de
    calidad distintas (fuente de verdad distinta): se mantienen separadas."""
    out = df
    if "phase_label_train" in out.columns:
        out = out[out["phase_label_train"].notna() & (out["phase_label_train"] != "")]
    if device == "cpu" and "training_quality_status" in out.columns:
        out = out[out["training_quality_status"] == "ok"]
    if device == "gpu":
        # F1-GPU-003: el dataset de fases GPU marca su propia elegibilidad.
        if "phase_quality_status" in out.columns:
            out = out[out["phase_quality_status"] == "ok"]
        elif "quality_status" in out.columns:
            out = out[out["quality_status"].isin(["ok", "gpu_telemetry"])]
    return out.copy()


def _numeric_candidates(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in ROOFLINE_TRUTH_COLUMNS:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            cols.append(c)
    return cols


def diagnose_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[ColumnDiagnosis]:
    out: list[ColumnDiagnosis] = []
    n = len(df)
    for c in columns:
        s = pd.to_numeric(df[c], errors="coerce")
        arr = s.to_numpy(dtype="float64", na_value=np.nan)
        n_missing = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        finite = s.replace([np.inf, -np.inf], np.nan).dropna()
        std = float(finite.std(ddof=0)) if len(finite) else 0.0
        mean = float(finite.mean()) if len(finite) else float("nan")
        is_const = len(finite) > 0 and std == 0.0
        eligible = True
        reason = None
        if c in ROOFLINE_TRUTH_COLUMNS:
            eligible, reason = False, "roofline_truth_column"
        elif len(finite) == 0:
            eligible, reason = False, "all_missing_or_non_numeric"
        elif n_missing > 0.5 * n:
            eligible, reason = False, "over_half_missing"
        elif is_const:
            eligible, reason = False, "constant"
        out.append(ColumnDiagnosis(
            name=c, n_total=n, n_missing=n_missing, n_inf=n_inf,
            is_constant=is_const, std=std, mean=mean,
            eligible_as_feature=eligible, reason_excluded=reason,
        ))
    return out


def correlation_pairs(df: pd.DataFrame, columns: list[str], threshold: float = 0.85) -> list[dict]:
    """Pares con |Pearson| o |Spearman| > threshold, sobre filas finitas comunes."""
    sub = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    pear = sub.corr(method="pearson")
    spear = sub.corr(method="spearman")
    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            rp = pear.loc[a, b]
            rs = spear.loc[a, b]
            rp = None if pd.isna(rp) else float(rp)
            rs = None if pd.isna(rs) else float(rs)
            hit = (rp is not None and abs(rp) > threshold) or (rs is not None and abs(rs) > threshold)
            if hit:
                pairs.append({
                    "a": a, "b": b, "pearson": rp, "spearman": rs,
                    "keep": _prefer(a, b), "drop": b if _prefer(a, b) == a else a,
                })
    return pairs


def _prefer(a: str, b: str) -> str:
    ra = _DIRECTNESS_RANK.get(a, 1)
    rb = _DIRECTNESS_RANK.get(b, 1)
    if ra != rb:
        return a if ra < rb else b
    return sorted((a, b))[0]  # determinista


def compute_vif(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """VIF_j = 1 / (1 - R^2_j) de regresar la columna j sobre el resto.
    Estandariza; descarta filas no finitas; columnas constantes -> inf."""
    sub = df[columns].apply(pd.to_numeric, errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < len(columns) + 2 or len(columns) < 2:
        return {c: float("nan") for c in columns}
    X = sub.to_numpy(dtype="float64")
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    out: dict[str, float] = {}
    for j, c in enumerate(columns):
        if std[j] == 0.0:
            out[c] = float("inf")
            continue
        Xs = (X - mean) / np.where(std == 0.0, 1.0, std)
        y = Xs[:, j]
        others = np.delete(Xs, j, axis=1)
        A = np.column_stack([np.ones(len(others)), others])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[c] = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return out


def analyse(
    df: pd.DataFrame,
    device: str,
    *,
    corr_threshold: float = 0.85,
    vif_threshold: float = 10.0,
) -> FeatureContractReport:
    if device not in ("cpu", "gpu"):
        raise ValueError("device debe ser 'cpu' o 'gpu' -- se mantienen separados")
    rep = FeatureContractReport(device=device)
    rep.n_rows_total = int(len(df))
    rep.roofline_truth_columns_seen = sorted(set(df.columns) & ROOFLINE_TRUTH_COLUMNS)

    elig = _eligible_rows(df, device)
    rep.n_rows_eligible = int(len(elig))
    if rep.n_rows_eligible == 0:
        rep.notes.append("cero filas elegibles: no se puede fijar contrato de features")
        return rep

    candidates = _numeric_candidates(elig)
    rep.candidate_columns = candidates
    rep.column_diagnosis = diagnose_columns(elig, candidates)
    usable = [c.name for c in rep.column_diagnosis if c.eligible_as_feature]

    rep.high_corr_pairs = correlation_pairs(elig, usable, corr_threshold)

    # Primer filtrado: quitar el lado 'drop' de cada par muy correlado antes de VIF.
    dropped_by_corr = {p["drop"] for p in rep.high_corr_pairs}
    after_corr = [c for c in usable if c not in dropped_by_corr]
    rep.vif = compute_vif(elig, after_corr)

    for c in dropped_by_corr:
        keep = next((p["keep"] for p in rep.high_corr_pairs if p["drop"] == c), None)
        rep.recommended_drops.append({
            "column": c, "reason": "high_corr", "prefer_instead": keep,
        })
    for c, v in rep.vif.items():
        if v != v:  # nan
            continue
        if v > vif_threshold or v == float("inf"):
            rep.recommended_drops.append({
                "column": c, "reason": "vif_over_threshold", "vif": None if v == float("inf") else v,
            })

    dropped_all = {d["column"] for d in rep.recommended_drops}
    rep.recommended_feature_set = [c for c in after_corr if c not in dropped_all]
    rep.notes.append(
        "recommended_feature_set es una PROPUESTA: la selección final se fija "
        "revisando este reporte sobre el dataset real y congelando con freeze()."
    )
    if not rep.recommended_feature_set:
        rep.notes.append("advertencia: el filtrado dejó 0 features -- revisar umbrales")
    leak = set(rep.recommended_feature_set) & ROOFLINE_TRUTH_COLUMNS
    assert not leak, f"fuga: {leak} en recommended_feature_set"
    return rep


def freeze_contract(rep: FeatureContractReport, features: list[str], out_path: Path) -> dict:
    """Congela un contrato de features revisado a mano. Falla si incluye una
    columna de verdad Roofline o una que el reporte marcó como no elegible."""
    leak = set(features) & ROOFLINE_TRUTH_COLUMNS
    if leak:
        raise ValueError(f"contrato con fuga de verdad Roofline: {sorted(leak)}")
    not_eligible = {
        c.name for c in rep.column_diagnosis
        if not c.eligible_as_feature and c.name in features
    }
    if not_eligible:
        raise ValueError(f"contrato incluye columnas no elegibles: {sorted(not_eligible)}")
    contract = {
        "schema": "f1-xdev-004/frozen_feature_contract/1",
        "device": rep.device,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": list(features),
        "source_report_generated_at": rep.generated_at_utc,
        "n_rows_eligible_when_frozen": rep.n_rows_eligible,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False))
    return contract


def write_report(rep: FeatureContractReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"feature_contract_{rep.device}.json"
    json_path.write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    csv_path = out_dir / f"feature_contract_{rep.device}_pairs.csv"
    pd.DataFrame(rep.high_corr_pairs or [{"a": "", "b": "", "pearson": "", "spearman": "", "keep": "", "drop": ""}]).to_csv(csv_path, index=False)
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="CSV intermedio final del dispositivo.")
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    parser.add_argument("--vif-threshold", type=float, default=10.0)
    parser.add_argument("--freeze", nargs="+", default=None,
                        help="Lista de features a congelar como contrato (revisada a mano).")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv, low_memory=False)
    rep = analyse(df, args.device, corr_threshold=args.corr_threshold,
                  vif_threshold=args.vif_threshold)
    json_path, csv_path = write_report(rep, args.out_dir)
    print(f"filas totales={rep.n_rows_total}  elegibles={rep.n_rows_eligible}")
    print(f"candidatas={len(rep.candidate_columns)}  pares |rho|>{args.corr_threshold}={len(rep.high_corr_pairs)}")
    for p in rep.high_corr_pairs:
        print(f"  {p['a']} ~ {p['b']}  pearson={p['pearson']}  spearman={p['spearman']}  -> conservar {p['keep']}")
    print("VIF:", {k: round(v, 2) if v == v and v != float('inf') else v for k, v in rep.vif.items()})
    print("feature set propuesto:", rep.recommended_feature_set)
    print(f"reporte: {json_path}  |  {csv_path}")

    if args.freeze:
        contract = freeze_contract(rep, args.freeze, args.out_dir / f"frozen_feature_contract_{args.device}.json")
        print(f"contrato congelado: {contract['features']}")
    else:
        print("(sin --freeze: la selección definitiva queda pendiente del dataset real)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
