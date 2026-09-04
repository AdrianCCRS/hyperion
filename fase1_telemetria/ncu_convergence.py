"""F1-GPU-004 -- convergencia y procedencia de la verdad Roofline GPU (`ncu`).

Los kernels GPU históricos tuvieron análisis de convergencia (p. ej.
`rodinia_lud`); los candidatos nuevos NO lo heredan. Antes de aceptar una
etiqueta Roofline para un kernel GPU hace falta:

- perfilarlo con cantidades crecientes de trabajo (launches / tamaño);
- registrar cantidad solicitada vs. realmente observada;
- calcular la intensidad operacional con FLOPs y bytes DRAM coherentes con su
  precisión;
- detectar FP32 / FP64 / mezcla / ausencia de FLOPs útiles;
- aplicar un criterio de convergencia DECLARADO ANTES: cambio relativo de la
  OI < 1% entre dos puntos consecutivos;
- conservar la salida cruda de `ncu`, los comandos, versiones (`ncu`, driver,
  CUDA), checksum, argumentos y precisión;
- NO permitir aceptar una etiqueta Roofline sin evidencia convergente;
- marcar kernels enteros / sin FLOPs útiles como NO aptos para esta verdad,
  en vez de asignarles una etiqueta ficticia.

`ncu` no está disponible en el entorno local: la ejecución real queda para
paccaA100. El parser y la lógica de convergencia SÍ están aquí y probados.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

REL_TOL_DEFAULT = 0.01           # 1% -- criterio declarado antes del resultado
OBSERVED_MATCH_TOL = 0.02        # launches observados vs. solicitados

# Métricas de `ncu` que buscamos (por subcadena, tolerante a versión).
# FLOPs: instrucciones de punto flotante ejecutadas por operación.
_FP32_KEYS = (
    "sass_thread_inst_executed_op_fadd_pred_on",
    "sass_thread_inst_executed_op_fmul_pred_on",
    "sass_thread_inst_executed_op_ffma_pred_on",
)
_FP64_KEYS = (
    "sass_thread_inst_executed_op_dadd_pred_on",
    "sass_thread_inst_executed_op_dmul_pred_on",
    "sass_thread_inst_executed_op_dfma_pred_on",
)
_INT_KEYS = (
    "sass_thread_inst_executed_op_iadd_pred_on",
    "sass_thread_inst_executed_op_imad_pred_on",
)
_FMA_KEYS = ("ffma", "dfma")
_DRAM_BYTES_KEYS = ("dram__bytes.sum", "dram__bytes_read.sum", "dram__bytes_write.sum")
_LAUNCH_KEYS = ("launch__", )  # solo para conteo de filas realmente


@dataclass
class NcuPoint:
    launch_count_requested: int
    launch_count_observed: int
    flops: float
    dram_bytes: float
    operational_intensity: float | None
    precision: str
    raw_metric_sums: dict = field(default_factory=dict)


@dataclass
class KernelConvergence:
    kernel_ref: str
    exec_path: str | None = None
    binary_checksum: str | None = None
    kernel_args: list[str] = field(default_factory=list)
    ncu_version: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    precision: str | None = None
    points: list[dict] = field(default_factory=list)
    converged: bool = False
    converged_at_launch_count: int | None = None
    final_operational_intensity: float | None = None
    roofline_label_eligible: bool = False
    status: str = "pending"     # pending | converged | not_converged | not_suitable_for_roofline_truth
    reason: str = ""
    generated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ------------------------------------------------------------------ parser

def _num(s: str) -> float | None:
    s = (s or "").strip().replace(",", "")
    if not s or s.lower() in ("n/a", "nan", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_ncu_csv(text: str) -> dict:
    """Parsea la salida `ncu --csv` (estilo `--metrics`, una fila por launch,
    una columna por métrica). Devuelve sumas por bucket sobre TODAS las filas
    (== todos los launches de esa corrida) + nº de filas (launches observados).

    Tolerante a versiones: mapea por subcadena del nombre de métrica.
    """
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r]
    if not rows:
        return {"launches_observed": 0, "fp32_inst": 0.0, "fp64_inst": 0.0,
                "int_inst": 0.0, "fma_inst": 0.0, "dram_bytes": 0.0, "by_metric": {}}
    header = [h.strip().strip('"') for h in rows[0]]
    data = rows[1:]

    def col_idx(pred) -> list[int]:
        return [i for i, h in enumerate(header) if pred(h.lower())]

    fp32_cols = col_idx(lambda h: any(k in h for k in _FP32_KEYS))
    fp64_cols = col_idx(lambda h: any(k in h for k in _FP64_KEYS))
    int_cols = col_idx(lambda h: any(k in h for k in _INT_KEYS))
    fma_cols = col_idx(lambda h: any(k in h for k in _FMA_KEYS))
    dram_cols = col_idx(lambda h: h in _DRAM_BYTES_KEYS or ("dram__bytes" in h and h.endswith(".sum")))
    # nombre de kernel para agrupar, si está
    kname_cols = col_idx(lambda h: h in ("kernel name", "\"kernel name\"", "demangled name"))

    sums = {"fp32_inst": 0.0, "fp64_inst": 0.0, "int_inst": 0.0, "fma_inst": 0.0,
            "dram_bytes": 0.0}
    by_metric: dict[str, float] = {}
    launches = 0
    for r in data:
        if all(not c.strip() for c in r):
            continue
        launches += 1
        for name, cols, key in (
            ("fp32", fp32_cols, "fp32_inst"), ("fp64", fp64_cols, "fp64_inst"),
            ("int", int_cols, "int_inst"), ("fma", fma_cols, "fma_inst"),
            ("dram", dram_cols, "dram_bytes"),
        ):
            for ci in cols:
                v = _num(r[ci]) if ci < len(r) else None
                if v is not None:
                    sums[key] += v
                    by_metric[header[ci]] = by_metric.get(header[ci], 0.0) + v
    return {"launches_observed": launches, "by_metric": by_metric, **sums,
            "has_kernel_name_col": bool(kname_cols)}


def flops_and_precision(parsed: dict) -> tuple[float, str]:
    """FLOPs = fadd + fmul + 2*ffma  (+ las dobles), es decir cada FMA cuenta
    como 2 FLOPs (mul + add). El parser ya suma `fp32_inst` = fadd+fmul+ffma,
    `fp64_inst` = dadd+dmul+dfma, y `fma_inst` = ffma+dfma; sumar `fma_inst`
    una vez más hace que las FMA cuenten doble.

    Precisión: fp64 si domina la doble; fp32 si domina la simple; mixed si
    ambas relevantes; integer_no_flops si solo hay enteros; no_flops si nada.
    """
    fp32 = parsed.get("fp32_inst", 0.0)
    fp64 = parsed.get("fp64_inst", 0.0)
    ints = parsed.get("int_inst", 0.0)
    fma = parsed.get("fma_inst", 0.0)
    total_fp = fp32 + fp64
    if total_fp <= 0:
        return 0.0, ("integer_no_flops" if ints > 0 else "no_flops")
    flops = fp32 + fp64 + fma
    frac64 = fp64 / total_fp
    if frac64 >= 0.9:
        prec = "fp64"
    elif frac64 <= 0.1:
        prec = "fp32"
    else:
        prec = "mixed"
    return float(flops), prec


def operational_intensity(flops: float, dram_bytes: float) -> float | None:
    if dram_bytes and dram_bytes > 0:
        return flops / dram_bytes
    return None


# ------------------------------------------------------------------ convergence

def assess_convergence(points: list[NcuPoint], *, rel_tol: float = REL_TOL_DEFAULT
                       ) -> dict:
    """Criterio declarado ANTES del resultado: convergió si los dos puntos con
    más trabajo tienen |OI_i - OI_{i-1}| / OI_{i-1} < rel_tol, y en ambos los
    launches observados ~= solicitados."""
    valid = [p for p in points if p.operational_intensity is not None]
    if len(valid) < 2:
        return {"converged": False, "reason": "menos de 2 puntos con OI definida"}
    ordered = sorted(valid, key=lambda p: p.launch_count_requested)
    a, b = ordered[-2], ordered[-1]
    for p in (a, b):
        req = max(p.launch_count_requested, 1)
        if abs(p.launch_count_observed - p.launch_count_requested) / req > OBSERVED_MATCH_TOL:
            return {"converged": False,
                    "reason": f"launches observados ({p.launch_count_observed}) != "
                              f"solicitados ({p.launch_count_requested})"}
    rel = abs(b.operational_intensity - a.operational_intensity) / a.operational_intensity
    if rel < rel_tol:
        return {"converged": True, "converged_at_launch_count": b.launch_count_requested,
                "final_operational_intensity": b.operational_intensity,
                "relative_change": rel}
    return {"converged": False, "reason": f"cambio relativo de OI {rel:.4f} >= {rel_tol}",
            "relative_change": rel}


def build_kernel_report(kernel_ref: str, points: list[NcuPoint], *,
                        exec_path: str | None = None, binary_checksum: str | None = None,
                        kernel_args: list[str] | None = None,
                        ncu_version: str | None = None, driver_version: str | None = None,
                        cuda_version: str | None = None,
                        rel_tol: float = REL_TOL_DEFAULT) -> KernelConvergence:
    rep = KernelConvergence(
        kernel_ref=kernel_ref, exec_path=exec_path, binary_checksum=binary_checksum,
        kernel_args=list(kernel_args or []), ncu_version=ncu_version,
        driver_version=driver_version, cuda_version=cuda_version,
        points=[asdict(p) for p in points],
    )
    precisions = {p.precision for p in points if p.precision not in (None, "")}
    rep.precision = ("mixed" if len(precisions) > 1
                     else (next(iter(precisions)) if precisions else None))

    if rep.precision in ("integer_no_flops", "no_flops") or precisions <= {"integer_no_flops", "no_flops"}:
        rep.status = "not_suitable_for_roofline_truth"
        rep.reason = ("sin FLOPs de punto flotante útiles: FLOPs/byte no describe "
                      "trabajo -- NO se asigna etiqueta Roofline")
        rep.roofline_label_eligible = False
        return rep

    conv = assess_convergence(points, rel_tol=rel_tol)
    rep.converged = bool(conv.get("converged"))
    rep.converged_at_launch_count = conv.get("converged_at_launch_count")
    rep.final_operational_intensity = conv.get("final_operational_intensity")
    if rep.converged:
        rep.status = "converged"
        rep.roofline_label_eligible = rep.precision in ("fp32", "fp64", "mixed")
        rep.reason = f"convergió (cambio relativo {conv.get('relative_change', 0):.4f})"
    else:
        rep.status = "not_converged"
        rep.roofline_label_eligible = False
        rep.reason = conv.get("reason", "no convergió")
    return rep


# ------------------------------------------------------------------ runner

_METRICS_ARG = ",".join([
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_iadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_imad_pred_on.sum",
    "dram__bytes.sum",
])


def _ncu_versions(ncu: str) -> dict:
    out = {}
    for key, cmd in (("ncu_version", [ncu, "--version"]),
                     ("driver_version", ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
                     ("cuda_version", ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            out[key] = (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else None
        except (OSError, subprocess.SubprocessError, IndexError):
            out[key] = None
    return out


def runbook(kernel_ref: str, exec_tmpl: str, launch_counts: list[int], out_dir: Path) -> Path:
    """Sin `ncu` local: deja el procedimiento exacto para paccaA100."""
    lines = [
        f"# F1-GPU-004 -- runbook de convergencia ncu para {kernel_ref}",
        "# Ejecutar en paccaA100 (ncu disponible), dentro de la asignación Slurm.",
        "set -euo pipefail",
        'OUT="$PWD/ncu_convergence"; mkdir -p "$OUT"',
        "",
    ]
    for lc in launch_counts:
        cmd = exec_tmpl.replace("{launches}", str(lc)).replace("{N}", str(lc))
        lines += [
            f'echo "=== launches={lc} ==="',
            f'ncu --csv --metrics {_METRICS_ARG} --target-processes all \\',
            f'    {cmd} > "$OUT/{kernel_ref}__lc{lc}.csv" 2> "$OUT/{kernel_ref}__lc{lc}.log"',
        ]
    lines += [
        "",
        "# Luego, en cualquier máquina:",
        f"python3 -m fase1_telemetria.ncu_convergence --kernel {kernel_ref} \\",
        f"    --from-csv \"$OUT/{kernel_ref}__lc*.csv\" --out-dir \"$OUT\"",
    ]
    path = out_dir / f"{kernel_ref}__ncu_runbook.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _points_from_csv_glob(pattern: str) -> list[NcuPoint]:
    import glob
    pts: list[NcuPoint] = []
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"lc(\d+)", Path(f).name)
        requested = int(m.group(1)) if m else len(pts) + 1
        parsed = parse_ncu_csv(Path(f).read_text())
        flops, prec = flops_and_precision(parsed)
        oi = operational_intensity(flops, parsed.get("dram_bytes", 0.0))
        pts.append(NcuPoint(
            launch_count_requested=requested,
            launch_count_observed=parsed.get("launches_observed", 0),
            flops=flops, dram_bytes=parsed.get("dram_bytes", 0.0),
            operational_intensity=oi, precision=prec,
            raw_metric_sums=parsed.get("by_metric", {}),
        ))
    return pts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", required=True)
    p.add_argument("--exec-template", default=None,
                   help="Comando del kernel con {launches}/{N} como placeholder del trabajo.")
    p.add_argument("--exec-path", default=None)
    p.add_argument("--binary-checksum", default=None)
    p.add_argument("--launch-counts", default="10,50,100,500")
    p.add_argument("--from-csv", default=None, help="glob de CSVs `ncu` ya generados (offline).")
    p.add_argument("--ncu", default="ncu")
    p.add_argument("--rel-tol", type=float, default=REL_TOL_DEFAULT)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    launch_counts = [int(x) for x in a.launch_counts.split(",")]

    if a.from_csv:
        points = _points_from_csv_glob(a.from_csv)
        versions = {}
    elif shutil.which(a.ncu):
        versions = _ncu_versions(a.ncu)
        points = []
        for lc in launch_counts:
            if not a.exec_template:
                raise SystemExit("--exec-template es obligatorio para invocar ncu")
            cmd = a.exec_template.replace("{launches}", str(lc)).replace("{N}", str(lc))
            csv_path = a.out_dir / f"{a.kernel}__lc{lc}.csv"
            r = subprocess.run(
                [a.ncu, "--csv", "--metrics", _METRICS_ARG, "--target-processes", "all",
                 *cmd.split()],
                capture_output=True, text=True, timeout=1200, check=False,
            )
            csv_path.write_text(r.stdout)
            (a.out_dir / f"{a.kernel}__lc{lc}.log").write_text(r.stderr)
            parsed = parse_ncu_csv(r.stdout)
            flops, prec = flops_and_precision(parsed)
            points.append(NcuPoint(lc, parsed.get("launches_observed", 0), flops,
                                   parsed.get("dram_bytes", 0.0),
                                   operational_intensity(flops, parsed.get("dram_bytes", 0.0)),
                                   prec, parsed.get("by_metric", {})))
    else:
        rb = runbook(a.kernel, a.exec_template or "<exec> {launches}", launch_counts, a.out_dir)
        print(f"ncu no disponible localmente -- runbook para paccaA100 en {rb}")
        return 3

    rep = build_kernel_report(
        a.kernel, points, exec_path=a.exec_path, binary_checksum=a.binary_checksum,
        kernel_args=(a.exec_template.split() if a.exec_template else []),
        rel_tol=a.rel_tol, **({} if a.from_csv else versions),
    )
    out = a.out_dir / f"{a.kernel}.json"
    out.write_text(json.dumps(asdict(rep), indent=2, ensure_ascii=False))
    print(f"{a.kernel}: status={rep.status}  precision={rep.precision}  "
          f"converged={rep.converged}  roofline_eligible={rep.roofline_label_eligible}")
    print(f"  {rep.reason}")
    print(f"  reporte: {out}")
    return 0 if rep.roofline_label_eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
