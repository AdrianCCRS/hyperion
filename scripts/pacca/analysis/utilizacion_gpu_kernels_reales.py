"""¿Los kernels REALES del catalogo GPU ejercitan la A100, o pasa lo mismo
que con los sinteticos dual_*?

Corre sobre pacca_gpu_final_dataset_v2_20260825, que contiene 13 de los 23
kernels GPU de la campana final (6 Rodinia, 6 RAJAPerf GPU, dgemm) mas
rodinia_dwt2d en cuatro tamanos extra.

PROBLEMA METODOLOGICO Y COMO SE RESUELVE: esta campana NO tiene el contrato
cold_warm_v1, asi que no se puede acotar la fase caliente como se hizo con
dual_*. Promediar gpu_util_pct sobre toda la corrida mezclaria el setup del
host (utilizacion ~0) y subestimaria. Por eso se usan estadisticos robustos
a esa cola:
  - util_p90      : percentil 90 de la utilizacion, aproxima el nivel
                    sostenido mientras la GPU trabaja de verdad
  - util_activa   : mediana de las muestras con utilizacion > 0
  - frac_util_50  : fraccion de muestras con utilizacion >= 50 %
Ninguno depende de saber donde empieza la fase caliente.

El nivel de GPU se FIJA: gpu_util_pct no es comparable entre relojes (a
menor reloj el mismo trabajo ocupa mas tiempo y la utilizacion aparente
sube; verificado en el analisis de dual_*).
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict
from statistics import median

ROOT = "/home/latorresn/hyperion-results/campaigns/pacca_gpu_final_dataset_v2_20260825"
GPU_FIJO = "F0"   # 1410 MHz, reloj maximo
CALIBRACION = {"ert_probe", "stream_official", "gpu_stream_bw",
               "gpu_ert_probe_fp32", "gpu_ert_probe_fp64"}

NOMBRE = re.compile(r"^.*?__(?P<kref>.+?)__(?P<cpu>REF|F\d+)__gpu(?P<gpu>[^_]+)__rep(?P<rep>\d+)$")


def percentil(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    i = int(round((p / 100.0) * (len(s) - 1)))
    return s[i]


def lee_util(path):
    utils = []
    try:
        with open(path, newline="") as fh:
            for fila in csv.DictReader(fh):
                try:
                    utils.append(float(fila.get("gpu_util_pct") or ""))
                except ValueError:
                    pass
    except OSError:
        return None
    return utils


def main():
    porKernel = defaultdict(list)
    for nombre in sorted(os.listdir(ROOT)):
        if nombre.endswith("__baseline"):
            continue
        ruta = os.path.join(ROOT, nombre)
        if not os.path.isdir(ruta):
            continue
        m = NOMBRE.match(nombre)
        if not m or m.group("gpu") != GPU_FIJO:
            continue
        kref = m.group("kref")
        if kref in CALIBRACION or "_calibration" in kref:
            continue
        utils = lee_util(os.path.join(ruta, "samples.csv"))
        if not utils:
            continue
        activos = [u for u in utils if u > 0]
        porKernel[kref].append({
            "p90": percentil(utils, 90),
            "activa": median(activos) if activos else 0.0,
            "frac50": sum(1 for u in utils if u >= 50) / len(utils),
            "n_muestras": len(utils),
        })

    if not porKernel:
        sys.exit("sin datos")

    print(f"kernels reales del catalogo, nivel GPU fijo = {GPU_FIJO} (1410 MHz)")
    print(f"corridas agregadas: {sum(len(v) for v in porKernel.values())}\n")
    print(f"{'kernel':<32} {'util p90':>9} {'util activa':>12} {'% muestras >=50':>16} {'n':>4}")
    print("-" * 78)

    bajos, altos = [], []
    for kref in sorted(porKernel):
        rs = porKernel[kref]
        p90 = median([r["p90"] for r in rs if r["p90"] is not None])
        act = median([r["activa"] for r in rs])
        f50 = median([r["frac50"] for r in rs]) * 100
        marca = ""
        if p90 < 50:
            marca = "  <-- infrautilizada"
            bajos.append((kref, p90))
        else:
            altos.append((kref, p90))
        print(f"{kref:<32} {p90:>9.1f} {act:>12.1f} {f50:>16.1f} {len(rs):>4}{marca}")

    print()
    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)
    print(f"kernels que SI ejercitan la GPU (p90 >= 50 %): {len(altos)}")
    for k, v in sorted(altos, key=lambda x: -x[1]):
        print(f"   {k:<32} p90={v:.1f} %")
    print(f"kernels infrautilizados (p90 < 50 %): {len(bajos)}")
    for k, v in sorted(bajos, key=lambda x: x[1]):
        print(f"   {k:<32} p90={v:.1f} %")

    # dwt2d por tamano: unica familia REAL con barrido de tamanos disponible
    dwt = {k: v for k, v in porKernel.items() if k.startswith("rodinia_dwt2d")}
    if len(dwt) > 1:
        print()
        print("=" * 78)
        print("BONUS: rodinia_dwt2d por tamano (unico kernel REAL con barrido)")
        print("=" * 78)
        for k in sorted(dwt):
            rs = dwt[k]
            p90 = median([r["p90"] for r in rs if r["p90"] is not None])
            print(f"   {k:<32} p90={p90:>6.1f} %   (n={len(rs)})")


if __name__ == "__main__":
    main()
