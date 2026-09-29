"""Tamaño de efecto por bloques de los confirmatorios E-A y CloverLeaf (Fase 4).

Complementa la prueba de signos (cuyo p mínimo con cinco bloques es 0.0625) con
la media geométrica de la razón brazo/base por bloque y su IC95 por t pareada
sobre el logaritmo. Imprime una fila por experimento, brazo y métrica.

Datos:
  datos/fase4_20260926/fase4_EA.csv            (E-A, brazo base = "base")
  datos/fase4_20260924/cloverleaf_confirm_7692.csv (CloverLeaf, base = "ref")
"""
import csv
import math
import statistics as st
from pathlib import Path

from scipy import stats

DATOS = Path(__file__).resolve().parent.parent / "datos"


def edp(r):
    return (float(r["e_cpu_j"]) + float(r["e_gpu_j"])) * float(r["wall_s"])


def e_gpu(r):
    return float(r["e_gpu_j"])


def resumen(razones):
    logs = [math.log(x) for x in razones]
    n = len(logs)
    m, s = st.mean(logs), st.stdev(logs)
    t = stats.t.ppf(0.975, n - 1)
    p = stats.ttest_1samp(logs, 0.0).pvalue
    return math.exp(m), math.exp(m - t * s / math.sqrt(n)), math.exp(m + t * s / math.sqrt(n)), p


def por_bloque(ruta, col_bloque, base, brazos):
    filas = {}
    for r in csv.DictReader(open(ruta)):
        if "cell" in r:  # el CSV de E-A repite filas; se deduplica por celda
            filas[r["cell"]] = r
        else:
            filas[(r[col_bloque], r["arm"])] = r
    bloques = {}
    for r in filas.values():
        bloques.setdefault(r[col_bloque], {})[r["arm"]] = r
    return {a: [(b[a], b[base]) for _, b in sorted(bloques.items())] for a in brazos}


def main():
    casos = [
        ("E-A", DATOS / "fase4_20260926/fase4_EA.csv", "rep", "base", ("activo_gpu", "sombra")),
        ("CloverLeaf", DATOS / "fase4_20260924/cloverleaf_confirm_7692.csv", "block", "ref", ("activo", "sombra")),
    ]
    print("experimento,brazo,metrica,n,media_geom,ic95_inf,ic95_sup,p_t_pareada,mediana_razones")
    for nombre, ruta, col, base, brazos in casos:
        for brazo, pares in por_bloque(ruta, col, base, brazos).items():
            for metrica, f in (("edp_nodo", edp), ("e_gpu", e_gpu)):
                razones = [f(a) / f(b) for a, b in pares]
                g, lo, hi, p = resumen(razones)
                print(f"{nombre},{brazo},{metrica},{len(razones)},{g:.4f},{lo:.4f},{hi:.4f},{p:.4g},{st.median(razones):.4f}")


if __name__ == "__main__":
    main()
