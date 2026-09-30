"""Descomposición del costo de observación del agente de CPU (revisión C6 del director), sin mediciones nuevas.

Para cada campaña con brazo sombra de CPU, separa la energía de CPU adicional de la sombra frente a la base en la parte
que se explica por la mayor duración (potencia de la base por el tiempo extra) y la que corresponde a mayor potencia, y
expresa esta última como potencia media adicional (W). Compara con el brazo sombra del agente de GPU, que usa el mismo
ONNX Runtime. Medianas de las repeticiones válidas por brazo.

Entradas: docs/libro/datos/fase4_20260926/{fase4_A.csv, fase4_C.csv} (agente final: ORT de un hilo, núcleos 0-5)
Salida: docs/libro/datos/fase4_20260930/costo_cpu_descomposicion.csv
Reproduce: python3 docs/libro/scripts/analisis_costo_cpu_20260930.py
"""
import csv
import statistics as st
from pathlib import Path

L = Path(__file__).resolve().parents[1]
FUENTES = {"matriz inicial": L / "datos/fase4_20260926/fase4_A.csv", "Escenario C": L / "datos/fase4_20260926/fase4_C.csv"}


def main():
    filas = []
    for nombre, path in FUENTES.items():
        g = {}
        for r in csv.DictReader(open(path)):
            if r.get("state_ok", "1") == "1":
                g.setdefault((r["scope"], r["set"], r["arm"]), []).append(r)
        m = lambda rows, c: st.median(float(r[c]) for r in rows)  # noqa: E731
        for scope in ("cpu", "gpu", "joint"):
            for s in ("known", "unseen"):
                if (scope, s, "sombra") not in g:
                    continue
                b, v = g[(scope, s, "base")], g[(scope, s, "sombra")]
                tb, eb = m(b, "wall_s"), m(b, "e_cpu_j")
                de, dt = m(v, "e_cpu_j") - eb, m(v, "wall_s") - tb
                potencia = de - eb / tb * dt
                filas.append({"campana": nombre, "alcance": scope, "kernels": s, "t_base_s": round(tb, 1),
                              "dt_s": round(dt, 2), "de_cpu_j": round(de), "fraccion_potencia": round(potencia / de, 2) if de > 0 else "",
                              "w_extra": round(potencia / tb, 1)})
    out = L / "datos/fase4_20260930/costo_cpu_descomposicion.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
    for r in filas:
        print(r)


if __name__ == "__main__":
    main()
