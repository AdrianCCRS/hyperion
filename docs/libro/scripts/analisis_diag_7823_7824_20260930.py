"""Diagnóstico del costo de observación del agente de CPU (job 7824, revisión C6) y réplica de E-A con EPP=power (job 7823).

Entradas: docs/libro/datos/fase4_20260930/costo_cpu_7824/{costo_matriz,reposo}/ y epppower_7823/
Salida (stdout): potencia extra de CPU por brazo frente a la base del bloque, contraste con la sombra de referencia
(prueba de signos exacta, 6 bloques), potencia en reposo con y sin agente, residencia en C6 e interrupciones entre núcleos
de la ventana de energía, y E-A bajo EPP=power (media geométrica por bloque con IC95 t sobre el logaritmo).
Reproduce: python3 docs/libro/scripts/analisis_diag_7823_7824_20260930.py
"""
import csv, glob, math, statistics as st
from pathlib import Path
from scipy import stats

D = Path(__file__).resolve().parents[1] / "datos" / "fase4_20260930"


def parse(f):
    sec, cur = {}, None
    for l in open(f).read().splitlines():
        if l.startswith("## "):
            cur = l[3:].split()[0]; sec[cur] = []
        elif cur:
            sec[cur].append(l)
    return sec


def idle_us(sec):
    d = {}
    for l in sec["cpuidle"]:
        p = l.split()
        if len(p) == 4:
            d.setdefault(p[0], {})[p[1]] = int(p[2])
    return d


def cal(sec):
    for l in sec["interrupts"]:
        if l.strip().startswith("CAL"):
            return [int(x) for x in l.split()[1:25] if x.isdigit()]


def costo():
    M = D / "costo_cpu_7824" / "costo_matriz"
    rows = list(csv.DictReader(open(M / "results.csv")))
    d = {(r["arm"], r["rep"]): r for r in rows}
    reps = sorted({r["rep"] for r in rows}, key=int)
    P = lambda a, k: float(d[(a, k)]["e_cpu_j"]) / float(d[(a, k)]["wall_s"])  # noqa: E731
    print(f"celdas válidas {sum(r['state_ok'] == '1' and r['app_rc'] == '0' for r in rows)}/{len(rows)}; "
          f"potencia de la base {st.median(P('base', k) for k in reps):.1f} W")
    for a in ("sombra", "sombra_10ms", "sombra_idle1ms", "sombra_pin"):
        dp = [P(a, k) - P("base", k) for k in reps]
        print(f"{a:15s} potencia extra mediana {st.median(dp):5.1f} W  por bloque {[round(x, 1) for x in dp]}")
    for a in ("sombra_10ms", "sombra_idle1ms", "sombra_pin"):
        dif = [P(a, k) - P("sombra", k) for k in reps]
        n = sum(x < 0 for x in dif)
        print(f"{a} frente a sombra: menor en {n}/6 bloques, p={stats.binomtest(n, 6).pvalue:.3f}, diferencia mediana {st.median(dif):+.1f} W")
    res = {}
    for c in sorted(glob.glob(str(M / "cells" / "cpu_known_*"))):
        arm = Path(c).name.split("cpu_known_")[1].rsplit("_", 1)[0]
        a, b = parse(c + "/snap_inicio.txt"), parse(c + "/snap_fin.txt")
        ia, ib = idle_us(a), idle_us(b)
        res.setdefault((arm, "cpu6_C6_s"), []).append((ib["cpu6"]["C6"] - ia["cpu6"]["C6"]) / 1e6)
        res.setdefault((arm, "CAL"), []).append(sum(y - x for x, y in zip(cal(a), cal(b))))
    for (arm, k), v in sorted(res.items()):
        print(f"{arm:15s} {k:10s} mediana {st.median(v):.1f}")
    R = list(csv.DictReader(open(D / "costo_cpu_7824" / "reposo" / "results.csv")))
    for arm in ("reposo", "reposo_sombra"):
        v = [float(x["e_cpu_j"]) / float(x["wall_s"]) for x in R if x["arm"] == arm]
        print(f"{arm:14s} {[round(x, 1) for x in v]} mediana {st.median(v):.1f} W")


def epp():
    rows = list(csv.DictReader(open(D / "epppower_7823" / "EA_confirmatorio_epppower" / "results.csv")))
    d = {(r["arm"], r["rep"]): r for r in rows}
    reps = sorted({r["rep"] for r in rows}, key=int)
    f = {"edp": lambda r: (float(r["e_cpu_j"]) + float(r["e_gpu_j"])) * float(r["wall_s"]), "egpu": lambda r: float(r["e_gpu_j"])}
    for a, b in (("activo_gpu", "base"), ("sombra", "base"), ("fijo_gpu_f1", "base"), ("activo_gpu", "fijo_gpu_f1")):
        for k in ("edp", "egpu"):
            lg = [math.log(f[k](d[(a, x)]) / f[k](d[(b, x)])) for x in reps]
            h = stats.t.ppf(.975, len(lg) - 1) * st.stdev(lg) / math.sqrt(len(lg))
            n = sum(x < 0 for x in lg)
            print(f"EPP=power {a}/{b} {k}: {math.exp(st.mean(lg)):.3f} (IC95 {math.exp(st.mean(lg) - h):.3f} a {math.exp(st.mean(lg) + h):.3f}; {n}/5)")


if __name__ == "__main__":
    costo(); epp()
