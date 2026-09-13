"""Dos analisis sobre la campana del selector (pacca_dual_gpu_full_20260828),
sin gastar computo nuevo. Solo libreria estandar, para correr en cualquier
nodo sin depender del venv.

A) ROBUSTEZ AL TAMANO DE LA POLITICA DE FRECUENCIA
   ¿El nivel de reloj de GPU que minimiza el EDP cambia con el tamano del
   problema? Si no cambia, la decision ya acordada (un solo tamano por
   familia dual_*) queda respaldada. Si cambia, hay que saberlo ANTES de
   derivar la tabla de politica.

B) ¿LOS TAMANOS SON DEMASIADO PEQUENOS?
   Utilizacion real de la GPU durante la FASE CALIENTE (acotada con el
   contrato cold_warm_v1, no sobre toda la corrida). Un tamano cuya fase
   caliente corre con utilizacion baja no esta ejercitando la GPU y es un
   mal punto de medida para una politica de frecuencia de GPU.

Energia: la de CPU sale de RAPL en metadata.json; la de GPU se integra de
gpu_power_mw sobre los timestamps de samples.csv (robusto, no depende de que
gpu_energy_mj sea acumulativo o por muestra).
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean, median

ROOT = "/home/latorresn/hyperion-results/campaigns/pacca_dual_gpu_full_20260828"
CPU_FIJO = "F0"          # se fija la CPU al maximo para aislar el eje de GPU
GPU_FIJO_UTIL = "REF"    # parte B: nivel de GPU fijo, si no se mezclan relojes
UTIL_PISO = 5.0          # % de utilizacion considerado "GPU realmente activa"

NOMBRE = re.compile(r"^.*?__(?P<kref>.+?)__(?P<cpu>[^_]+)__gpu(?P<gpu>[^_]+)__rep(?P<rep>\d+)$")
TAM = re.compile(r"^dual_(?P<fam>[a-z]+)_gpu_N(?P<n>\d+)$")


def lee_samples(path, warm_t0=None, warm_t1=None):
    """Devuelve (util_media_fase_caliente, sm_clock_medio, energia_gpu_J)."""
    utils, clocks = [], []
    energia_mj = 0.0
    prev_ts = None
    try:
        with open(path, newline="") as fh:
            for fila in csv.DictReader(fh):
                try:
                    ts = int(fila["timestamp_ns"])
                except (KeyError, ValueError, TypeError):
                    continue
                try:
                    pw = float(fila.get("gpu_power_mw") or 0.0)
                except ValueError:
                    pw = 0.0
                if prev_ts is not None and pw > 0:
                    energia_mj += pw * ((ts - prev_ts) / 1e9)  # mW * s = mJ
                prev_ts = ts
                # utilizacion y reloj solo dentro de la fase caliente
                if warm_t0 is not None and not (warm_t0 <= ts <= warm_t1):
                    continue
                try:
                    u = float(fila.get("gpu_util_pct") or "")
                    utils.append(u)
                except ValueError:
                    pass
                try:
                    clocks.append(float(fila.get("gpu_sm_clock_mhz") or ""))
                except ValueError:
                    pass
    except OSError:
        return None, None, None
    return (mean(utils) if utils else None,
            mean(clocks) if clocks else None,
            energia_mj / 1000.0)


def main():
    registros = []
    revisados = 0
    for nombre in sorted(os.listdir(ROOT)):
        ruta = os.path.join(ROOT, nombre)
        meta = os.path.join(ruta, "metadata.json")
        if not os.path.isdir(ruta) or not os.path.exists(meta):
            continue
        m = NOMBRE.match(nombre)
        if not m or m.group("cpu") != CPU_FIJO:
            continue
        t = TAM.match(m.group("kref"))
        if not t:
            continue  # solo kernels dual_*_gpu_N*, que son los que barren tamano
        try:
            d = json.load(open(meta))
        except (OSError, ValueError):
            continue
        elapsed_s = (d.get("telemetry_elapsed_ns_mean") or 0) / 1e9
        if elapsed_s <= 0:
            continue
        e_cpu_J = ((d.get("rapl_pkg_total_delta_uj") or 0)
                   + (d.get("rapl_dram_total_delta_uj") or 0)) / 1e6
        dt = d.get("dispatch_timing") or {}
        w0, w1 = dt.get("warm_t0_ns"), dt.get("warm_t1_ns")
        util, clock, e_gpu_J = lee_samples(os.path.join(ruta, "samples.csv"), w0, w1)
        revisados += 1
        registros.append({
            "fam": t.group("fam"), "n": int(t.group("n")),
            "gpu": m.group("gpu"), "rep": m.group("rep"),
            "t_s": elapsed_s, "warm_s": dt.get("warm_total_seconds"),
            "e_cpu_J": e_cpu_J, "e_gpu_J": e_gpu_J or 0.0,
            "util": util, "clock": clock,
        })

    print(f"corridas procesadas (CPU={CPU_FIJO}, kernels dual_*_gpu_N*): {revisados}\n")
    if not registros:
        sys.exit("sin datos")

    # ---------- B) utilizacion de GPU por tamano ----------
    print("=" * 78)
    print(f"B) UTILIZACION DE GPU EN FASE CALIENTE (nivel GPU fijo = {GPU_FIJO_UTIL})")
    print("=" * 78)
    print(f"{'familia':<10} {'N':>10} {'util % (med)':>13} {'SM MHz':>9} {'warm s':>9} {'n':>4}")
    print("-" * 78)
    porTam = defaultdict(list)      # todos los niveles: se usa en la parte A
    porTamUtil = defaultdict(list)  # un solo nivel: se usa en la parte B
    for r in registros:
        porTam[(r["fam"], r["n"])].append(r)
        if r["gpu"] == GPU_FIJO_UTIL:
            porTamUtil[(r["fam"], r["n"])].append(r)
    sospechosos = []
    for (fam, n) in sorted(porTamUtil):
        rs = porTamUtil[(fam, n)]
        us = [r["util"] for r in rs if r["util"] is not None]
        cs = [r["clock"] for r in rs if r["clock"] is not None]
        ws = [r["warm_s"] for r in rs if r["warm_s"] is not None]
        u = median(us) if us else float("nan")
        marca = ""
        if us and u < 50:
            marca = "  <-- GPU infrautilizada"
            sospechosos.append((fam, n, u))
        print(f"{fam:<10} {n:>10} {u:>13.1f} {mean(cs) if cs else 0:>9.0f} "
              f"{mean(ws) if ws else 0:>9.3f} {len(rs):>4}{marca}")

    print()
    print("B2) MISMA UTILIZACION, PERO AGRUPADA POR NIVEL DE GPU")
    print("    (si sube al bajar el reloj, es artefacto de medida, no mas trabajo)")
    porNivelUtil = defaultdict(list)
    porNivelClock = defaultdict(list)
    for r in registros:
        if r["util"] is not None:
            porNivelUtil[r["gpu"]].append(r["util"])
        if r["clock"] is not None:
            porNivelClock[r["gpu"]].append(r["clock"])
    print(f"{'nivel':<8} {'util % (med)':>13} {'SM MHz real':>13} {'n':>6}")
    print("-" * 44)
    for lvl in sorted(porNivelUtil):
        us = porNivelUtil[lvl]
        cs = porNivelClock.get(lvl, [])
        print(f"{lvl:<8} {median(us):>13.1f} {mean(cs) if cs else 0:>13.0f} {len(us):>6}")

    # ---------- A) ¿cambia el nivel optimo con el tamano? ----------
    print()
    print("=" * 78)
    print("A) NIVEL DE RELOJ DE GPU QUE MINIMIZA EL EDP, POR TAMANO")
    print("=" * 78)
    print(f"{'familia':<10} {'N':>10} {'mejor nivel':>12} {'EDP mejor':>12} {'EDP en REF':>12} {'ganancia %':>11}")
    print("-" * 78)
    argmin_por_fam = defaultdict(list)
    for (fam, n) in sorted(porTam):
        porNivel = defaultdict(list)
        for r in porTam[(fam, n)]:
            edp = (r["e_cpu_J"] + r["e_gpu_J"]) * r["t_s"]
            porNivel[r["gpu"]].append(edp)
        if not porNivel:
            continue
        medios = {k: mean(v) for k, v in porNivel.items()}
        mejor = min(medios, key=medios.get)
        ref = medios.get("REF")
        gan = (ref - medios[mejor]) / ref * 100.0 if ref else float("nan")
        argmin_por_fam[fam].append((n, mejor))
        print(f"{fam:<10} {n:>10} {mejor:>12} {medios[mejor]:>12.1f} "
              f"{ref if ref else float('nan'):>12.1f} {gan:>11.1f}")

    print()
    print("=" * 78)
    print("VEREDICTO DE ROBUSTEZ AL TAMANO")
    print("=" * 78)
    for fam in sorted(argmin_por_fam):
        pares = argmin_por_fam[fam]
        niveles = sorted({lvl for _, lvl in pares})
        estado = "ESTABLE" if len(niveles) == 1 else f"CAMBIA ({len(niveles)} niveles distintos)"
        detalle = ", ".join(f"N{n}->{lvl}" for n, lvl in pares)
        print(f"{fam:<10} {estado}")
        print(f"           {detalle}")

    if sospechosos:
        print()
        print("TAMANOS CON GPU INFRAUTILIZADA (utilizacion mediana < 50 % en fase caliente):")
        for fam, n, u in sospechosos:
            print(f"  dual_{fam}_gpu_N{n}: {u:.1f} %")


if __name__ == "__main__":
    main()
