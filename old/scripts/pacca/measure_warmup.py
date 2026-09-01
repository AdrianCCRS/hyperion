#!/usr/bin/env python3
"""Mide el transitorio de arranque real de un kernel a partir de un
windows.csv ya generado -- no hace falta re-correr nada: warmup_seconds
nunca se le pasa al harness C++ (runner.py no lo incluye en build_command),
así que toda corrida ya capturó el transitorio completo desde el principio;
hoy solo queda marcado como quality_status="warmup_excluded" en vez de
usarse para medir cuánto dura de verdad.

Criterio (ARC-81, acordado antes de correr esto): se busca el primer punto
a partir del cual el coeficiente de variación (CV%) de la señal relevante
(IPC para CPU, gpu_util_pct para GPU) se mantiene por debajo de un umbral
en dos ventanas móviles consecutivas -- exigir estabilidad en dos ventanas
seguidas, no en el resto completo de la corrida, para no confundir el fin
del transitorio de arranque con un cambio de fase legítimo más adelante
(ej. npb_bt/npb_ft alternan fases real y deliberadamente, ver ARC-71).
El umbral (5%) es el mismo que CAL-10 ya usa para juzgar estabilidad de la
referencia IPC/IPS -- reutilizado, no inventado de nuevo para esto.

Uso:
    python measure_warmup.py windows1.csv [windows2.csv ...]

Imprime una fila JSON por archivo con el warmup detectado, la fracción de
la corrida que representa, y si no se pudo detectar un punto claro.
"""
import csv
import json
import statistics
import sys

CV_THRESHOLD_PCT = 5.0
MARGIN = 1.2  # +20% de margen de seguridad sobre el punto detectado


def _cv_pct(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    return (statistics.pstdev(values) / abs(mean)) * 100.0


def _adaptive_window_size(n: int) -> int:
    # GPU: pocas muestras (cadencia ~100 ms) -- ventana chica.
    # CPU: miles de muestras (cadencia ~1 ms) -- ventana de 15 alcanza.
    return max(3, min(15, n // 4))


def detect_warmup_ns(
    series: list[tuple[int, float]], *, min_mean_floor: float = 0.0
) -> tuple[int | None, bool]:
    """series: [(t_ns, valor), ...] ordenada por t_ns. Devuelve
    (t_ns_relativo_al_inicio, detectado).

    ARC-86: una ventana de puros ceros tiene CV%=0 por construcción (ver
    `_cv_pct`), así que sin `min_mean_floor` el detector confunde el reposo
    inicial (antes de que arranque el trabajo real) con la señal ya
    estabilizada -- exactamente lo opuesto de lo que se busca. Con el piso,
    una ventana "estable en cero" no cuenta como fin de arranque."""
    n = len(series)
    window = _adaptive_window_size(n)
    if n < window * 2:
        return None, False
    values = [v for _, v in series]
    t0 = series[0][0]
    for i in range(n - window * 2 + 1):
        w1, w2 = values[i:i + window], values[i + window:i + 2 * window]
        first = _cv_pct(w1)
        second = _cv_pct(w2)
        if first is None or second is None:
            continue
        if first <= CV_THRESHOLD_PCT and second <= CV_THRESHOLD_PCT:
            if statistics.fmean(w1 + w2) < min_mean_floor:
                continue
            return series[i][0] - t0, True
    return None, False


def _sse(values: list[float]) -> float:
    """Suma de errores cuadráticos respecto a la media -- la métrica que
    minimiza la segmentación binaria en cada corte candidato."""
    if not values:
        return 0.0
    mean = statistics.fmean(values)
    return sum((v - mean) ** 2 for v in values)


def _best_split(values: list[float], min_size: int) -> tuple[int, float] | None:
    n = len(values)
    if n < 2 * min_size:
        return None
    total_sse = _sse(values)
    best_i, best_gain = None, 0.0
    for i in range(min_size, n - min_size + 1):
        gain = total_sse - _sse(values[:i]) - _sse(values[i:])
        if gain > best_gain:
            best_gain, best_i = gain, i
    if best_i is None:
        return None
    return best_i, best_gain


def detect_changepoints(
    values: list[float], *, min_size: int = 10, min_relative_gain: float = 0.10, max_depth: int = 6
) -> list[int]:
    """ARC-83: segmentación binaria (Scott & Knott 1974; misma familia que
    el PELT de Barrett et al. 2017, sin la optimización de velocidad que
    PELT aporta -- no hace falta para series de este tamaño). Encuentra
    recursivamente el corte que más reduce la suma de errores cuadráticos
    dentro de cada segmento, y sigue partiendo mientras la reducción sea al
    menos `min_relative_gain` de la varianza del segmento -- a diferencia
    del umbral de CV%, no asume que el estado estable es "plano": solo pide
    que haya un cambio real y grande en el comportamiento de la señal, así
    que separa fin-de-arranque de cambios de fase legítimos (los detecta a
    todos como changepoints distintos, no los confunde entre sí)."""
    changepoints: list[int] = []

    def recurse(lo: int, hi: int, depth: int) -> None:
        if depth <= 0 or hi - lo < 2 * min_size:
            return
        segment = values[lo:hi]
        split = _best_split(segment, min_size)
        if split is None:
            return
        i, gain = split
        total = _sse(segment)
        if total == 0 or gain / total < min_relative_gain:
            return
        changepoints.append(lo + i)
        recurse(lo, lo + i, depth - 1)
        recurse(lo + i, hi, depth - 1)

    recurse(0, len(values), max_depth)
    return sorted(changepoints)


def detect_warmup_via_changepoints(
    series: list[tuple[int, float]], *, plateau_ratio: float = 0.8
) -> tuple[int | None, bool]:
    """ARC-86 -- usar el PRIMER changepoint a secas es incorrecto cuando el
    arranque tiene ráfagas espurias (ej. GPU: transferencias H2D o los
    primeros lanzamientos de kernel generan blips de 3-4% de utilización
    antes de que el trabajo real empiece a rendir; ver rodinia_hotspot,
    donde el primer changepoint cae en t=0.33s pero el régimen estable real
    -- 100% de utilización sostenida -- no arranca hasta t=2.13s). En vez de
    eso: se calcula la media de cada segmento entre changepoints, se toma el
    máximo como "meseta" de régimen cargado, y se busca el PRIMER segmento
    cuya media alcance `plateau_ratio` de esa meseta -- los blips previos,
    al no sostenerse, quedan descartados sin necesidad de un umbral fijo de
    duración."""
    values = [v for _, v in series]
    min_size = max(5, len(series) // 50)
    changepoints = detect_changepoints(values, min_size=min_size)
    if not changepoints:
        return None, False

    bounds = [0, *changepoints, len(values)]
    segments = list(zip(bounds, bounds[1:]))
    segment_means = [statistics.fmean(values[a:b]) for a, b in segments]
    plateau_mean = max(segment_means)
    if plateau_mean <= 0:
        return None, False

    threshold = plateau_ratio * plateau_mean
    for (a, _b), mean in zip(segments, segment_means):
        if mean >= threshold:
            if a == 0:
                return None, False
            t0 = series[0][0]
            return series[a][0] - t0, True
    return None, False


def analyze(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    is_gpu = any(r.get("gpu_power_mw") not in (None, "") for r in rows)
    if is_gpu:
        series = sorted(
            (int(r["t_end_ns"]), float(r["gpu_util_pct"]))
            for r in rows
            if r.get("quality_status") == "gpu_telemetry"
            and r.get("t_end_ns") not in (None, "")
            and r.get("gpu_util_pct") not in (None, "")
        )
        signal = "gpu_util_pct"
    else:
        series = sorted(
            (int(r["t_start_ns"]), float(r["ipc"]))
            for r in rows
            if r.get("t_start_ns") not in (None, "") and r.get("ipc") not in (None, "")
        )
        signal = "ipc"

    result = {"path": path, "signal": signal, "n_samples": len(series)}
    if not series:
        result["detected"] = False
        return result

    total_span_s = (series[-1][0] - series[0][0]) / 1e9
    result["total_span_s"] = total_span_s

    # ARC-86: piso de ruido solo aplica a gpu_util_pct (escala 0-100%, con
    # ceros reales durante el reposo inicial); IPC de CPU no tiene ese modo
    # degenerado -- un IPC "estable en cero" implicaría que no se ejecutó
    # ninguna instrucción, lo cual ya se filtra antes de llegar aquí.
    min_mean_floor = 5.0 if is_gpu else 0.0
    t_ns, ok = detect_warmup_ns(series, min_mean_floor=min_mean_floor)
    result["method"] = "cv_threshold"
    result["detected"] = ok
    if not ok:
        # ARC-83: el umbral de CV% (Georges et al. 2007) tiene una falla de
        # detección documentada en la literatura -- antes de dejarlo en "no
        # detectado", se intenta con segmentación binaria (changepoints),
        # que no asume que el estado estable sea "plano dentro de un
        # umbral", solo que haya un cambio grande y real en la señal.
        t_ns, ok = detect_warmup_via_changepoints(series)
        if ok:
            result["method"] = "changepoint"
        result["detected"] = ok
    if ok:
        raw_s = t_ns / 1e9
        result["raw_warmup_s"] = round(raw_s, 4)
        result["proposed_warmup_s"] = round(raw_s * MARGIN, 4)
        result["fraction_of_run"] = round(raw_s / total_span_s, 4) if total_span_s else None
    return result


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(json.dumps(analyze(path)))
