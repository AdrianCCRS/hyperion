#!/bin/bash
# Restaura scaling_min_freq/scaling_max_freq al rango nativo completo en
# cpu0-5 y sus pares SMT (16-21 en paccaA100) ANTES de que el orquestador
# capture su propio snapshot "original" al arrancar una campaña.
#
# POR QUE EXISTE. `orchestrator.campaign.run_campaign` restaura frecuencia
# en su `finally:` sin importar como termine la corrida (CAM-07) -- pero
# restaura al snapshot que capturo AL EMPEZAR esa misma corrida. Si una
# corrida previa crasheo dentro de la calibracion (o de cualquier punto
# antes del snapshot) y dejo min=max pinneados en un valor bajo, la
# siguiente corrida captura ESE estado ya roto como su propio "original" y
# lo restaura fielmente al terminar -- la corrupcion se propaga en silencio
# de job en job sin que ningun chequeo la detecte (descubierto 2026-08-27,
# job 6651 crasheo en D03 y dejo cpu0-5/16-21 en min=max=800000; el job
# 6657 siguiente midio REF muy por debajo de F0 por esta razon exacta).
#
# Este script NO reemplaza esa proteccion, la complementa: se ejecuta ANTES
# de invocar el orquestador, para que el snapshot que este SI capture ya
# sea el rango nativo sano, no una corrupcion heredada.
set -eo pipefail

MIN_KHZ="${1:-800000}"
MAX_KHZ="${2:-3200000}"  # turbo deshabilitado por convencion (no_turbo=1);
                          # cpuinfo_max_freq reporta 3600000 (con turbo).

for c in 0 1 2 3 4 5 16 17 18 19 20 21; do
  path_min="/sys/devices/system/cpu/cpu${c}/cpufreq/scaling_min_freq"
  path_max="/sys/devices/system/cpu/cpu${c}/cpufreq/scaling_max_freq"
  if [[ ! -w "$path_min" || ! -w "$path_max" ]]; then
    echo "E01: sin permiso de escritura sobre cpu${c} (min o max)" >&2
    exit 65
  fi
  # max primero, luego min: si el rango actual es mas estrecho que el
  # objetivo (p.ej. min=max=800000), escribir min antes que max fallaria
  # con EINVAL (min propuesto > max actual).
  echo "$MAX_KHZ" > "$path_max"
  echo "$MIN_KHZ" > "$path_min"
  actual_min="$(<"$path_min")"
  actual_max="$(<"$path_max")"
  if [[ "$actual_min" != "$MIN_KHZ" || "$actual_max" != "$MAX_KHZ" ]]; then
    echo "E01: cpu${c} no confirmo el rango (min=$actual_min max=$actual_max, esperado min=$MIN_KHZ max=$MAX_KHZ)" >&2
    exit 70
  fi
done
echo "RESET_CPU_FREQ_RANGE_OK: cpu0-5,16-21 min=${MIN_KHZ} max=${MAX_KHZ}"
