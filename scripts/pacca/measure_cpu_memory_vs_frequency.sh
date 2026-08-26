#!/bin/bash
# ¿La memoria se frena cuando bajamos la frecuencia del NUCLEO?
#
# LA PREGUNTA, Y POR QUE ES LA PREGUNTA DEL PROYECTO. Toda la tesis de la
# parte CPU descansa en una premisa: si una carga esta limitada por
# memoria, bajar el reloj del nucleo no deberia alargar su tiempo, porque
# el cuello de botella no es el nucleo. Si eso fuera cierto, alpha
# (fraccion del tiempo que escala con el reloj) valdria ~0 para una carga
# perfectamente saturada en ancho de banda.
#
# NO ES LO QUE SE MIDIO. `stream_official` satura el ancho de banda al
# 100% (76.80 GB/s, la referencia del propio nodo) y aun asi da
# alpha = 0.154, no 0. Traducido: a 800 MHz, STREAM tarda un 46% mas que
# a 3200 MHz PESE a estar completamente limitado por memoria. Esa brecha
# es la falla que hay que explicar, y hasta ahora solo se habia observado
# el SINTOMA (el tiempo) sin medir la CAUSA (que hace el subsistema de
# memoria mientras tanto).
#
# LA HIPOTESIS QUE ESTE SCRIPT PONE A PRUEBA. En este Xeon el uncore
# --controlador de memoria, malla, LLC-- tiene su PROPIO dominio de
# frecuencia, verificado en el nodo: /sys/devices/system/cpu/
# intel_uncore_frequency/package_00_die_00 declara un rango de 800 a
# 2400 MHz, distinto del rango de nucleo (800-3200 MHz). Si la gestion de
# energia baja tambien el uncore cuando bajan los nucleos, entonces el
# subsistema de memoria SI se frena, la premisa del proyecto es falsa tal
# como esta enunciada, y alpha > 0 en STREAM deja de ser un misterio.
#
# `current_freq_khz` de ese sysfs es ilegible sin root en este nodo, asi
# que el uncore se mide POR SUS PROPIOS TICKS: `uncore_imc_0/clockticks/`
# contado sobre un intervalo de reloj de pared conocido da la frecuencia
# real del controlador de memoria. Es medicion directa, no inferencia.
#
# QUE PRODUCE. Por cada (carga, nivel de frecuencia de nucleo):
#   - frecuencia REAL del controlador de memoria (GHz), de sus clockticks
#   - ancho de banda alcanzado (GB/s), de cas_count_read+write x 64 B
#   - tiempo transcurrido
#   - frecuencia de nucleo observada bajo carga (control C2 de siempre)
#
# Con eso se separan las dos explicaciones posibles del alpha=0.154 de
# STREAM, que hoy estan confundidas:
#   (a) el uncore baja con los nucleos -> la memoria se frena de verdad
#   (b) el uncore se mantiene -> la memoria va igual de rapido y el 15.4%
#       es trabajo del lado del nucleo (calculo de direcciones, control de
#       bucle, incapacidad de sostener suficientes fallos en vuelo)
# Son diagnosticos distintos y llevan a decisiones distintas: (a) es un
# limite de plataforma que hay que declarar, (b) es un limite del catalogo
# que se puede atacar con cargas mejor elegidas.
#
# CARGAS. Tres, elegidas para cubrir los tres regimenes y poder contrastar:
#   - stream_official : satura ancho de banda (100% de referencia)
#   - ptrchase        : limitado por LATENCIA, no por ancho de banda --
#                       persecucion de punteros con dependencia serial
#   - ert_probe       : control COMPUTE-BOUND, debe mostrar el
#                       comportamiento opuesto (BW ~0, tiempo escalando
#                       con el reloj). Sin este control no se puede saber
#                       si el instrumento distingue algo.
#
# EVENTOS EN FORMATO CRUDO, no por alias. Leccion del 2026-08-25 (fix de
# E13 en preflight.py): el alias simbolico `cas_count_read` trae metadato
# de unidad y perf lo AUTOESCALA a MiB incluso en modo intervalo, asi que
# el campo de valor deja de ser un conteo entero. Los formatos crudos se
# leen de sysfs en tiempo de ejecucion, igual que hace uncore_reader.cpp.
#
# Uso: sbatch run_measure_cpu_memory_vs_frequency.sbatch
set -o pipefail
# ARC-126: sin "-u" a proposito.

# bin/stream_c, no bin/stream_official: `stream_official` es el ID del
# catalogo, el ejecutable se llama distinto (catalog.yaml exec_path).
STREAM_BIN=/home/latorresn/hyperion-kernels/bin/stream_c
PTRCHASE_BIN=/home/latorresn/hyperion-kernels/bin/ptrchase
ERT_BIN=/home/latorresn/hyperion-kernels/bin/ert_probe

DELEGATED_CPUS=(0 1 2 3 4 5)

# Mismos 5 niveles que la campaña de CPU ya validada
# (pacca_cpu_final_attempt03_20260820_arc174).
LEVELS=(F0:3200000 F1:2600000 F2:2000000 F3:1400000 F4:800000)

# --- Frecuencia: hermanos SMT incluidos (ARC-162/163) --------------------
# intel_pstate coordina el P-state a nivel de NUCLEO FISICO. Pinear solo
# 0-5 sin sus hermanos deja el reloj compartido al limite del hermano
# libre: es el bug que invalido la corrida 6475 entera.
smt_siblings_for() {
  local cpu="$1"
  local path="/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list"
  [[ -r "$path" ]] && tr ',' '\n' < "$path" | tr '-' '\n'
}

declare -a FREQ_CPUS
_expand_with_smt_siblings() {
  local -A seen=()
  FREQ_CPUS=()
  for cpu in "${DELEGATED_CPUS[@]}"; do
    [[ -z "${seen[$cpu]:-}" ]] && { FREQ_CPUS+=("$cpu"); seen[$cpu]=1; }
    while read -r sib; do
      [[ -z "$sib" ]] && continue
      [[ -z "${seen[$sib]:-}" ]] && { FREQ_CPUS+=("$sib"); seen[$sib]=1; }
    done < <(smt_siblings_for "$cpu")
  done
}
_expand_with_smt_siblings
echo "CPUs de frecuencia (delegados + hermanos SMT): ${FREQ_CPUS[*]}" >&2

set_freq() {
  local target_khz="$1"
  for cpu in "${FREQ_CPUS[@]}"; do
    local base="/sys/devices/system/cpu/cpu${cpu}/cpufreq"
    local current_min
    current_min="$(cat "${base}/scaling_min_freq")"
    if [[ "$target_khz" -lt "$current_min" ]]; then
      echo "$target_khz" > "${base}/scaling_min_freq"
      echo "$target_khz" > "${base}/scaling_max_freq"
    else
      echo "$target_khz" > "${base}/scaling_max_freq"
      echo "$target_khz" > "${base}/scaling_min_freq"
    fi
  done
  sleep 2
}

restore_freq() {
  for cpu in "${FREQ_CPUS[@]}"; do
    local base="/sys/devices/system/cpu/cpu${cpu}/cpufreq"
    echo "$(cat "${base}/cpuinfo_max_freq")" > "${base}/scaling_max_freq"
    echo "$(cat "${base}/cpuinfo_min_freq")" > "${base}/scaling_min_freq"
  done
}
trap restore_freq EXIT

# --- Eventos de uncore en formato CRUDO ----------------------------------
read_raw() {
  local path="$1"
  [[ -r "$path" ]] && tr -d '\n' < "$path"
}
IMC_CLOCK_FMT="$(read_raw /sys/bus/event_source/devices/uncore_imc_0/events/clockticks)"
IMC_READ_FMT="$(read_raw /sys/bus/event_source/devices/uncore_imc_0/events/cas_count_read)"
IMC_WRITE_FMT="$(read_raw /sys/bus/event_source/devices/uncore_imc_0/events/cas_count_write)"
if [[ -z "$IMC_CLOCK_FMT" || -z "$IMC_READ_FMT" || -z "$IMC_WRITE_FMT" ]]; then
  echo "ERROR: no se pudieron leer los formatos de evento de uncore_imc_0" >&2
  exit 1
fi
echo "formatos crudos: clock='$IMC_CLOCK_FMT' read='$IMC_READ_FMT' write='$IMC_WRITE_FMT'" >&2

# clockticks de UNA sola unidad (uncore_imc_0): es el reloj del
# controlador, no trafico -- sumarlo sobre las 12 unidades daria 12x la
# frecuencia real. El trafico SI se suma sobre todas (`uncore_imc/`).
EVENTS="uncore_imc_0/${IMC_CLOCK_FMT}/,uncore_imc/${IMC_READ_FMT}/,uncore_imc/${IMC_WRITE_FMT}/"

sample_freq_bg() {
  local out="$1"
  : > "$out"
  while :; do
    for cpu in "${DELEGATED_CPUS[@]}"; do
      cat "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq" 2>/dev/null
    done
    sleep 0.02
  done >> "$out"
}

export OMP_NUM_THREADS=6
export OMP_PROC_BIND=true
export OMP_PLACES=cores

run_one() {
  local name="$1"; shift
  local level_id="$1"; shift
  local khz="$1"; shift
  # El resto de argumentos es el comando completo de la carga.

  local perf_out freq_log
  perf_out="$(mktemp -p /home/latorresn hyperion_memfreq_perf_XXXXXX)"
  freq_log="$(mktemp -p /home/latorresn hyperion_memfreq_freq_XXXXXX)"

  sample_freq_bg "$freq_log" &
  local sampler_pid=$!

  local t0 t1
  t0=$(date +%s%N)
  perf stat -a -x ';' -e "$EVENTS" -o "$perf_out" -- "$@" >/dev/null 2>&1
  local rc=$?
  t1=$(date +%s%N)

  kill "$sampler_pid" 2>/dev/null
  wait "$sampler_pid" 2>/dev/null

  python3 - "$name" "$level_id" "$khz" "$t0" "$t1" "$perf_out" "$freq_log" "$rc" <<'PY'
import sys

name, level_id, khz, t0, t1, perf_path, freq_path, rc = sys.argv[1:9]
elapsed_s = (int(t1) - int(t0)) / 1e9

clockticks = cas_read = cas_write = None
with open(perf_path) as handle:
    for line in handle:
        if line.startswith("#") or ";" not in line:
            continue
        fields = line.split(";")
        if len(fields) < 3:
            continue
        raw, _unit, event = fields[0].strip(), fields[1].strip(), fields[2].strip()
        if not raw or raw.startswith("<"):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if event.startswith("uncore_imc_0/"):
            clockticks = value
        elif "umask=0xf" in event:
            cas_read = value
        elif "umask=0x30" in event:
            cas_write = value

freqs = []
try:
    with open(freq_path) as handle:
        for line in handle:
            line = line.strip()
            if line.isdigit():
                freqs.append(int(line))
except OSError:
    pass

def fmt(value, digits=4):
    return "" if value is None else f"{value:.{digits}f}"

imc_ghz = clockticks / elapsed_s / 1e9 if clockticks and elapsed_s > 0 else None
total_cas = (cas_read or 0) + (cas_write or 0)
bw_gbs = total_cas * 64 / elapsed_s / 1e9 if elapsed_s > 0 else None
freq_mean = sum(freqs) / len(freqs) if freqs else None
target_khz = int(khz)
within = "yes" if freq_mean and abs(freq_mean - target_khz) <= 0.05 * target_khz else "NO"

print(f"{name},{level_id},{target_khz},{elapsed_s:.6f},{fmt(imc_ghz)},"
      f"{fmt(bw_gbs,3)},{fmt(freq_mean,0)},{within},{len(freqs)},{rc}")
PY

  rm -f -- "$perf_out" "$freq_log"
}

echo "carga,nivel,khz_objetivo,tiempo_s,imc_ghz,bw_gbs,freq_nucleo_khz,freq_ok,n_muestras,rc"

for lv in "${LEVELS[@]}"; do
  level_id="${lv%%:*}"
  khz="${lv##*:}"
  set_freq "$khz"

  # Warmup descartado por nivel: deja que el reloj se asiente bajo carga
  # real antes de cronometrar (ARC-160/164 -- el decaimiento tarda
  # segundos bajo intel_pstate+HWP).
  [[ -x "$STREAM_BIN" ]] && "$STREAM_BIN" >/dev/null 2>&1

  [[ -x "$STREAM_BIN" ]]   && run_one stream_official "$level_id" "$khz" "$STREAM_BIN"
  [[ -x "$PTRCHASE_BIN" ]] && run_one ptrchase "$level_id" "$khz" \
      "$PTRCHASE_BIN" --size-mib 512 --iterations 6 --seed 20260806
  [[ -x "$ERT_BIN" ]]      && run_one ert_probe "$level_id" "$khz" "$ERT_BIN"
done

echo "CPU_MEMORY_VS_FREQUENCY_DONE" >&2
