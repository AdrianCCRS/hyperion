#!/bin/bash
# Ejecuta un comando con Turbo Boost deshabilitado mediante el helper
# delegado por el administrador de pacca y restaura exactamente el estado
# inicial al salir, también ante INT/TERM. Debe invocarse dentro de una
# asignación Slurm exclusiva, nunca directamente sobre el nodo compartido.
set -o pipefail

if [[ $# -eq 0 ]]; then
  echo "Uso: $0 COMANDO [ARGUMENTOS...]" >&2
  exit 64
fi

state_file="/sys/devices/system/cpu/intel_pstate/no_turbo"
# El reset defensivo de los launchers toca estos CPU logicos. El wrapper
# debe fotografiar el mismo conjunto ANTES de ese reset; si solo conserva
# no_turbo, una entrada normal 0.8--3.6 GHz puede salir capada a 3.2 GHz
# aunque la campana haya restaurado correctamente su snapshot interno.
state_cpus_csv="${HYPERION_CPU_STATE_CPUS:-0,1,2,3,4,5,16,17,18,19,20,21}"
# ARC-147: la ruta absoluta es obligatoria, no cosmética. sudoers en pacca
# define "Defaults secure_path=/sbin:/bin:/usr/sbin:/usr/bin" para
# latorresn -- ese secure_path NO incluye /usr/local/bin (confirmado con
# `sudo -n -l`), así que "sudo set_turbo_state ..." (nombre pelado) nunca
# puede resolverlo sin importar el $PATH del usuario ni si `command -v` lo
# encuentra antes de invocar sudo. La regla NOPASSWD en sudoers también está
# declarada con la ruta absoluta (`NOPASSWD: /usr/local/bin/set_turbo_state`),
# que es lo único contra lo que sudo compara el comando. Antes de este fix,
# este script resolvía el helper con `command -v` (búsqueda por $PATH) y
# luego invocaba `sudo -n "$helper"` -- si /usr/local/bin no estaba en el
# $PATH de la shell no interactiva de srun, command -v no encontraba nada y
# el script fallaba con "no está instalado", indistinguible en la práctica
# del bloqueo de permiso real reportado en ARC-136/138/146. Confirmado por
# el administrador (correo, 2026-08-18): hay que invocar con la ruta
# completa. Nunca fue un permiso ausente.
helper="/usr/local/bin/set_turbo_state"

if [[ ! -r "$state_file" ]]; then
  echo "E01: no se puede leer $state_file" >&2
  exit 65
fi
if [[ ! -x "$helper" ]]; then
  echo "E01: $helper no está instalado o no es ejecutable" >&2
  exit 65
fi

initial_state="$(<"$state_file")"
if [[ "$initial_state" != "0" && "$initial_state" != "1" ]]; then
  echo "E01: estado inicial no_turbo inválido: $initial_state" >&2
  exit 65
fi

state_snapshot="$(mktemp /tmp/hyperion_cpu_state.XXXXXX)" || exit 65
IFS=',' read -r -a state_cpus <<< "$state_cpus_csv"
for c in "${state_cpus[@]}"; do
  cpu_dir="/sys/devices/system/cpu/cpu${c}/cpufreq"
  for field in scaling_governor scaling_min_freq scaling_max_freq; do
    if [[ ! -r "$cpu_dir/$field" ]]; then
      echo "E01: no se puede fotografiar $cpu_dir/$field" >&2
      rm -f "$state_snapshot"
      exit 65
    fi
  done
  governor="$(<"$cpu_dir/scaling_governor")"
  min_khz="$(<"$cpu_dir/scaling_min_freq")"
  max_khz="$(<"$cpu_dir/scaling_max_freq")"
  printf '%s %s %s %s\n' "$c" "$governor" "$min_khz" "$max_khz" >> "$state_snapshot"
  echo "CPU_FREQ_STATE_INITIAL: cpu${c} governor=${governor} min=${min_khz} max=${max_khz}"
done
echo "TURBO_STATE_INITIAL: no_turbo=${initial_state}"

restore_cpu_state() {
  restore_failed=0
  while read -r c governor min_khz max_khz; do
    cpu_dir="/sys/devices/system/cpu/cpu${c}/cpufreq"
    current_governor="$(<"$cpu_dir/scaling_governor")"
    current_min="$(<"$cpu_dir/scaling_min_freq")"
    current_max="$(<"$cpu_dir/scaling_max_freq")"

    # Mantener min<=max en cada escritura aunque una interrupcion deje un
    # rango fijo completamente distinto al original.
    if (( min_khz > current_max )); then
      echo "$max_khz" > "$cpu_dir/scaling_max_freq" || restore_failed=1
      echo "$min_khz" > "$cpu_dir/scaling_min_freq" || restore_failed=1
    elif (( max_khz < current_min )); then
      echo "$min_khz" > "$cpu_dir/scaling_min_freq" || restore_failed=1
      echo "$max_khz" > "$cpu_dir/scaling_max_freq" || restore_failed=1
    else
      echo "$max_khz" > "$cpu_dir/scaling_max_freq" || restore_failed=1
      echo "$min_khz" > "$cpu_dir/scaling_min_freq" || restore_failed=1
    fi
    # En pacca min/max son delegados al usuario, pero scaling_governor no
    # necesariamente es escribible. No convertir una restauracion ya
    # correcta en fallo por intentar reescribir el mismo valor; si de verdad
    # cambio, el intento se mantiene y la verificacion posterior falla
    # cerrado si el permiso no alcanza.
    if [[ "$current_governor" != "$governor" ]]; then
      echo "$governor" > "$cpu_dir/scaling_governor" || restore_failed=1
    fi

    actual_governor="$(<"$cpu_dir/scaling_governor")"
    actual_min="$(<"$cpu_dir/scaling_min_freq")"
    actual_max="$(<"$cpu_dir/scaling_max_freq")"
    if [[ "$actual_governor" != "$governor" || "$actual_min" != "$min_khz" || "$actual_max" != "$max_khz" ]]; then
      echo "E01: restauracion CPU no verificada en cpu${c} (esperado ${governor}/${min_khz}/${max_khz}, observado ${actual_governor}/${actual_min}/${actual_max})" >&2
      restore_failed=1
    else
      echo "CPU_FREQ_STATE_RESTORED: cpu${c} governor=${actual_governor} min=${actual_min} max=${actual_max}"
    fi
  done < "$state_snapshot"
  return "$restore_failed"
}

finish() {
  status=$?
  trap - EXIT INT TERM
  if ! sudo -n "$helper" "$initial_state"; then
    echo "E01: no se pudo restaurar no_turbo=$initial_state" >&2
    [[ $status -eq 0 ]] && status=70
  fi
  restored_state="$(<"$state_file")"
  if [[ "$restored_state" != "$initial_state" ]]; then
    echo "E01: restauración de Turbo no verificada (esperado=$initial_state, observado=$restored_state)" >&2
    [[ $status -eq 0 ]] && status=70
  else
    echo "TURBO_STATE_RESTORED: no_turbo=${restored_state}"
  fi
  if ! restore_cpu_state; then
    echo "E01: no se pudo restaurar exactamente governor/min/max de todos los CPU fotografiados" >&2
    [[ $status -eq 0 ]] && status=70
  fi
  rm -f "$state_snapshot"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! sudo -n "$helper" 1; then
  echo "E01: el helper existe, pero sudo no permite ejecutarlo sin contraseña" >&2
  exit 69
fi
applied_state="$(<"$state_file")"
if [[ "$applied_state" != "1" ]]; then
  echo "E01: set_turbo_state 1 no dejó no_turbo=1 (observado=$applied_state)" >&2
  exit 70
fi

"$@"
command_status=$?

final_active_state="$(<"$state_file")"
if [[ "$final_active_state" != "1" ]]; then
  echo "E01: no_turbo cambió durante la ejecución (observado=$final_active_state)" >&2
  [[ $command_status -eq 0 ]] && command_status=70
fi

exit "$command_status"
