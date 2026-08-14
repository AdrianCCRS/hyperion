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
helper="$(command -v set_turbo_state 2>/dev/null || true)"

if [[ ! -r "$state_file" ]]; then
  echo "E01: no se puede leer $state_file" >&2
  exit 65
fi
if [[ -z "$helper" || ! -x "$helper" ]]; then
  echo "E01: set_turbo_state no está instalado o no es ejecutable" >&2
  exit 65
fi

initial_state="$(<"$state_file")"
if [[ "$initial_state" != "0" && "$initial_state" != "1" ]]; then
  echo "E01: estado inicial no_turbo inválido: $initial_state" >&2
  exit 65
fi

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
  fi
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
