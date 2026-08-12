#!/bin/bash
#
# Campana completa de caracterizacion Roofline con Intel Advisor, en
# paccaA100. Ver docs/advisor/pipeline_advisor_diseno_y_arquitectura.md
# para el diseno completo y docs/advisor/estudio_intel_advisor_roofline.md
# para la metodologia de medicion.
#
# Regla dura: nunca cancela, señaliza ni interfiere con trabajos de otros
# usuarios. Si el nodo esta ocupado, este job se queda en cola.
#
#SBATCH --job-name=advisor_roofline_characterization
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --output=/home/%u/raperezp/logs/slurm_%j.out
#SBATCH --error=/home/%u/raperezp/logs/slurm_%j.err
#
# --- Datos reales de tiempo acumulados hasta 2026-08-11 (ver README.md
# "Datos reales de tiempo por clase" para la tabla completa y como se
# actualiza) -- el default de este script (clase A) es el unico con
# cobertura completa (6/6 kernels). Clase C se piloteo parcialmente (job
# 5024): ep.C y cg.C completaron real (6.0 min y 23.7 min en la pasada
# tripcounts+cache-sim respectivamente), bt.C SUPERO 1800s y se tuvo que
# matar (dato real: "no cabe en 30 min", no un numero exacto). ft/lu/mg.C
# quedaron sin correr. NO extrapolar estos numeros a ciegas para B -- B es
# el turno pendiente de medir, ver el calculo de --time mas abajo.
#
# --- Calculo de --time=10:00:00 (clase A, 6 kernels x 2 repeticiones) -----
#   Techo conservador por unidad (no un promedio esperado):
#     baseline clase A       <= 60s   (EP.A real=1.2s; otros kernels A
#                                       podrian ser mas grandes, nunca
#                                       medidos, se deja margen amplio)
#     factor de sobrecosto total del pipeline completo (4 pasadas: survey+
#     tripcounts sin simular, survey+tripcounts con --enable-cache-simulation)
#     ~= 40x  (17x medido para SOLO tripcounts+cache-sim; se redondea hacia
#              arriba para cubrir las otras 3 pasadas, ninguna medida
#              todavia de forma aislada)
#     => techo por unidad ~= 60s x 40 = 2400s = 40 min
#
#   12 unidades (6 kernels x 2 repeticiones) x 40 min      = 480 min
#   Generacion de 5 reportes x 12 unidades (rapido, <1min c/u, margen 20%)= 96 min
#   Preflight + arranque de Slurm/entorno                   =  15 min
#   Total                                                    ~= 591 min ~= 9h51
#   Redondeado a 10h.
#
#   Este --time se pasa por linea de comandos al enviar (ver README.md),
#   no hace falta editar este archivo para cambiar clase/kernels/tiempo.

set -eo pipefail
# NUNCA "set -u" aqui -- el propio script de inicializacion de Lmod
# (init/bash) referencia $LD_PRELOAD sin comprobar si esta definida, y bajo
# nounset eso aborta "module load" en seco, en silencio (confirmado en
# corrida real, job 5023: 'module load advisor/2023.0.0' fallaba con
# "LD_PRELOAD: unbound variable" y el resto del script seguia como si nada,
# hasta que "advisor" resultaba "command not found" varios pasos despues).

echo "=== Job Slurm $SLURM_JOB_ID en $(hostname), $(date -u +%FT%TZ) ==="

module purge
module load devtools/intel/oneapi/2023
module load advisor/2023.0.0

echo "advisor: $(command -v advisor || echo NO_ENCONTRADO)"
advisor --version || true

NPB_ROOT="${ADVISOR_NPB_ROOT:-$HOME/raperezp/NPB3.4-OMP}"
BIN_DIR="${ADVISOR_BIN_DIR:-$NPB_ROOT/bin}"
CONFIG_DIR="${ADVISOR_CONFIG_DIR:-$NPB_ROOT/config}"
OUTPUT_DIR="${ADVISOR_OUTPUT_DIR:-$HOME/raperezp/results/job_${SLURM_JOB_ID}}"
REPETITIONS="${ADVISOR_REPETITIONS:-2}"
KERNELS="${ADVISOR_KERNELS:-ep,cg,mg,ft,lu,bt}"
CLASSES="${ADVISOR_CLASSES:-A}"
# 1800s (default de run_characterization.py) resulto insuficiente para bt.C
# real (job 5024, 2026-08-11: broke el timeout, elapsed real desconocido
# porque se mato a los 1800s). 3600s es el nuevo default de ESTE script
# (no del argparse de run_characterization.py, que se deja en 1800 para
# quien lo corra suelto) -- generoso pero no arbitrario: cg.C real tardo
# 1422s en la pasada pesada, bt.C supero 1800s. Ver README.md para cuando
# conviene subirlo mas (clase C con kernels pesados) o se puede bajar
# (clase A, todo corre en <100s).
TIMEOUT="${ADVISOR_TIMEOUT:-3600}"

echo "BIN_DIR=$BIN_DIR"
echo "CONFIG_DIR=$CONFIG_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "KERNELS=$KERNELS CLASSES=$CLASSES REPETITIONS=$REPETITIONS TIMEOUT=$TIMEOUT"

mkdir -p "$OUTPUT_DIR"
# NUNCA "${BASH_SOURCE[0]}" aqui -- Slurm copia el script enviado a un
# directorio de spool efimero (/var/spool/slurmd/jobNNNNN/) antes de
# ejecutarlo, asi que BASH_SOURCE apunta ahi, no al directorio real del
# pipeline (confirmado en la misma corrida real: "run_characterization.py:
# No such file or directory" buscandolo en el spool). SLURM_SUBMIT_DIR es
# la variable correcta para volver al directorio real de envio.
cd "$SLURM_SUBMIT_DIR"

python3 run_characterization.py \
  --bin-dir "$BIN_DIR" \
  --config-dir "$CONFIG_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --kernels "$KERNELS" \
  --classes "$CLASSES" \
  --repetitions "$REPETITIONS" \
  --core-range 0-5 \
  --threads 6 \
  --timeout "$TIMEOUT"

status=$?
echo "=== run_characterization.py termino con codigo $status ==="
echo "Resultados en: $OUTPUT_DIR"
echo "CSV de kernels: $OUTPUT_DIR/consolidated_characterization.csv"
echo "CSV de loops:   $OUTPUT_DIR/consolidated_loops.csv"
exit $status
