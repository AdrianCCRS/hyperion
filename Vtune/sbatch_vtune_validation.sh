#!/bin/bash
#
# Campana de validacion cruzada VTune (Microarchitecture Exploration) para
# Hyperion, en paccaA100 (Cartagena). Ver docs/vtune/vtune_cross_validation.md
# para la metodologia completa y Vtune/README.md para como lanzar y revisar
# esta campana.
#
# Regla dura: este job NUNCA cancela, señaliza ni interfiere con trabajos de
# otros usuarios. Si el nodo esta ocupado, este job se queda en cola -- eso
# es exactamente lo que --nodelist=paccaA100 sin --begin/--immediate produce
# por default en Slurm, no hace falta logica adicional para lograrlo.
#
#SBATCH --job-name=vtune_uarch_validation
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/vtune_validation/logs/slurm_%j.out
#SBATCH --error=/home/%u/vtune_validation/logs/slurm_%j.err

# --- Sobre --output/--error con %u en vez de una ruta fija -----------------
# La plantilla original traia "/home/latorres/salida_%j.out" (sin la 'n'
# final). El resto de la documentacion ya presente en el repo
# (pipelinevtune/CLAUDE.md) registra el usuario remoto como "latorresn"
# ("/home/latorresn/vtune_selfcheck/"). No se puede resolver esa
# discrepancia por inspeccion de codigo -- son dos strings distintos, no un
# typo evidente de uno solo. En vez de adivinar, este script usa el
# placeholder %u de Slurm (usuario que envia el job), que resuelve el
# directorio correcto sin importar cual de los dos es el real. Si tu $HOME
# remoto no es literalmente /home/<usuario>, ajusta --output/--error a mano
# antes de enviar.
#
# --- Sobre --gres=gpu:1 en una campana CPU-only ----------------------------
# Esta campana no usa la GPU. Se mantiene --gres=gpu:1 junto con --exclusive
# porque es el patron de reserva ya usado y confirmado en este nodo/particion
# para este proyecto (ver docs/retoma/pacca/Auditoria_PaccaA100_Unicartagena.md
# seccion "Acceso") -- --exclusive por si solo ya entrega el nodo completo
# (CPUs + GPU), pero desviarse del patron de reserva ya probado en un cluster
# compartido sin necesidad no aporta nada y sí introduce una variable no
# probada. Si se confirma que --gres=gpu:1 es innecesario en esta particion,
# es un cambio de una linea, no estructural.
#
# --- Calculo de --time=06:00:00 (NO es un numero arbitrario) ---------------
# Alcance de esta campana (ver Vtune/README.md y run_validation.py):
#   6 kernels NPB (ep,cg,mg,ft,lu,bt) x clase C x 2 repeticiones = 12 unidades
#   2 anclas (STREAM, DGEMM)         x 2 repeticiones           =  4 unidades
#   Total: 16 "unidades de carga de trabajo", cada una = 1 corrida baseline
#   (sin VTune) + 1 coleccion 'uarch-exploration' + generacion de reportes.
#
# Presupuesto por unidad (techo conservador, no un promedio esperado):
#   - baseline:            <= 180 s  (el unico dato real medido en este
#                           proyecto es EP clase C = 13.6 s bajo un analisis
#                           de peso similar -hpc-performance-, ver
#                           pipelinevtune/context/04_vtune_selfchecker_resultados.md;
#                           180 s deja margen amplio para MG/FT/LU/BT/CG,
#                           mas pesados que EP y sin medicion previa en
#                           este nodo especifico)
#   - vtune -collect uarch-exploration: <= 300 s (180 s x 1.5: uarch-exploration
#                           arma mas eventos de PMU simultaneos que hpc-performance/
#                           hotspots -- fracción con multiplexacion mas probable,
#                           ver docs/vtune/vtune_cross_validation.md seccion E)
#   - reportes (summary texto+csv, hw-events csv): <= 30 s
#   => techo por unidad ~= 510 s, redondeado a 600 s (10 min) por unidad
#
#   16 unidades x 10 min                    = 160 min
#   Preflight (modulos + smoke uarch-exploration real) =  10 min
#   Arranque de Slurm/entorno                =   5 min
#   Subtotal                                 = 175 min
#
# Factor de seguridad x2 sobre el subtotal, porque uarch-exploration NUNCA se
# corrio con exito en este nodo antes de este permiso (a diferencia de
# hotspots/hpc-performance, que si tienen tiempos reales medidos) -- mismo
# principio que RUN-03 del orquestador principal (timeout >= 3x lo esperado,
# nunca confiar en "deberia funcionar" sin margen, ver
# docs/retoma/Guia_Maestra_Fase1_DVFS.md seccion 10.3):
#   175 min x 2 ~= 350 min ~= 5h50 -> redondeado a 6h.
#
# Si la primera corrida real confirma tiempos mucho menores (probable, dado
# que el techo de arriba es deliberadamente generoso), reducir --time en
# corridas futuras usando el dato empirico real en vez de este calculo --
# ver campaign_metadata / consolidated_validation.csv de la corrida ya hecha.

set -euo pipefail

echo "=== Job Slurm $SLURM_JOB_ID en $(hostname), $(date -u +%FT%TZ) ==="

# --- Modulos: secuencia confirmada por el proyecto (modulo padre primero,
# el modulo vtune es jerarquico y no aparece sin el, ver
# pipelinevtune/context/04_vtune_selfchecker_resultados.md) -----------------
module purge
module load devtools/intel/oneapi/2023
module load vtune/2023.0.0

echo "vtune: $(command -v vtune || echo NO_ENCONTRADO)"
vtune --version || true

# --- Rutas: mismo convenio remoto que pipelinevtune (context/00), pero
# resueltas via $HOME en vez de hardcodeadas, por la misma razon que
# --output/--error arriba -----------------------------------------------
WORKDIR="${VTUNE_VALIDATION_WORKDIR:-$HOME/vtune_selfcheck}"
BIN_DIR="${VTUNE_VALIDATION_BIN_DIR:-$WORKDIR/bin}"
ANCHOR_DIR="${VTUNE_VALIDATION_ANCHOR_DIR:-$WORKDIR/anchor_bin}"
OUTPUT_DIR="${VTUNE_VALIDATION_OUTPUT_DIR:-$HOME/vtune_validation/results_vtune/job_${SLURM_JOB_ID}}"
REPETITIONS="${VTUNE_VALIDATION_REPETITIONS:-2}"
KERNELS="${VTUNE_VALIDATION_KERNELS:-ep,cg,mg,ft,lu,bt}"
CLASSES="${VTUNE_VALIDATION_CLASSES:-C}"

echo "BIN_DIR=$BIN_DIR"
echo "ANCHOR_DIR=$ANCHOR_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

cd "$(dirname "${BASH_SOURCE[0]}")"

python3 run_validation.py \
  --bin-dir "$BIN_DIR" \
  --anchor-dir "$ANCHOR_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --kernels "$KERNELS" \
  --classes "$CLASSES" \
  --repetitions "$REPETITIONS" \
  --core-range 0-5 \
  --threads 6

status=$?
echo "=== run_validation.py termino con codigo $status ==="
echo "Resultados en: $OUTPUT_DIR"
echo "CSV consolidado: $OUTPUT_DIR/consolidated_validation.csv"
exit $status
