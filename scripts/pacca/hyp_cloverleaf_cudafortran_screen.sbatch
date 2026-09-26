#!/bin/bash
# CloverLeaf CUDA Fortran: compilacion oficial fijada, validacion y pilotos REF/sombra.
# Job exclusivo en paccaA100; sin --time intencionalmente.
#SBATCH --job-name=hyp_cloverleaf_screen
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/latorresn/hyperion-results/final/logs/%x_%j.out
#SBATCH --error=/home/latorresn/hyperion-results/final/logs/%x_%j.err
set -euo pipefail
ROOT="${HYPERION_ROOT:-$HOME/hyperion_c8}"
OUT="${OUT:-$HOME/hyperion-results/final/cloverleaf_cudafortran_screen_${SLURM_JOB_ID}}"
SRC="$HOME/hyperion-kernels/external_src/CloverLeaf_CUDAFortran"
COMMIT=6558a980d6094a70e815d445d0f561c11c0c8ef6
CUDA=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0
# Algunos modulefiles del cluster expanden LD_PRELOAD sin valor bajo `set -u`.
# El compilador se referencia por ruta absoluta para que ese detalle no altere
# la compilación reproducible.
set +u; module load cmake/4.3.4 gnu12/12.4.0 devtools/nvidia/hpc_sdk/nvhpc/23.1 2>&1 || true; set -u
NVFORTRAN=/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvfortran
export LD_LIBRARY_PATH="$CUDA/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"; source "$HOME/hyperion-venv/bin/activate"
mkdir -p "$OUT" "$SRC"
hostname; "$NVFORTRAN" --version | head -2; "$CUDA/bin/nvcc" --version | tail -1
if [ ! -d "$SRC/.git" ]; then git clone https://github.com/UK-MAC/CloverLeaf_CUDAFortran.git "$SRC"; fi
git -C "$SRC" fetch --depth=1 origin "$COMMIT"; git -C "$SRC" checkout --detach "$COMMIT"; git -C "$SRC" rev-parse HEAD | tee "$OUT/source_commit.txt"
# API CUDA Fortran antigua: en NVHPC 23.1 `cudaGetStreamDefault()` fue
# retirada; stream=0 es el stream por defecto CUDA y conserva esa semántica.
# Se registra el diff para que la fuente oficial y la adaptación sean auditables.
sed -i 's/cudaGetStreamDefault()/0/g' "$SRC"/*.cuf
git -C "$SRC" diff -- '*.cuf' > "$OUT/nvhpc23_default_stream_compat.patch"
test -s "$OUT/nvhpc23_default_stream_compat.patch"
mkdir -p "$OUT/toolbin"; ln -sf "$NVFORTRAN" "$OUT/toolbin/pgfortran"
(cd "$SRC" && PATH="$OUT/toolbin:$PATH" make clean && PATH="$OUT/toolbin:$PATH" make FLAGS='-O3 -cuda -gpu=cc80,nordc') |& tee "$OUT/build.log"
test -x "$SRC/clover_leaf"
run_case() { local name=$1 input=$2; local d="$OUT/$name"; mkdir -p "$d"; cp "$input" "$d/clover.in"; (cd "$d" && /usr/bin/time -f 'wall_s=%e' "$SRC/clover_leaf") >"$d/stdout.log" 2>"$d/stderr.log"; grep -Ei 'error from|kinetic energy' "$d/clover.out" | tail -4 > "$d/check.txt" || true; }
run_case reference "$SRC/clover_bm16_short.in"
# Escala fijada antes de ver energia: mismo problema 3840x3840, 400 pasos (objetivo 1--3 min).
sed -e 's/end_time=0.5/end_time=15.5/' -e 's/end_step=87/end_step=400/' "$SRC/clover_bm16_short.in" > "$OUT/clover_screen.in"
run_case ref_duration "$OUT/clover_screen.in"
python3 - "$OUT" <<'PY'
import pathlib, re, sys
o=pathlib.Path(sys.argv[1]); txt=(o/'reference'/'clover.out').read_text(errors='replace')
if not re.search(r'kinetic energy', txt, re.I): raise SystemExit('referencia sin kinetic energy')
for d in ('reference','ref_duration'):
 t=(o/d/'stderr.log').read_text(errors='replace'); m=re.search(r'wall_s=([0-9.]+)',t); print(d, 'wall_s='+ (m.group(1) if m else 'NA'))
PY
echo "Solo se compilo, valido y calibro duracion. Los pilotos sombra requieren revisar el resultado de este job antes de iniciar mediciones con daemon."
