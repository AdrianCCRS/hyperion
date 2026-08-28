#!/bin/bash
# Recompila los 12 binarios del selector CPU/GPU y resincroniza la CADENA
# COMPLETA de checksums, que tiene dos niveles y por eso se rompio antes
# (2026-08-27, job 6669: una recompilacion cambio 4 binarios GPU sin cambio
# de fuente y dejo 4 wrappers apuntando a un checksum viejo, en silencio):
#
#   libexec/dual/<name>   binario real       -> su sha va DENTRO del wrapper
#   bin/<name>            wrapper de guarda  -> su sha va en catalog.yaml
#
# Recompilar cambia (1), lo que obliga a regenerar (2), lo que a su vez
# cambia el sha del wrapper y obliga a parchear el catalogo. Hacer solo uno
# de los tres pasos deja el sistema roto de una forma que no falla hasta la
# mitad de una campana. Este script hace los tres y luego VERIFICA.
#
# ARC-126: `set -e -o pipefail` SIN `-u` -- Lmod referencia variables no
# definidas y `-u` aborta al cargar cualquier modulo.
set -e -o pipefail

REPO="${REPO:-$HOME/hyperion}"
BIN_DIR="${BIN_DIR:-$HOME/hyperion-kernels/bin}"
LIB_DIR="${LIB_DIR:-$HOME/hyperion-kernels/libexec/dual}"
CPU_LIBS="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:/opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib"
NAMES=(gemm_cpu gemm_gpu fft_cpu fft_gpu axpy_cpu axpy_gpu
       stencil_cpu stencil_gpu cholesky_cpu cholesky_gpu spmv_cpu spmv_gpu)

echo "== 1/4: compilando los 12 binarios =="
bash "$REPO/scripts/pacca/build_dual_kernels.sh"

echo
echo "== 2/4: regenerando wrappers con los checksums nuevos =="
mkdir -p "$BIN_DIR"
for name in "${NAMES[@]}"; do
  sha="$(sha256sum "$LIB_DIR/$name" | awk '{print $1}')"
  cat > "$BIN_DIR/$name" <<WRAP
#!/bin/bash
set -euo pipefail
binary="$LIB_DIR/$name"
expected_binary_sha256="$sha"
actual_binary_sha256="\$(sha256sum "\$binary" | awk '{print \$1}')"
if [[ "\$actual_binary_sha256" != "\$expected_binary_sha256" ]]; then
  echo "ERROR: checksum de $name no coincide (\$actual_binary_sha256)" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$CPU_LIBS:\${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=6 OMP_PROC_BIND=true OMP_PLACES=cores
run_dir="\$(mktemp -d -p "\$HOME" hyperion_${name}_run_XXXXXX)"
trap 'rm -rf -- "\$run_dir"' EXIT
cd "\$run_dir"
"\$binary" "\$@"
WRAP
  chmod +x "$BIN_DIR/$name"
  echo "  wrapper $name -> binario $sha"
done

echo
echo "== 3/4: parcheando binary_checksum en catalog.yaml =="
python3 - "$REPO" "$BIN_DIR" <<'PYEOF'
import hashlib, pathlib, re, sys

repo, bin_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
names = ["gemm_cpu","gemm_gpu","fft_cpu","fft_gpu","axpy_cpu","axpy_gpu",
         "stencil_cpu","stencil_gpu","cholesky_cpu","cholesky_gpu","spmv_cpu","spmv_gpu"]
sha = {n: hashlib.sha256((bin_dir/n).read_bytes()).hexdigest() for n in names}

path = repo/"orchestrator/schemas/kernels/catalog.yaml"
lines = path.read_text().splitlines(keepends=True)

# Recorre entradas: recuerda el ultimo exec_path visto y, al llegar a la linea
# pacca-a100 de esa entrada, la reescribe con el sha del wrapper que le toca.
# Solo toca entradas cuyo exec_path sea uno de los 12 -- el resto del catalogo
# (kernels de Fase 1, calibraciones) queda intacto.
current, patched = None, 0
for i, line in enumerate(lines):
    m = re.match(r'\s*exec_path:\s*bin/(\S+)\s*$', line)
    if m:
        current = m.group(1)
        continue
    if current in sha:
        m2 = re.match(r'(\s*pacca-a100:\s*")sha256:[0-9a-f]{64}(")\s*$', line)
        if m2:
            lines[i] = f"{m2.group(1)}sha256:{sha[current]}{m2.group(2)}\n"
            patched += 1
            current = None
path.write_text("".join(lines))
print(f"  entradas parcheadas: {patched}")
for n in names:
    print(f"    {n}: {sha[n]}")
PYEOF

echo
echo "== 4/4: verificando que el catalogo carga y los checksums cuadran =="
cd "$REPO"
PYTHONPATH="$REPO" python3 - "$BIN_DIR" <<'PYEOF'
import hashlib, pathlib, sys
from orchestrator.catalog import load_catalog

bin_dir = pathlib.Path(sys.argv[1])
cat = load_catalog("orchestrator/schemas/kernels/catalog.yaml")
bad = []
for k in cat.values():
    if not k.exec_path.startswith("bin/"):
        continue
    f = bin_dir/k.exec_path[4:]
    declared = (k.binary_checksum or {}).get("pacca-a100")
    if not f.exists() or not declared:
        continue
    real = "sha256:" + hashlib.sha256(f.read_bytes()).hexdigest()
    if real != declared:
        bad.append((k.id, k.exec_path, declared, real))
print(f"  catalogo carga OK: {len(cat)} kernels")
if bad:
    print(f"  DESAJUSTES: {len(bad)}")
    for b in bad[:10]:
        print("   ", b)
    sys.exit(1)
print("  todos los binary_checksum presentes cuadran con el disco")
PYEOF

echo
echo "REBUILD_AND_RESYNC_OK"
