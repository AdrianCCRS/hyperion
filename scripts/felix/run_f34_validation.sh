#!/bin/bash
# F3.4: compila el harness C++ (cmake+gnu14), corre CTest, y ejecuta STREAM
# bajo telemetry_kernel_launcher --exec con el esquema de cores de F4.2
# (delegated_cpus=0-5, collector_cpu=6, consumer_cpu=7) para la validacion
# de bytes movidos. Ver docs/retoma/Informe_Piloto_F3_2026-07-31.md para la
# interpretacion de resultados (el sesgo de -33.8% encontrado y por que).
#
# Resultados en ~/hyperion-results/validation/<run_id>/.
set -euo pipefail

module load gnu14 2>&1 || true
module load cmake/3.29.3 2>&1 || true

cd ~/hyperion/telemetry
mkdir -p build-felix
cd build-felix
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. 2>&1 | tail -20
make -j8 2>&1 | tail -40

echo "=== CTest ==="
ctest --output-on-failure 2>&1 | tail -40

echo "=== F3.4: STREAM bajo el launcher ==="
RUN_ID="f34_stream_$(date +%s)"
OUT_DIR=~/hyperion-results/validation
mkdir -p "$OUT_DIR"

export OMP_NUM_THREADS=6
./telemetry_kernel_launcher \
  --exec ~/hyperion-kernels/bin/stream_c \
  --repetitions 1 \
  --perf-cpus 0-5 \
  --pin-workload-cpus 0-5 \
  --collector-cpu 6 \
  --consumer-cpu 7 \
  --interval-ns 1000000 \
  --output-dir "$OUT_DIR" \
  --run-id "$RUN_ID"

RUN_DIR="$OUT_DIR/$RUN_ID"
echo "=== metadata.json ==="
cat "$RUN_DIR/metadata.json"
echo "RUN_DIR_MARKER=$RUN_DIR"
echo "DONE"
