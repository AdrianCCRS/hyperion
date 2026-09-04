# F1-GPU-002 — Probe de transición de reloj GPU y cadencia efectiva de NVML

Infraestructura reproducible para medir, **bajo carga real en la A100**:

- `T_actuacion = t_estable − t_solicitud` para una transición dirigida del
  reloj **graphics** que fija `nvidia-smi -lgc`, con `t_command_return`
  registrado por separado. El reloj SM se conserva como señal auxiliar; y
- la **cadencia efectiva observada** de las señales NVML (reloj SM, utilización,
  potencia, temperatura, energía), reportada siempre como *cota inferior de
  actualizaciones físicas*, nunca como tasa de refresco confirmada.

Ver `Seguimiento_Cambios_Plan_Director.md` § `F1-GPU-002` y
`Plan_Detallado_Realineacion_Hyperion.md` § 2.4.1 para el diseño y el porqué.

> Este módulo es independiente. No cambia la cadencia de las campañas
> existentes, ni el entrenador GPU, ni `derive_policy_table.py`. Solo produce
> el número que se le pasa a mano por `--t-transicion-gpu-ns`.

---

## 1. Componentes

| Archivo | Rol |
|---|---|
| `common/telemetry/include/telemetry/gpu_transition_analysis.hpp` | Lógica pura (detección de estabilidad, percentiles de cadencia, `T_actuacion`, análisis de escalones). Sin NVML/CUDA. Unit-tested sin GPU. |
| `common/telemetry/experiments/gpu_clock_transition_probe.cpp` | Ejecutable. Verifica actuación con `NVML_CLOCK_GRAPHICS` y exporta además `NVML_CLOCK_SM`; cada timestamp de estabilidad se toma después de leer graphics. `nvidia-smi` **solo** actúa (`-lgc`/`-rgc`) y usa `sudo -n` con timeout. |
| `common/telemetry/tests/test_gpu_transition_analysis.cpp` | Pruebas de la lógica pura (convergencia, primer toque + salida de tolerancia, timeout, timestamps irregulares, NVML ausente, throttling, `T_actuacion`). |
| `fase1_telemetria/gpu_transition/aggregate_transition_matrix.py` | Junta varios `gpu_clock_transition_summary.json` y deriva `T_transicion_gpu_ns_conservative` = **máximo** sobre pares y réplicas (nunca promedio). |
| `fase1_telemetria/tests/test_aggregate_transition_matrix.py` | Pruebas del agregador, incluidas matriz incompleta, dry-run y restauración no confirmada. |

Artefactos que produce **cada corrida** del probe (en `--out-dir`):

- `gpu_clock_transition_raw.csv` — una fila por lectura NVML: `seq`, fase
  (`pre_request`/`post_request`), timestamp monotónico, intervalo real desde la
  lectura previa, relojes graphics y SM, utilización, memoria, potencia, temperatura,
  energía, `throttle_reasons` (hex) — cada una con su bit `*_valid`.
- `gpu_clock_transition_summary.json` — metadatos (UUID/modelo GPU, driver,
  CUDA, clocks soportados, checksum y comando de carga, origen/destino,
  criterio de estabilidad), línea de tiempo (`t_solicitud`,
  `t_command_return`, `t_estable`, …), cadencia real (p50/p95/min/max de
  `delta_timestamp_ns`), análisis de escalones por señal, métricas de
  transición y `result` + `failure_reason` + estado de restauración.
- `gpu_clock_transition_matrix.csv` — una fila resumen de esa transición,
  pensada para concatenar entre corridas.

---

## 2. Build (en paccaA100, dentro de la asignación de Slurm)

```bash
# nodo con el driver NVIDIA + NVML; mismo tipo de nodo donde correrá la campaña
cmake -S common/telemetry -B common/telemetry/build -DWITH_GPU=ON
cmake --build common/telemetry/build -j --target gpu_clock_transition_probe

# (opcional) si el módulo CUDA del clúster no trae el symlink .so sin versión:
#   -DNVML_LIB=/ruta/libnvidia-ml.so.1 -DNVML_INCLUDE_DIR=/ruta/include
```

Sin `-DWITH_GPU=ON` el binario compila igual pero al ejecutarse solo imprime
que necesita NVML y sale con código 2 (esto es lo que corre en CI).

Verificar la lógica pura (no necesita GPU):

```bash
cmake --build common/telemetry/build --target gpu_transition_analysis_test
ctest --test-dir common/telemetry/build -R 'gpu_transition_analysis|gpu_clock_controller'
```

---

## 3. Requisitos de la carga CUDA sostenida

- Debe mantener la GPU **activa** (util ≥ `--active-util-threshold-pct`, por
  defecto 5 %) durante, al menos, `--warmup-ns` + `--workload-min-active-ns` +
  `--max-wait-ns` + margen. Sugerencia inicial: warmup 2 s, activa 6 s,
  max-wait 3 s → carga de ≥ 12 s.
- Debe ser un binario de terceros sin instrumentar (p. ej. un kernel GPU del
  catálogo en bucle, o `cuda-samples`/`gpu-burn`). El probe la lanza vía
  `sh -c "<cmd>"`, la vigila por NVML y la termina al final (SIGTERM→SIGKILL).
- Si la carga no alcanza actividad sostenida, o el reloj no se estabiliza, el
  probe **falla explícitamente**, conserva el crudo y escribe `result` +
  `failure_reason`.

---

## 4. Procedimiento en paccaA100

### 4.0 Descubrir los clocks soportados

```bash
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" -q -d SUPPORTED_CLOCKS | sed -n '1,40p'
```

o correr el probe una vez y leer `supported_sm_clocks_mhz` del summary JSON.
El probe **ajusta** `--from-clock`/`--to-clock` al soportado más cercano y lo
anota; la **tolerancia** debe ser menor que la mitad del salto al soportado
vecino, o el probe aborta.

### 4.1 Etapa A — cadencia efectiva (elegir `q_produccion`)

El bloque `observed_cadence` + `signal_step_analysis` del summary ya da, para
los relojes **graphics y SM**, la distribución real de `delta_timestamp_ns`, los cambios
consecutivos (cota inferior) y la duración de escalones. Para cubrir el resto
de señales (util, potencia, memoria, temperatura, energía) a varias cadencias,
correr el probe con `--dry-run-actuation` a `--probe-interval-ns` de
5, 10, 50 y 100 ms sobre la misma carga y comparar el `signal_step_analysis` de
cada una:

```bash
for Q in 5000000 10000000 50000000 100000000; do
  ./common/telemetry/build/gpu_clock_transition_probe \
    --workload-cmd "$WORKLOAD" --gpu "$CUDA_VISIBLE_DEVICES" \
    --from-clock REF --to-clock "$MID_MHZ" --tolerance-mhz 15 \
    --probe-interval-ns "$Q" --dry-run-actuation \
    --warmup-ns 2000000000 --workload-min-active-ns 6000000000 --max-wait-ns 1000000000 \
    --out-dir "$RESULTS/etapaA/q_${Q}"
done
```

`q_produccion` = la cadencia más gruesa que, frente a 5 ms, no pierde bordes
observables ni reduce materialmente los escalones post-warmup en la carga más
corta. **Esto no cambia ninguna campaña**: es una entrada para una decisión
futura, versionada aparte.

> `--dry-run-actuation` **no** produce una medición de transición válida (no
> toca el reloj). Es solo para la caracterización de cadencia de la Etapa A.

### 4.2 Etapa B — matriz de transiciones

Para **cada par dirigido** que la política puede solicitar, y para
**`REF → cada nivel fijo candidato` por separado**, con **≥ 3 réplicas**:

```bash
WORKLOAD='<comando de carga CUDA sostenida>'
RESULTS="$HOME/hyperion-results/gpu_transition/$(date +%Y%m%d)"

run_pair () {   # $1=from  $2=to_mhz  $3=label
  for R in 1 2 3; do
    ./common/telemetry/build/gpu_clock_transition_probe \
      --workload-cmd "$WORKLOAD" \
      --gpu "$CUDA_VISIBLE_DEVICES" \
      --from-clock "$1" --to-clock "$2" \
      --tolerance-mhz 15 \
      --stable-consecutive 3 \
      --probe-interval-ns 5000000 \
      --warmup-ns 2000000000 \
      --workload-min-active-ns 6000000000 \
      --request-at-ns 5000000000 \
      --max-wait-ns 3000000000 \
      --replicate-id "$R" \
      --label "$3" \
      --out-dir "$RESULTS/${3}/r${R}"
  done
}

# Ejemplo con los niveles de las campañas GPU (REF / F3≈mid / F6≈min).
# Sustituir MID_MHZ y MIN_MHZ por soportados reales del nodo.
run_pair REF        "$MID_MHZ" "REF_to_F3"
run_pair REF        "$MIN_MHZ" "REF_to_F6"
run_pair "$MID_MHZ" "$MIN_MHZ" "F3_to_F6"
run_pair "$MIN_MHZ" "$MID_MHZ" "F6_to_F3"
```

Reglas (del plan y del `Seguimiento`):

- La **dirección importa**: `A→B` y `B→A` son pares distintos, medir ambos si
  la política puede pedir ambos.
- `REF → fijo` va **por separado** de los pares fijo↔fijo.
- La **tolerancia** se documenta en el summary; si la primera lectura estable
  cae al límite de la resolución de `gpu_sm_clock_mhz`, el resultado es una
  **cota superior observable**, no una latencia exacta — y sigue siendo válida
  para `min_dwell_ns`.
- El probe **restaura** el reloj (`nvidia-smi -rgc`) al terminar, ante SIGINT,
  fallo de comando o timeout. Una segunda señal no salta la restauración.

### 4.3 Agregar y derivar la cota conservadora

```bash
python3 -m fase1_telemetria.gpu_transition.aggregate_transition_matrix \
    "$RESULTS" \
    --require-pair "REF->${MID_MHZ}" \
    --require-pair "REF->${MIN_MHZ}" \
    --require-pair "${MID_MHZ}->${MIN_MHZ}" \
    --require-pair "${MIN_MHZ}->${MID_MHZ}" \
    --output "$RESULTS/transition_matrix_aggregate.json"
```

Imprime la matriz por par, los avisos (réplicas < 3, timeouts, resultados no
estables) y:

```
T_transicion_gpu_ns_conservative = <N>  (cota superior observable; par peor: <A->B>)
```

`<N>` es el **máximo** de `conservative_upper_bound_ns` sobre todos los pares y
réplicas estables — nunca un promedio ni un percentil. El agregador falla
cerrado para política si falta un par declarado, hay timeout/dry-run, la
restauración no fue confirmada o cambian UUID, driver o checksum de carga.

### 4.4 Alimentar la derivación de política

```bash
python3 fase3_daemon/policy/derive_policy_table.py \
    <windows.csv de la campaña de barrido GPU...> \
    --t-transicion-gpu-ns <N> \
    --output policy_table.yaml
```

- Si `<N>` es **comparable o mayor** que la duración de las corridas/fases
  elegibles del barrido → `derive_policy_table.py` deja GPU en `no_actuar`
  (resultado válido; el motivo queda distinguido de "rango de potencia
  angosto").
- Si es **menor** → el filtro `filter_gpu_transition_not_settled` excluye las
  filas cuyo `run_id` dura menos que `<N>`, conservando el motivo.
- `min_dwell_ns` del daemon se fija **como mínimo** a `<N>`. Un multiplicador
  extra solo tras medir sensibilidad/histéresis, nunca como constante
  implícita.

### 4.5 Criterio de salida (del `Seguimiento`)

El barrido GPU completo solo arranca tras guardar, versionados: (a) la
selección de `q_produccion` con su reporte de Etapa A, y (b) el
`transition_matrix_aggregate.json` reproducible. O bien tras documentar que la
resolución A100/driver solo permite una cota tan alta que la actuación GPU no
es viable. En ambos casos la política GPU **no** pasa silenciosamente a
`actuar`.

---

## 5. Qué queda por validar en hardware (NO verificado desde el entorno local)

Lo implementado aquí se probó en un entorno **sin GPU**:

- ✅ Build CPU-only del probe (imprime el aviso de `-DWITH_GPU` y sale 2).
- ✅ `gpu_transition_analysis_test` — 8 grupos de casos de la lógica pura.
- ✅ `test_aggregate_transition_matrix.py` — 9 casos del agregador.
- ✅ Suite completa `common/telemetry` (`ctest`, 15/15) y
  `pytest fase1_telemetria common` (530).

Pendiente en paccaA100 con NVML real:

1. Build `-DWITH_GPU=ON` contra el `nvml.h` / `libnvidia-ml` del nodo
   (verificar que `nvmlDeviceGetCurrentClocksThrottleReasons`,
   `nvmlDeviceGetSupportedGraphicsClocks` y
   `nvmlSystemGetCudaDriverVersion_v2` resuelven con esa versión de driver).
2. Confirmar que `sudo nvidia-smi -i <sel> -lgc/-rgc` funciona sin contraseña
   bajo la asignación de Slurm (mismo permiso que usa `gpu_freqctl.py`).
3. Elegir una carga CUDA sostenida real y calibrar `--warmup-ns` /
   `--workload-min-active-ns` / `--request-at-ns` a su duración.
4. Ejecutar la Etapa A y fijar `q_produccion`.
5. Ejecutar la Etapa B (matriz dirigida, ≥ 3 réplicas) y agregar.
6. Registrar el resultado (número o bloqueo) en `Seguimiento` § `F1-GPU-002`
   y, si procede, pasar `<N>` a `derive_policy_table.py`.
