# Estado de tareas — sesión 2026-09-15/16 (mejora clasificadores CPU y GPU)

Verificado contra `git log origin/main` y `squeue` reales. Hay OTRO agente
trabajando en paralelo sobre `paccaA100` (mismo repo, mismo `job 7144`) —
lo que sigue incluye SU trabajo cuando quedó confirmado en commits reales.

---

## MODELO CPU

### Ejecutado, con resultado real
- **Campaña final CPU**: 1290/1290 corridas reales. Cerrada.
- **Calibración warmup CPU** (43 kernels, `measure_warmup.py` sobre `windows.csv` real): tabla completa. Warmup real CPU minúsculo (máx 4.8%, mayoría <2%). `catalog.yaml` actualizado (commit `3c9fd42`).
- **Reprocesamiento CPU con warmup calibrado**: `COMPLETED, exit 0` (job `7360`, confirmado por `sacct`). Dataset regenerado.
- **`srad`/`particlefilter` agregados a `catalog.yaml`** (commit `937237f`): tamaño, warmup (0.0036s / 0.0096s) y checksum medidos en hardware real. `particlefilter` reveló ~6x de overhead bajo instrumentación completa (`perf` attach) — tuvo que recalibrarse el tamaño por esto. **Pendiente**: agregarlos a `cpu_final.yaml` para que entren al dataset de entrenamiento.

### Código escrito, sin ejecución real que confirme el efecto todavía
- **Fix `scale_pos_weight` XGBoost por pliegue** (commit `cb84d33`): aplica también a CPU. Sin corrida completa exitosa con el código nuevo todavía.

### Corriendo ahora mismo
- **`7355` (`hyp_cpu_optuna`)**: búsqueda de hiperparámetros, 4h11m activo (confirmado con uso real de CPU, no colgado). Corre con el código VIEJO de `scale_pos_weight` (antes de `cb84d33`) — su resultado no reflejará el fix.

### Bloqueado / no empezado
- **`kmeans` CPU**: mismo bloqueo que la variante GPU (ver abajo) — falta el archivo `kdd_cup`, no vendorizado en el clúster.
- **Análisis completo de resultados CPU**: pendiente hasta que cierren `7355` y el reentrenamiento.

---

## MODELO GPU

Contexto: el problema real de GPU es la **escasez de `compute_bound`** (73 de 334 corridas, 5 kernels). Nada de esta sección aplica a CPU.

### Ejecutado, con resultado real
- **Verificación parche overflow `nw`** (N=47008, hardware real): sin crash. (No aporta al catálogo: `nw` es aritmética entera, sin FLOPs, descalificado del catálogo Roofline en ambos dispositivos.)
- **Perfilado por-lanzamiento `gaussian`** (`ncu` real, Fan1 vs Fan2): OI 0.153/0.284 FLOP/byte. NEGATIVO, ambos memory_bound.
- **Extensión `lavamd`** (boxes1d 70→100, hardware real): warmup 6.26s/9.10s (68.8%). NEGATIVO, no mejora.
- **`dwt2d` a REF** (hardware real): warmup 1.26s/1.69s. INCONCLUSO, falta probar a frecuencia reducida.
- **RAJAPerf `Apps_LTIMES`** como candidato `compute_bound` (hardware real, `ncu` convergió): OI 0.696 FLOP/byte FP64. NEGATIVO, memory_bound.
- **`srad`/`nw` GPU vía `ncu`**: falla `LaunchFailed`/CUDA error 700, cualquier tamaño. Sin diagnosticar.
- **[OTRO AGENTE] CUTLASS SIMT Conv2d fprop** como candidato `compute_bound` (commit `792b645`): **OI=438.06 FLOP/byte medida y verificada** (incluye chequeo anti-artefacto-de-caché L2). Warmup real: 1.1737s. **La ancla `compute_bound` más profunda del catálogo hasta ahora.** Agregado a `gpu_final.yaml` (commit `54e6b10`) — todavía NO corrido dentro de una campaña completa.

### Código escrito, EN EJECUCIÓN ahora mismo (ver abajo)
- **F1 macro sobre pliegues mixtos** (commit `f4f6d19` + fix `033dde0`): código listo, corriendo por primera vez en `7366` — todavía no termina, cero resultado numérico hasta que cierre.
- **`T_transición_gpu` wireado** en `gpu_phases.py` (commit `cea8935`, otro agente): código listo, no verificado en esta sesión que ya se haya ejercitado con datos reales.

### Corriendo ahora mismo
- **`7366` (`hyp_gpu_mixedf2`)**: reentrenamiento con fix `scale_pos_weight` + F1 mixtos, primera corrida real de ambos, LANZADA y corriendo. 1h16m+ activo (confirmado con uso real de CPU, no colgado). Falta que termine para tener el número.

### Ejecutado, con resultado real (noche 2026-09-16, tras cerrar `gpu_conv2d_extra` 30/30)
- **`kmeans` desbloqueado**: `kdd_cup` no vendorizado → generador sintético propio (`gen_kmeans_input.py`, N puntos ~ K centros gaussianos reales). Encontrado y corregido: pedir más clusters (`-k`) de los que el generador realmente crea hace oscilar el criterio de parada de `kmeans_clustering.c` sin converger (probado con `k=15`/`k=50` sobre datos de 5 clusters reales, timeout >120s ambos). Con K=200 real converge limpio.
- **`kmeans` CPU (`rodinia_kmeans_omp`)**: calibrado en hardware real, N=350000/D=34/K=200 → 27.18s cómputo puro / 28.6-28.7s con harness completo (3 reps), warmup real 0.001-0.002s (negligible, consistente con el resto de CPU). Agregado a `catalog.yaml`. **No** es el objetivo de esta búsqueda (CPU no tiene el problema de escasez), se agregó de paso por ya estar medido.
- **`kmeans` GPU (`rodinia_kmeans_gpu_N350000_D34_K200`)**: la interfaz CUDA difiere de la OMP (`-m`/`-n` fijan K, no `-k`) y el criterio de convergencia se comporta totalmente distinto (2 iteraciones siempre, vs decenas en CPU con los mismos datos) — la calibración de CPU no transfirió. `-l` (nloops, repite la corrida completa determinista) escala perfectamente lineal (~0.019s/loop medido), usado para extender a ~26s. **OI real medida con `ncu_convergence.py`: 68.9495 FLOP/byte, fp32, convergencia real a 2000 lanzamientos** (cambio relativo 0.76% < 1% tolerancia) — muy por encima del ridge fp32 REF (14.965852). Actividad Tensor Core detectada es ruido (~0.03%, muy debajo del umbral de contaminación real 10%). Warmup real medido vía `telemetry_kernel_launcher` (`gpu_util_pct`, 3 reps): 1.92-2.15s (6.1-6.9%). Agregado a `catalog.yaml` (commit `de920e4`, pusheado a `origin/main`) y a la copia de catálogo de producción en el clúster (`catalog.screening.yaml`, con backup `.bak_20260916`).
- **Campaña dedicada `pacca_gpu_kmeans_extra_20260916`** (mismo diseño que `gpu_conv2d_extra`: 10 niveles GPU × 3 reps = 30 combos): **CERRADA, 30/30 aceptadas, 0 rechazadas, matriz completa, frecuencia restaurada verificada.** Corrió limpio de 03:23 a 04:02 (2026-09-16), 3.35 core-hours. Log en `~/kmeans_extra.log`, datos en `~/hyperion-results/final/campaigns/gpu_kmeans_extra/`.
- **Intento de fusionar kmeans en la campaña oficial `gpu_final.yaml`/`output_dir=.../campaigns/gpu`**: BLOQUEADO por `CAM-09` (a propósito, no es bug) — el fingerprint del protocolo incluye el set completo de `kernel_refs`, así que agregar kmeans a la lista SIEMPRE cambia el fingerprint frente a las 540 corridas ya aceptadas de los otros 18 kernels, y el resume se niega a mezclar protocolos distintos bajo el mismo `output_dir`. Verificado que `conv2d` tiene el mismo problema (tampoco tiene corridas dentro de esa carpeta oficial, solo en su propia campaña aislada). **Conclusión real**: el patrón de este proyecto es fusionar a nivel del CSV de entrenamiento (`training_gpu_phases.csv` combinado entre campañas), no reanudar dentro del mismo `output_dir` de Slurm. `gpu_final.yaml` queda como documentación de la composición completa deseada (19 kernels), útil solo si algún día se rehace la campaña entera en un `output_dir` nuevo desde cero. **Pendiente real**: hacer el merge de `training_gpu_phases.csv` de `gpu_kmeans_extra` (y de `gpu_conv2d_extra`, que tampoco está mezclado) con el de la campaña oficial, a nivel de Fase 2 (entrenamiento), no de Fase 1 (medición).

### Ejecutado, con resultado real (sesión diurna 2026-09-16)
- **`b+tree` GPU**: DESCARTADO (confirmado, sin cambios). El kernel no tiene ningún uso de `float`/`double` (verificado leyendo `kernel_gpu_cuda.cu`), mismo motivo que excluyó a `nw` de este catálogo.
- **`miniBUDE` (Rodinia-ajeno, UoB-HPC/miniBUDE)**: CERRADO. Repo clonado (Apache-2.0, commit `570f66c`), datos reales incluidos (`data/bm1`, con `ref_energies.out`) — sin riesgo de datos inventados. Build requirió vendorizar `libjsoncpp` (ausente en paccaA100) y forzar `CMAKE_CXX_COMPILER=g++` (el `nvc++` autodetectado agrega `-fast`, incompatible con el host-compiler real de `nvcc`). El binario se auto-valida contra la referencia (`valid: true`, `max_diff_%=0.014`). **OI real medida con `ncu_convergence.py`: 25582.5597 FLOP/byte, fp32, convergencia real confirmada entre lc=50 y lc=100** (cambio relativo 0.20%, muy por debajo de la tolerancia 1%) — la ancla `compute_bound` más extrema del catálogo con enorme margen (~1700x el ridge fp32 de referencia, ~373x `gpu_cutlass_simt_conv2d`). Overhead de instrumentación real ~1.84x (731.6ms/iter instrumentado vs 397.8ms/iter en bruto) — recalibrado `-i` de 68 a 37 para mantener ~30s instrumentados. Warmup real medido vía `telemetry_kernel_launcher`: 1.07-1.20s (2.9-3.3%). Agregado a `catalog.yaml` (commit `1f9be2b`, pusheado) y a la copia de producción en el clúster.
- **Campaña dedicada `pacca_gpu_minibude_extra_20260916`** (mismo diseño: 10 niveles GPU × 3 reps = 30 combos): **CERRADA, 30/30 aceptadas, 0 rechazadas, matriz completa, frecuencia restaurada verificada.** 3.14 core-hours. Primeros 4 intentos fallaron reproduciblemente en la transición de nivel `F4→F5` con `GpuFrequencyControlError` ("el candado no parece haberse aplicado", utilización residual 7-11%) — diagnosticado como un bug real en `common/hpc/gpu_freqctl.py`: `utilization.gpu` de NVML es un promedio sobre la última ventana de muestreo (~1s), no instantáneo, así que justo después de que el kernel de calibración del nivel anterior termina, esa ventana todavía reporta >0% por un momento aunque no haya nada corriendo. Corregido con una espera de ~1.5s antes de releer utilización (commit `783ed21`, pusheado, con inyección de dependencia para no pagar el sleep real en tests — 26/26 tests de `gpu_freqctl` bajaron de 15.14s a 0.15s). El 5º intento, con el fix aplicado y sincronizado al clúster, cerró limpio sin ningún reintento adicional.
- **`cfd` GPU**: investigado a fondo, PAUSADO por decisión explícita del usuario (no por bloqueo técnico irresoluble). Se generó una malla sintética real (cadena 1D, formato `.domn` confirmado leyendo `euler3d.cu`) y se probó en hardware real: la primera versión (frontera `wing` en 2/4 direcciones) produjo NaN en las 100 celdas — riesgo confirmado, no solo sospechado. Diagnosticado y corregido: cambiar a frontera `far field` (relaja hacia el estado de flujo libre conocido, en vez de solo empujar presión sin amortiguación) eliminó el NaN por completo (densidad estable en 1.4, el equilibrio esperado). Escalado a N=5,000,000 elementos: 29.57s totales, pero el I/O de parseo domina (22.7s de 29.57s, mismo patrón de `iostream` lento visto con kmeans) — el cómputo real GPU es solo 6.87s (2000 iteraciones × 3.44ms). Complicación adicional sin resolver: `cfd` lanza 4 kernels CUDA distintos por iteración (`compute_step_factor`, `compute_flux`, `time_step`, más `initialize` una vez), y `ncu_convergence.py` no filtra por nombre de kernel — mediría las 4 mezcladas, distorsionando la OI agregada. Arreglarlo requiere adaptar el script (filtro `--kernel-name` o postprocesado que separe métricas por kernel) o medir cada kernel por separado a mano. Dado que la literatura de CFD en mallas no estructuradas suele inclinarse a memory-bound de todas formas, el usuario decidió no seguir invirtiendo tiempo esta noche. **Para retomar**: el generador (`gen_cfd_mesh.py`, en el scratchpad de la sesión, no comprometido al repo) y el binario compilado (`~/hyperion-kernels/bin/rodinia_cfd_euler3d`, checksum `f4b1015541384da4d426c744a73b514b071a85c556225a203d6b6e4c311ba650`) siguen en el clúster, listos para continuar sin repetir el trabajo de diagnóstico de NaN.

---

## Incidentes de esta sesión (para no repetir)

- Falso positivo de "job terminado" por un filtro `squeue -n` mal armado (coincidencia de prefijo, no exacta) → llevó a lanzar un reprocesamiento duplicado (`7363`) sobre los mismos archivos en paralelo con `7360`. Detectado y cancelado a tiempo. Verificar con `sacct -j <id>` o el nombre completo, no con `-n` parcial.
- Uso repetido de `--overlap` en `paccaA100` sin verificar primero si había una campaña activa de OTRO agente (barrido de `gpu_conv2d_extra`) → riesgo real de contaminar su telemetría (RAPL de paquete + GPU compartida). El usuario detuvo la sesión al notarlo. Memoria `feedback-interactive-shell-contaminates-telemetry.md` ampliada para cubrir `--overlap` explícitamente. Regla: verificar `squeue`/estado del contenedor ANTES de cada `--overlap`, no asumir que "solo es CPU, no debería importar".
