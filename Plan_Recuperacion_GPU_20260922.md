# Plan de recuperación GPU — 2026-09-22

Escrito para que lo ejecute una sesión nueva sin contexto previo. Cada
afirmación de aquí se verificó contra datos reales el 2026-09-22; **verificar
de nuevo antes de asumir que algo sigue igual**.

## -1. Alcance de H1: bloquea más que CUPTI

**Importante para no confundir dos bloqueos distintos**: H1 (candado de
reloj roto) no solo bloquea la recuperación de la campaña CUPTI (§3, T3).
Bloquea **cualquier cosa que necesite variar la frecuencia de GPU de
verdad**, incluida la tabla de política clase→frecuencia por EDP (el
equivalente GPU de lo que CPU cerró en su sección 15-20 del informe de
calidad). "Bloqueado por CUPTI" y "bloqueado por el candado roto" no son
la misma cosa — el segundo es más amplio. Mientras H1 no se resuelva, no
hay política DVFS de GPU posible, exista o no la campaña CUPTI.

## 0. Los tres hechos que condicionan todo

### H1. El candado de reloj de GPU está roto en `paccaA100`

Verificado en hardware (job `srun -p GPU --exclusive`, nodo libre, sin
medición activa): `sudo -n nvidia-smi -i 0 -lgc 700,700` y luego
`-lgc 1200,1200` devuelven `exit 0` pero `clocks.sm` **se queda siempre en
765 MHz** — en idle, bajo carga real al 100 % y después de `-rgc`. El
rendimiento bajo carga (`ert_probe_gpu fp32` → 10178.03 GFLOP/s) coincide
con el pico histórico del catálogo medido *sin* candado: la GPU corre a su
reloj nativo pase lo que pase.

No es el síntoma de ARC-113/115 (que era "bajo carga ignora el candado y
sube", corregido en ARC-137). Es peor: el candado no tiene efecto en ninguna
dirección. `Driver Version: 610.57.04`, con campos marcados "Deprecated" por
NVIDIA. No se investigó la causa.

**Fecha del daño, acotada con datos:**
- Campaña `pacca_gpu_final_20260913`: relojes **reales y distintos** por
  nivel (`dual_cholesky`: 960/810/660/510/360/210 MHz). Candado funcionando.
- Campaña `gpu_oi_window_20260916`: **765.0 MHz en todos los niveles**
  (REF/F0/F2/F4/F6/F8 verificados). Candado ya roto.

→ Se rompió entre el **2026-09-13** y el **2026-09-16**.

### H2. La campaña CUPTI de 2026-09-16 tiene 0 filas utilizables (pero es recuperable)

`~/hyperion-results/final/campaigns/gpu_oi_window_20260916` en pacca: 544
corridas con `training_gpu_phases.csv`, 1092 trazas CUPTI. Todo el
postproceso ya corrió. Pero al inspeccionar una corrida real de
`rodinia_gaussian`:

```
filas: 51   granularity: time_window
label: '' en las 51        training_eligible: False en las 51
quality: gpu_frequency_invalid=46, no_cuda_activity=5
verdict: "0 ventanas GPU elegibles logradas, por debajo de
          target_windows_per_repetition=5"
```

La cadena causal es: reloj roto (H1) → observado 765 MHz ≠ solicitado →
`gpu_frequency_quality_status=invalid` → ventana no elegible → etiqueta
nunca calculada → verdict rechaza la corrida. **El eje de frecuencia
envenenó todas las filas, no solo el eje de frecuencia.**

**Lo bueno, y es mucho:** los artefactos caros están intactos.
- Traza CUPTI real y completa: **8197 eventos** en esa corrida de gaussian
  (≈ los 8190 lanzamientos `Fan1`/`Fan2` esperados para N=4096, más memcpys),
  con `start_ns`/`end_ns`/`grid`/`transfer_bytes` por evento.
- `samples.csv` + `windows.csv` NVML por corrida.
- Calibraciones Roofline completas: `roofline_calibration_gpu_{fp32,fp64}_{REF,F0..F8}.json`.

Lo que falla es una capa de *gates y metadatos* aguas abajo, recomputable sin
volver a medir. El mecanismo CUPTI + modelos analíticos (la parte difícil)
funcionó.

### H3. El dataset histórico es el único con eje de frecuencia válido

`pacca_gpu_final_20260913` (+ las 3 campañas extra fusionadas) es el único
dataset GPU con relojes reales por nivel. Mientras H1 no se resuelva, **no se
puede medir ninguna campaña GPU nueva con eje de frecuencia**. Ese dataset es
irreemplazable hoy: tratarlo como tal (no sobrescribir, no re-postprocesar
encima).

Estado actual de la calidad del clasificador sobre él (16 familias, 483
corridas, tras las acciones A1/A2 de la ronda anterior): exactitud balanceada
por celda ≈ 0.72, **pero 7 de 16 familias fallan sistemáticamente**
(`dual_cholesky` 0.0, `rodinia_lud` 0.0, `rajaperf_cuda_gemm` 0.0 en su celda
memory, `rodinia_gaussian` 0.1, `rodinia_heartwall` 0.4, `rodinia_kmeans`
0.47). El agregado promedia aciertos perfectos con fallos totales.

## 1. Recomendaciones (orden de importancia)

1. **Commitear lo pendiente antes que nada.** Hay ~1500 líneas sin versionar
   (infraestructura CUPTI + scripts de calidad). Es riesgo de pérdida real y
   sin eso nada de lo demás es auditable.
2. **Reportar H1 a los administradores del clúster.** Es una regresión a
   nivel de nodo (probable actualización de driver), no algo corregible desde
   este repo. Cuanto más se tarde, más difícil correlacionarlo con el cambio
   que lo causó. Sin esto, el eje de frecuencia de GPU está muerto de forma
   permanente.
3. **Recuperar la campaña CUPTI (H2) es la tarea de mayor valor** — 544
   corridas ya medidas, con las trazas intactas. Pero es *diagnóstico
   primero*, no "entrenar y ya".
4. **No lanzar campañas nuevas** hasta que H1 esté resuelto. Cualquier
   campaña con eje de frecuencia hoy reproduce exactamente los mismos datos
   envenenados.
5. **Reencuadrar el capítulo GPU alrededor de lo defendible.** El resultado
   honesto es que la telemetría NVML de dispositivo clasifica bien unas
   familias y nada en otras, con explicación física (intensidad aritmética y
   firma de ocupación son ejes ortogonales). Eso es un resultado legítimo de
   tesis, igual que CPU reportó "no actuar" como hallazgo negativo válido.
   El capítulo debe mostrar la tabla por familia, no esconderse tras el 0.72.

## 2. Plan de ejecución

### T1 — Commitear el trabajo pendiente (primero, sin excepción)

Estado: `git status` muestra modificados `common/hpc/{catalog,config}.py`,
`common/telemetry/CMakeLists.txt`,
`common/telemetry/experiments/telemetry_kernel_launcher.cpp`,
`fase1_telemetria/{campaign,gpu_phases,postprocess,repostprocess_campaign,runner,validation}.py`
y sus tests; sin versionar
`common/telemetry/src/cupti_activity_preload.cpp`,
`fase1_telemetria/{gpu_oi_models,gpu_window_oi,audit_gpu_oi_coverage}.py` +
tests, `fase2_clasificador/analysis/{gpu_quality_report,rebuild_historical_gpu_relaxed_fraction}.py`,
`scripts/pacca/build_cupti_activity.sh`,
`scripts/pacca/final_campaign/gpu_oi_window_final.yaml`.

- Commits separados por tema: (a) infraestructura CUPTI/OI analítica,
  (b) scripts de calidad GPU de la ronda A1-A5.
- `git add` con rutas explícitas, **nunca `-A`** (ya colgó archivos ajenos
  del usuario en un commit una vez).
- No commitear `tmp/`, `local_datasets/`, `docs/libro.zip`.
- Correr los tests antes: `python3 -m pytest fase1_telemetria/tests -q`.

### T2 — Redactar el reporte de H1 para los administradores

Entregable: un texto corto (no un commit) que el autor pueda enviar. Debe
incluir: nodo (`paccaA100`), driver (`610.57.04`), comando exacto, salida
observada (`exit 0`, `clocks.sm` fijo en 765 MHz en los tres estados),
evidencia de que antes funcionaba (relojes reales por nivel en la campaña del
13-sep) y la ventana de fechas 13→16 de septiembre. Dejarlo en
`docs/general/` o donde el autor prefiera; **no enviarlo por cuenta propia**.

### T3 — Recuperar la campaña CUPTI (la tarea grande)

**T3.1 — Diagnóstico (antes de tocar nada).** Confirmar sobre 3-5 corridas de
kernels distintos que el único motivo de rechazo es `gpu_frequency_invalid` y
que la etiqueta vacía es *consecuencia* de la inelegibilidad, no un fallo
propio del join analítico. Leer
`fase1_telemetria/gpu_phases.py::build_gpu_time_window_rows` y seguir en el
código dónde se calcula (o se omite) `phase_label_train`.

*Gate:* si la etiqueta resulta vacía por una razón independiente del gate de
frecuencia (p. ej. el join con `gpu_oi_models` nunca produjo trabajo
analítico), **detenerse y reportar** — el alcance cambia por completo.

**T3.2 — Reproceso con la frecuencia reinterpretada.** Todas las corridas
corrieron en realidad al reloj nativo (765 MHz). Reprocesar tratándolas como
**un solo nivel de frecuencia nativo**, no como REF/F0..F8.

Detalle de corrección crítico, fácil de equivocar: el *ridge* usado para
etiquetar depende del nivel (`roofline_calibration_gpu_fp32_F4.json`, etc.).
Si todas corrieron a reloj nativo, el ridge correcto para **todas** es el de
`REF`, no el del nivel nominal. Usar el ridge del nivel nominal produciría
etiquetas calculadas contra un techo que el hardware nunca tuvo.

Marcar cada fila recuperada con trazabilidad explícita (columna tipo
`frequency_axis_invalidated=True` + el nivel nominal original), mismo criterio
que `rebuild_historical_gpu_relaxed_fraction.py` usó con
`quality_gate_relaxed`. **No modificar el gate de producción en
`gpu_phases.py`** — que siga rechazando; esto es un paso aparte y trazable.

**T3.3 — La pregunta científica que este dataset existe para responder.**
Con las ventanas ya etiquetadas, medir si la OI por ventana **varía de forma
material dentro de una corrida** y si eso produce familias de clase mixta.
Caso de validación: `rodinia_gaussian` (8197 eventos CUPTI, submatriz
decreciente). Graficar OI(t) y contar cuántas ventanas caen a cada lado del
ridge.

*Gate:* si la OI(t) resulta esencialmente constante y las familias siguen
siendo de una sola clase, **ese es el resultado** — documentarlo como
hallazgo negativo honesto y **no** seguir iterando. El plan previo
(`Plan_OI_Ventana_GPU_Reentrenamiento_20260916.md`, §4.3) ya anticipó
explícitamente este desenlace como válido.

**T3.4 — Entrenar solo si T3.3 pasa el gate.** Reutilizar
`fase2_clasificador/analysis/gpu_quality_report.py` tal cual (ya implementa
celda familia×clase, LOFO 5 semillas, IC95 por bootstrap, comparación de 7
modelos, umbral de abstención y auditoría de celdas). Dos avisos:
- `gpu_sm_clock_mhz_median` será **constante** en este dataset (765 MHz) →
  es una feature muerta aquí; correr sin ella y decirlo.
- No mezclar con el dataset histórico por corrida: contratos distintos
  (`granularity=run` vs `time_window`), y `merge_historical_gpu_phase_runs.py`
  lo rechaza a propósito.

### T4 — Capítulo del libro, con lo que haya

Independiente de cómo salga T3. `docs/libro/secciones/03_resultados.tex`.
- Reportar la **tabla por familia**, no solo el agregado.
- Explicar el mecanismo físico de los fallos (OI real vs firma NVML como ejes
  ortogonales; `rodinia_kmeans` es el caso ya documentado).
- Documentar H1 como restricción operativa y su ventana de fechas.
- Convención vigente del libro: solo metodología final y resultado final, sin
  narrativa de depuración, sin tablas ni figuras "antes/después" del propio
  proceso de arreglo. Sin em-dash `---` en la prosa.

## 3. Qué NO hacer

- No lanzar campañas GPU con eje de frecuencia mientras H1 siga abierto.
- No tocar `protocol.py::derive_kernel_family` (compartido con CPU, congelado;
  el override de `rajaperf_cuda` va local al script de diagnóstico).
- No relajar el gate de frecuencia dentro de `gpu_phases.py` para producción.
- No borrar ni re-postprocesar encima de `pacca_gpu_final_20260913` (H3).
- No perseguir `lavamd`/`dwt2d`/`cfd`/2MM-3MM: fuera de alcance por decisión
  explícita previa.
- No usar `--time` en `sbatch`/`srun`; todo `--job-name` empieza con `hyp_`;
  nunca `scancel` sobre jobs ajenos.
- No correr nada en CPU/GPU desde una shell adjunta mientras haya una
  medición activa en el nodo.

## 4. Verificación

- T1: `git log --stat` muestra los dos commits; `git status` limpio salvo
  `tmp/`, `local_datasets/`, `docs/libro.zip`.
- T3.2: número de ventanas recuperadas con etiqueta no vacía > 0, y reporte
  por kernel de cuántas se recuperaron contra las 544 corridas.
- T3.3: figura o tabla de OI(t) para `rodinia_gaussian` + conteo de ventanas
  a cada lado del ridge, por familia.
- T3.4: `gpu_quality_report.py --stages inventory,matrix,threshold,cells`,
  comparando contra el baseline histórico en `tmp/gpu_quality_20260922/` y
  `tmp/gpu_quality_relaxed020_split_20260922/`.
