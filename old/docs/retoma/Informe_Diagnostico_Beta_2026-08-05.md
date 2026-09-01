# Informe de diagnóstico — ¿funciona el orquestador? (2026-08-05)

Contexto de la solicitud: quedan ~3 semanas para el proyecto y la Fase 1
(según la numeración del plan de tesis, no la del repo) todavía no está
cerrada. Este informe responde con evidencia concreta, no impresiones, a
la pregunta: **¿esta beta del pipeline está funcionando bien?**

## Veredicto ejecutivo

**Sí, el pipeline funciona correctamente de punta a punta — pero llegar
a "sí" tomó encontrar y corregir 6 bugs reales en la primera ronda de
pruebas contra hardware de verdad**, uno de ellos crítico (dejaba vacío
el 67% de los datos de cualquier campaña con más de una repetición, sin
ningún error visible). Con los 6 corregidos, una campaña real de 7
kernels × 3 repeticiones (21 corridas) terminó **21/21 aceptadas**, con
~305 000 filas de datos cuya clasificación física (`compute_bound` vs
`memory_bound`) coincide con lo esperado en los 7 kernels.

**El riesgo más grande para el cronograma de 3 semanas no es el código:
es H1** — el permiso de escritura de `cpufreq` en SC3, todavía sin
respuesta. Sin él, el pipeline solo puede medir en el punto de operación
nativo del nodo (governor `performance`, un único nivel de frecuencia).
**No existe hoy ningún dato real de DVFS multi-frecuencia** — todo lo
verificado en este informe es a una sola frecuencia. Ver sección 6.

**Corrección a este informe (agregada después de una pregunta directa del
usuario):** la afirmación de "los datos tienen sentido físico" de la
sección 3 no menciona que `bytes_moved_window`/`operational_intensity` —
la base de `phase_label_train` — sigue calculándose con el mismo
contador que F3.4 (ARC-33) ya demostró que subestima el tráfico real de
memoria en ~30-34% en STREAM, sin cuantificar en los otros 6 kernels.
**Ese sesgo nunca se corrigió ni se validó con uncore** — P4 (el permiso
para leer contadores de uncore) sigue sin respuesta de SC3, así que la
validación cruzada propuesta en F3.4 nunca se hizo. Que la clasificación
haya salido bien no prueba que el sesgo desapareció; ver sección 6 para
el detalle completo.

## 1. Qué se probó

Campaña real (`campaign_felix_ref_full.yaml`) contra felix, vía
`python -m orchestrator.cli run-campaign` (el comando real de producción,
no un script de prueba): 7 kernels del catálogo (`npb_ep`, `npb_mg`,
`npb_cg`, `npb_is`, `npb_ft`, `npb_lu`, `dgemm_n2048`) × nivel `REF`
(governor nativo, el único disponible sin H1) × 3 repeticiones = 21
combinaciones. Calibración Roofline (`stream_official`/`ert_probe`) +
5 repeticiones de referencia (`npb_mg`) antes de la matriz.

## 2. Comportamiento del orquestador

| Aspecto | Resultado |
|---|---|
| Preflight completo (46 checks) | Todos en verde, corrido a mano antes de la campaña (el CLI no lo invoca automáticamente todavía — ver sección 6) |
| Calibración D03 (BW/FLOPs pico vs ficha de referencia) | Pasa — `i_ridge=2.24 FLOP/byte` |
| Calibración D04 (estabilidad de referencia) | Pasa — `cv_pct=0.88%` (ver nota abajo) |
| Corridas aceptadas | **21/21**, 0 rechazadas |
| `windows.csv` con datos reales | **21/21** (antes del fix de esta ronda: solo 7/21) |
| Total de filas de `windows.csv` | ~305 554 |
| `perf_running_ratio_min` | **1.0000 en las 21 corridas** — el PMU nunca tuvo que multiplexar los 4 contadores |
| `push_retries` | **0 en las 21 corridas** — el ring buffer productor/consumidor nunca se llenó, cero muestras perdidas |
| `sampling_interval_cv_pct` | 0.43%–11.16%, media 2.42% — el intervalo de 1 ms es preciso y estable |
| Overhead de instrumentación (telemetry vs baseline) | 0.8%–7.7% según el kernel (media global 5.2%, ver nota abajo) |
| `total_core_hours` de la campaña | ≈1.0 |

**Nota sobre D04:** en el primer intento se usó `npb_ep` como kernel de
referencia y dio `cv_pct=56.99%` (falla, aunque no bloqueante). Se
diagnosticó con los contadores reales: `ipc`/`ips` estaban perfectamente
estables (CV 0.09%/0.90%), pero `mpki`/`miss_rate` no, porque `npb_ep`
genera solo 979–4220 cache-misses totales en toda la corrida — un número
tan chico que cualquier variación normal del contador se ve enorme en
términos relativos. No es un bug del pipeline, es una mala elección de
kernel de referencia. Cambiado a `npb_mg` (con tráfico de memoria real),
`cv_pct` bajó a 0.88%.

**Nota sobre overhead:** el CV global (48-51% entre corridas) es alto,
pero mezclar kernels muy distintos en una sola cifra de CV es engañoso —
`dgemm_n2048` (que usa OpenBLAS, con su propio manejo interno de hilos)
tiene overhead casi nulo (0.8%, con una repetición incluso ligeramente
negativa por ruido normal), mientras los NPB rondan 5-8%. Calculado
*por kernel*, cada uno es razonablemente consistente entre sus 3
repeticiones. El gate original de F4.4 ("CV < 10%") asumía overhead
homogéneo entre kernels distintos, un supuesto que no se sostiene — ver
sección 6.

## 3. ¿Los datos tienen sentido físico?

Comparando `phase_label_train` (calculado por el modelo Roofline real,
`operational_intensity` vs `i_ridge`) contra `phase_label_hint`
(la expectativa declarada en el catálogo), agregado sobre las 3
repeticiones de cada kernel:

| kernel | hint | ventanas `ok` | resultado dominante | % coincide con hint |
|---|---|---|---|---|
| `npb_ep` | compute_bound | 18 (de 19 338 — ver nota) | compute_bound | **100%** |
| `dgemm_n2048` | compute_bound | 35 629 | compute_bound | **99.6%** |
| `npb_cg` | memory_bound | 74 346 | memory_bound | **100.0%** |
| `npb_mg` | memory_bound | 8 681 | memory_bound | **92.8%** |
| `npb_is` | memory_bound | 4 429 | memory_bound | 69.9% (mezclado, mayoría correcta) |
| `npb_ft` | intermedio | 38 760 | compute_bound | 74.8% / 25.2% memory — mezcla real |
| `npb_lu` | intermedio | 113 541 | compute_bound | 68.0% / 32.0% memory — mezcla real |

**Lectura:** los kernels con hint definido separan limpio (EP/DGEMM
compute-bound casi puro, CG memory-bound casi puro, MG mayormente
memory-bound). Los "intermedios" (FT/LU) muestran mezcla real de fases
en vez de caer arbitrariamente de un lado — exactamente el comportamiento
esperado de un roofline que funciona, no ruido. Esto es, informalmente,
una confirmación del criterio **INT-T08** (F4.5 del plan), aunque no se
corrió todavía como gate formal con el umbral ≥80% que propone el plan.

**Nota sobre `npb_ep` (18 ventanas `ok` de 19 338):** no es un defecto —
EP genera números aleatorios sin apenas tocar memoria, así que
`bytes_moved_window≈0` en casi toda la corrida y la intensidad
operacional queda indefinida (`quality_status=intensity_undefined`) para
la inmensa mayoría de sus ventanas. Es la misma razón física que explica
por qué es mal candidato a kernel de referencia (nota de la sección 2).
Las pocas ventanas válidas sí clasifican correctamente.

## 4. Bugs reales encontrados y corregidos en esta ronda de pruebas

Ninguno de estos existía como sospecha — todos aparecieron al correr
código real contra hardware real por primera vez, algo que hasta esta
semana nunca se había hecho (todo el trabajo previo eran tests con
mocks). Es información importante para calibrar cuánta confianza dar al
resto del pipeline que **todavía no** se ha ejercitado así:

1. **`_parse_cpu_list()` no entendía el formato real de `freqdomain_cpus`
   de felix** (lista separada por espacios, no por comas) — el check de
   seguridad E10 (protección contra fuga de control de frecuencia a otro
   usuario del socket) nunca tenía datos con qué bloquear, pese a que los
   tests con mocks pasaban.
2. **El launcher C++ nunca escribía el stdout del kernel medido en el
   caso exitoso** — rompía en silencio toda la extracción de BW/FLOPs
   (CAL-02/CAL-03/POST-09).
3. **`calibration.py` no convertía las unidades nativas del stdout**
   (STREAM imprime MB/s, no B/s; `ert_probe` imprime GFLOP/s, no FLOP/s)
   — habría sesgado el ridge point de *cualquier* calibración futura por
   un factor de ~1000.
4. **El check de "procesos ajenos" (E06) filtraba por `Cpus_allowed`**
   (podría correr ahí) en vez de por actividad real (`processor`+`state`
   de `/proc/stat`, está corriendo ahí *ahora*) — rechazó el 100% de las
   combinaciones del primer intento real por daemons del sistema en
   reposo, cero contención real.
5. **`cgroup_path` exigía un valor no nulo para `environment_tier:
   hpc_sc3`**, un requisito que felix no puede satisfacer (sin
   delegación de cgroup) y que además contradecía el propio texto de la
   regla MAN-01 ("opcional en todos los tiers") — resabio de antes de la
   migración a PID+inherit.
6. **`windows.csv` quedaba completamente vacío en toda repetición de
   campaña >1** (el más grave): `runner.py` nunca fija `--repetitions`
   en el launcher, así que la columna interna `repetition` de
   `samples.csv` siempre vale "1" — pero el post-procesamiento filtraba
   por el índice de repetición de la *campaña* (1/2/3). Para repetición
   ≥2 el filtro nunca encontraba nada. La corrida quedaba "aceptada"
   igual, porque la validación nunca revisa el contenido de
   `windows.csv`. Afectó 14 de 21 corridas (67%) en el primer intento
   real de esta campaña, sin ningún error visible.

Los 6 están corregidos, con tests de regresión, y verificados de nuevo
contra hardware real después de cada fix (ver ARC-42 a ARC-48 en
`Registro_Cambios_Fuera_Plan_Original.md`).

## 5. Estado del checklist técnico

**119 de 124 reglas (96%)** de la Guía Maestra están marcadas ☑, con
referencia a test o verificación en hardware real. Las 4 pendientes:

- `ENV-12` (detección de `gpu_vendor` real) — no crítico para DVFS de
  CPU, GPU no está en el alcance actual.
- `FRQ-07` (calibración a F0 fijo) y `FRQ-08` (prueba de caos en
  bare-metal) — bloqueadas por H1 (permiso) y por una acción humana
  fuera de este entorno, respectivamente.
- `MLT-06` (commit hash del protocolo en cada metadata) — trazabilidad
  menor, no bloqueante.

Ninguna de las 4 impide correr campañas reales en el nivel `REF` de hoy.

## 6. Brechas y riesgos que siguen abiertos (no resueltos en este informe)

- **El ground truth de `bytes_moved_window` sigue sesgado y sin
  corregir — afecta directamente la clasificación reportada en la
  sección 3.** `bytes_moved_window = delta_cache_misses ×
  cache_line_size_bytes` (`postprocess.py`), usando el mismo contador
  genérico per-core (`PERF_COUNT_HW_CACHE_MISSES` vía PID+inherit) desde
  Fase 1 — sin cambios desde F3.4 (ARC-33), donde se midió que subestima
  el tráfico real de memoria en ~30-34% en STREAM (probablemente el
  prefetcher de hardware de Nehalem-EX ocultando accesos reales al
  contador de demanda). **No existe ninguna ruta de uncore en el
  pipeline** — P4 (permiso para leer contadores de uncore, la validación
  cruzada propuesta para cuantificar/corregir este sesgo) sigue sin
  respuesta de SC3, y confirmado bloqueado por `perf_event_paranoid=1`
  (ARC-35). El sesgo se cuantificó *solo* en STREAM; no se sabe si es
  igual, mayor o menor en los 6 kernels del dataset, cada uno con un
  patrón de acceso distinto. Que `phase_label_train` haya coincidido
  bien con `phase_label_hint` en la campaña real (sección 3) es
  tranquilizador pero no es una prueba de que el sesgo no está
  presente — los kernels probados están mayormente lejos del ridge
  point; un kernel genuinamente cercano al límite compute/memory-bound
  sería mucho más sensible a un error sistemático de esta magnitud en el
  denominador de `operational_intensity`.
- **H1 (permiso de escritura cpufreq) sin respuesta de SC3 — el riesgo
  más urgente dado el cronograma.** Todo lo verificado en este informe
  es a una sola frecuencia (governor nativo). El pipeline está listo
  para escribir frecuencias en cuanto se conceda, pero **no hay ningún
  dato real de DVFS multi-nivel todavía**, que es presumiblemente el
  objetivo central del proyecto.
- El CLI (`orchestrator.cli run-campaign`) todavía no invoca
  `run_campaign_preflight()` automáticamente — hay que correrlo a mano
  aparte, como se hizo para este informe. Fácil de olvidar antes de una
  campaña real.
- La calibración (`stream_official`/`ert_probe`) corre con el conteo de
  hilos por defecto del sistema, no necesariamente igual a
  `delegated_cpus` de la campaña — puede sub- o sobre-estimar el pico
  real según cuántos cores tenga la campaña.
- El gate de overhead de F4.4 ("CV < 10%") no tiene sentido calculado
  across-kernel como está implementado hoy (sección 2) — probablemente
  haya que redefinirlo por kernel o eliminarlo como gate agregado.
- `run_reduced_preflight()` (E01/E02/E07/E08/I07/C01/C02 por corrida)
  sigue sin conectarse a `campaign.py` — solo se extrajo y conectó E06
  cuando se pidió explícitamente.
- H2/H3/H4 (energía externa, prueba de caos, decisiones de alcance del
  director) siguen pendientes, ninguna depende de código.

## 7. Recomendación dado el cronograma de 3 semanas

1. **Enviar/escalar H1 ya** (`Solicitud_Permisos_SC3.md` ya está
   redactado) — es la dependencia externa más larga y la que más
   directamente bloquea tener un dataset DVFS real. Todo lo demás en
   este informe confirma que el pipeline está listo para usarlo en
   cuanto llegue.
2. Mientras se espera H1: el pipeline ya puede generar un dataset REF
   grande y confiable (features absolutas y relativas, clasificación
   Roofline validada) — útil como línea base y para adelantar cualquier
   análisis/modelo que no dependa estrictamente de variar frecuencia.
3. No dar por buena ninguna corrida futura solo por `accepted:true` en
   `verdict.json` — como muestra el bug #6, eso no garantiza que
   `windows.csv` tenga datos. Vale la pena, si hay tiempo, agregar una
   validación de "al menos N filas en windows.csv" al veredicto mismo.

## 8. Datos crudos de referencia

`~/hyperion-results/campaigns/felix_ref_full_20260805/` en felix.
`roofline_calibration.json`, `calibration_references.json`,
`campaign_metadata.json`, y 21 subcarpetas con `windows.csv`/
`samples.csv`/`metadata.json`/`verdict.json` completos.
