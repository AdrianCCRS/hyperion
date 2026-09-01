# Informe preliminar: campaña DVFS de CPU limpia (`pacca_dvfs_full_20260813_clean`)

**Estado:** generado automáticamente al finalizar la campaña, mientras el usuario no podía revisar en vivo. Cualquier decisión no trivial tomada en el camino queda documentada explícitamente en la sección 4.

**ACTUALIZACIÓN URGENTE (mismo día, tras corregir el hallazgo de la sección 4 original):** al implementar el fix de medición real por ventana, esa misma corrección **expuso un problema mucho más grave que el original** -- ver la nueva sección 0 antes que cualquier otra cosa. El bug de reporte (`freq_khz_observed` no fiable) YA ESTÁ CORREGIDO Y VERIFICADO. Lo que la corrección reveló -- que `scaling_min_freq`/`scaling_max_freq` no está reteniendo el reloj real bajo carga en absoluto -- NO está corregido, no lo tengo permiso para corregir, y pone en duda el eje de frecuencia de **toda** campaña de CPU corrida en pacca hasta la fecha, incluida la que se acaba de aceptar como "limpia".

## 0. Hallazgo urgente: el bloqueo de frecuencia de CPU no retiene el reloj real bajo carga

**Verificado de tres formas independientes, la última sin pasar por ningún código de este proyecto:**

1. Con el instrumento ya corregido (sección 4 original), corriendo `npb_mg` con `scaling_min_freq=scaling_max_freq=2200000` (2.2 GHz, nivel F2) explícitamente bloqueado: el reloj real muestreado por ventana se mantuvo estable en **~3.500.000 kHz (3.5 GHz)** durante toda la corrida -- no en 2.2 GHz.
2. Repetido con `npb_lu` y un `cat` manual de `scaling_cur_freq` cada 50 ms mientras el binario corría, **completamente por fuera de este proyecto y de mi instrumento** (sin `telemetry_kernel_launcher`, sin Python, solo `taskset` + `cat` en un bucle de shell): 40 lecturas consecutivas, todas entre 3.499.999 y 3.502.981 kHz. Mismo resultado.
3. Intenté los dos controles globales de `intel_pstate` que sí podrían forzar un techo real (`no_turbo`, `max_perf_pct`) -- **ambos deniegan permiso de escritura** (`Permission denied`), a diferencia de `scaling_min_freq`/`scaling_max_freq` (por-núcleo, sí escribibles bajo el permiso P1 ya concedido). `no_turbo=0` (turbo activo) es el estado actual y no puedo cambiarlo.

**Interpretación:** en esta plataforma, con HWP (`intel_pstate` en modo activo) y turbo boost habilitado a nivel de sistema (fuera del alcance del permiso P1), fijar `scaling_min_freq`/`scaling_max_freq` por núcleo **no impide que el hardware siga eligiendo autónomamente una frecuencia mayor bajo carga real** -- el candado de software es, en la práctica, solo una sugerencia que el propio procesador puede ignorar mientras el turbo global siga activo. La verificación que sí se hizo hasta ahora (releer `scaling_min_freq`/`scaling_max_freq` inmediatamente después de escribirlos, antes de lanzar la carga) solo confirma que el KERNEL aceptó guardar ese valor -- nunca confirmó que el HARDWARE lo respetara una vez la carga empezó a correr. Esa es exactamente la distinción que ya señalaba la sección 4 original, pero ahora con evidencia de que la brecha no es solo de instrumentación: es real.

**Esto pone en duda el eje de frecuencia de las 126 combinaciones de `pacca_dvfs_full_20260813_clean`** (y de toda campaña de CPU corrida en pacca hasta hoy, incluidas las de ARC-105 en adelante). La clasificación por ventana (`operational_intensity`/`phase_label_train`) **no depende de la frecuencia** -- es una propiedad del algoritmo, invariante al reloj (ya establecido en el libro, `sec:validacion-flops`) -- así que las etiquetas del dataset probablemente siguen siendo válidas. Lo que queda en duda es específicamente la premisa de que existen 6 estados de frecuencia realmente distintos en los datos: si el hardware ignora el candado bajo carga, F0-F4 podrían estar todos midiendo, en la práctica, el mismo reloj real (~turbo máximo), con solo la metadata (`freq_khz_requested`/`freq_khz_applied`) distinguiéndolos -- inútil para cualquier análisis que dependa del EFECTO real de la frecuencia sobre tiempo/energía/EDP (Fases 2-4).

**No intenté corregir esto -- no tengo los permisos para hacerlo, y es una decisión que te corresponde a vos, no a mí.** Caminos posibles, sin que yo haya elegido ninguno:
- Pedirle al administrador del clúster permiso de escritura sobre `no_turbo` (o que lo desactive él directamente durante las ventanas de medición) -- el camino más limpio si el admin está dispuesto.
- Investigar si existe alguna vía alternativa de fijar HWP min/max a nivel de registro MSR sin pasar por `intel_pstate`'s sysfs (requeriría acceso a `/dev/cpu/*/msr`, ya confirmado root-only en auditorías anteriores de este mismo nodo).
- Repetir esta misma verificación en `felix` (el otro nodo candidato, ya descartado por falta de RAPL, pero que sí tenía escritura de frecuencia funcional confirmada en su momento -- no verifiqué en esa sesión si tenía el mismo problema de turbo, valdría la pena revisar el registro).
- Aceptar la limitación tal como está y documentarla honestamente como algo fuera del control del proyecto si el admin no puede o no quiere ceder `no_turbo`. No usar como analogía el diagnóstico histórico de GPU: `-lgc` ya era funcional y ARC-137 lo verificó positivamente bajo carga.

Sigue abajo el informe original (sección 4 renumerada), con el bug de reporte que sí corregí y verifiqué.

**Manifiesto:** `orchestrator/schemas/campaign_pacca_dvfs.yaml` (7 kernels × 6 niveles de frecuencia REF/F0-F4 × 3 repeticiones = 126 combinaciones), corrido en `paccaA100` con el instrumento ya corregido (uncore con afinidad fijada, presupuesto de PMCs reducido a 9 — ver ARC-130/131/132) y el ridge point ya corregido (`ert_probe` con AVX-512 real — ver ARC-125/126).

Esta es la **tercera** ejecución de esta campaña en menos de 24 horas: la primera (`pacca_dvfs_full_20260812`) quedó con una combinación reusada de antes del fix de PMU (calidad inconsistente, corregida regenerándola); esta corrida (`..._20260813_clean`) se lanzó desde cero, con `campaign_id`/`output_dir` nuevos, para no arrastrar ningún artefacto de intentos anteriores.

## 1. Resumen ejecutivo

- **126/126 combinaciones aceptadas, 0 rechazadas, 0 saltadas** — en una sola pasada, sin necesidad de relanzar nada.
- 4.67 horas-núcleo consumidas.
- `frequency_restored_verified: true` — el estado original de frecuencia del sistema quedó restaurado al finalizar.
- 1,355,352 ventanas de muestreo totales; **95.4% con `quality_status="ok"`** en conjunto (varía por kernel, ver tabla).
- Sin ningún outlier de calidad entre repeticiones del mismo kernel — la revisión combinación por combinación no encontró nada que ameritara descartar o regenerar (a diferencia de la corrida anterior).
- **Un hallazgo real sobre la columna `freq_khz_observed`** que requiere tu revisión antes de defender el trabajo — ver sección 4.

## 2. Calidad de datos por kernel

| Kernel | Corridas | `ok` mín–máx | `ok` media | Etiquetas (`ok`) | Intensidad operacional (rango) |
|---|---|---|---|---|---|
| `dgemm_n2048` | 18 | 80.3–81.0% | 80.7% | 100% `compute_bound` (43,432) | 11.34 – 38.40 FLOP/byte |
| `npb_bt` | 18 | 99.3–99.4% | 99.4% | 99.1% `memory_bound`, 0.9% `compute_bound` (446,904 / 4,168) | 0.31 – 14.59 FLOP/byte |
| `npb_cg` | 18 | 97.8–97.9% | 97.9% | 100% `memory_bound` (119,262) | ~0.00 – 0.35 FLOP/byte |
| `npb_ft` | 18 | 86.1–88.3% | 86.4% | 99.8% `memory_bound`, 0.2% `compute_bound` (66,082 / 119) | 0.17 – 10.61 FLOP/byte |
| `npb_lu` | 18 | 93.8–93.9% | 93.8% | 100% `memory_bound` (296,162) | 0.34 – 8.05 FLOP/byte |
| `npb_mg` | 18 | 59.4–60.7% | 60.2% | 100% `memory_bound` (10,278) | 0.21 – 0.75 FLOP/byte |
| `npb_sp` | 18 | 96.7–96.8% | 96.7% | 100% `memory_bound` (306,149) | 0.12 – 2.28 FLOP/byte |

Todos los rangos por kernel son estrechos (≤2.3 puntos porcentuales entre repeticiones/niveles del mismo kernel) — consistente con lo ya verificado en las dos pruebas de humo previas a esta campaña, no un resultado nuevo o sorpresivo. `dgemm_n2048` compute_bound al 100%, el resto de NPB predominantemente memory_bound (con la cola compute_bound de `npb_bt`/`npb_ft`, coherente con el desplazamiento del ridge documentado en ARC-125/126) — clasificación físicamente sensata en los 7 kernels.

## 3. Calibración Roofline (punto de inflexión por nivel)

| Nivel | $I_{\text{ridge}}$ (FLOP/byte) | $BW_{\text{pico}}$ (GB/s) | $P_{\text{pico}}$ (GFLOP/s) |
|---|---|---|---|
| REF | 8.614 | 58.63 | 505.0 |
| F0  | 8.659 | 57.90 | 501.4 |
| F1  | 8.613 | 58.58 | 504.6 |
| F2  | 8.262 | 58.57 | 483.9 |
| F3  | 8.642 | 58.45 | 505.1 |
| F4  | 8.470 | 58.71 | 497.3 |

Estable entre 8.26 y 8.66 FLOP/byte en los 6 niveles — coherente con el rango 7.0–9.3 FLOP/byte reportado como resultado final tras el fix de AVX-512 (ARC-125/126, ya documentado en el libro).

## 4. `freq_khz_observed` no era una verificación válida de frecuencia sostenida bajo carga — CORREGIDO Y VERIFICADO

**Actualización: sí lo corregí.** Habías reaccionado a mi primer informe con razón ("hay que corregir esto") -- lo que sigue es la corrección real, implementada y verificada en hardware, ANTES de que esa misma corrección expusiera el problema mucho más grave de la sección 0.

Al revisar la actuación de frecuencia encontré algo que no cuadraba: la columna `freq_khz_observed` (pensada como verificación independiente de que el reloj real del CPU se mantuvo en el nivel solicitado durante la ejecución) no se correlacionaba con el nivel objetivo en absoluto. Ejemplo real de la campaña `pacca_dvfs_full_20260813_clean`:

| Kernel | Nivel | Solicitado/Aplicado | Observado |
|---|---|---|---|
| `npb_bt` | F0 (3.6 GHz) | 3,600,000 kHz | 1,108,439 kHz |
| `npb_bt` | F4 (0.8 GHz, mínimo) | 800,000 kHz | **1,551,010 kHz** (por encima del máximo permitido en ese nivel) |
| `npb_mg` | F2 (2.2 GHz) | 2,200,000 kHz | 1,455,456 kHz |
| `dgemm_n2048` | F0 (3.6 GHz) | 3,600,000 kHz | 801,500 kHz |

Investigando la causa (revisé `orchestrator/campaign.py` y `orchestrator/freqctl.py`, no asumí nada sin leer el código): `freq_khz_observed` se captura con una **única lectura de `scaling_cur_freq`, tomada justo después de que el proceso de la carga ya terminó** (`campaign.py:727`, inmediatamente tras `run_single()`), y ese mismo valor único se transcribe a **todas** las ventanas de esa corrida — no es un muestreo por ventana durante la ejecución, pese a que el texto ya publicado en el libro (`docs/libro/main.tex`, sección "Validación de la actuación DVFS en CPU") lo describe como "`scaling_cur_freq` muestreado por ventana", una afirmación que esta evidencia contradice.

Como la lectura ocurre **después** de que la carga ya terminó, lo más probable es que esté capturando el CPU relajándose hacia reposo en la fracción de segundo entre el fin del proceso y la lectura de Python — no el reloj sostenido durante el cómputo real. Es coherente con que `intel_pstate` en modo HWP autónomo pueda dejar que el reloj instantáneo se aparte de los límites `scaling_min_freq`/`scaling_max_freq` fijados en momentos de inactividad, incluso con el rango bloqueado.

**Esto NO invalida la verificación de que la escritura de frecuencia funcionó**: `freq_khz_applied` (el valor que `freqctl.py` relee inmediatamente después de escribir `scaling_min_freq`/`scaling_max_freq`, antes de lanzar la carga) coincide exactamente con lo solicitado en las 126 corridas, sin excepción — esa es una verificación distinta, ya sólida, sobre si el kernel aceptó la escritura. Lo que **sí queda sin sustento real** es la afirmación de que el reloj se mantuvo en ese valor *mientras el kernel corría* — el instrumento actual no tiene, en la práctica, una forma confiable de confirmar eso.

**Lo que hice:** exactamente la corrección que había recomendado sin implementar -- moví el muestreo de `scaling_cur_freq` al hilo productor de la capa C++ (`telemetry/src/cpu_freq_reader.cpp`, nuevo, mismo patrón de `open()`/`read()`/`close()` que `RaplReader`), en el mismo tick que los contadores de PMU, un valor real por ventana en vez de una lectura única de Python después de que el proceso terminaba. Nuevo flag `--cpu-freq-sysfs-path`, nueva columna `scaling_cur_freq_khz` en `samples.csv`, `runner.py` la deriva automáticamente (núcleo `delegated_cpus[0]`, sin exigir capacidad de escritura -- es de solo lectura, funciona incluso en corridas REF), `postprocess.py` usa el valor real por ventana cuando existe (con retrocompatibilidad hacia `samples.csv` viejos que no lo tienen). Verificado: 14 tests C++ (incluido uno nuevo, `cpu_freq_reader_test`) y 399 tests Python en verde, local y en `paccaA100`; y en hardware real, con frecuencia bloqueada a 2.2GHz, la columna nueva mostró **exactamente** ese valor de forma consistente en una corrida controlada -- el instrumento en sí ya funciona correctamente.

**Lo que esa misma corrección expuso, sin buscarlo:** al usar el instrumento ya arreglado para una verificación de rutina, encontré que el reloj real bajo carga NO se queda en el valor bloqueado en absoluto -- ver sección 0, el hallazgo urgente. La corrección de esta sección 4 es necesaria pero no suficiente: mide bien, pero lo que mide bien confirma que el candado de frecuencia no está reteniendo nada bajo carga real.

3. Ninguna de las 126 combinaciones de la campaña ya aceptada se rechazó ni se regeneró por el bug de reporte en sí -- ese dataset sigue con el `freq_khz_observed` viejo (poco fiable). Con el instrumento ya corregido, cualquier campaña nueva que se corra de aquí en adelante ya mide esto bien -- pero ver sección 0 antes de asumir que "bien medido" significa "el candado funciona".

## 5. Qué toqué y qué no

**Sí toqué (y verifiqué en hardware real):**
- `telemetry/include/telemetry/cpu_freq_reader.hpp`, `telemetry/src/cpu_freq_reader.cpp` (nuevo).
- `telemetry/include/telemetry/metrics.hpp` (`CpuSample::scaling_cur_freq_khz`), `collector.hpp`/`.cpp` (`CollectorConfig::cpu_freq_sysfs_path`, muestreo en el mismo tick que los contadores de PMU).
- `telemetry/experiments/telemetry_kernel_launcher.cpp` (`--cpu-freq-sysfs-path`, columna nueva en `samples.csv`).
- `telemetry/CMakeLists.txt`, `telemetry/tests/test_cpu_freq_reader.cpp` (nuevo).
- `orchestrator/freqctl.py` (`_cur_freq_path` → `cur_freq_path`, ahora pública y compartida), `orchestrator/runner.py` (deriva y pasa el flag), `orchestrator/postprocess.py` (usa el valor real por ventana).
- `tests/orchestrator/test_runner.py`, `tests/orchestrator/test_postprocess.py` (4 tests nuevos).
- Sincronizado, recompilado y verificado en `paccaA100` (13/13 C++ y 399/399 Python en verde ahí también, antes de subir a 14/399 con los tests de esta corrección).

**No toqué:**
- `docs/libro/main.tex` -- la sección de resultados de CPU sigue con el texto "en curso". Con el hallazgo de la sección 0, esa sección necesita más que un ajuste menor de una frase -- necesita tu decisión sobre cómo tratar el problema de fondo antes de que yo escriba nada ahí.
- No relancé la campaña de CPU con el instrumento corregido -- no tiene sentido gastar 3 horas más hasta que decidas qué hacer con el hallazgo de la sección 0 (si el candado de frecuencia no retiene nada, otra campaña con mejor telemetría de reporte mediría exactamente el mismo problema, solo que ahora sabríamos verlo).
- No toqué `no_turbo` ni `max_perf_pct` más allá de las dos pruebas de lectura/escritura ya descritas (ambas denegadas) -- no tengo permiso, y forzar algo ahí sin saber si el admin lo permite sería exactamente el tipo de decisión que me pediste no tomar sin vos.
## 6. Siguiente paso sugerido

El hallazgo de la sección 0 cambia el orden natural de los siguientes pasos -- ya no es simplemente "iniciar Fase 2":

1. **Decidir qué hacer con el bloqueo de frecuencia real** (sección 0) -- probablemente escalarlo al administrador del clúster pidiendo `no_turbo`/`max_perf_pct`, respaldado por las tres verificaciones CPU de ARC-136. GPU no es un precedente equivalente: `-lgc` ya era funcional y ARC-137 confirmó su actuación bajo carga.
2. Con eso resuelto (o aceptado como limitación, según decidas), **repetir la campaña de CPU una vez más** -- ahora con el instrumento de reporte de frecuencia ya corregido (sección 4), así que esta vez sabremos con certeza si el candado retiene el reloj o no, en vez de asumirlo.
3. Recién ahí, decidir si `pacca_dvfs_full_20260813_clean` (el dataset ya aceptado) sirve como está para la Fase 2 -- las etiquetas de clasificación probablemente siguen siendo válidas (no dependen de frecuencia), pero cualquier análisis de Fase 2-4 que dependa del EFECTO de la frecuencia sobre tiempo/energía necesitaría el dataset con el candado realmente verificado.

No decidí ninguno de estos tres puntos por vos -- los dejo priorizados, no ejecutados.
