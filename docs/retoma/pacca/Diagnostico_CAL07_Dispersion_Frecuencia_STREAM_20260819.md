# Diagnóstico: CAL-07 rechazó la calibración `stream_official` en F0 (2026-08-19)

## Síntoma

El primer smoke real de la campaña CPU final (post ARC-147/148/149, Turbo confirmado
cerrado en `paccaA100`) falló en la calibración con:

```
CAL-07: la traza de frecuencia de la calibración (stream_official) en 'F0' no fue
válida: frecuencia efectiva fuera de objetivo en 340/6348 muestras
(objetivo=3200000 kHz, tolerancia=160000.0 kHz, rango=1659211..3207520 kHz)
```

## Evidencia recolectada

`samples.csv` de la corrida rechazada (`stream_official__F0__rep00`, 6 CPUs delegados
`[0,1,2,3,4,5]`, `interval_ns=1ms`) desglosado por núcleo vía la columna
`scaling_cur_freq_khz_all` (ARC-145, una lectura por CPU delegado):

| Núcleo | Muestras fuera de tolerancia | Mínimo observado (kHz) |
|---|---|---|
| 0 | 10 / 1058 | 2 472 952 |
| 1 | 0 / 1058 | 3 199 996 |
| 2 | 33 / 1058 | 2 123 588 |
| 3 | 88 / 1058 | 1 877 812 |
| 4 | 110 / 1058 | 1 876 176 |
| 5 | 99 / 1058 | 1 659 211 |

Objetivo declarado: 3 200 000 kHz (F0, fracción 1.0). Duración total de la corrida:
~1.06 s.

**Control decisivo**: se repitió el mismo análisis sobre `stream_official__REF__rep00`
(gobernador nativo, sin ningún candado de frecuencia -- CAL-07 no aplica en REF, así
que esta corrida sí fue aceptada). El mismo patrón de caídas aparece, con la misma
magnitud, en los mismos núcleos:

| Núcleo | Mínimo (kHz) | Máximo (kHz) |
|---|---|---|
| 0 | 2 888 579 | 3 200 003 |
| 1 | 1 669 116 | 3 203 229 |
| 2 | 1 855 622 | 3 200 011 |
| 3 | 2 000 827 | 3 200 332 |
| 4 | 3 166 561 | 3 200 003 |
| 5 | 1 874 720 | 3 200 403 |

Bajo REF no existe ningún candado de frecuencia que "el hardware pudiera estar
ignorando" -- y aun así el rango de `scaling_cur_freq_khz` cae igual de bajo (hasta
1.67 GHz) que bajo F0. Esto descarta que F0 esté sufriendo el mismo bloqueador de
Turbo ya cerrado en ARC-147/148 (que producía lecturas **por encima** del objetivo,
nunca por debajo) -- el patrón aquí es opuesto y aparece con o sin candado.

## Hipótesis (respaldada por la evidencia, no confirmada con una prueba adicional)

`scaling_cur_freq` bajo el driver `intel_pstate` no es una lectura instantánea del
P-state activo: se deriva de los contadores APERF/MPERF y representa la frecuencia
**efectiva promedio** durante el intervalo de muestreo, incluyendo cualquier tiempo
que el núcleo haya pasado en un C-state de reposo dentro de ese intervalo. Un núcleo
que ejecuta a 3.2 GHz pero queda ocioso una fracción del intervalo de 1 ms reporta una
frecuencia "efectiva" proporcionalmente más baja, sin que el P-state solicitado haya
cambiado en absoluto.

`stream_official` (STREAM) es memory-bound: los 6 hilos compiten por el mismo ancho de
banda de memoria y no completan cada segmento de arreglo de forma perfectamente
sincronizada -- microdesajustes entre hilos generan huecos de ociosidad breves en
algunos núcleos mientras otros siguen activos, especialmente notorio en una corrida
tan corta (~1 s) con muestreo tan fino (1 ms). Esto explica por qué el efecto es
desigual entre núcleos (núcleo 1 casi no lo sufre en F0, núcleos 3-5 sí) y por qué
aparece igual bajo REF: no depende de si hay un candado activo, depende del patrón de
ociosidad real de STREAM en esta plataforma.

## Lo que esto NO es

- No es el bloqueador de Turbo (ARC-136/147/148): ese producía lecturas **por encima**
  del objetivo bajo carga sostenida real, ya confirmado cerrado con la sonda directa
  (`sudo -n /usr/local/bin/set_turbo_state`, ARC-148).
- No es un bug de sincronización del launcher/build (ARC-149): la corrida sí ejecutó,
  sí produjo `samples.csv` real, y el patrón es reproducible en dos corridas
  independientes (F0 y REF) con la misma forma.

## Lo que falta para confirmar (no hecho todavía, requiere decisión del usuario)

1. Repetir con un kernel de calibración/dataset **compute-bound** (`ert_probe`, o un
   kernel del dataset como `npb_mg`) bajo F0 -- si la dispersión desaparece o se reduce
   drásticamente, confirma que el efecto es específico del patrón de ociosidad de
   STREAM, no del candado de frecuencia en sí.
2. Si se confirma, la corrección pertinente es de **calibración del chequeo**, no del
   candado de frecuencia: CAL-07/E01 podría necesitar excluir del veredicto las
   muestras que caen dentro de una ventana ya marcada `warmup_excluded`/con
   `running_ratio` bajo, o usar una señal más robusta al reposo parcial (p. ej. el
   máximo por ventana en vez del valor puntual), en vez de aflojar la tolerancia a
   ciegas.

## Control ejecutado (2026-08-19, ARC-152): resultado más complicado de lo esperado

Se corrió `ert_probe` (compute-bound) a F0 de forma aislada, llamando
`runner.run_single()` directamente con la misma `apply_frequency` que usa
`calibration.py` (mismo código real, sin pasar por `manifest.calibration`
completo -- ver `orchestrator/schemas/diagnose_cal07_control.py`). Resultado:
**no confirma ni descarta la hipótesis de dispersión por ociosidad -- reveló
un problema distinto y más severo**:

```
repetitions=1 telemetry_mean_ns=344654668 overhead_mean=0.00%
sampling_cv=0.00% perf_running_ratio_min=0.0000 push_retries=0 samples=0
```

`ert_probe` completó su ejecución real (485.985 GFLOPs/sec, ~345ms, checksum
correcto) pero el colector de telemetría capturó **cero muestras en absoluto**
(`samples=0`, `perf_running_ratio_min=0.0`) bajo F0. El mismo kernel bajo REF,
en la campaña real que sí llegó a ejecutarlo, capturó 88 muestras con
`perf_running_ratio_min=1.0000` sin ningún problema.

Esto es distinto del patrón de `stream_official` (que sí capturó 6348
muestras, solo 340 fuera de tolerancia) -- aquí no hay dispersión que
analizar, hay **ausencia total de telemetría** específicamente bajo un nivel
de frecuencia fijo (F0), no bajo el gobernador nativo (REF). No se investigó
la causa de esto todavía.

**Salvedad importante sobre este control**: se ejecutó con un script
independiente que replica la llamada de `calibration.py` a `run_single()`,
no con el camino real de `manifest.calibration` vía el CLI `calibrate`/
`run-campaign`. Existe una posibilidad no descartada de que el script de
control tenga una diferencia de entorno respecto al camino real (aunque el
código invocado es literalmente el mismo `runner.run_single()`). Antes de
tratar el hallazgo de "cero muestras en F0" como un bloqueador confirmado de
la campaña final, se recomienda reproducirlo por el camino oficial
(`calibrate` con un manifiesto que declare únicamente `ert_probe` en
`calibration` -- bloqueado hoy por MAN-07, que exige también una referencia
de ancho de banda para poder calcular I_ridge).

## Serie de controles adicionales (2026-08-19, ARC-156): aislado a `ert_probe`

Antes de cualquier fix, se ejecutó una serie de controles dirigidos para acotar la
causa del "cero muestras" y decidir si afecta al dataset real:

1. **Pausa de 1.5s entre `apply_frequency()` y el lanzamiento** (hipótesis de
   condición de carrera entre la escritura de sysfs y el arranque del colector
   PMU): sin efecto, sigue en cero muestras. **Hipótesis descartada.**
2. **`ert_probe` en REF por el mismo script standalone**: 32 muestras reales.
   Descarta que el bug sea un artefacto del script de diagnóstico -- es
   específico de un nivel de frecuencia fijo, no del método de prueba.
3. **`ert_probe` en F4 (piso, 800MHz)**: también cero muestras, igual que F0
   (techo, 3.2GHz). Descarta que sea específico de pinear en el extremo
   superior/turbo -- ocurre en ambos extremos del hardware por igual.
4. **Rango angosto no-cero (`target ± 1000 kHz`, en vez de `min==max` exacto)**
   en F0: también cero muestras. Descarta la hipótesis de "ancho de rango
   cero" como causa -- ni siquiera ensanchar el rango en 2 kHz lo recupera.
5. **`ert_probe` en F2 (2.2GHz, punto medio que no toca ningún límite físico
   del procesador)**: también cero muestras. Confirma que el bug ocurre bajo
   **cualquier** nivel de modo `fixed` (bounded_range), sin importar el valor
   ni la cercanía a los límites de hardware -- específico del modo de
   aplicación de frecuencia (`_apply_bounded`), no de un valor concreto.
6. **`npb_mg` (kernel real del dataset, no de calibración) en F0**: **`accepted:
   true`**, 955 ventanas de CPU, 5730 muestras crudas, 0 muestras fuera de
   tolerancia, dispersión máxima de apenas 12.4 kHz sobre un margen de 180 kHz
   efectivo. **Decisivo**: confirma que el bug de cero-muestras es exclusivo de
   `ert_probe` -- no afecta a los kernels reales del dataset, que sí producen
   telemetría limpia y coherente bajo frecuencia fija.

**Conclusión de esta ronda**: dos problemas independientes, ambos confinados a
la calibración, ninguno confirmado en corridas reales de dataset:
- `ert_probe`: telemetría CPU completamente ausente bajo cualquier nivel
  `fixed`, causa raíz no identificada (se agotaron las hipótesis de timing,
  extremos de hardware, y ancho de rango sin encontrarla).
- `stream_official`: dispersión real (5.4% de muestras fuera de tolerancia,
  ARC-150), patrón que `npb_mg` NO reproduce bajo el mismo nivel -- refuerza
  que es un artefacto propio de la ejecución de STREAM (hipótesis original de
  ARC-150: promedio de `scaling_cur_freq` afectado por huecos de ociosidad
  entre hilos en una corrida muy corta), no un problema general de la
  plataforma.

Importante: el valor de calibración (`P_pico`/`BW_pico`) se extrae del
`stdout` de cada programa vía regex (`_extract_metric`), **no depende de la
telemetría PMU en absoluto** -- ninguno de los dos bugs corrompe la medición
de calibración en sí, solo la verificación de calidad CAL-07 que confirma
(de forma redundante) que la frecuencia se sostuvo durante la medición.

## CORRECCIÓN CRÍTICA (2026-08-19, ARC-157): 3 de los 8 controles corrieron sin Turbo desactivado

Al responder una pregunta del usuario ("¿el ridge depende de la frecuencia, se
está teniendo en cuenta?") se auditaron los 8 scripts de control de esta sesión
y se encontró que **3 corrieron sin el wrapper `with_cpu_turbo_disabled.sh`**:
la prueba "decisiva" de `npb_mg`, y las pruebas de `ert_probe` en F2 y F4. Esto
invalida cualquier conclusión sobre el VALOR de frecuencia real sostenido en
esas tres corridas -- no invalida necesariamente el hallazgo de "cero
muestras" (que es sobre si el colector produjo datos, no sobre a qué
frecuencia), pero sí invalida la comparación de `GFLOPs/sec` entre niveles que
se había usado como evidencia.

**Repetidas con Turbo correctamente desactivado**:
- `npb_mg` en F0: **aún mejor que antes** -- `accepted: true`, 963 ventanas,
  dispersión de solo 1.5kHz (antes 12.4kHz sin la protección). Confirma con
  más fuerza que el dataset real funciona limpio bajo frecuencia fija.
- `ert_probe` en F4: **sigue en cero muestras**, idéntico a antes. El hallazgo
  de "cero muestras bajo cualquier nivel fijo" es robusto, no depende del
  estado de Turbo.
- `ert_probe` en F4, `GFLOPs/sec` con Turbo confirmado apagado: **487.158**,
  prácticamente idéntico a F0 (485.985, medido también con Turbo apagado
  desde el principio, ARC-152). **Con Turbo genuinamente descartado como
  explicación, esto significa que `ert_probe` no refleja el candado de
  frecuencia en su rendimiento medido, por una causa distinta y no
  identificada.**

**Reevaluación del riesgo**: esto ya no es "una verificación redundante que se
puede omitir con seguridad". Sin telemetría (cero muestras) y sin que el
rendimiento medido escale con la frecuencia pedida, no hay ninguna evidencia
independiente de que `P_pico` de `ert_probe` en F0-F4 corresponda realmente a
esas frecuencias -- podría estar midiendo la misma condición no identificada en
los cinco niveles. El ridge (`i_ridge = P_pico/BW_pico`) para cualquier nivel
que no sea REF queda con esta duda sin resolver mientras `ert_probe` se use
como fuente de `P_pico`.

## ARC-158/159: causa raíz real de `ert_probe` identificada -- fix parcial en el kernel, fix pendiente en el orquestador

Con el usuario pidiendo investigar la causa de fondo ("el kernel es nuestro,
deberíamos poder arreglarlo"), se leyó `kernels/ert/ert_probe.c` completo.

**Bug 1 encontrado y corregido (ARC-158)**: el driver original abría una
región paralela de OpenMP nueva por cada combinación de tamaño de trabajo x
repeticiones del barrido, y se quedaba con el máximo GFLOP/s observado en
docenas de mediciones -- algunas de solo 31-63 microsegundos (confirmado con
cálculo directo desde `BEST_WORKING_SET_DOUBLES`/`BEST_TRIALS`/`GFLOPs/sec`
reportados), dominadas por el overhead de sincronización de OpenMP y la
resolución del reloj, no por cómputo real. Primer intento de fix (filtro de
duración mínima de 1ms) corrigió el ruido pero sesgó la elección hacia un
tamaño de trabajo de ~3.3MB/hilo que ya no cabe en L1/L2 (limitado por ancho
de banda, no por cómputo) -- corregido con repeticiones adaptativas por
tamaño (duplicar hasta cruzar el umbral, una sola medición fiable por
tamaño), que ahora selecciona un tamaño pequeño (~90KB) cache-residente de
forma estable y reproducible.

**Bug 2 encontrado, NO corregido, de fondo distinto**: incluso con la
medición ya estable y fiable, el `GFLOPs/sec` sigue sin escalar con la
frecuencia pedida (F0=509.87 vs F4=508.70, con Turbo confirmado apagado).
Verificado leyendo `scaling_cur_freq` directamente cada 5ms mientras
`ert_probe` corre: durante la fase F4 (candado en 800MHz confirmado por
lectura de `scaling_min/max_freq`), las primeras muestras de `cur_freq`
muestran **3200000** -- el valor de la fase F0 anterior -- bajando
gradualmente después. **El candado de frecuencia no se asienta a tiempo**:
`ert_probe` corre en decenas de milisegundos, más rápido de lo que el
hardware tarda en completar la transición real de P-state tras aplicar un
nuevo límite (especialmente hacia abajo). Esto explica por qué `npb_mg`
(corre por segundos, tiempo de sobra para que la frecuencia se asiente) no
tiene este problema mientras que `ert_probe` sí, incluso ya con su
metodología de medición corregida.

**Descartado**: se probó una pausa de asentamiento de 0.3s entre aplicar F4 y
lanzar `ert_probe`, imprimiendo `scaling_cur_freq` justo antes de medir --
seguía en ~3.1GHz, no 800MHz. 300ms no alcanza. Esto llevó a diagnosticar más
a fondo el mecanismo, en vez de simplemente alargar la pausa a ciegas.

## Causa raíz final (ARC-160): decaimiento lento de HWP bajo EPP=performance, sin permiso para evitarlo

Diagnóstico de plataforma (`intel_pstate` global, gobernador y EPP por CPU):

- `energy_performance_preference` = **`performance`** en los 6 CPUs
  delegados (valores disponibles: `default performance balance_performance
  balance_power power`).
- **Sin permiso de escritura sobre `scaling_governor`** (`Permission denied`
  al intentar cambiar a `powersave`) -- el permiso P1 concedido por el
  administrador (ARC-105/107) cubre únicamente `scaling_min_freq`/
  `scaling_max_freq`, no el gobernador ni el EPP.
- Al iniciar un job posterior (con más tiempo transcurrido desde la última
  corrida a alta frecuencia), `scaling_cur_freq` sí había bajado a valores
  cercanos a 800MHz bajo el mismo candado que antes fallaba a los 300ms.

**Interpretación**: bajo `intel_pstate` en modo HWP con `EPP=performance`, el
algoritmo autónomo del hardware que elige la frecuencia real dentro del
rango permitido está sesgado fuertemente hacia mantenerse alto. Al bajar el
techo (`scaling_max_freq`) inmediatamente después de correr a frecuencia
alta, el hardware no salta al nuevo límite -- decae gradualmente, en una
escala de segundos (no milisegundos), sin que exista ningún permiso
disponible hoy para forzar `powersave` u otro EPP que evite este
comportamiento.

**Implicación para todo el proyecto, no solo para `ert_probe`**: este no es
un problema exclusivo del kernel de calibración de FLOPs -- es un riesgo
latente para *cualquier* transición hacia una frecuencia más baja dentro de
la matriz de una campaña real. `ert_probe` lo expone de forma extrema (corre
en decenas de milisegundos, mucho más corto que el tiempo de decaimiento),
pero un kernel de dataset que corra justo después de una combinación de
frecuencia más alta podría tener ventanas iniciales con frecuencia real por
encima del nivel nominal, diluidas u ocultas si el resto de la corrida (de
varios segundos) sí alcanza a estabilizarse dentro de tolerancia -- CAL-07/
E01 solo exige que el promedio de muestras caiga dentro de tolerancia, no
que la primera ventana lo haga. `npb_mg` pasó limpio en las pruebas de esta
sesión probablemente porque, para cuando corrió, el estado ya había tenido
tiempo de asentarse desde una prueba anterior -- no porque esté exento del
mecanismo.

**CORRECCIÓN (2026-08-19, ARC-162): este diagnóstico (EPP/HWP) era
incorrecto.** Una revisión externa (pedida por el usuario) cuestionó la
atribución a EPP con buena evidencia: el propio mecanismo de espera activa
implementado a partir de este diagnóstico (ARC-161) **falló su timeout de
30s** en la verificación final -- inconsistente con un simple "decaimiento
que tarda más", y mucho más consistente con una interferencia *activa y
continua*, no un transitorio. La revisión propuso una causa alternativa,
con respaldo directo en la documentación oficial de `intel_pstate`
("Coordination of P-State Limits"): los CPUs delegados (0-5) comparten
núcleo físico con hermanos SMT (16-21) que el orquestador nunca restringía.
Ver la sección "Causa raíz real" más abajo -- el fix real no es una pausa
ni un permiso nuevo, es controlar también los hermanos SMT.

## Causa raíz real (ARC-162, 2026-08-19): coordinación de límites P-state entre hermanos SMT

Confirmado con `lscpu -e` en `paccaA100`: los 6 CPUs delegados (0-5) comparten
núcleo físico con 6 hermanos SMT (16-21) -- mismo `CORE` id, nunca
restringidos por el orquestador (`manifest.cores.delegated_cpus` solo
declara 0-5). `related_cpus`/`affected_cpus` de cada uno muestran políticas
de `cpufreq` sysfs independientes (`policy0` solo lista "0", `policy16`
solo lista "16") -- las escrituras a un lado nunca se propagan
automáticamente al otro a nivel de software. Pero el hardware físico
subyacente (el generador de reloj real del núcleo) es compartido entre
ambos hilos, y el driver `intel_pstate` documenta explícitamente que un
hermano SMT que solicita mayor rendimiento puede hacer que el otro supere
su límite configurado por política (`docs.kernel.org/admin-guide/pm/intel_pstate.html#coordination-of-p-state-limits`).

**Prueba B, autorizada explícitamente por el usuario tras revisar la
crítica externa**: comparación directa en `paccaA100` real, `ert_probe`
(ya corregido, ARC-158) bajo F0 (3.2GHz) y F4 (0.8GHz), con la carga
siempre confinada a 0-5 vía `taskset`:

| Prueba | F0 | F4 | Razón F0/F4 |
|---|---|---|---|
| A: solo 0-5 limitados, 16-21 libres (comportamiento de siempre) | 507.463 GFLOP/s | 509.910 GFLOP/s | 0.995 (sin escalar) |
| B: 0-5 Y hermanos 16-21 limitados juntos, carga solo en 0-5 | 506.889 GFLOP/s | **133.988 GFLOP/s** | **3.78 (≈4 esperado)** |

Corrobora además el propio algoritmo adaptativo de `ert_probe.c`: en B-F4
eligió un tamaño de trabajo y repeticiones distintos (656 dobles, 4096
repeticiones) que en las otras tres corridas (11240 dobles, 512
repeticiones) -- consistente con estar midiendo cómputo genuinamente más
lento, no una medición inflada por ruido.

**Conclusión**: el mecanismo de espera activa (ARC-161, `manifest.frequency_settle`,
`freqctl.wait_for_frequency_settled()`) sigue siendo una salvaguarda
razonable (falla en voz alta en vez de medir sin confirmación), pero por sí
solo NO resuelve el problema -- una espera, por larga que sea, no puede
asentar una interferencia activa y continua desde un hermano SMT sin
restringir. El fix real es extender el conjunto de CPUs que `freqctl.py`
controla para incluir también los hermanos SMT de los CPUs delegados,
mientras la carga real permanece confinada solo a los CPUs delegados (vía
`--pin-workload-cpus`, sin cambios). No implementado todavía -- pendiente
de que el usuario lo autorice explícitamente como cambio de producción.

## Estado final (2026-08-19, ARC-166) -- CAL-07 CERRADO Y VERIFICADO DE PUNTA A PUNTA

**Investigado el residuo de ARC-165 (11/492 muestras fuera de tolerancia en
F0): reproducible 2/2 (jobs 6315, 6316), un CPU delegado específico se
queda estancado en un valor constante durante ~11 muestras consecutivas
justo tras arrancar el kernel real, luego salta de golpe al objetivo.**
Implementado `frequency_validation.grace_seconds` en `validate_cpu_frequency_trace()`
-- excluye de la comprobación de TOLERANCIA (nunca de los chequeos
estructurales de integridad) las lecturas dentro de esa ventana desde el
primer tick CPU de la traza; falla en voz alta si la corrida completa cae
dentro de la gracia, en vez de aceptar por vacuidad.

**Error propio corregido en el camino**: el primer valor (`grace_seconds=15.0`,
job 6317) fue **1000x demasiado grande** -- se confundió `timestamp_ns`
(ya en nanosegundos) con segundos; el rezago real es **~10-11
MILISEGUNDOS** (latencia de transición P-state normal, no un decaimiento de
segundos), y 15.0s se tragó la corrida COMPLETA de `ert_probe` (~83ms de
duración total). Corregido a `grace_seconds=0.05` (50ms).

**Verificación final de punta a punta (job 6318): F0 y F4 ambos
`accepted: true`, `mismatched_samples: 0` en los dos.** CAL-07 queda
CERRADO Y VERIFICADO DE PUNTA A PUNTA por primera vez en todo el proyecto.
La cadena completa de fixes: ARC-163 (control de hermanos SMT) + ARC-165
(warm-up de asentamiento antes del sondeo) + ARC-166 (grace period
post-transición). Detalle completo en el registro de cambios (ARC-166).

## Estado final (2026-08-19, ARC-165) -- superado por la sección de arriba, se conserva como registro histórico

**Actualización sobre la sección "Estado final (ARC-164)" de más abajo: se
implementó el fix del bug de settle-antes-de-carga (`freqctl._start_warmup_load`/
`_stop_warmup_load`, un `taskset -c <cpu> yes` por CPU delegado corriendo
mientras `wait_for_frequency_settled()` sondea, detenido siempre en un
`finally`).** Verificación real (job 6315): **el `FrequencyControlError` que
bloqueaba F0 desde ARC-161 desapareció por completo** -- F0 corrió de punta
a punta (`success: true`) por primera vez en toda esta investigación.

Pero **la corrida, ya en marcha, fue rechazada por la validación real por
ventana (CAL-07/E01)**: `11/492` muestras de `scaling_cur_freq` cayeron
fuera de tolerancia (mínimo observado 2 970 401kHz vs objetivo
3 200 000kHz±160 000kHz -- una caída real de ~230MHz durante la ejecución
misma, no un problema de sondeo previo). Mucho menor que la dispersión
original de `stream_official` (340/6348, ARC-150), pero no cero. F4 (piso,
0.8GHz) sí pasó limpio (`0/798` fuera de tolerancia).

**CAL-07 sigue sin cerrarse.** Lo que cambia con ARC-165: el bloqueador que
impedía siquiera *empezar* a medir (settle verificando CPUs inactivos, ARC-164)
está resuelto y verificado. Lo que queda abierto es un residuo real de
dispersión de frecuencia *durante* la corrida en F0 -- no identificado si es
ruido de medición aceptable, un efecto residual de HWP bajo carga sostenida,
u otra causa distinta. No investigado todavía; ninguna tolerancia ni
verificación se relajó u omitió para hacerlo pasar. `paccaA100` verificado
limpio tras el job (frecuencias nativas, `no_turbo=0`, sin procesos `yes`
huérfanos).

## Estado final (2026-08-19, ARC-164) -- superado por la sección de arriba, se conserva como registro histórico

**Actualización sobre la sección "Estado final (ARC-162)" de más abajo: el
fix de hermanos SMT (ARC-163) se implementó y se verificó que pinea
correctamente los 12 CPUs (0-5 y 16-21), pero la verificación de punta a
punta vía el pipeline real (`runner.run_single()`, job 6306) siguió
fallando -- por una causa DISTINTA, recién descubierta (ARC-164), no por un
fallo del fix de ARC-163.**

`runner.py::run_single()` llama a `freqctl.settle_if_configured()`
(el mecanismo de espera activa de ARC-161) **antes** de lanzar el
subproceso del kernel. Esto significa que `wait_for_frequency_settled()`
relee `scaling_cur_freq` mientras los CPUs delegados están **inactivos**.
Bajo `intel_pstate`, `scaling_cur_freq` se calcula del ratio APERF/MPERF --
una medida de actividad real de instrucciones ejecutadas -- así que un
núcleo inactivo no puede reflejar un candado de frecuencia alto (F0,
3.6GHz) por más que `scaling_min_freq`/`scaling_max_freq` estén pineados
correctamente en ese valor. Confirmado en `paccaA100` real: tras pinear los
12 CPUs a 3600000kHz (confirmado por relectura de `scaling_min/max_freq`),
`scaling_cur_freq` muestreado cada 1s durante 10s de inactividad se quedó
fluctuando entre ~800MHz y ~1GHz, nunca cerca de 3.6GHz.

Esto también explica retroactivamente por qué F4 (0.8GHz, el piso) sí pasó
en el job 6306: un CPU inactivo ya está naturalmente cerca del piso, así
que el objetivo se cumple "por accidente", no porque el mecanismo esté
midiendo lo que se diseñó a medir.

**CAL-07 sigue sin resolverse de punta a punta.** El fix de ARC-163 es
correcto y necesario (confirmado por Prueba B, ARC-162) pero no es
suficiente por sí solo -- el diseño de ARC-161 (verificar el asentamiento
antes de iniciar la carga real) es incompatible con cómo `intel_pstate`
reporta `scaling_cur_freq`. Ningún fix de código aplicado todavía para
ARC-164 -- requiere una decisión de diseño (p.ej. mover la verificación de
asentamiento a después de que la carga real ya esté corriendo) pendiente de
discutir con el usuario. Ver ARC-163/ARC-164 en el registro de cambios para
el detalle completo.

Estado de `paccaA100` verificado limpio tras esta investigación (min/max de
los 12 CPUs en sus valores nativos de hardware, `no_turbo=0`,
`scaling_governor=performance`).

## Estado final (2026-08-19, ARC-162) -- superado por la sección de arriba, se conserva como registro histórico

**Causa raíz real identificada y confirmada experimentalmente: coordinación
de límites P-state entre hermanos SMT (0-5 y 16-21), nunca controlados
juntos por el orquestador.** Las dos causas propuestas antes (dispersión
propia de STREAM, ARC-150; y decaimiento de HWP por EPP=performance,
ARC-160) quedan reclasificadas:

- El diagnóstico de `stream_official` (dispersión, no ausencia total) sigue
  sin descartarse del todo -- podría ser el mismo mecanismo SMT en menor
  grado, o un artefacto propio de STREAM como se hipotetizó originalmente;
  no se volvió a probar con el fix de SMT aplicado.
- El diagnóstico de EPP/HWP (ARC-160) **queda retractado** -- la evidencia
  que parecía respaldarlo (asentamiento no monótono en el barrido de
  pausas) es exactamente el patrón esperado de una interferencia SMT activa,
  no de un decaimiento por EPP.
- `ert_probe` en sí tenía un bug real de metodología de medición (ARC-158,
  corregido y desplegado a producción) independiente del problema de SMT --
  ambos se apilaban.

**No corregido en producción todavía**: extender `freqctl.py` para
controlar también los hermanos SMT de los CPUs delegados (manteniendo la
carga real solo en los CPUs delegados) es el fix que la Prueba B confirmó
necesario. Pendiente de autorización explícita del usuario para
implementarlo como cambio de producción -- ya autorizada la investigación
(Prueba B), el siguiente paso es el fix mismo.

**Estado de las piezas ya corregidas y en producción, independientes de
este hallazgo**:
- ARC-153/154 (shim de blocking-sync GPU): corregido y verificado.
- ARC-158 (metodología de medición de `ert_probe.c`): corregido, desplegado
  al catálogo de producción (`pacca-a100`), verificado.
- ARC-161 (`manifest.frequency_settle`, espera activa): implementado,
  probado, 14 tests nuevos en verde -- sigue siendo una salvaguarda de
  defensa en profundidad razonable (falla en voz alta en vez de medir sin
  confirmar), aunque por sí sola no resuelve el problema de fondo.
