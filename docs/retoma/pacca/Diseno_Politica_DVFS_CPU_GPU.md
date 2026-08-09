# Diseño de la política DVFS heterogénea CPU+GPU

**Estado de este documento:** versión 2, reescrita por completo el 2026-08-06
después de una discusión que descartó varias ideas de la versión 1 (código
propio para los kernels GPU, CUPTI para detectar límites de kernel, GPU
funcionando solo offline con una tabla estática). La versión 1 llegó a
conclusiones que **contradecían el plan de trabajo de grado aprobado**
(`docs/general/plan_trabajo_grado.md`) sin que nadie se diera cuenta hasta que
se releyó el plan con cuidado. Este documento parte del plan como **verdad
absoluta, no negociable**, y solo diseña lo que el plan no especifica al
detalle (implementación concreta del harness).

Si en algún punto de aquí en adelante algo parece contradecir el plan, es un
error de este documento, no una corrección al plan.

---

## 0. Lo que el plan aprobado obliga (cita textual, no interpretación)

Esto es lo que no se puede tocar. Cuatro citas, con la sección exacta:

1. **Objetivo específico 1** (sección 3.2): recolectar telemetría "mediante
   contadores de rendimiento por hardware e interfaces de potencia estándar
   (**Perf y RAPL para CPU y NVML para GPU**)".
2. **Sección 5.1 (Fase 1):** la muestra cubre "cuatro escenarios base: CPU
   compute-bound, CPU memory-bound, GPU compute-bound y GPU memory-bound"; y
   para GPU, la telemetría es específicamente "la utilización de
   Multiprocesadores de Streaming (SM), el uso de memoria y el consumo de
   potencia" — **NVML, nada más**. No menciona `ncu`, CUPTI, ni FLOPs/bytes.
3. **Sección 5.2 (Fase 2):** "el modelo recibe como entradas vectores de
   telemetría del hardware y produce como salida una etiqueta... si el sistema
   se encuentra en un régimen dominado por cómputo o por memoria" — **la
   salida es binaria**, no hay una tercera ni cuarta clase.
4. **Sección 5.3 (Fase 3):** el daemon, "en cada instante de ejecución,
   captura las métricas actuales del sistema, construye el vector de entrada
   para el modelo y ejecuta la inferencia correspondiente" — **la inferencia
   es en vivo, también para GPU**, usando NVML como entrada.

De estas cuatro citas se derivan tres restricciones duras para todo lo que
sigue:

- **GPU no puede depender de `ncu` en tiempo de ejecución.** El plan fija
  NVML como la única fuente de telemetría de GPU. `ncu` sigue siendo útil,
  pero solo puede jugar un papel *fuera* del daemon.
- **La salida sigue siendo binaria** (`compute_bound`/`memory_bound`). Los
  "cuatro escenarios" de la Fase 1 son un criterio de **muestreo** para tener
  buena cobertura en el dataset de entrenamiento, no cuatro clases de salida.
  Ver sección 2 para el razonamiento completo.
- **Tiene que haber inferencia en vivo para GPU**, no una tabla estática
  poblada offline. Una tabla kernel→etiqueta no generaliza a cargas no vistas
  (el plan pide "aplicaciones científicas", no "los kernels de Rodinia que
  usamos para entrenar") — este fue precisamente el error de la versión 1 de
  este documento.

---

## 1. Hallazgos empíricos en el A100 real (2026-08-06)

Estos hallazgos siguen siendo válidos sin importar el rediseño de arriba —
son hechos del hardware, no decisiones de arquitectura. Verificados en
`paccaA100` dentro de un `srun` real.

| Hecho medido | Evidencia | Consecuencia |
|---|---|---|
| **Application clocks deprecados** | `nvidia-smi -q -d CLOCK` → `Applications Clocks: Requested functionality has been deprecated` | La única vía de control de reloj es `nvmlDeviceSetGpuLockedClocks`/`nvidia-smi -lgc`, que **requiere root**. Sin ruta no-root. |
| **Reloj de memoria no ajustable** | `SUPPORTED_CLOCKS` lista 1 valor de memoria (1215 MHz) y **81** de SM (765-1410 MHz) | El espacio DVFS de GPU es unidimensional: solo el reloj de SM. |
| **`ncu` funciona sin root** | Kernel CUDA propio trivial perfilado con `ncu --metrics dram__bytes.sum`: 388.80 MB medidos vs ≈402 MB analíticos | `ncu` es viable como herramienta de **etiquetado offline** (ver sección 2) — no está bloqueado por permisos. |
| **DCGM no instalado** | `which dcgmi nv-hostengine` → ausentes | No diseñar asumiendo métricas de profiling continuas de DCGM. |
| **Persistence mode deshabilitado**, reposo a 765 MHz, límite de potencia 250 W | `nvidia-smi --query-gpu=...` | Afecta reproducibilidad de mediciones de energía/latencia; considerar pedir `-pm 1` como mejora opcional (ya incluido en la solicitud de permisos, P4). |
| **765 MHz no es solo el reloj de reposo -- persiste bajo carga real** (ARC-77) | `nvidia-smi --query-gpu=clocks.sm` muestreado *mientras* `ert_probe_gpu` corría (no en reposo): 765 MHz de 1410 MHz máximos (54.3%), con solo 62 W de 250 W usados y 26°C -- sin límite térmico ni de potencia que lo justifique | La GPU no hace *boost* aunque haya trabajo de sobra para justificarlo y margen térmico/energético amplio. Sin `P4` no hay manera de pedirle que suba. Explica por qué toda calibración de cómputo medida hoy (`gpu_ert_probe_fp32/fp64`, `gpu_dgemm_calibration`) ronda 48-54% de los picos teóricos de NVIDIA -- no es un límite del código de calibración, es el reloj real bajo el que corre *todo* en este nodo hoy. |

**Consecuencia estratégica:** el control de reloj de GPU está bloqueado por
permisos igual que el de CPU — la GPU no es un plan B si el permiso de CPU
tarda, cae en el mismo bloqueador. Ver `Solicitud_Permisos_Pacca_Unicartagena.md`
P4 (ya redactado). **Consecuencia metodológica (ARC-77):** como ningún
kernel del catálogo tiene tampoco el permiso para forzar el reloj máximo,
el `i_ridge` de GPU debe derivarse de picos *medidos bajo esta misma
limitación* (`gpu_ert_probe_fp32/fp64`), no de picos teóricos ni de
literatura con reloj desbloqueado -- de lo contrario se compararía a los
kernels reales contra una velocidad que ninguno puede alcanzar hoy, el
mismo tipo de error ya corregido para Tensor Cores.

---

## 2. Por qué la salida sigue siendo binaria, no cuatro clases

Este punto se discutió explícitamente porque la sección 5.1 del plan, leída
rápido, parece sugerir "cuatro escenarios" como si fueran cuatro clases de
salida. Vale la pena dejar el razonamiento completo por escrito:

**Los "cuatro escenarios base" (CPU-compute, CPU-memory, GPU-compute,
GPU-memory) son una estrategia de muestreo, no un espacio de etiquetas.** Su
función es asegurar que el dataset tenga ejemplos representativos de ambas
etiquetas en ambos dispositivos, para que el entrenamiento no aprenda una
correlación espuria (por ejemplo, "si el vector viene de GPU, es
compute-bound" porque los benchmarks de GPU elegidos resultaron ser todos
compute-bound por casualidad). La sección 5.2 lo confirma: la salida del
modelo es "si el sistema se encuentra en un régimen dominado por cómputo o
por memoria" — dos clases, sin mención de dispositivo.

Una salida de cuatro clases sería, además, peor diseño, no solo una
desviación del plan:

1. **El dispositivo nunca es una incógnita en tiempo de inferencia.** El
   daemon siempre sabe si está leyendo Perf o NVML — jamás confunde de dónde
   viene el vector. Pedirle al modelo que además adivine el dispositivo es
   informationalmente redundante.
2. **Las *features* de cada dispositivo son disjuntas** (IPC/miss-rate/stalls
   de Perf vs. util/potencia/memoria de NVML). Un modelo de 4 clases entrenado
   sobre ese espacio mixto en la práctica termina comportándose como dos
   clasificadores binarios pegados, pero con más superficie para errores de
   clase cruzada (predecir `CPU_memory` sobre un vector que vino de GPU).

**Decisión: dos clasificadores binarios independientes**, uno por
dispositivo, mismo espacio de salida (`compute_bound`/`memory_bound`), misma
metodología (Random Forest / Árbol de Decisión, comparados igual que exige la
Fase 2). El plan dice "el modelo" en singular porque describe la metodología
una vez, no porque exija una única instancia compartida entre CPU y GPU.

---

## 3. El rol correcto de `ncu`: etiquetador de Fase 1, nunca señal de runtime

Esta es la corrección central frente a la versión 1 de este documento.

**`ncu` no calcula la intensidad operacional que ve el modelo en producción.**
El modelo en producción solo ve NVML (restricción del plan, sección 0). El
papel de `ncu` es exactamente el mismo que ya cumplen `bytes_moved_window` y
`flops_window_estimate` del lado CPU: producir la **etiqueta de verdad**
(ground truth) que se usa para *entrenar* el clasificador, nunca para
alimentarlo en producción.

**Corrección de diseño (ARC-79):** la versión anterior de esta sección decía
"se etiqueta una sola vez y esa etiqueta se reusa en todos los niveles
`FG_n`" -- **eso es falso en general**, mismo error que ARC-78 encontró y
corrigió del lado CPU. `i_ridge = P_pico/BW_pico` depende del reloj:
`P_pico` escala con la frecuencia de SM, `BW_pico` casi no (el reloj de
memoria es un dominio aparte que este proyecto no toca -- sección 1). Un
kernel con intensidad operacional fija (`rodinia_lud`, ARC-76/77: 7.6-7.8
FLOP/byte, a menos de 7% del ridge medido a la frecuencia de reposo) puede
cruzar la frontera compute/memory-bound solo por un cambio de reloj de GPU,
sin que el kernel mismo cambie en nada. La corrección separa lo que SÍ es
invariante de frecuencia (la intensidad operacional, `FLOPs/byte`, que
depende del algoritmo y el tamaño de datos, no del reloj) de lo que NO lo es
(el ridge point contra el cual se compara esa intensidad):

Concretamente, en Fase 1:

1. Por cada kernel candidato (de Rodinia u otra fuente), se corre `ncu` **una
   sola vez** y se obtiene `FLOPs/byte` real, medido con contadores de
   hardware, no estimado. Esto sigue siendo cierto sin cambios: la
   intensidad operacional de un kernel no depende del reloj al que corra.
2. **`P_pico_gpu` se calibra UNA VEZ POR CADA nivel `FG_n`** (sección 6),
   fijando el reloj de SM a ese nivel y corriendo
   `gpu_ert_probe_fp32`/`gpu_ert_probe_fp64` (sección 5) ahí -- no una sola
   vez para toda la campaña. `BW_pico_gpu` sí se mide una sola vez
   (`gpu_stream_bw`): el reloj de memoria no es parte del espacio DVFS de
   este proyecto (sección 1), así que no hay ninguna razón física para que
   cambie entre niveles.
3. Se calcula `i_ridge_gpu(FG_n) = P_pico_gpu(FG_n) / BW_pico_gpu` -- un
   ridge point **por nivel**, mismo principio que `calibration.py` ya
   implementa del lado CPU (ARC-78).
4. La etiqueta de cada kernel se recalcula **por combinación (kernel, FG_n)**:
   `compute_bound` si `FLOPs/byte > i_ridge_gpu(FG_n)`, `memory_bound` en
   caso contrario. Un kernel borderline como `rodinia_lud` puede terminar
   con etiquetas distintas en `FG0` (reloj alto) que en `FG4` (reloj bajo)
   -- es lo físicamente correcto, no un error de medición.
5. Cada etiqueta (kernel, FG_n) se le asigna a las muestras NVML recolectadas
   mientras ese kernel corre **a ese FG_n específico**, nunca a las de otro
   nivel.
6. El modelo de GPU (Fase 2) se entrena sobre esas muestras NVML con esa
   etiqueta como target — **`ncu` nunca vuelve a ejecutarse después de Fase
   1**, ni en el daemon, ni en ninguna corrida de validación. Lo que sí se
   repite por nivel es la calibración de `P_pico_gpu` (paso 2), no `ncu`.

Esto resuelve limpio la restricción del plan (NVML es la única entrada de
producción) sin perder rigor en el etiquetado (no se etiqueta a ojo ni con un
hint de literatura, se mide) -- y ahora sin el gap metodológico de asumir un
ridge point constante entre niveles de frecuencia.

**Estado de esta corrección (ARC-80): implementada en código, no solo en
diseño.** La corrección de ARC-79 (arriba) se escribió primero como un
cambio de diseño puro, con el razonamiento de que "esto es Fase 2, sin
construir todavía" -- **ese razonamiento estaba mal**: los pasos 1-4 de
esta misma sección están explícitamente bajo "Fase 1" (es la etiqueta de
verdad para *entrenar* el futuro modelo, no el modelo en sí), así que
dejarlos sin código era un hueco real de Fase 1, no algo que pudiera
esperar. Implementado: `KernelEntry` gana `operational_intensity_flops_per_byte`
(el `FLOPs/byte` medido con `ncu`, ver catálogo) y `gpu_precision`
("fp32"/"fp64", determina cuál de los dos ridge points aplica);
`calibration.run_gpu_calibration()` calibra `P_pico_gpu` fp32 y fp64 por
separado, una vez por cada nivel de `manifest.frequency_levels` (fuentes
declaradas explícitamente en `manifest.gpu["calibration"]`, nunca
inferidas del catálogo, para no confundir `gpu_dgemm_calibration` --
referencia informativa de cuBLAS, ARC-76 -- con la fuente real del ridge);
`postprocess.py` calcula `phase_label_train`/`operational_intensity` para
las filas `gpu_telemetry` cuando esa calibración está disponible. **Sin
esto, el dataset de GPU que se viene recolectando desde ARC-72 tenía
*features* (potencia/utilización) pero ningún target con el cual entrenar
-- ya corregido y verificado con una campaña real en `paccaA100`**
(`gpu_ert_probe_fp32`/`fp64` calibrados a REF, `rodinia_lavamd` etiquetado
`compute_bound`, `rodinia_backprop`/`rodinia_lud` etiquetados
`memory_bound`, cada uno contra el ridge de su propia precisión). Lo que
sigue bloqueado por `P4` es únicamente *recalibrar* por más de un nivel de
frecuencia real (hoy solo existe el nivel `REF`) -- el mecanismo en sí ya
está listo para eso, no falta código, falta el permiso.

---

## 4. Arquitectura de runtime: un daemon, dos loops independientes

Pregunta que se discutió explícitamente: ¿"dos modelos" significa que compiten
o que se coordinan en cada decisión? **No — son dos ciclos de decisión
independientes, cada uno controla su propio dispositivo, sin negociar entre
sí, dentro de un mismo proceso daemon:**

```
daemon DVFS (un proceso)
│
├── Loop CPU (cadencia ~1 ms, el mecanismo que ya existe)
│     lee Perf (IPC, miss-rate, stalls...)
│  →  vector de features CPU
│  →  modelo_cpu.predict(vector) → compute_bound | memory_bound
│  →  si cambia la etiqueta: aplica vía scaling_min_freq/max_freq (cpupower)
│
└── Loop GPU (cadencia ~100 ms-1 s, la que realmente actualiza NVML)
      lee NVML (util_pct, power_mw, memoria, clock actual)
   →  vector de features GPU
   →  modelo_gpu.predict(vector) → compute_bound | memory_bound
   →  histéresis + tiempo mínimo de permanencia (evita flapping, sección 7)
   →  si corresponde: nvidia-smi -lgc / nvmlDeviceSetGpuLockedClocks
```

Cada modelo es un artefacto serializado distinto (Fase 2 los entrena, valida
y serializa por separado), con vectores de entrada de forma distinta. No se
combinan en una sola inferencia ni se promedian sus salidas — cada uno decide
sobre su propio actuador.

**La única comunicación entre los dos loops es una señal unidireccional:**
"GPU ocupada" (del loop GPU) hace que el loop CPU fuerce su frecuencia a
mínimo **sin siquiera consultar a `modelo_cpu`** en esa ventana. El loop GPU
nunca necesita saber nada del loop CPU. El porqué de esta regla, en detalle:

### 4.0 Aclaración importante (2026-08-06, ARC-66): este diagrama es el daemon de Fase 3, no el colector de Fase 1

El diagrama de arriba describe el **daemon** (Python, todavía sin construir,
Fase 3) — ahí "dos loops independientes" es literal: cada uno solo consume
telemetría ya recolectada y decide sobre su propio actuador, así que sí
pueden ser dos hilos (o incluso dos procesos) genuinamente separados.

**Eso no aplica al colector C++ que ya existe hoy** (`telemetry/src/collector.cpp`,
usado en Fase 1 para recolectar el dataset de entrenamiento). Ahí hay una
restricción de hardware/diseño que no se puede rodear: el `SPSCRing` que
conecta el hilo productor con el consumidor es **estrictamente de un solo
productor** (`spsc_ring.hpp`, verificado antes de intentar nada). No se puede
agregar un segundo hilo productor de GPU sin romper esa garantía lock-free.

**Por eso, a nivel del colector, CPU/RAPL/GPU comparten el mismo hilo
productor** — lo único que se logró (ARC-66) fue que NVML ya no se consulta
en cada tick de 1 ms junto con Perf/RAPL, sino que ese mismo hilo tiene una
compuerta de tiempo (`CollectorConfig::gpu_interval_ns`, 100 ms por defecto):
solo llama a NVML cuando ya pasó ese tiempo desde la última lectura. Es una
cadencia distinta dentro de un solo hilo, no un segundo loop físico. La
"arquitectura de dos loops" solo se vuelve literalmente cierta un nivel más
arriba, en el daemon de Fase 3, que consume filas ya escritas por este único
colector.

### 4.1 El caso spin-wait, resuelto con precisión

Cuando el CPU espera a que la GPU termine (`cudaDeviceSynchronize()` o
equivalente), hay dos escenarios distintos:

**(a) Espera por spin (comportamiento por defecto de CUDA).** El hilo de CPU
ejecuta un bucle activo mientras espera — consume ciclos reales. Perf ve IPC
alto y casi cero cache-misses, lo que haría que `modelo_cpu` clasifique esa
ventana como `compute_bound` **incorrectamente** y suba la frecuencia justo
cuando el CPU no hace ningún trabajo útil. Aquí forzar el mínimo es una
**corrección activa** contra un error real del clasificador.

**(b) Espera bloqueante (`cudaDeviceScheduleBlockingSync`).** El hilo se
bloquea de verdad — el sistema operativo lo saca de la cola de ejecución. En
sentido estricto, una vez bloqueado, la frecuencia P-state casi no afecta el
consumo (la potencia dinámica depende de que haya conmutación de compuertas,
y un hilo bloqueado no ejecuta nada — el ahorro real en reposo lo dan los
C-states, no el P-state). Aun así, **conviene forzar el mínimo igual, como
medida defensiva, no correctiva**: el cambio de frecuencia en CPU es casi
gratis (~1-10 ms, sección 4.1.6 del plan), y muchos mecanismos de
"blocking sync" en la práctica hacen un spin corto antes de bloquear de
verdad — cubrir ese margen no cuesta nada.

**En ambos casos, la ventana de espera se trata como si no hubiera fase que
clasificar** (mismo principio que ya aplica `quality_status=intensity_undefined`
del lado CPU): no se le pregunta nada a `modelo_cpu`, y la política aplica el
piso mínimo por default — en (a) porque corrige un error, en (b) porque es
gratis y cubre el margen de un bloqueo imperfecto.

**Acción de código necesaria (todavía no hecha):** los benchmarks GPU deben
usar `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` antes de cualquier
`cudaDeviceSynchronize()`. Sin esto, se cae siempre en el caso (a).

### 4.2 Por qué no hace falta CUPTI ni detectar límites de kernel

La versión 1 de este documento asumía que hacía falta saber "qué kernel
específico está corriendo ahora mismo" (vía NVTX o inyección CUPTI) para
poder aplicar una tabla de intensidad estática. **Con el rediseño de la
sección 3, esa necesidad desaparece por completo:** el modelo de GPU no
necesita saber qué kernel corre, igual que el modelo de CPU no necesita saber
qué kernel de NPB corre — ambos predicen directo desde el vector de telemetría
actual. Esto también resuelve limpio el hecho de que **los kernels de GPU son
de Rodinia (terceros), no código propio** — no hace falta instrumentar nada
dentro de esos binarios, exactamente como hoy no se instrumenta NPB.

---

## 5. Calibración Roofline de GPU

Necesaria para dar el paso de "intensidad medida con `ncu`" a "etiqueta
compute/memory-bound" en la sección 3. Estructura recomendada, preservando la
simetría metodológica con CPU:

- **Ancho de banda:** BabelStream (equivalente aceptado de STREAM en GPU).
- **Pico de FLOPs:** microbenchmark propio (`ert_probe_gpu.cu`, ARC-76),
  análogo directo de `ert_probe.c` en CPU -- un bucle multiplicar-sumar que
  cada hilo corre enteramente en un registro, en aritmética CUDA corriente
  (sin ninguna librería de terceros).

**Corrección de diseño (ARC-76)**: la primera versión de esta calibración
usaba `cuBLAS DGEMM` como fuente de `P_pico` -- se descartó porque `cuBLAS`
puede elegir, sin que se le pida, una ruta de hardware acelerada que los
demás kernels del catálogo no usan, dando un `P_pico` que no representa lo
que esos kernels pueden alcanzar. El microbenchmark propio evita ese riesgo
por construcción (mismo principio que ya rige `ert_probe` en CPU: nunca
depender de una librería optimizada de terceros para medir un techo que
se va a usar como vara de comparación).

Un riesgo sigue vigente:

- **Sin permiso de reloj, `P_pico` sale al boost que la GPU elija en el
  momento de calibrar** — no reproducible, análogo al problema turbo/HWP que
  CPU ya controla (check D01). Mitigación mínima: registrar `clocks.current.sm`
  durante la calibración junto al valor medido.

Anticipar que el ridge de GPU es más alto que el de CPU (BW alcanzable
~1.3-1.4 TB/s, FP64 vanilla → `i_ridge` ≈ 7 FLOP/byte): más kernels van a
clasificar `memory_bound` en GPU que en CPU. Es resultado esperable, no una
señal de error.

---

## 6. Niveles de frecuencia de GPU (FG_n)

Mismo esquema que ya usan para CPU (`Guia_Maestra_Fase1_DVFS.md` sección 8.1:
F0-F4 a 100/75/50/25/mín% del rango `[f_min, f_max]`). Aplicado al rango
confirmado en el A100 (765-1410 MHz de reloj de SM; el de memoria no cuenta,
solo hay un valor soportado):

| Nivel | % del rango | MHz objetivo (antes de redondear) |
|---|---|---|
| FG0 | 100% | 1410 |
| FG1 | 75% | ~1169 |
| FG2 | 50% | ~1088 |
| FG3 | 25% | ~926 |
| FG4 | mín (0%) | 765 |

**Diferencia importante frente a CPU:** el A100 tiene 81 valores soportados,
no necesariamente espaciados uniformemente (a diferencia de los pasos limpios
de 100 MHz que sí tiene el `intel_pstate` de pacca). Cada FG_n calculado por
porcentaje debe redondearse al valor soportado más cercano consultando
`nvmlDeviceGetSupportedGraphicsClocks()` en vivo — nunca asumir espaciado
uniforme.

---

## 7. Campaña de caracterización de Fase 1 (GPU)

Es el equivalente exacto de `campaign_pacca_dvfs.yaml` (CPU, F0-F4), pero
para GPU. **No necesita nada de lo que se descartó** (CUPTI, límites de
kernel en vivo) — el reloj se fija para toda la corrida del binario, igual
que CPU fija frecuencia para toda una corrida de NPB:

**Por cada FG_n, antes de correr ningún kernel Rodinia (ARC-79, corrige el
gap de la sección 3):**
0. Fijar el reloj de SM a FG_n y calibrar `P_pico_gpu(FG_n)` corriendo
   `gpu_ert_probe_fp32`/`gpu_ert_probe_fp64` a ese reloj -- análogo exacto
   de lo que `calibration.py` ya hace por nivel del lado CPU (ARC-78).
   `BW_pico_gpu` (`gpu_stream_bw`) se mide una sola vez para toda la
   campaña, no por nivel (sección 3, paso 2). Calcular
   `i_ridge_gpu(FG_n) = P_pico_gpu(FG_n) / BW_pico_gpu` y, con la intensidad
   `FLOPs/byte` de cada kernel (medida una sola vez con `ncu`, fuera de este
   loop), derivar la etiqueta `compute_bound`/`memory_bound` que le
   corresponde a CADA kernel **en este `FG_n` específico**.

Por cada kernel Rodinia × cada FG_n × repeticiones:
1. Fijar el reloj de SM a FG_n (`nvidia-smi -lgc <mhz>,<mhz>`).
2. Correr el kernel, muestreando NVML en vivo durante toda la corrida
   (util_pct, power_mw, memoria, clock alcanzado) — **estas muestras son las
   *features* de entrenamiento**, no un subproducto.
3. Medir energía con `nvmlDeviceGetTotalEnergyConsumption()` (delta,
   milijulios acumulados — el análogo exacto de RAPL; nunca integrar
   `power_mw`, que es un *gauge* filtrado y con lag, la peor señal posible
   para acumular energía a través de fronteras de fase).
4. Restaurar el reloj original al terminar (mismo patrón de snapshot/restore
   que ya existe para CPU).
5. Etiquetar las muestras NVML de esta combinación (kernel, FG_n) con la
   etiqueta calculada en el paso 0 **para este mismo FG_n** -- nunca con la
   etiqueta de otro nivel, y nunca asumiendo que es la misma en todos los
   niveles.

La intensidad `FLOPs/byte` de cada fila sale de `ncu`, corrido una sola vez
por kernel (invariante de frecuencia, sección 3 paso 1) -- lo que SÍ se
recalcula por `FG_n` es el ridge contra el cual se compara esa intensidad
(paso 0 de arriba), y por lo tanto la etiqueta final.

**Esto no necesita el permiso P4 para diseñarse ni para escribirse** — solo
para ejecutarse. El manifiesto puede prepararse hoy (mismo patrón que
`campaign_pacca_dvfs.yaml`, preparado antes de tener el permiso de CPU).

---

## 8. Estado de implementación

**Hecho y verificado (2026-08-06, ARC-66):**

- `telemetry/include/telemetry/gpu_clock_controller.hpp` **ya refactorizado**:
  `on_phase_begin()` recibe `GpuPhaseLabel` (la etiqueta ya decidida por
  `ncu` en Fase 1 o por `modelo_gpu.predict()` en el daemon), no una
  intensidad — no clasifica nada internamente, solo posee la lógica de
  histéresis por tiempo mínimo de permanencia (`min_dwell_ns`) y aplica el
  cambio de reloj a través de un `ClockSetter` inyectado.
  `telemetry/tests/test_gpu_clock_controller.cpp` actualizado a la nueva
  firma (primera aplicación incondicional, misma etiqueta no cambia nada,
  supresión por dwell insuficiente, aplicación al superar el dwell, setter
  que falla no corrompe el estado).
- **Muestreo de NVML sacado del tick de 1 ms** en `collector.cpp`/`.hpp`:
  nuevo campo `CollectorConfig::gpu_interval_ns` (100 ms por defecto). El
  productor sigue siendo un único hilo (el ring `SPSCRing` es estrictamente
  de un solo productor, así que no se puede meter un segundo hilo/ring sin
  romper esa garantía) — la solución es una compuerta de tiempo dentro del
  mismo hilo: solo se llama a NVML cuando ya pasó `gpu_interval_ns` desde la
  última lectura, en vez de en cada tick de 1 ms.
- **Test nuevo `test_collector_gpu_cadence.cpp`** que instancia un
  `Collector` real con GPU habilitada y confirma empíricamente (no solo por
  inspección de código) que en una corrida de 300 ms con
  `gpu_interval_ns=50ms` se reciben ~6 muestras de GPU, no ~300 — se
  saltea (exit 77) en builds sin `TELEMETRY_WITH_GPU`, mismo patrón que
  `perf_reader_pid_live_test`.
- **Verificado en dos configuraciones locales:** build normal (sin GPU, 12
  tests, el nuevo se saltea correctamente) y build con `WITH_GPU=ON` contra
  un stub local de `nvml.h`/`libnvidia-ml.so` (símbolos mínimos, sin
  hardware real) para confirmar que la rama `#ifdef TELEMETRY_WITH_GPU`
  compila y corre — 11/11 en verde en esa configuración, incluida la prueba
  de cadencia real. El stub se usó solo para compilar/probar localmente y
  se descartó, no se comprometió al repo.

**Hecho y verificado en hardware real (2026-08-06, ARC-72/73/74) — Fase 1 de
GPU completa de punta a punta:**

- **Kernels Rodinia reales**: `hotspot` y `backprop` (floats confirmados;
  `pathfinder`, candidato inicial, se descartó por ser enteros puros —
  mismo error que `npb_is`/`npb_ep`, ARC-57). Compilados y corridos en
  `paccaA100`.
- **Shim de blocking-sync implementado y verificado**:
  `orchestrator/native/blocking_sync_shim.cpp` + `orchestrator/gpu_shim.py`
  (compilación on-demand, mismo patrón que `pmc_multiplex_probe.c`).
  Confirmado con un probe dedicado: 99.8% CPU (spin) sin el shim, 0.0%
  (bloqueo real) con él, sin alterar la salida de los kernels.
- **Calibración Roofline de GPU real**: BabelStream (Triad ≈1.399 TB/s,
  89.9% del pico teórico de una A100-PCIe-40GB confirmada por SKU) +
  `ert_probe_gpu.cu` (ARC-76, microbenchmark propio en aritmética CUDA
  corriente, sin `cuBLAS`) dando FP64=4698.6 GFLOP/s, FP32=10178.2 GFLOP/s.
  **Corrección respecto al primer intento (ver
  `Consolidacion_Kernels_Dataset_Fase1.md` sección 0)**: se probó primero
  con `cublas_dgemm_bench` (≈10.4 TFLOP/s), pero `cuBLAS` eligió por su
  cuenta un kernel Tensor Core, dando un techo que los kernels Rodinia
  (sin Tensor Cores) no pueden alcanzar -- se reemplazó por el probe
  propio. Ridge resultante: **≈3.36 FLOP/byte para FP64 vainilla, ≈7.28
  FLOP/byte para FP32 vainilla** (con BW=1.399 TB/s).
- **Caracterización `ncu` real** (no asumida de literatura): `hotspot`
  (FP32) 5.03 FLOP/byte -- memory_bound contra el ridge FP32 (7.28),
  contradice nuestro hint de catálogo, no una clasificación oficial de
  Rodinia (que solo describe el dominio de la aplicación, no asigna
  compute/memory-bound); `backprop` (FP32) 0.087 FLOP/byte (memory_bound,
  margen amplio); `cublas_dgemm_bench` 68.0 FLOP/byte (compute_bound, muy
  por encima del ridge -- sigue en el catálogo como dataset, ya no como
  fuente de calibración del ridge).
- **Integración completa al orquestador**: `KernelEntry.device`
  (catalog.py), `--enable-gpu`/shim/`LD_LIBRARY_PATH` wireados en
  `runner.py`, `postprocess.py` ya no descarta las filas GPU
  (`quality_status=gpu_telemetry`, passthrough puro). 5 kernels GPU en
  `catalog.yaml` con checksums reales de `pacca-a100`.
- **Campaña real completa corrida en `paccaA100`** (`campaign_pacca_gpu_ref.yaml`,
  REF, 5 kernels × 3 repeticiones): **6/6 aceptadas, 0 rechazadas**
  (encontrado y corregido en el camino un bug real de rutas CUDA en
  `gpu_shim.py` — `nvcc` en PATH resolvía al del NVIDIA HPC SDK, sin
  `cuda_runtime.h`/`libcublas`, ver ARC-74). `windows.csv` verificado con
  filas `gpu_telemetry` reales (`gpu_power_mw` 36-39 W, `gpu_util_pct` no
  nulo), sin contaminar las ventanas de CPU del mismo run.
- **Gap real de ARC-58 encontrado y corregido en el camino (ARC-73)**:
  `manifest.py` nunca definía `projected_campaign_bytes`/
  `remaining_core_hours`/`projected_core_hours` como campos parseables,
  aunque `preflight.py` los exige desde siempre — **ningún manifiesto
  podía pasar el preflight automático de ARC-58**, ni siquiera los ya
  validados de CPU. Corregido con los tres campos opcionales en `Manifest`
  + `_parse_optional_non_negative_number()`.

**No implementado, y ya no hace falta implementarlo** (descartado en el
rediseño de la sección 0-4): inyección CUPTI, detección de límites de
kernel, tabla estática de intensidad por kernel como mecanismo de
producción.

**No implementado, pendiente de permiso P4 (bloqueado, no es cuestión de
tiempo):** el wrapper real a `nvmlDeviceSetGpuLockedClocks`/`nvidia-smi
-lgc` — no se puede probar contra hardware real sin el permiso. Todo lo
demás de Fase 1 ya no depende de este permiso.

---

## 9. Plan concreto, en orden

1. **Enviar el correo de permisos** (P1-P4 ya redactados en
   `Solicitud_Permisos_Pacca_Unicartagena.md`) — bloquea DVFS de CPU y de GPU.
   Único punto de esta lista que sigue sin resolverse.
2. ~~Refactorizar `GpuClockController`~~ — **hecho (ARC-66)**.
3. ~~Preparar el manifiesto de la campaña de caracterización GPU~~ — **hecho
   (ARC-72/73)**: `campaign_pacca_gpu_ref.yaml`, corrido en hardware real.
4. ~~Resolver `cudaDeviceScheduleBlockingSync` en binarios de terceros~~ —
   **hecho (ARC-72)**: shim `LD_PRELOAD`, verificado (99.8%→0.0% CPU).
5. ~~Sacar el muestreo NVML del loop de 1 ms de `collector.cpp`~~ — **hecho
   (ARC-66)**.
6. ~~Caracterizar con `ncu` los kernels Rodinia elegidos~~ — **hecho
   (ARC-72)**: `hotspot`/`backprop`/`cublas_dgemm_bench`, valores reales en
   sección 8.
7. ~~Correr la campaña de caracterización~~ — **hecho (ARC-72/73/74)**: 6/6
   aceptadas en `paccaA100`, `windows.csv` con datos NVML reales. No hizo
   falta esperar P4 — corrió íntegramente a frecuencia REF.
8. Medir `T_transición` real de GPU el día que llegue el permiso, para fijar
   el tiempo mínimo de permanencia con datos reales, no con el valor de
   literatura citado en el plan (~10 ms). **Único paso que sigue bloqueado
   por P4** — todo lo demás de Fase 1 está listo.

---

## 10. Qué queda explícitamente sin resolver

- `T_transición` real del cambio de reloj de GPU — no medible sin el permiso
  (único punto de Fase 1 realmente bloqueado por P4).
- ~~El etiquetado `ncu` de GPU (sección 3) no recalcula la etiqueta por
  nivel de frecuencia~~ — **corregido en diseño (ARC-79) y en código
  (ARC-80)**: `calibration.run_gpu_calibration()` calibra `P_pico_gpu`
  (fp32 y fp64 por separado, ARC-76) por cada nivel de
  `manifest.frequency_levels`, y `postprocess.py` deriva `phase_label_train`
  por combinación (kernel, freq_level_id, precisión) -- mismo principio que
  `calibration.py` ya implementa del lado CPU (ARC-78), verificado con una
  campaña real en `paccaA100`. **Sigue sin poder *recalibrar por más de un
  nivel real* hasta que llegue `P4`** (no hay manera de fijar el reloj de
  SM a distintos niveles hoy, así que solo existe el nivel `REF`) — pero
  el código ya soporta más niveles sin cambios, solo falta el permiso para
  que `manifest.frequency_levels` declare más de uno de verdad.
- ~~Qué techo FP64 corresponde a reportar como `P_pico`~~ — **resuelto
  (ARC-76)**: se dejó de usar `cuBLAS` (elige Tensor Cores por su cuenta)
  y se mide con un microbenchmark propio en aritmética CUDA corriente
  (`ert_probe_gpu.cu`), mismo criterio que `ert_probe` en CPU.
- Si el modelo de GPU debe ser exactamente la misma familia de algoritmo que
  el de CPU (Random Forest, dice la Fase 2 como ejemplo) o si conviene
  comparar candidatos distintos por dispositivo — la Fase 2 del plan ya prevé
  comparar varios modelos, esto es una instancia normal de esa comparación,
  no una decisión nueva.
- El presupuesto térmico/de potencia compartido a nivel de nodo,
  deliberadamente no modelado (fuera del alcance del plan, sección 6.2).
- El checkout usado para verificar en hardware real fue uno aislado
  (`~/hyperion-gpu-fase1` en pacca), no el `~/hyperion` real del nodo — la
  próxima sesión que retome esto en pacca debe sincronizar el código real
  antes de seguir, no asumir que `~/hyperion` ya tiene estos cambios.
