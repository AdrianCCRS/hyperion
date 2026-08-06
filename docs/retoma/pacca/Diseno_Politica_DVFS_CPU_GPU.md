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

**Consecuencia estratégica:** el control de reloj de GPU está bloqueado por
permisos igual que el de CPU — la GPU no es un plan B si el permiso de CPU
tarda, cae en el mismo bloqueador. Ver `Solicitud_Permisos_Pacca_Unicartagena.md`
P4 (ya redactado).

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

Concretamente, en Fase 1:

1. Por cada kernel candidato (de Rodinia u otra fuente), se corre `ncu` **una
   sola vez** (la intensidad operacional de un kernel no cambia con la
   frecuencia — depende del algoritmo y el tamaño de datos, no del reloj) y
   se obtiene `FLOPs/byte` real, medido con contadores de hardware, no
   estimado.
2. Esa intensidad se compara contra `i_ridge_gpu` (de la calibración Roofline
   de GPU, sección 5) para producir una etiqueta `compute_bound`/`memory_bound`
   — el análogo de `phase_label_hint`/`phase_label_train`, pero con `ncu`
   jugando un papel más fuerte (medición real, no hint de literatura).
3. **Esa etiqueta se le asigna a todas las muestras NVML** recolectadas
   mientras ese kernel corre, en **todos los niveles de frecuencia FG_n**
   (sección 6) — porque la etiqueta física del kernel no depende del reloj al
   que se lo mida.
4. El modelo de GPU (Fase 2) se entrena sobre esas muestras NVML con esa
   etiqueta como target — **`ncu` nunca vuelve a ejecutarse después de Fase
   1**, ni en el daemon, ni en ninguna corrida de validación.

Esto resuelve limpio la restricción del plan (NVML es la única entrada de
producción) sin perder rigor en el etiquetado (no se etiqueta a ojo ni con un
hint de literatura, se mide).

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
- **Pico de FLOPs:** cuBLAS DGEMM (análogo de `dgemm_n2048`).

Dos riesgos a anticipar:

- **La elección del techo FP64 cambia `i_ridge` por 2×.** El A100 tiene FP64
  vanilla (~9.7 TFLOP/s) y FP64 Tensor Core (~19.5 TFLOP/s). Debe declararse
  cuál se usa y verificar que coincide con lo que los kernels reales ejercitan.
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

La etiqueta de cada fila (sección 3) sale de `ncu`, corrido una sola vez por
kernel, no por combinación FG_n/repetición.

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

**No implementado, y ya no hace falta implementarlo** (descartado en el
rediseño de la sección 0-4): inyección CUPTI, detección de límites de
kernel, tabla estática de intensidad por kernel como mecanismo de
producción.

**No implementado, pendiente de permiso P4 (bloqueado, no es cuestión de
tiempo):** el wrapper real a `nvmlDeviceSetGpuLockedClocks`/`nvidia-smi
-lgc` — no se puede probar contra hardware real sin el permiso.

**No implementado, pendiente de que se elijan los kernels Rodinia
concretos:** el manifiesto de la campaña de caracterización GPU (sección
7) y el mecanismo de `cudaDeviceScheduleBlockingSync` (sección 4.1) — este
último en particular necesita decidirse cómo se aplica a binarios de
terceros sin tocar su fuente (candidato: un shim vía `LD_PRELOAD` que
intercepte la primera llamada de contexto CUDA e inserte
`cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` antes — no
implementado ni probado todavía, es una hipótesis de mecanismo, no una
decisión cerrada).

---

## 9. Plan concreto, en orden

1. **Enviar el correo de permisos** (P1-P4 ya redactados en
   `Solicitud_Permisos_Pacca_Unicartagena.md`) — bloquea DVFS de CPU y de GPU.
2. ~~Refactorizar `GpuClockController`~~ — **hecho (ARC-66)**.
3. Preparar el manifiesto de la campaña de caracterización GPU (sección 7) —
   no necesita el permiso para escribirse, solo para correrse. Pendiente de
   que se elijan los kernels Rodinia concretos.
4. Resolver `cudaDeviceScheduleBlockingSync` en binarios de terceros
   (sección 4.1, hipótesis: shim `LD_PRELOAD`) antes de cualquier corrida
   heterogénea real.
5. ~~Sacar el muestreo NVML del loop de 1 ms de `collector.cpp`~~ — **hecho
   (ARC-66)**.
6. Caracterizar con `ncu` los kernels Rodinia elegidos, una vez que estén
   seleccionados (sección 3).
7. Correr la campaña de caracterización (sección 7) en cuanto llegue P4.
8. Medir `T_transición` real de GPU el día que llegue el permiso, para fijar
   el tiempo mínimo de permanencia con datos reales, no con el valor de
   literatura citado en el plan (~10 ms).

---

## 10. Qué queda explícitamente sin resolver

- `T_transición` real del cambio de reloj de GPU — no medible sin el permiso.
- Qué techo FP64 (vanilla vs Tensor Core) corresponde a los kernels Rodinia
  que finalmente se elijan — depende de qué kernels queden seleccionados.
- Si el modelo de GPU debe ser exactamente la misma familia de algoritmo que
  el de CPU (Random Forest, dice la Fase 2 como ejemplo) o si conviene
  comparar candidatos distintos por dispositivo — la Fase 2 del plan ya prevé
  comparar varios modelos, esto es una instancia normal de esa comparación,
  no una decisión nueva.
- El presupuesto térmico/de potencia compartido a nivel de nodo,
  deliberadamente no modelado (fuera del alcance del plan, sección 6.2).
