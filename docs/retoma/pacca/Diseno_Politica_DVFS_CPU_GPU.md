# Diseño de la política de inferencia DVFS con CPU y GPU — respuesta razonada

**Propósito de este documento:** respuesta a la pregunta de diseño planteada en
`Prompt_Diseno_Politica_DVFS_CPU_GPU.md`. No es un informe de estado ni un plan
de implementación cerrado: es criterio de arquitectura con la evidencia empírica
que se pudo levantar del nodo real antes de decidir. Donde una afirmación viene
de una medición propia se dice explícitamente; donde viene de conocimiento de
literatura o de la semántica documentada de una API, también.

**Fecha:** 2026-08-06. **Nodo:** `paccaA100` (NVIDIA A100-PCIE-40GB, driver
595.45.04, CUDA 13.2).

---

## 0. Resumen ejecutivo

La premisa del prompt es que el problema difícil es *coordinar* dos políticas de
control. La evidencia levantada hoy dice otra cosa:

1. **El control de reloj de GPU está bloqueado por permisos, igual que el de
   CPU.** La vía no-root que uno esperaría (application clocks) fue *deprecada*
   en este driver. Sólo queda `nvmlDeviceSetGpuLockedClocks`, que exige root.
2. **La medición de GPU, en cambio, está abierta hoy sin pedir nada**: `ncu`
   (Nsight Compute) corre como usuario normal en este nodo y entrega FLOPs y
   bytes de DRAM reales por kernel.
3. **La clasificación Roofline de GPU tal como está planteada no es
   implementable con el harness actual**, y no por falta de calibración: `util_pct`
   y `power_mw` no dan ni el numerador ni el denominador de la intensidad
   operacional.

De ahí la recomendación central: **la intensidad operacional de GPU debe venir de
caracterización estática offline por kernel (`ncu`), no de medición online**; la
telemetría NVML de runtime queda para segmentar/atribuir fases y medir energía,
nunca para calcular intensidad. Y el correo de permisos (todavía sin enviar) pasa
a ser camino crítico de dos de los tres entregables, no de uno.

---

## 1. Hallazgos empíricos en el nodo real (2026-08-06)

Todos verificados en `paccaA100` dentro de un `srun` real, consultas de lectura
salvo el test de `ncu` (que ejecuta un binario propio trivial, sin tocar estado
del nodo).

| Hecho medido | Evidencia | Consecuencia de diseño |
|---|---|---|
| **Application clocks deprecados** | `nvidia-smi -q -d CLOCK` → `Applications Clocks: Requested functionality has been deprecated` | `nvmlDeviceSetApplicationsClocks` no sirve en driver 595. La única vía de control es `nvmlDeviceSetGpuLockedClocks` / `nvidia-smi -lgc`, **que requiere root**. No hay ruta no-root al control de frecuencia de GPU en este nodo. |
| **Reloj de memoria no ajustable** | `SUPPORTED_CLOCKS` lista **1** valor de memoria (1215 MHz) y **81** de SM (765→1410 MHz) | El espacio DVFS de GPU es **unidimensional**: sólo SM clock. No hay segundo eje que explorar; la matriz experimental de GPU es más simple de lo que se podría haber asumido. |
| **`ncu` funciona sin root** | `ncu --metrics dram__bytes.sum,...` sobre un axpy propio de 2²⁴ doubles → `dram__bytes.sum = 388.80 MB` (analítico ≈402 MB, coherente) | **La caracterización real de FLOPs/bytes por kernel está disponible hoy, sin pedir permisos.** Es la pieza que faltaba para poder calcular intensidad operacional de GPU. |
| **DCGM no instalado** | `which dcgmi nv-hostengine` → ausentes | La vía de métricas de profiling continuas (`DCGM_FI_PROF_*`) requiere instalación por parte del administrador. No conviene diseñar asumiéndola. |
| **Persistence mode deshabilitado**, SM a 765 MHz en reposo, límite de potencia 250 W | `nvidia-smi --query-gpu=...` | Sin persistence mode el driver se descarga entre usos: las llamadas NVML son más lentas y el reloj cae a reposo. Relevante para medir latencia de transición y para la reproducibilidad de la calibración. |

**Implicación estratégica.** La GPU **no es un seguro** contra el bloqueo de
`scaling_min_freq` en CPU: cae en la misma clase de permiso. Lo que sí es un
seguro real es la *caracterización* de GPU, que no depende de nadie.

---

## 2. La brecha que hay que resolver antes que la coordinación

El acuerdo #2 del prompt dice que si la GPU está activa, la ventana "se clasifica
con el Roofline de GPU". Con el harness de hoy eso no tiene entrada: `GpuSample`
sólo tiene `power_mw` y `util_pct` (ver `telemetry/include/telemetry/metrics.hpp`),
y ninguno de los dos es una medida de trabajo ni de tráfico de memoria. Además,
`orchestrator/postprocess.py` **hoy descarta las filas GPU por completo** — no hay
ninguna columna de GPU en `REQUIRED_OUTPUT_COLUMNS`; el launcher las escribe a
`samples.csv` y nadie las consume.

La solución compatible con el cronograma y con la arquitectura existente es
reconocer que **en el lado CPU la intensidad tampoco se mide del todo**: los FLOPs
salen del stdout del propio binario y se prorratean por `delta_instructions` (una
aproximación declarada, POST-08/09/10). El análogo GPU es:

- **Intensidad operacional estática por kernel**, medida una vez con `ncu`
  (`dram__bytes.sum` para bytes; los contadores `smsp__sass_thread_inst_executed_op_*`
  o `sm__sass_thread_inst_executed_op_dfma_pred_on.sum` para FLOPs), almacenada en
  `catalog.yaml` junto a los campos que ya existen para CPU.
- **Telemetría NVML de runtime usada sólo para segmentar y atribuir** (¿está la
  GPU ocupada? ¿desde cuándo?) y para energía — nunca para calcular intensidad.

Esto además no pierde información real: los kernels de un benchmark GPU son
homogéneos dentro de un launch, así que una intensidad por-ventana de 1 ms sería
la misma cifra repetida con ruido de muestreo encima.

---

## 3. Respuestas a las cinco preguntas

### 3.1 Granularidad de la decisión y anti-flapping

Con intensidad estática por kernel, la granularidad de decisión cae naturalmente
en **la fase** — la corrida maximal de kernels consecutivos con la misma
etiqueta — no en el launch individual. Es una buena señal de diseño que las dos
restricciones (costo alto de conmutación, e intensidad constante dentro del
kernel) apunten al mismo lugar.

Mecanismo anti-flapping, en dos partes que hacen falta juntas:

- **Histéresis de doble umbral**: no volver a cruzar la etiqueta hasta que la
  intensidad supere el ridge por un margen ±δ. Evita oscilación en kernels que
  caen justo sobre el punto de inflexión.
- **Tiempo mínimo de permanencia**: `T_dwell ≥ K · T_transición`, con K ≈ 10 como
  punto de partida. Sin esto, la histéresis sola no impide pagar el costo de
  transición en una fase demasiado corta para amortizarlo.

**El número que gobierna todo esto —`T_transición`— hoy no está medido.** El
prompt lo cita como "decenas de ms según literatura". Ese es exactamente el tipo
de dato que este proyecto ya decidió dos veces no dar por supuesto (ARC-52: no
adivinar encodings crudos de memoria; ARC-59: no asumir que uncore estaba
bloqueado sin probarlo). Medirlo es ~1 hora de trabajo el día que llegue el
permiso: fijar el reloj, hacer polling de `nvmlDeviceGetClockInfo` hasta alcanzar
el objetivo, y registrar el tiempo hasta que se estabiliza. Hasta entonces, diseñar
asumiendo 10–100 ms y dejar `K` como parámetro del manifiesto, no como constante
en el código.

### 3.2 Atribución CPU vs GPU

**`util_pct` no significa lo que parece.** `nvmlDeviceGetUtilizationRates()`
devuelve el *porcentaje de tiempo, durante el último período interno de muestreo,
en que hubo al menos un kernel ejecutándose*. Es ocupación temporal, no
throughput: un kernel usando 1 SM de 108 reporta 100%. Para "¿hay algo
corriendo?" sirve; para "¿cuánto trabajo?" no dice nada — lo cual es consistente
con usarlo sólo como señal de atribución, como propone el prompt.

El problema es la resolución: **el período interno de muestreo de NVML es del
orden de ~1 s** en muchos drivers. El collector hoy lo consulta cada 1 ms
(`collector.cpp`, mismo loop que CPU y RAPL), así que recibe **el mismo valor
rancio repetido ~1000 veces seguidas**. No hay atribución sub-segundo posible sólo
con NVML, por mucho que se muestree rápido.

Señales mejores, en orden de costo de implementación:

1. **Marcadores explícitos (NVTX o un log con timestamp) en los límites de
   kernel.** El proyecto compila sus propios benchmarks, así que instrumentarlos
   es legítimo y es de lejos lo más robusto para un experimento controlado. Costo
   casi nulo, cero permisos.
2. **CUPTI *Activity* API** — importante no confundirla con la API de
   *profiling*: la Activity API entrega timestamps de inicio/fin de kernel con
   overhead bajo y **sin kernel replay ni serialización**. Es mucho más barata de
   lo que el prompt sugiere y es la respuesta "correcta" si aparece tiempo.
3. **DCGM `SM_ACTIVE`/`GR_ENGINE_ACTIVE`**, si el administrador instala DCGM.

### 3.3 Calibración Roofline de GPU

Estructura recomendada, preservando la simetría metodológica con el lado CPU (que
es lo que hace defendible el capítulo):

- **Ancho de banda: BabelStream**, no un kernel propio. Es el equivalente aceptado
  de STREAM en GPU, tiene backend CUDA, y usar el estándar de la comunidad es
  exactamente el mismo argumento por el que hoy usan STREAM oficial en CPU.
- **Pico de FLOPs: cuBLAS DGEMM**, análogo directo al `dgemm_n2048` que ya está
  en el catálogo.

Dos riesgos que pesan más que el clásico "teórico vs alcanzable":

- **La elección del techo FP64 cambia `i_ridge` por 2×.** El A100 tiene FP64
  vanilla (~9.7 TFLOP/s) y FP64 Tensor Core (~19.5 TFLOP/s). Un factor 2 en
  `P_pico` mueve el punto de inflexión por 2 y **voltea la etiqueta de cualquier
  kernel cercano al ridge**. Hay que declarar cuál se usa y que coincida con lo
  que los kernels medidos realmente ejercitan (si cuBLAS selecciona tensor cores
  automáticamente, el `P_pico` observado ya es el de TC).
- **Sin root no se puede fijar el reloj durante la calibración**, así que `P_pico`
  sale al boost que la GPU elija en ese momento — no reproducible. Es el análogo
  exacto del problema turbo/HWP que en CPU ya se controla con el check D01.
  Mitigación mínima mientras no haya permiso: ya se muestrea NVML, así que
  **registrar `clocks.current.sm` durante la calibración y reportarlo junto al
  valor**, en vez de publicar un `P_pico` sin estado de reloj asociado.

Anticipar además que **el ridge de GPU es bastante más alto que el de CPU**: con
BW alcanzable del orden de 1.3–1.4 TB/s y FP64 vanilla, `i_ridge` ≈ 7 FLOP/byte.
Muchos más kernels van a clasificar `memory_bound` en GPU que en CPU. Es un
resultado esperable y reportable, pero conviene saberlo antes de sorprenderse con
la distribución de etiquetas.

### 3.4 ¿Dos políticas o un optimizador conjunto?

**Dos políticas independientes con una regla de coordinación**, sin dudarlo, por
asimetría de riesgo:

- Un optimizador conjunto necesita una recompensa conjunta = energía total, que a
  su vez necesita atribución de energía por dispositivo confiable y sincronizada.
  Este proyecto **acaba de demostrar lo difícil que es eso**: ARC-56 documentó
  `power_w` con picos de hasta 106 kW por un desajuste de cadencia entre RAPL y
  las ventanas de CPU. Construir un optimizador sobre una señal de potencia de GPU
  que además es *filtrada y con lag* sería optimizar ruido.
- Quedan ~3 semanas y hay un pipeline de CPU funcionando y validado que no
  conviene arriesgar.
- El acoplamiento físico real es débil: relojes y dominios de potencia separados.

La regla de coordinación que cubre la mayor parte del acoplamiento es un solo
condicional: **GPU ocupada ⟹ reloj de CPU al mínimo**, con prioridad sobre la
etiqueta Roofline del CPU. La sección siguiente explica por qué esto no es una
optimización oportunista sino una corrección obligatoria.

El único acoplamiento que queda sin modelar es el presupuesto térmico/de potencia
a nivel de nodo. A esta escala es de segundo orden; conviene nombrarlo como
conocido-y-diferido en el escrito, no modelarlo.

### 3.5 Trampas y precedentes

**(a) El spin-wait del CPU se ve `compute_bound` — la trampa más grave, y es
específica de este código.** `cudaDeviceSynchronize()` hace *spin* por defecto
(`cudaDeviceScheduleAuto` gira cuando el número de hilos no supera al de cores).
Un CPU en spin-wait muestra IPC alto y prácticamente cero cache misses, lo que en
este pipeline da `bytes_moved_window ≈ 0` y por lo tanto una intensidad operacional
enorme ⟹ **`compute_bound` ⟹ la política sube el reloj justo en la ventana donde
el CPU no hace nada más que quemar potencia esperando a la GPU**. Es la decisión
máximamente equivocada, producida por un clasificador que funciona correctamente
alimentado con una entrada sin sentido físico.

Esto convierte el acuerdo #3 del prompt ("bajar la frecuencia ahí parece gratis")
en algo más fuerte: **no es gratis, es obligatorio**, y no por ahorro sino para
evitar un error sistemático. Dos defensas, ambas baratas:
`cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` para que el hilo realmente se
bloquee, y que "GPU ocupada" sea un override explícito de la etiqueta del CPU.
(Nota: cuando los bytes son exactamente 0 el pipeline ya marca
`intensity_undefined` y no entrena con esa ventana — el peligro real es el régimen
de bytes pequeños-pero-no-cero, donde la intensidad sale grande y creíble.)

**(b) Muestrear NVML dentro del loop de 1 ms degrada la calidad del muestreo de
CPU.** Hoy `collector.cpp` hace dos ioctls NVML por iteración, en el mismo hilo que
lee los contadores de CPU y RAPL. El proyecto ya rastrea `sampling_interval_cv_pct`
como indicador de salud, y ARC-55/56 mostró la cadena de daño: jitter ⟹ ventanas
anómalamente cortas ⟹ métricas derivadas sin sentido. Recomendación: sacar el
muestreo de GPU a su propia cadencia (~50–100 ms, acorde a la tasa real de
actualización de NVML), lo que además elimina el almacenamiento de ~1000 valores
rancios duplicados por segundo.

**(c) No integrar `power_mw` para obtener energía.** Es un *gauge* filtrado
internamente, con lag — la peor clase de señal para integrar a través de fronteras
de fase. Existe **`nvmlDeviceGetTotalEnergyConsumption()`** (Volta en adelante):
milijulios acumulados desde la carga del driver, es decir **el análogo exacto de
RAPL**, un contador del que se toman deltas exactos. Hoy no se está usando y
debería reemplazar a `power_mw` como fuente de energía (dejando `power_mw` como
señal auxiliar de forma, si acaso). Es el mismo salto de calidad que en CPU
significó pasar a leer `energy_uj`.

**(d) Incluir el governor por defecto como brazo baseline.** En GPUs modernas el
power management interno ya captura buena parte de la ganancia fácil, y hay
precedentes en la literatura de políticas ingenuas que *pierden* contra el
comportamiento por defecto. La maquinaria ya existe: `schedule_runs` ya trata
baseline y telemetry como par atómico.

**(e) No dejar que el alcance GPU hunda el entregable CPU.** Con el permiso de
escritura de frecuencia de CPU todavía sin conceder, el riesgo real del cronograma
es terminar sin una historia completa de CPU *ni* de GPU. El orden de adición
debería ser: caracterización GPU (sin permisos) → energía GPU (sin permisos) →
control GPU (con permisos), y nunca al revés.

---

## 3.6 Decisión final de arquitectura (2026-08-06, tras discusión con el usuario)

El usuario confirmó dos cosas que cambian el punto de partida de las secciones
anteriores: **el permiso de reloj de GPU sí va a llegar** (P4 no es un "por si
acaso"), y **el harness no debe tratarse como fijo** — se construyó pensando
sólo en CPU (un solo loop de muestreo a 1 ms, una sola tabla de ventanas en
`postprocess.py`) y esa arquitectura hay que replantearla para GPU, no
parchearla.

**Decisión:** el harness pasa de "un loop, una tabla" a **dos dominios de
control desacoplados**, coordinados por una sola señal compartida:

- **CPU+RAPL**: sigue exactamente igual que hoy — tick de 1 ms, decisión por
  ventana, cambio de reloj casi instantáneo.
- **GPU**: deja de vivir dentro de `collector.cpp`/el tick de 1 ms. La unidad
  de decisión es la **fase** (una corrida maximal de kernels con la misma
  clasificación), no una ventana de tiempo fijo — porque cambiar el reloj de
  GPU es caro, decidir cada 1 ms sobre datos de GPU desperdicia más en
  overhead de transición del que ahorra.
- **Señal compartida única**: "GPU ocupada" gatea la política de CPU (GPU
  ocupada ⟹ CPU a mínimo, sin importar la clasificación de la fase GPU — ver
  3.5.a sobre el peligro del spin-wait). No hay más acoplamiento que ése.

**Mecanismo de límite de fase elegido: llamada directa en el punto de
lanzamiento, dentro del mismo proceso — no un colector externo leyendo NVML.**
A diferencia de CPU (donde el kernel es un binario opaco y el harness lo
observa desde afuera vía PID+`inherit`), los benchmarks de GPU (BabelStream,
cuBLAS DGEMM, y los que se escriban para el catálogo) **sí son código propio,
compilado por el proyecto** — así que no hace falta instrumentación externa
(NVTX leído por otro proceso, o inyección CUPTI) para saber cuándo empieza y
termina una fase: el propio código que hace `cudaLaunchKernel` puede llamar
directamente al motor de decisión antes de lanzar el siguiente lote de
kernels. Esto elimina toda la complejidad de correlacionar eventos entre
procesos. La inyección CUPTI (`CUDA_INJECTION64_PATH`) queda anotada como
alternativa futura únicamente para si el proyecto necesitara medir binarios
GPU de terceros sin acceso al código fuente — no es necesaria para el plan
actual.

**Implementado hoy:** `telemetry/include/telemetry/gpu_clock_controller.hpp`
(+ `telemetry/tests/test_gpu_clock_controller.cpp`, 11/11 tests C++ en verde
incluyendo el nuevo). Es el motor de decisión puro: clasifica la intensidad
estática de la fase contra `i_ridge` con una banda de histéresis (evita
reclasificar por ruido cerca del punto de inflexión), aplica un piso de
permanencia mínima (`min_dwell_ns`) antes de permitir un nuevo cambio de
reloj, y **no depende de NVML/CUDA en absoluto** — el cambio de reloj real se
inyecta como una función (`ClockSetter`), igual que `campaign.py` inyecta
`apply_frequency()` en vez de llamar a `freqctl` directo. Esto permite
testear toda la lógica de histéresis/dwell/fallo-de-aplicación en cualquier
máquina, sin GPU. Deliberadamente NO implementado todavía (fuera de alcance
de "listo hoy"): el wrapper real que llama a
`nvmlDeviceSetGpuLockedClocks`/`nvidia-smi -lgc` (bloqueado por P4, sin
permiso no se puede probar en hardware real), y la tabla de intensidad
estática por kernel (requiere corridas de `ncu` sobre los kernels GPU
definitivos, que todavía no están escritos).

**Qué falta para que esto corra en hardware real**, en orden:
1. Que llegue el permiso P4 (bloqueante para probar el `ClockSetter` real).
2. Escribir los benchmarks GPU del catálogo (BabelStream, cuBLAS DGEMM) con
   la llamada a `on_phase_begin()` en sus puntos de lanzamiento.
3. Caracterizar cada uno con `ncu` (ya confirmado que funciona sin permisos,
   sección 1) para poblar la tabla de intensidad estática.
4. Medir `T_transición` real en el nodo el día que llegue el permiso, para
   fijar `min_dwell_ns` con datos en vez de con el valor de literatura.

---

## 4. Plan sugerido para la semana

1. **Enviar hoy el correo de permisos**, ahora incluyendo el ítem de GPU locked
   clocks (ver `Solicitud_Permisos_Pacca_Unicartagena.md`, P4). Bloquea DVFS de
   CPU *y* de GPU: es el camino crítico del proyecto entero.
2. **Caracterizar con `ncu` los kernels GPU candidatos** — funciona ya, sin
   permisos — y llevar la intensidad estática al catálogo. Esto es entregable y
   reportable aunque el permiso nunca llegue.
3. Sacar NVML del loop de 1 ms y cambiar `power_mw` por
   `nvmlDeviceGetTotalEnergyConsumption()`.
4. Blindar el spin-wait antes de que contamine cualquier corrida heterogénea.
5. Consumir las filas GPU en `postprocess.py` (hoy se escriben y se descartan).

El punto 2 es el seguro real del cronograma: **caracterización más evaluación
offline de la política es una contribución completa que no depende de ningún
permiso**. El lazo cerrado es lo que se agrega si el permiso llega, no la base
sobre la que se apuesta el proyecto.

---

## 5. Qué queda explícitamente sin resolver

- `T_transición` real del cambio de reloj de GPU (no medible sin permiso de
  escritura; todo el parámetro `K` de anti-flapping depende de ese número).
- Qué techo FP64 (vanilla vs Tensor Core) corresponde a los kernels que
  finalmente se midan — decisión pendiente de tener el conjunto de kernels GPU
  cerrado.
- Si vale la pena instrumentar CUPTI Activity API para atribución fina, o si los
  marcadores explícitos en los benchmarks propios alcanzan. Depende de cuánto
  margen quede después de los puntos 1–5 de arriba.
- El presupuesto térmico/de potencia compartido a nivel de nodo, deliberadamente
  no modelado.
