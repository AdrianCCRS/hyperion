# Prompt — diseño de la política de inferencia DVFS cuando hay CPU y GPU

**Propósito de este documento:** no es un informe de estado, es una pregunta
de diseño preparada para llevarse tal cual (o casi) a un modelo más potente
(Opus, Fable) que pueda pensarla con más profundidad de la que dio el
tiempo en esta sesión. Contiene todo el contexto necesario para que ese
modelo pueda responder sin haber visto el resto de la conversación.

---

## Contexto del proyecto

Hyperion es un pipeline de medición e inferencia para decidir políticas de
frecuencia de CPU (DVFS — Dynamic Voltage and Frequency Scaling) en función
de si una fase de ejecución es `compute_bound` o `memory_bound`, usando el
modelo Roofline: se mide la intensidad operacional real (FLOPs/bytes
movidos) de cada ventana de tiempo y se compara contra el punto de
inflexión (`i_ridge = P_pico/BW_pico`) del hardware. Si la intensidad está
por debajo del ridge, la fase es `memory_bound`; si está por encima, es
`compute_bound`. Esa etiqueta binaria es la que alimenta la política: en
fases `memory_bound` normalmente conviene bajar la frecuencia de CPU (el
cuello de botella es el ancho de banda de memoria, no la velocidad del
core, así que bajar el reloj ahorra energía con poco o ningún costo de
tiempo), y en fases `compute_bound` conviene mantenerla alta.

Hoy esto está implementado y validado en hardware real **solo para CPU**:
un harness en C++ abre contadores de hardware (`perf_event_open`, PID +
`inherit=1`) sobre el proceso medido y produce ventanas con IPC, tasa de
fallos de caché, ciclos de stall, etc. Un orquestador en Python calcula
`operational_intensity` por ventana y la clasifica contra el Roofline del
nodo. Ya se corrió una campaña real de 21 corridas (7 kernels NPB/DGEMM ×
3 repeticiones) en un nodo CPU-only y la clasificación salió coherente con
lo esperado.

El proyecto ahora tiene acceso a un nodo con GPU (NVIDIA A100). El harness
**ya captura muestras de GPU** vía NVML (`GpuSample`: `power_mw`,
`util_pct`, con timestamp propio en el mismo ring buffer que las muestras
de CPU), pero eso es lo único que existe del lado GPU — no hay
calibración Roofline de GPU, no hay clasificación de fase GPU, y no hay
ningún mecanismo de control de frecuencia de GPU implementado.

**Restricción de cronograma:** quedan ~3 semanas para cerrar el proyecto.
Se quería tener un piloto real de la parte GPU corriendo antes de decidir
la arquitectura de esto, pero el tiempo apremia y hay que decidir el diseño
ahora, sin ese piloto previo.

---

## Lo que ya se discutió y se dio por acordado en esta sesión (no re-abrir sin razón)

1. **La etiqueta sigue siendo binaria** (`compute_bound`/`memory_bound`),
   no se necesitan 4 clases combinando dispositivo × tipo. El concepto
   Roofline es agnóstico de arquitectura; lo que cambia entre CPU y GPU es
   qué métricas alimentan el cálculo de intensidad operacional y contra qué
   `i_ridge` se compara (cada dispositivo necesita su propia calibración:
   `P_pico`/`BW_pico`/`i_ridge` propios, obtenidos con un microbenchmark
   análogo al que ya se usa para CPU — para GPU, algo como cuBLAS GEMM para
   el pico de FLOPs y un kernel de ancho de banda puro para el pico de
   memoria).

2. **La atribución de "a qué dispositivo pertenece esta ventana" es una
   señal aparte del label**, no una tercera clase que el modelo tenga que
   predecir. La idea de partida: usar `util_pct` de GPU (ya capturado por
   NVML) como señal de atribución — si la GPU está activa en una ventana,
   esa ventana se clasifica con el Roofline de GPU y controla el reloj de
   GPU; si no, es una ventana puramente CPU y sigue el camino que ya existe
   hoy.

3. **Mientras la GPU trabaja, el CPU normalmente está bloqueado esperando**
   (sincronización), así que su política en esa ventana no depende del
   label Roofline — es un estado ocioso, no una decisión de intensidad
   operacional. Bajar la frecuencia del CPU ahí parece "gratis"
   energéticamente, sin relación con si la fase GPU es compute o
   memory-bound.

4. **En ejecución asíncrona (CPU haciendo trabajo real mientras la GPU
   computa en paralelo)**, los dos labels aplicarían en paralelo a sus
   propios relojes, sin interferencia entre sí — son relojes físicamente
   independientes, a diferencia del riesgo de dominio de frecuencia
   compartido dentro de un mismo procesador (SMT/socket) que sí existe en
   el lado CPU.

5. **Asimetría de latencia de control:** cambiar la frecuencia de CPU
   (`intel_pstate`, escribiendo `scaling_min_freq`/`scaling_max_freq`) es
   casi instantáneo y por core. Cambiar el reloj de GPU (`nvidia-smi`/NVML,
   `nvmlDeviceSetGpuLockedClocks` o equivalente) es mucho más lento (del
   orden de decenas de milisegundos reportado en literatura previa) y
   normalmente aplica al dispositivo completo, no a un SM ni a un kernel
   individual. Esto probablemente obliga a que la política de GPU decida
   por invocación de kernel o por fase completa, no por ventana de muestreo
   de 1 ms como hace hoy la política de CPU.

6. **El costo de reconfigurar frecuencia de GPU es alto** — esto ya se
   sabía de antemano por estudios/papers previos revisados por el equipo
   (no de esta sesión, información previa del proyecto). No se tiene el
   número exacto a mano en este momento para citarlo aquí; el modelo que
   reciba este prompt debe tratarlo como una restricción fuerte a
   considerar en el diseño (alta latencia + costo, probablemente energético
   y no solo temporal, de cada cambio de reloj de GPU), no como un detalle
   menor.

---

## La pregunta concreta para el modelo más potente

Diseñar el **proceso de inferencia** completo que decide, en tiempo real o
casi real, sobre qué dispositivo actuar y con qué política, dado:

- Ventanas de muestreo CPU a granularidad de ~1 ms (ya funcionando).
- Muestras de GPU (`util_pct`, `power_mw`) a la cadencia que NVML permita
  (más gruesa que 1 ms en la práctica).
- Un costo alto y una latencia alta para cada cambio de frecuencia de GPU,
  que hace que reaccionar por cada ventana de 1 ms sea probablemente
  contraproducente (el costo de recalibrar podría superar el ahorro).
- La necesidad de que el CPU y la GPU tengan políticas coherentes entre sí
  cuando trabajan simultáneamente (ninguno debería tomar una decisión que
  ignore lo que está haciendo el otro en ese mismo instante).

Preguntas específicas que se le pide resolver o al menos aportar criterio
razonado:

1. **¿A qué granularidad temporal/de evento debería tomarse la decisión de
   frecuencia de GPU** (por kernel individual, por fase agregada de varios
   kernels, por ventana de tiempo fija más gruesa que la de CPU, por algún
   evento de sincronización explícito), dado el costo alto de cambiar el
   reloj? ¿Cómo se evita "flapping" (cambiar de frecuencia demasiado
   seguido, pagando el costo de transición sin ganar suficiente tiempo en
   el nuevo estado para amortizarlo)?

2. **¿Cómo se hace la atribución CPU-vs-GPU de forma robusta**, más allá de
   un umbral simple sobre `util_pct`? ¿Hay una señal mejor o
   complementaria (por ejemplo, si el proyecto llegara a instrumentar CUPTI
   más adelante) que el equipo debería considerar agregar al harness, sabiendo
   que hoy solo se tiene `util_pct`/`power_mw` de NVML?

3. **¿Qué pasa con la calibración Roofline de GPU en sí** — cómo estructurar
   ese microbenchmark de calibración (equivalente al STREAM/ERT que ya
   existe para CPU) sabiendo que no hubo tiempo para un piloto real todavía,
   y qué riesgos de ese enfoque (peak FLOPs/BW teóricos vs. alcanzables)
   son más importantes de anticipar dado el cronograma ajustado.

4. **¿Conviene tratar esto como dos políticas independientes con una regla
   de coordinación simple (el enfoque que se venía discutiendo), o hay una
   razón de peso para modelarlo como un problema conjunto** (por ejemplo, un
   solo optimizador que vea ambas señales a la vez), considerando que el
   proyecto tiene un cronograma de 3 semanas y ya tiene un pipeline CPU
   funcionando que no se quiere arriesgar a romper?

5. Cualquier riesgo, trampa o precedente de la literatura de DVFS
   heterogéneo CPU+GPU que el equipo debería conocer antes de comprometerse
   con una arquitectura, dado que la decisión se está tomando sin el piloto
   real que originalmente se quería tener primero.

---

## Lo que NO se le pide al modelo

- No se pide código todavía, es una pregunta de arquitectura/diseño.
- No se pide que resuelva la calibración Roofline de CPU (eso ya está
  hecho y validado en hardware real, ver `docs/retoma/Guia_Maestra_Fase1_DVFS.md`
  y `docs/retoma/Informe_Diagnostico_Beta_2026-08-05.md`).
- No se pide que cuestione la decisión ya tomada de mantener el label
  binario (`compute_bound`/`memory_bound`) — esa parte del diseño se
  considera cerrada; el foco es el proceso de inferencia/coordinación
  alrededor de esa etiqueta cuando hay dos dispositivos.
