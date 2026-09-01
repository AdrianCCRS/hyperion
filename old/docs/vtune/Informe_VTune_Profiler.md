# Intel VTune Profiler — qué es, cómo mide, y qué devuelve realmente

Informe técnico de referencia para el proyecto Hyperion. Escrito a partir de la
implementación real de `pipelinevtune/` (ver esa carpeta para el código) y de
evidencia empírica capturada en el nodo `paccaA100` (Ice Lake-SP, HPC
Universidad de Cartagena). Cada afirmación de este documento que depende del
comportamiento real del profiler está respaldada por una captura concreta —
no es un resumen genérico de la documentación de Intel.

---

## 1. Qué es VTune, en una frase

Intel VTune Profiler es una herramienta de **medición de rendimiento por
muestreo de contadores de hardware** (no de simulación, no de análisis
estático de código): ejecuta el programa real, mientras corre, y va anotando
periódicamente en qué instrucción/función estaba y qué estaban haciendo los
contadores de la CPU en ese instante. Con miles de esas muestras reconstruye
estadísticamente dónde se va el tiempo y por qué el hardware se detiene
(esperando memoria, mal-predicción de saltos, etc.).

No es un profiler de un solo propósito ("dime la función más lenta") — es una
familia de más de diez **tipos de análisis** distintos, cada uno activando un
subconjunto distinto de contadores de hardware y aplicando un modelo de
interpretación distinto sobre esos números.

---

## 2. Cómo se obtiene

- Forma parte del **Intel oneAPI Base Toolkit** (gratuito, sin licencia de
  pago, desde 2020). También existe como paquete standalone
  ("Intel VTune Profiler") descargable directo del sitio de Intel, y como
  paquete de Linux (`.rpm`/`.deb`) o vía `conda`/`pip` (paquete
  `intel-vtune` en canales de Intel).
- En clústeres HPC administrados (como este) normalmente llega como **módulo
  del sistema de modules/Lmod**, a veces anidado bajo el toolkit de oneAPI
  padre. En este proyecto, confirmado empíricamente:

  ```bash
  module load devtools/intel/oneapi/2023   # modulo padre -- sin este, "module avail vtune" no muestra nada
  module load vtune/2023.0.0               # el modulo real de vtune, jerarquico
  ```

  Este comportamiento jerárquico (el módulo hijo invisible sin el padre) no
  está documentado en ningún sitio genérico de Intel — es una particularidad
  de cómo este clúster organiza sus módulos Lmod, y costó descubrirla por
  ensayo y error (ver `pipelinevtune/context/04_vtune_selfchecker_resultados.md`).
- **No requiere privilegios de administrador para su uso básico** (el modo
  "driverless" que se describe en la sección 3) — importante en un nodo
  compartido donde no se puede pedir `sudo` ni instalar drivers de kernel.

---

## 3. Cómo mide: la pregunta de fondo ("¿envuelve/instrumenta el binario?")

**No.** VTune no recompila, no re-linkea, ni inyecta código dentro del binario
que se está midiendo (a diferencia de herramientas de *instrumentación*, como
`gprof` o algunos modos de Intel Advisor, que sí insertan contadores dentro
del código). El binario NPB/STREAM/DGEMM que este proyecto usa se compiló
exactamente igual con o sin VTune de por medio — VTune lo lanza (o se
"adjunta" a un proceso ya corriendo) y observa desde afuera.

### 3.1 Las dos técnicas de muestreo que usa VTune

| Técnica | Qué mide | Overhead | Cuándo la usa VTune |
|---|---|---|---|
| **User-mode sampling** | Interrumpe el proceso periódicamente por temporizador de SO y anota la pila de llamadas | Bajo, pero menos preciso — el temporizador del SO no está sincronizado con lo que hace la CPU | Modo por defecto de `hotspots` sin `-knob sampling-mode=hw` |
| **Hardware Event-Based Sampling (EBS)** | Configura un contador de hardware de la CPU (la PMU, *Performance Monitoring Unit*) para que se desborde cada N eventos (ej. cada 2 millones de instrucciones retiradas) y genere una interrupción justo ahí | Bajo (típicamente 2-5% del tiempo de ejecución) y **preciso a nivel de instrucción/ciclo** porque el disparo lo hace el hardware, no un reloj de SO | `-collect hotspots -knob sampling-mode=hw` y todo lo que use contadores de microarquitectura (`hpc-performance`, `uarch-exploration`, `memory-access`) |

Este proyecto usa **EBS** exclusivamente (`sampling-mode=hw`), porque es la
única forma de que los porcentajes de Top-Down (`Memory Bound`, `Cache
Bound`, etc.) signifiquen algo real — el modo por temporizador de SO no puede
atribuir tiempo a categorías de microarquitectura, solo a funciones.

### 3.2 El mecanismo real detrás de EBS: PMU + interrupción, dos caminos posibles

El hardware (la PMU) tiene un puñado de registros contadores (`PMC0`,
`PMC1`, ... — típicamente 4-8 por core en procesadores modernos) que se
pueden programar para contar un evento específico (ej.
`MEM_LOAD_RETIRED.L3_MISS`) y disparar una interrupción de "overflow" cuando
llegan a un umbral. VTune necesita un mecanismo del sistema operativo para:
(a) programar esos registros, y (b) capturar la interrupción y anotar qué
instrucción/pila estaba ejecutando el proceso en ese instante. Hay dos rutas:

1. **Driver propio de Intel (`sep`/`socperf`)** — un módulo de kernel que
   Intel distribuye junto a VTune, instalable con privilegios de root. Da
   acceso a más eventos (incluyendo algunos de uncore) y menor overhead.
2. **"Driverless", vía `perf_event_open` del kernel Linux** — VTune usa la
   misma interfaz estándar de Linux que usan `perf stat`/`perf record`
   (la que expone `/proc/sys/kernel/perf_event_paranoid`). No necesita
   ningún driver adicional ni privilegios especiales, siempre que
   `perf_event_paranoid` no esté configurado de forma demasiado restrictiva
   para el tipo de evento pedido.

**Este proyecto usa la ruta 2 — confirmado literalmente en cada captura**:

```
Collector Type: Driverless Perf per-process sampling
```

Esto es lo que hace viable correr VTune en este nodo compartido sin pedir
nada al administrador: es exactamente el mismo mecanismo de kernel
(`perf_event_open`) que el harness de telemetría propio de Hyperion
(`telemetry/src/perf_reader.cpp`) usa para su medición vía PID+inherit — ver
`docs/retoma/pacca/Auditoria_PaccaA100_Unicartagena.md` sección 5, donde se
confirmó empíricamente que ese mecanismo funciona en este nodo incluso con
`perf_event_paranoid=2`. VTune, al no depender de su driver propio, hereda
la misma viabilidad.

**Consecuencia práctica importante:** como ambas herramientas (VTune y el
harness del orquestador) están limitadas por el mismo mecanismo del kernel,
comparten también la misma limitación — no pueden leer contadores de
**uncore** (memoria, caché L3 compartida a nivel de PMU física, no del lado
del core) sin `CAP_PERFMON`/`perf_event_paranoid<1`, que este nodo no
otorga. Ver sección 9.

---

## 4. El fundamento teórico detrás de los números: Top-Down Microarchitecture Analysis Method (TMAM)

Los análisis más relevantes de este proyecto (`hpc-performance`,
`uarch-exploration`) no reportan contadores crudos sueltos — los organizan
según una metodología de Intel llamada **TMAM** (a veces "Top-Down
Analysis"), publicada originalmente por Yasin (Intel, 2014). La idea central:

En cada ciclo de reloj, el front-end de la CPU puede *emitir* hasta N
"pipeline slots" (N = ancho de emisión del core, ej. 4-6 en Ice Lake) hacia
las unidades de ejecución. TMAM clasifica **cada slot posible** en una de
cuatro categorías, que suman 100%:

```
Pipeline Slots = Retiring + Bad Speculation + Front-End Bound + Back-End Bound
```

- **Retiring** — el slot se usó para una instrucción que efectivamente
  terminó (trabajo útil). Un valor alto es bueno.
- **Bad Speculation** — el slot se gastó en instrucciones que luego se
  descartaron (mal-predicción de salto, por ejemplo).
- **Front-End Bound** — el slot quedó vacío porque el front-end (fetch +
  decode) no logró alimentar instrucciones a tiempo.
- **Back-End Bound** — el slot quedó vacío porque el back-end (ejecución)
  no pudo aceptar más trabajo. **Esta categoría se subdivide en:**
  - **Core Bound** — cuellos de botella dentro del core (puertos de
    ejecución saturados, latencias de instrucciones complejas).
  - **Memory Bound** — cuellos de botella esperando datos de la jerarquía
    de memoria (L1/L2/L3/DRAM).

`Memory Bound` a su vez se desglosa en niveles cada vez más finos:
`L1 Bound`, `L2 Bound`, `L3 Bound`/`Cache Bound`, `DRAM Bound`, `Store
Bound`, etc. — cuanto más fino el nivel, más cerca está de necesitar
contadores de **uncore** (el controlador de memoria vive fuera del core).

### 4.1 Qué de todo esto expone VTune en este nodo — y qué no

Esto es evidencia empírica de este proyecto, no teoría genérica:

| Nivel TMAM | ¿Aparece en `vtune -report summary` de `hpc-performance` en este nodo? |
|---|---|
| Retiring / Bad Speculation / Front-End Bound / Back-End Bound (nivel 1 completo) | **No.** Requiere el análisis `uarch-exploration` ("Microarchitecture Exploration"), confirmado no disponible en este nodo (`context/04`). |
| Memory Bound (agregado, "% of Pipeline Slots") | **Sí**, siempre poblado con un número real. |
| Cache Bound ("% of Clockticks") | **Sí.** |
| DRAM Bound ("% of Clockticks") | **Sí, y con valores creíbles** (0.0% en un kernel sin tráfico de memoria relevante, 67.7% en STREAM) — pese a que el acceso directo a contadores *uncore* está bloqueado en este nodo (ver sección 9, resuelve la aparente contradicción). |
| Core Bound (aislado) | **No.** No aparece como métrica independiente en este reporte/versión — es la pieza que faltaba para replicar el TMAM de 4 categorías completo, y la razón por la que este proyecto tuvo que calibrar `classification_vtune_native` contra anclas STREAM/DGEMM en vez de comparar directamente contra un "Core Bound" leído del reporte (ver `pipelinevtune/context/02_decisiones.md`, decisión D3-v3). |

Esto no es un defecto de VTune — es consecuencia directa de que
`uarch-exploration` (el análisis que sí da las 4 categorías completas)
depende de eventos que este nodo no permite leer sin privilegios. `hpc-performance`
da un subconjunto more limitado pero suficiente pensado específicamente para
código HPC de cómputo intensivo (de ahí su nombre).

---

## 5. Catálogo de tipos de análisis (lo que aparece en `vtune -collect-list`)

Confirmado contra la instalación real (VTune 2023.0.0) en este nodo:

| Análisis | Para qué sirve | ¿Disponible aquí? |
|---|---|---|
| `performance-snapshot` | Vista rápida de 30 segundos para decidir qué análisis más profundo correr después. Punto de entrada recomendado por Intel, no usado en este proyecto porque ya sabíamos qué buscar. | Sí |
| `hotspots` | Identifica las funciones/líneas de código que más tiempo de CPU consumen. Con `-knob sampling-mode=hw` usa EBS real; sin ese knob, muestreo por temporizador. **No decide memory-bound/compute-bound por sí solo** — solo dice "dónde", no "por qué". | Sí (con y sin EBS) |
| `anomaly-detection` | (Preview/experimental en esta versión) detecta anomalías de rendimiento a nivel de microsegundo usando Intel Processor Trace. | Listado, no usado |
| `memory-consumption` | Cuánta memoria (RAM) consume el proceso y sus objetos — **no** es lo mismo que "memory-bound"; esto es sobre uso de RAM, no sobre cuellos de botella de acceso a memoria. | Listado, no usado |
| `uarch-exploration` ("Microarchitecture Exploration") | El TMAM de 4 categorías completo (sección 4). Requiere eventos de microarquitectura amplios, algunos cercanos a uncore. | **No disponible en este nodo** |
| `memory-access` | Desglose fino de acceso a memoria por objeto de datos (qué estructura/array genera los cache misses), ancho de banda de memoria por *bandwidth utilization histogram*. Depende fuertemente de uncore. | **No disponible en este nodo** |
| `threading` | Eficiencia de paralelismo (OpenMP/MPI/TBB) — balance de carga entre hilos, tiempo de espera en barreras/locks. | Listado, no usado (el proyecto usa `hotspots`+`hpc-performance`, no `threading`, porque el foco es compute-vs-memory, no eficiencia de paralelización per se) |
| `hpc-performance` ("HPC Performance Characterization") | Pensado específicamente para código científico/HPC de cómputo intensivo: `DP/SP GFLOPS`, CPI, utilización de cores físicos/lógicos, el Memory Bound descrito arriba, y vectorización (qué fracción del cómputo usa SIMD 128/256/512-bit). **Es el análisis central de este proyecto.** | Sí |
| `io` | Utilización de subsistemas de E/S, CPU y buses del procesador durante operaciones de I/O. | Listado, no usado (los kernels NPB no hacen I/O significativo) |
| `gpu-offload` | Para código que corre parcialmente en GPU — identifica si el código está CPU-bound o GPU-bound y estima beneficio de offloading. | Listado, no usado (el pipeline es CPU-only a propósito) |

---

## 6. Qué datos concretos devuelve `hotspots` y `hpc-performance` (con ejemplos reales)

### 6.1 `hotspots` (con `sampling-mode=hw`)

Captura real de este proyecto (`ep.C.x`, ver
`pipelinevtune/tests/unit/fixtures/real_hotspots_ep_C.txt`):

```
Elapsed Time: 13.470s
    CPU Time: 106.174s
    Instructions Retired: 579,328,000,000
    Microarchitecture Usage: 43.5% of Pipeline Slots
        CPI Rate: 0.640

Top Hotspots
Function          Module        CPU Time  % of CPU Time(%)
MAIN__._omp_fn.1  ep.C.x         55.505s             52.3%
func@0x777b0      libm-2.28.so   36.110s             34.0%
vranlc_           ep.C.x         10.239s              9.6%
```

**Qué significa cada cosa:** `CPU Time` (106.174s) es la suma del tiempo de
CPU de los 8 hilos, no el tiempo de reloj (`Elapsed Time`, 13.470s) — con 8
hilos perfectamente paralelos, `CPU Time ≈ 8 × Elapsed Time` (aquí da ~7.9x,
coherente con una pequeña porción secuencial). La tabla de "Top Hotspots"
identifica que más de la mitad del tiempo (52.3%) se va en el bucle
principal de generación aleatoria de EP, y un tercio (34.0%) en `libm`
(funciones matemáticas del sistema, `sqrt`/`log`) — esto es exactamente el
patrón que `context/03_kernels_notas.md` anticipaba para EP: trabajo
dominado por generador congruencial + funciones trascendentales.

`Microarchitecture Usage: 43.5% of Pipeline Slots` es, de hecho, un
**sustituto parcial de "Retiring"** (la categoría de TMAM que hpc-performance
no expone) — mide qué fracción de slots produjo trabajo útil, aunque no
desglosa el resto en Bad Speculation/Front-End/Back-End por separado.

### 6.2 `hpc-performance`

Dos capturas reales, kernel compute-ish (EP) vs. memory-bound por
construcción (STREAM):

| Métrica | EP (`ep.C.x`) | STREAM (`stream_omp`) | Lectura |
|---|---|---|---|
| `DP GFLOPS` | 9.841 | 2.746 | EP hace más FLOP/s en términos absolutos — pero ver sección 7, esto es engañoso para EP específicamente |
| `Memory Bound` | 6.1% | 51.9% | STREAM tiene ~8.5x más fracción de slots detenidos por memoria |
| `Cache Bound` | 11.5% | 19.2% | Ambos tocan caché, STREAM más |
| `DRAM Bound` | 0.0% | 67.7% | La señal más contundente: EP casi no llega a DRAM (working set diminuto), STREAM está fundamentalmente limitado por ancho de banda de DRAM |
| `CPI Rate` | 0.641 | 5.345 | EP retira instrucciones cada 0.64 ciclos en promedio (eficiente); STREAM necesita 5.3 ciclos por instrucción retirada — síntoma directo de estar esperando memoria constantemente |
| `Vectorization` | 10.7% | 100.0% | STREAM (copias/sumas de arrays) vectoriza perfecto; EP (ramas condicionales del generador aleatorio) vectoriza poco |

### 6.3 Sobre `-report hw-events`: el nivel más crudo

Además de los reportes ya agregados (`summary`, `hotspots`), VTune puede
volcar los **contadores de hardware crudos por función**, sin ninguna
interpretación TMAM encima:

```
TOPDOWN.SLOTS, TOPDOWN.BACKEND_BOUND_SLOTS, CYCLE_ACTIVITY.STALLS_L1D_MISS,
CYCLE_ACTIVITY.STALLS_L2_MISS, CYCLE_ACTIVITY.STALLS_L3_MISS,
MEM_LOAD_RETIRED.L1_HIT/L2_HIT/L3_HIT/L3_MISS,
MEM_LOAD_L3_MISS_RETIRED.LOCAL_DRAM/REMOTE_DRAM,
FP_ARITH_INST_RETIRED.*, OFFCORE_REQUESTS_OUTSTANDING.*, ...
```

Confirmado disponible en este nodo (`pipelinevtune` lo usa como respaldo
opcional, `hpc_hw_events.csv`). Con estos eventos crudos sería posible
reconstruir a mano el TMAM completo de 4 categorías (incluyendo un "Core
Bound" real) siguiendo las fórmulas públicas de Intel — se evaluó y se
descartó por alcance para este proyecto (ver `PLAN.md` Fase 4.2), pero queda
documentado como camino viable si en el futuro hiciera falta.

---

## 7. Justificando los valores: por qué EP "engaña" si solo se mira DP GFLOPS

Este es el caso más instructivo del proyecto para explicar por qué no basta
con leer un solo número. Naturaleza de EP (`context/03_kernels_notas.md`):
genera pares gaussianos con el método polar de Marsaglia — su trabajo
dominante es un generador congruencial de 64 bits, más `sqrt`/`log`
(confirmado en la sección 6.1: `libm` se lleva 34-45% del tiempo).

El contador `DP GFLOPS` de VTune cuenta instrucciones de punto flotante
**empaquetadas y escalares de suma/multiplicación** (`FP_ARITH_INST_RETIRED.*`).
La unidad de división/raíz cuadrada de la CPU es una unidad de ejecución
distinta, con throughput mucho menor, y sus operaciones **no
necesariamente se cuentan en el mismo bucket de "FLOPs" que suma/multiplicación**
dependiendo de qué eventos arma VTune para esa métrica. El resultado
observable: EP puede aparecer con `DP GFLOPS` bajo (9.8, comparado con
DGEMM en el mismo nodo: 463-729) **sin que eso signifique que EP sea poco
intensivo en cómputo** — significa que gran parte de su trabajo de cómputo
(las raíces cuadradas y logaritmos) no entra en ese contador específico.

Esto es exactamente la razón detrás de la decisión D4 del proyecto
(`context/02_decisiones.md`): EP nunca se descarta, pero se marca para
revisión manual si su clasificación sale `memory_bound`/`ambiguous`, en vez
de aceptarse a ciegas — el número que lo llevaría a esa clasificación
(`DP GFLOPS` bajo) puede ser un artefacto de qué cuenta el contador, no una
propiedad real del kernel.

---

## 8. Cómo ver los resultados: línea de comandos vs. interfaz gráfica

### 8.1 Lo que este proyecto usa: `vtune -report` (CLI, texto/CSV)

Es la única vía practicable en este flujo de trabajo: el nodo se accede por
SSH sin interfaz gráfica reenviada, y la campaña corre desacoplada dentro de
un `sbatch` (decisión D7). Todo el pipeline (`pipelinevtune/run_vtune_pipeline.py`)
se apoya exclusivamente en:

```bash
vtune -collect hotspots -knob sampling-mode=hw -r RESULT_DIR -- BINARIO
vtune -report hotspots -r RESULT_DIR                    # texto legible
vtune -report hotspots -format=csv -r RESULT_DIR        # para parseo automatico
vtune -report summary -r RESULT_DIR                     # equivalente para hpc-performance
```

### 8.2 La interfaz gráfica (`vtune-gui`), para exploración manual

VTune también trae una GUI de escritorio (Eclipse-based en Linux, standalone
en Windows/macOS) con vistas mucho más ricas que el texto: línea de tiempo
interactiva por hilo, "Bottom-up"/"Top-down tree"/"Caller-Callee" navegables,
mapas de calor de código fuente línea por línea, histogramas de ancho de
banda de memoria, etc. Dos formas de usarla con un resultado generado en un
clúster remoto sin GUI:

1. **Copiar el directorio de resultado** (`-r RESULT_DIR`, la carpeta
   `.vtune`/`sqlite-db`/etc. completa) a una máquina local con VTune
   instalado, y abrirlo con `vtune-gui` → *File → Open Result* (o
   `vtune-gui RESULT_DIR` directo desde la terminal). No requiere que el
   nodo remoto tenga GUI ni X11 — el resultado es autocontenido.
2. **X11 forwarding** (`ssh -X`) si el clúster lo permite y hay suficiente
   ancho de banda — más lento, no probado en este proyecto porque el flujo
   por `sbatch` no mantiene una sesión interactiva.

Para este proyecto, la vía 1 es la recomendada si en algún momento hace
falta inspeccionar visualmente un caso atípico (por ejemplo, confirmar a ojo
en la línea de tiempo por qué un kernel salió `ambiguous`) — los resultados
ya quedan guardados en `vtune_results/<KERNEL>/class_<C>/rep_<NN>/{hotspots,hpc}/`
exactamente para ese propósito.

---

## 9. Limitaciones reales de este nodo, y por qué no invalidan el proyecto

- **Sin acceso a contadores *uncore*** (confirmado por dos vías independientes:
  la auditoría del orquestador con `perf_event_open` directo da `EACCES`, y
  LIKWID silenciosamente reporta `0 GB` de tráfico de memoria porque su
  daemon de acceso no tiene privilegios). Esto bloquea `uarch-exploration` y
  `memory-access` (sección 5), y con ellos, el TMAM de 4 categorías completo.
- **Pero `DRAM Bound` sí funciona en `hpc-performance`** — porque no lee el
  controlador de memoria (uncore) directamente, sino que lo **estima** con
  eventos del lado del core (`MEM_LOAD_L3_MISS_RETIRED.LOCAL_DRAM`,
  `OFFCORE_REQUESTS_OUTSTANDING.*` — confirmados presentes en
  `-report hw-events`, sección 6.3). Es una estimación por atribución de
  latencia (metodología TMAM estándar de Intel), no un conteo directo de
  bytes movidos — suficientemente fiable para comparar kernels entre sí
  (que es todo lo que este proyecto necesita), no para afirmar un número
  absoluto de GB/s de tráfico real a DRAM.
- **No hay "Core Bound" aislado** en el reporte disponible — es la
  consecuencia práctica más relevante para el diseño del clasificador de
  este proyecto (ver sección 4.1 y `context/02_decisiones.md` D3-v3).

Ninguna de estas limitaciones es específica de VTune — son restricciones de
permisos del nodo que afectarían a *cualquier* herramienta basada en
`perf_event_open` sin privilegios (LIKWID incluido, confirmado). VTune, al
degradar con gracia a lo que sí puede medir (en vez de fallar por completo),
sigue siendo la fuente de validación más completa disponible en este nodo —
de ahí la decisión D2 del proyecto de basarse exclusivamente en VTune.

---

## 10. Referencia rápida de comandos

```bash
# Descubrir que analisis estan realmente disponibles en esta instalacion
vtune -collect-list

# Smoke test minimo (mecanica, no contenido)
vtune -collect hotspots -knob sampling-mode=hw -r /tmp/smoke -- sleep 1
vtune -report summary -r /tmp/smoke

# Los dos analisis que usa este proyecto
vtune -collect hotspots -knob sampling-mode=hw -r RESULT_DIR -- ./binario
vtune -collect hpc-performance -r RESULT_DIR -- ./binario

# Reportes: texto (para leer), csv (para parsear), hw-events (crudo)
vtune -report hotspots -r RESULT_DIR
vtune -report summary  -r RESULT_DIR
vtune -report hotspots -format=csv -r RESULT_DIR
vtune -report summary  -format=csv -r RESULT_DIR
vtune -report hw-events -format=csv -r RESULT_DIR

# Ver un resultado en la GUI (en una maquina con VTune instalado)
vtune-gui RESULT_DIR
```

---

## Fuentes de este documento

- Evidencia empírica propia: `pipelinevtune/context/04_vtune_selfchecker_resultados.md`
  (addendum de Fase 0), `pipelinevtune/tests/unit/fixtures/real_*` (capturas
  reales), `pipelinevtune/context/02_decisiones.md` (decisiones D3-v3, D4).
- `docs/retoma/pacca/Auditoria_PaccaA100_Unicartagena.md` — auditoría
  independiente de `perf_event_open`/uncore/RAPL en el mismo nodo, usada
  para explicar la sección 9.
- Metodología TMAM: Ahmad Yasin, *"A Top-Down method for performance
  analysis and counters architecture"*, IEEE ISPASS 2014 — la publicación
  original de Intel detrás de la clasificación de Pipeline Slots descrita
  en la sección 4 (referencia externa, no específica de este proyecto).
