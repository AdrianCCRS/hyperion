# Validación cruzada de Hyperion con Intel VTune — Microarchitecture Exploration

Documento metodológico de la campaña implementada en `Vtune/` (raíz del
repo). Escrito para poder defender esta parte del trabajo de grado sin tener
que reconstruir el razonamiento de memoria. A diferencia de
`docs/vtune/Informe_VTune_Profiler.md` (que documenta Hotspots + HPC
Performance Characterization, los dos análisis que sí estaban disponibles
antes de este permiso) y de `pipelinevtune/` (que implementa ese pipeline
anterior y queda congelada como referencia histórica), este documento asume
que **Microarchitecture Exploration ya es funcional en `paccaA100`** — el
permiso que antes bloqueaba ese análisis específico
(`pipelinevtune/context/04_vtune_selfchecker_resultados.md`) se resolvió.

**Advertencia de honestidad metodológica que aplica a todo este documento:**
Microarchitecture Exploration nunca se había corrido con éxito en este nodo
antes de este permiso. Todo lo que sigue sobre nombres de campo exactos,
tiempos de ejecución y comportamiento observado de cada kernel está escrito
a partir de (a) documentación pública de Intel sobre TMAM y el viewpoint
"Microarchitecture Exploration" de VTune, y (b) el patrón ya confirmado
empíricamente en este mismo nodo para Hotspots/HPC Performance
Characterization (mismo VTune 2023.0.0, mismo mecanismo `perf_event_open`).
La primera corrida real de `Vtune/run_validation.py` debe reconciliar los
nombres de campo de `Vtune/uarch_parser.py` contra la salida real — si algo
no coincide, se corrige el parser y se anota aquí, siguiendo exactamente el
mismo patrón que este proyecto ya siguió con `hpc-performance`
(`pipelinevtune/context/02_decisiones.md`, D3-native → D3-v3).

---

## A. Propósito

Hyperion etiqueta y entrena su clasificador ligero `compute_bound`/
`memory_bound` a partir de telemetría capturada con `perf_event_open`
(PID+`inherit=1`) sobre el proceso real del kernel, siguiendo la calibración
Roofline descrita en `docs/retoma/Guia_Maestra_Fase1_DVFS.md`. Esa
telemetría es el mecanismo *oficial* del proyecto — el que alimenta
`windows.csv`, `phase_label_train` y en última instancia el modelo que el
agente DVFS va a usar en producción.

Esta campaña responde a una pregunta distinta y más específica: **¿los
kernels que el proyecto eligió para ese dataset (NPB EP/CG/MG/FT/LU/BT, más
STREAM y DGEMM como referencias) de verdad se comportan como se espera antes
de gastar presupuesto de campaña completo etiquetándolos?** Es una
validación de la *selección de kernels*, no una segunda fuente de
entrenamiento ni un segundo clasificador en competencia con el de Hyperion.

## B. Por qué VTune NO es el mecanismo principal del proyecto

Esta es la pregunta que hay que responder con más cuidado, porque la
respuesta obvia ("VTune es de Intel y pesado") no basta — VTune también usa
PMU vía `perf_event_open`, exactamente el mismo mecanismo de kernel que usa
el harness propio de Hyperion (`telemetry/src/perf_reader.cpp`). Si ambos
leen la misma fuente física, ¿por qué no usar directamente VTune como
clasificador en tiempo de ejecución?

| Criterio | VTune (Microarchitecture Exploration) | Harness propio de Hyperion |
|---|---|---|
| **Costo de inicialización** | Segundos: arranca su propio proceso de recolección (`vtune` como padre), inicializa el motor de análisis, arma múltiples grupos de eventos de PMU. No pensado para adjuntarse/desprenderse en microsegundos. | El `Collector` ya vive dentro del proceso del agente; abrir un `fd` de `perf_event_open` adicional es del orden de microsegundos. |
| **Overhead en régimen continuo** | Diseñado para una *sesión* de profiling (minutos), no para correr indefinidamente pegado a una aplicación de producción. Cuantos más eventos arma (y `uarch-exploration` arma bastantes más que `hotspots`), más probable que necesite multiplexar grupos de PMU entre sí (ver sección E.6) — introduce ruido estadístico si la ventana de observación es corta, justo el régimen (~1 ms) en el que el agente DVFS de Hyperion necesita decidir. | El agente lee 4 contadores fijos (`instructions`, `cycles`, `cache_references`, `cache_misses`) por ventana de ~1 ms, sin necesidad de multiplexar — cabe en los PMCs físicos disponibles sin rotar grupos (ver `PRE-D05`/`probe_pmc_count()` en el orquestador principal). |
| **Frecuencia de muestreo útil** | Pensado para agregar estadística durante toda una corrida y reportar un resumen al final (o una línea de tiempo gruesa). No expone una API para "dame el vector de features de los últimos 1 ms" que un daemon pueda consultar en un loop. | Por diseño: cada ventana ya es una fila con deltas listos para alimentar un clasificador en vivo. |
| **Dependencia de Intel** | Analysis engine, formato de resultado (`.vtune`/SQLite) y CLI son propietarios de Intel. Portar el *mecanismo de decisión* del agente a un nodo AMD (fuera del alcance de este trabajo, pero relevante para el diseño) no tendría equivalente de VTune. | `perf_event_open` es una interfaz del kernel Linux, portable entre fabricantes (los *nombres* de evento de PMU cambian, el mecanismo no). |
| **Dependencia de símbolos/binarios** | Para atribución fina a función/línea, se beneficia de símbolos de depuración; sin ellos igual reporta agregados Top-Down (que es todo lo que este proyecto necesita), pero la resolución de hotspots sí se degrada. | El agente nunca necesita símbolos — solo contadores agregados por PID/ventana. |
| **Disponibilidad en clúster** | Requiere módulo Lmod cargado, licencia/instalación de oneAPI, y —hasta este permiso— PMU restringida en nodos compartidos. No garantizado en cualquier nodo del clúster. | `perf_event_open` sin driver propio (modo *driverless*, confirmado en este mismo nodo desde antes de este permiso) — más portable dentro del propio clúster. |
| **Integración como daemon ligero** | No es su caso de uso: es una herramienta de *profiling offline* (se lanza, corre, termina, se analiza el resultado después), no un componente embebible dentro de un proceso de control. Ejecutarlo como subproceso persistente para cada job sería reinventar, con más peso, lo que `perf_event_open` ya da directo. | Encaja exactamente en el patrón "agente por ejecución" ya diseñado para Hyperion (ver conversación de arquitectura del agente DVFS): el propio proceso agente abre los `fd` que necesita, sin depender de un proceso externo. |
| **Reproducibilidad** | Alta *dentro de una sesión de profiling* (el `result-dir` es autocontenido y reproducible), pero el formato y CLI pueden cambiar entre versiones de VTune — un riesgo de mantenimiento a largo plazo si se volviera dependencia dura del pipeline de producción. | El formato de `windows.csv`/`Sample` es propio del proyecto, versionado junto con el resto del código. |
| **Restricciones de permisos** | Hasta este permiso, bloqueado por completo para este análisis en este nodo — la dependencia de un profiler comercial introduce un punto de fallo administrativo fuera del control del proyecto. | Ya funcionaba con `perf_event_paranoid=2` (modo driverless) antes de pedir ningún permiso adicional — ver `docs/retoma/pacca/Auditoria_PaccaA100_Unicartagena.md`. |
| **Profiling offline vs. control online** | Categoría de herramienta: mide, reporta, un humano interpreta después. No tiene un modo "responde en el loop de control mientras la aplicación corre". | Categoría de componente: alimenta una decisión de política DVFS en el mismo loop de ejecución. |

**La diferencia de fondo, en una frase:** VTune es una herramienta de
*profiling* — completa, rica, pensada para que un humano (o esta campaña de
validación) inspeccione un resultado después de una corrida. El clasificador
de Hyperion necesita *features mínimas, baratas, en vivo*, no la
reconstrucción completa del Top-Down de 4 categorías con sub-niveles que
VTune ofrece. Pedirle a VTune que haga el trabajo del agente sería usar una
herramienta de banco de pruebas como si fuera un sensor embebido — funciona
en el banco, no en el instrumento final.

## C. Independencia metodológica (y el riesgo de circularidad)

Comparten la fuente física (PMU vía `perf_event_open`), y aun así hay que
distinguir cuatro capas distintas para saber si esta validación es
realmente independiente o si es "preguntarle lo mismo dos veces":

```
misma fuente física de observación   (PMU, perf_event_open)
        ≠
mismo pipeline de procesamiento      (VTune: motor de análisis propietario
                                       de Intel, arma grupos de eventos y
                                       multiplexa por su cuenta;
                                       Hyperion: Collector propio, SPSCRing,
                                       postprocess.py con deltas explícitos)
        ≠
mismas métricas derivadas            (VTune: Top-Down Retiring/Front-End/
                                       Bad Speculation/Back-End con
                                       Memory Bound vs Core Bound;
                                       Hyperion: IPC, MPKI, LLC miss rate,
                                       operational intensity vs. i_ridge
                                       Roofline)
        ≠
mismo modelo de clasificación        (VTune: reglas de Intel sobre su
                                       propio Top-Down, ver sección E;
                                       Hyperion: Random Forest/Decision Tree
                                       entrenado sobre windows.csv;
                                       esta campaña: reglas propias sobre
                                       Memory Bound vs Core Bound, ver
                                       Vtune/validation_classifier.py)
```

**Dónde sí hay riesgo real de circularidad:** si el veredicto de esta
campaña y el `phase_label_train` de Hyperion vinieran ambos, en el fondo, de
"cuántos misses de LLC hubo" medido con el mismo evento de PMU y la misma
fórmula, estaríamos validando un número contra sí mismo con un nombre
distinto. Eso NO es lo que pasa aquí, por diseño:

- Hyperion deriva `phase_label_train` comparando `operational_intensity`
  (FLOPs del binario ÷ bytes movidos estimados por `perf`) contra `i_ridge`
  de una calibración Roofline propia (STREAM/ERT). Es un criterio de
  **intensidad aritmética**.
- Esta campaña deriva `vtune_validation_class` comparando **fracciones de
  slots de pipeline** perdidas por stalls de memoria vs. stalls de núcleo
  (`Memory Bound` vs. `Core Bound`, ambos como % de *Pipeline Slots* según
  TMAM). Es un criterio de **dónde se pierde tiempo de ejecución**, no de
  cuánto FLOP hay por byte.

Ambos criterios son teóricamente consistentes entre sí (un kernel realmente
memory-bound debería, en general, mostrar baja intensidad aritmética *y*
alta fracción de slots perdidos por memoria) pero se calculan con fórmulas,
eventos de PMU agregados de forma distinta, y ningún paso comparte código
entre los dos pipelines. Que coincidan es evidencia real; que uno "sepa" el
resultado del otro de antemano no es posible con este diseño.

**Qué SÍ comparten, y qué implica:** ambos dependen en última instancia de
que el hardware de este nodo exponga los eventos de PMU necesarios sin
distorsión. Si `perf_event_paranoid` degradara silenciosamente un evento
específico para ambos por igual, esta validación no lo detectaría (heredaría
el mismo sesgo). Por eso `Vtune/preflight_uarch.py` no se limita a confirmar
que VTune "corre" — confirma con un smoke test real que las 4 categorías de
Nivel 1 del Top-Down salen pobladas con números creíbles antes de aceptar
ningún resultado de la campaña.

## D. Validación de los kernels

Principio explícito para esta sección: **ningún kernel se declara de una
clase antes de correr la campaña.** La columna `expected_behavior` de
`consolidated_validation.csv` es un *hint* de literatura/diseño (idéntico en
espíritu a `phase_label_hint` en `docs/retoma/Guia_Maestra_Fase1_DVFS.md`
sección 6) — la columna que importa es `vtune_validation_class`, la que
observó VTune en esa corrida específica.

| Kernel | `expected_behavior` (hint, no verdad asumida) | Razón del hint |
|---|---|---|
| STREAM | `memory_bound` | Construido explícitamente para saturar ancho de banda de memoria (copia/suma de arreglos, sin reuso de caché) — referencia clásica de la literatura de Roofline. |
| DGEMM (OpenBLAS) | `compute_bound` | Multiplicación de matrices densa con alto reuso de datos en caché — referencia clásica compute-bound. |
| EP | `compute_bound` | Generador congruencial + `sqrt`/`log`, casi sin tráfico de memoria (ver `pipelinevtune/context/03_kernels_notas.md`). |
| MG | `memory_bound` | Recorre una jerarquía de mallas; los niveles finos no caben en LLC. |
| CG | `memory_bound` | SpMV sobre matriz dispersa aleatoria — poca localidad. |
| FT | `intermedio` | Alterna FFT local (buena localidad) con transposiciones globales (tráfico masivo) — mezcla de fases dentro de una corrida. |
| LU | `intermedio` | Paralelismo *wavefront* con sincronización fuerte — el limitante real puede ser sincronización, algo que ni TMAM ni Roofline capturan como categoría propia. |
| BT | `intermedio` (candidato compute-bound) | Solver estructurado, aritmética densa sin división/transcendentales pesados. |

Un resultado `ambiguous` o que contradiga el hint **no se descarta ni se
fuerza** — se reporta con su `validation_reason` completo (qué categoría del
Top-Down domina y por qué) para que quien lea `consolidated_validation.csv`
pueda decidir si el kernel sigue siendo apto para el dataset de
entrenamiento, necesita revisión manual (mismo trato que el proyecto ya le
da a EP por su riesgo de subconteo de FLOPs, decisión D4), o se excluye.

## E. Funcionamiento detallado de VTune y de Microarchitecture Exploration

### E.1 Qué mide, y cómo (el mecanismo, no solo el resultado)

VTune no instrumenta el binario — lo lanza (o se adjunta a un proceso ya
corriendo) y observa desde afuera mediante muestreo. Para Microarchitecture
Exploration, el mecanismo es **Hardware Event-Based Sampling (EBS)** vía
**PEBS** (*Precise Event-Based Sampling*): un contador de la PMU se programa
para desbordar cada N eventos (p. ej. cada ciertos millones de ciclos), y en
cada desborde el hardware captura el *instruction pointer* y el estado
exacto de ese instante — no una aproximación por temporizador de SO. En este
nodo, ese mecanismo pasa por `perf_event_open` del kernel Linux, en modo
*driverless* (sin el driver propio `sep`/`socperf` de Intel, confirmado para
Hotspots/HPC Performance en `docs/vtune/Informe_VTune_Profiler.md` §3.2) —
`Vtune/preflight_uarch.py` registra el `Collector Type` real de la primera
corrida para confirmar si esto sigue siendo así con el permiso nuevo, o si
ahora hay un driver propio detrás (cambiaría qué eventos de *uncore* son
alcanzables, ver E.7).

### E.2 Diferencia entre eventos de hardware crudos y métricas derivadas

Un **evento de hardware** (p. ej. `CYCLE_ACTIVITY.STALLS_L3_MISS`,
`TOPDOWN.BACKEND_BOUND_SLOTS`) es un contador físico de la PMU: un número
entero que se incrementa por una condición exacta del silicio, definido en
el manual de eventos de Intel para esta microarquitectura (Ice Lake-SP).
Una **métrica derivada** (`Memory Bound: 34.5% of Pipeline Slots`) es una
fórmula que VTune calcula combinando *varios* eventos crudos y normalizando
contra `TOPDOWN.SLOTS` (el total de slots de pipeline disponibles en la
ventana medida). El usuario nunca ve `Memory Bound` como un evento de PMU
suelto — es Intel, vía las fórmulas públicas de TMAM, quien define esa
combinación. `Vtune/run_validation.py` guarda ambas cosas para cada corrida:
`report.csv`/`summary.txt` (métricas ya derivadas, lo que usa
`validation_classifier.py`) y `raw_hw_events.csv` (los eventos crudos que
VTune configuró para llegar a esas métricas, vía `vtune -report hw-events`)
— este segundo archivo es la fuente real y verificable de "qué contadores
participaron", no una lista genérica de documentación.

### E.3 Metodología de descomposición: Top-Down Microarchitecture Analysis Method (TMAM)

Sí, es el método que usa Microarchitecture Exploration — a diferencia de
`hpc-performance` (que en este nodo solo exponía niveles parciales del
Top-Down, ver `docs/vtune/Informe_VTune_Profiler.md` §4.1), este análisis da
las 4 categorías de Nivel 1 completas. La idea (Yasin, Intel ISPASS 2014):
en cada ciclo, el front-end puede *emitir* hasta N "pipeline slots" (N =
ancho de emisión del core; Ice Lake-SP es 5-wide) hacia las unidades de
ejecución. TMAM clasifica cada slot posible en una de cuatro categorías
mutuamente excluyentes que suman ~100% de los slots disponibles:

```
Pipeline Slots = Retiring + Bad Speculation + Front-End Bound + Back-End Bound
```

- **Retiring** — el slot produjo una instrucción que efectivamente terminó
  (trabajo útil). Subcategorías: `Light Operations` (instrucciones simples,
  de bajo costo) vs. `Heavy Operations` (microcódigo, operaciones complejas).
- **Bad Speculation** — el slot se gastó en instrucciones luego descartadas.
  Subcategorías: `Branch Mispredict` (predicción de salto incorrecta) vs.
  `Machine Clears` (otras razones de descarte, p. ej. violaciones de orden
  de memoria).
- **Front-End Bound** — el slot quedó vacío porque fetch+decode no
  alimentó instrucciones a tiempo. Subcategorías: `Front-End Latency`
  (arranque lento, p. ej. tras un salto) vs. `Front-End Bandwidth`
  (decodificador saturado en régimen estable).
- **Back-End Bound** — el slot quedó vacío porque el back-end no pudo
  aceptar más trabajo. Se subdivide en las dos categorías que esta campaña
  usa como eje central de decisión (ver `Vtune/validation_classifier.py`):
  - **Memory Bound** — esperando datos de la jerarquía de memoria.
    Sub-niveles: `L1 Bound`, `L2 Bound`, `L3 Bound`, `DRAM Bound`,
    `Store Bound` — cuanto más fino, más cerca de necesitar contadores de
    *uncore* (ver E.7).
  - **Core Bound** — cuellos de botella dentro del propio core: puertos de
    ejecución saturados (`Ports Utilization`) o latencia de unidades
    específicas como la de división (`Divider`) — la pieza que
    `hpc-performance` no podía aislar en este nodo y que obligó al pipeline
    anterior (`pipelinevtune/classifier.py`, D3-v3) a calibrar contra
    anclas STREAM/DGEMM en su lugar. Con Microarchitecture Exploration
    funcional, `Core Bound` es un número directo del mismo reporte —
    razón central por la que esta campaña ya NO necesita esa calibración
    externa (ver `Vtune/validation_classifier.py`, docstring del módulo).

### E.4 Interpretación de cada porcentaje

Cada porcentaje es **fracción de Pipeline Slots** (no fracción de tiempo de
reloj, aunque para código de un solo hilo bien paralelizado ambas nociones
casi coinciden) — un slot no usado por Retiring y no perdido por Bad
Speculation está, por definición del modelo, perdido por Front-End o
Back-End. Un valor alto de `Retiring` es deseable (trabajo útil); valores
altos de cualquiera de las otras tres son, en principio, oportunidad de
mejora — pero para los propósitos de esta campaña, lo relevante no es
"cuánto se puede optimizar" (el uso normal de VTune) sino **cuál de las
cuatro categorías domina**, como señal de qué está limitando al kernel.
`DRAM Bound`, a diferencia de las categorías de Nivel 1/2, se reporta como
`% of Clockticks`, no `% of Pipeline Slots` (mismo patrón ya confirmado para
`hpc-performance` en este nodo, `pipelinevtune/context/04`) — no son
directamente comparables entre sí sin normalizar, por eso
`validation_classifier.py` usa `DRAM Bound` solo como detalle cualitativo
dentro de la justificación de `Memory Bound`, nunca en la comparación
numérica principal (`Memory Bound` vs. `Core Bound`, ambas ya en la misma
escala de Pipeline Slots).

### E.5 Qué métricas son directas y cuáles calculadas

| Directa (un solo evento de PMU, sin combinar) | Calculada (combina ≥2 eventos) |
|---|---|
| `Instructions Retired`, `Clockticks` | `CPI Rate` = Clockticks / Instructions Retired |
| `TOPDOWN.SLOTS` (crudo, ver `raw_hw_events.csv`) | `Retiring`/`Bad Speculation`/`Front-End Bound`/`Back-End Bound` = fracciones de `TOPDOWN.SLOTS` combinando varios contadores `TOPDOWN.*` y `PERF_METRICS.*` (registro dedicado de Ice Lake para TMAM, ver E.6) |
| `MEM_LOAD_RETIRED.L3_MISS` | `DRAM Bound` = estimación por atribución de latencia (no bytes movidos directos), combinando `MEM_LOAD_L3_MISS_RETIRED.LOCAL_DRAM`/`OFFCORE_REQUESTS_OUTSTANDING.*` — mismo mecanismo ya confirmado para `hpc-performance` en `docs/vtune/Informe_VTune_Profiler.md` §9 |
| `Average CPU Frequency` (vía MSR de frecuencia) | `IPC` = 1 / CPI Rate (no la reporta VTune como campo propio; `Vtune/uarch_parser.py` la deriva) |

### E.6 Multiplexación: por qué Microarchitecture Exploration es más pesado que Hotspots

Un core moderno tiene un número limitado de contadores de PMU programables
de propósito general (típicamente 4-8, confirmado en este nodo para otros
análisis vía `probe_pmc_count()` del orquestador principal — ARC-37). TMAM
completo necesita más eventos simultáneos de los que caben en esos
registros a la vez. Intel resuelve esto con **`PERF_METRICS`**, un registro
fijo específico (disponible desde Ice Lake) que entrega las 4 categorías de
Nivel 1 en un solo conjunto sin tener que multiplexar — pero los
sub-niveles (`L1 Bound`, `Divider`, etc.) sí pueden requerir contadores de
propósito general adicionales, y si el número de eventos pedidos supera los
disponibles, VTune **multiplexa**: rota qué eventos mide en ventanas de
tiempo muy cortas dentro de la misma corrida y extrapola estadísticamente el
resto. Esto no invalida el resultado, pero sí introduce ruido — es la razón
por la que el techo de tiempo de `Vtune/sbatch_vtune_validation.sh` asume
1.5× el costo de una corrida de `hpc-performance` para `uarch-exploration`,
y por la que `Vtune/uarch_parser.py` expone `topdown_sum_pct` (la suma de
las 4 categorías de Nivel 1, que debería rondar 100%) como bandera de
calidad — una desviación grande de 100% es síntoma de multiplexación
degradada, no un bug del parser.

### E.7 Qué depende de la microarquitectura concreta (Ice Lake-SP)

- El registro `PERF_METRICS` que da las 4 categorías de Nivel 1 sin
  multiplexar existe desde Ice Lake — en microarquitecturas anteriores
  (como el Westmere descartado para este proyecto, ver
  `pipelinevtune/CLAUDE.md`) TMAM se reconstruye completamente por
  multiplexación de eventos de propósito general, mucho más ruidoso.
- Los nombres y el número exacto de eventos por sub-categoría (`L1 Bound`
  vs. `L2 Bound` vs. `L3 Bound`) dependen del manual de eventos específico
  de Ice Lake-SP (Xeon Gold 5315Y) — no son portables literalmente a AMD ni
  a otra generación de Intel sin remapear.
- El ancho de emisión (5-wide en Ice Lake) determina el total de
  `TOPDOWN.SLOTS` disponibles por ciclo — cambia el punto de referencia de
  "100% de slots" entre microarquitecturas.
- Acceso a *uncore* (memoria, LLC compartida a nivel de PMU física, fuera
  del core) sigue siendo la restricción ya documentada para este nodo
  (`docs/vtune/Informe_VTune_Profiler.md` §9) — el permiso nuevo habilita
  Microarchitecture Exploration, pero `Vtune/preflight_uarch.py` no asume
  que también resolvió el acceso a uncore; lo que el smoke test confirma es
  si las 4 categorías de Nivel 1 (que Ice Lake puede calcular sin uncore,
  vía `PERF_METRICS`) salen pobladas — `DRAM Bound` específicamente puede
  seguir viniendo por estimación de latencia, no por conteo directo de
  bytes, exactamente como ya se confirmó para `hpc-performance`.

## F. Reproducibilidad

Cada corrida individual queda documentada en su propio `metadata.json`
(`Vtune/run_validation.py::_write_metadata`) con, como mínimo:

- Versión de VTune (`vtune --version`, registrada en el log del job).
- Nodo (`paccaA100`, fijado por `--nodelist` del sbatch) y CPUs reales
  (`affinity_at_run_time`, leído con `sched_getaffinity` en el momento de la
  corrida, no asumido).
- Hilos y dominio OMP (`omp_num_threads`, `omp_places`, `omp_proc_bind`,
  más el `pin_prefix` de `taskset` realmente aplicado).
- Comando exacto de VTune ejecutado (`exact_vtune_command`), sin abreviar.
- Binario (`binary_path`) y su hash `sha256` (`binary_checksum`) — mismo
  mecanismo que `catalog.py` ya usa en el orquestador principal (CAT-02),
  para poder confirmar después que la corrida se hizo contra el binario que
  se cree que se hizo.
- Argumentos del binario (relevante para DGEMM: `["4096", "5"]`, tamaño de
  problema y repeticiones internas).
- Fecha/hora UTC de la corrida (`timestamp_utc`).
- ID del job de Slurm (`slurm_job_id`) y nodo real (`slurm_nodelist`).
- `result_dir`: ruta al resultado completo de VTune, para abrir después en
  la GUI (sección de abajo) sin tener que re-ejecutar nada.

---

## 9. Flujo manual de referencia (un solo kernel)

Antes de confiar en la automatización, así es como se correría **un solo**
kernel a mano — el pipeline automático (`Vtune/run_validation.py`) hace
exactamente esto mismo, repetido por kernel/clase/repetición, con manejo de
errores y metadata alrededor.

```bash
# 1. Módulos (secuencia jerárquica confirmada para este cluster -- el
#    módulo hijo no aparece en 'module avail' sin el padre cargado primero)
module purge
module load devtools/intel/oneapi/2023
module load vtune/2023.0.0

# 2. Confirmar version y que el analisis esta listado
vtune --version
vtune -collect-list | grep -i uarch

# 3. Dominio D6 del proyecto: 6 cores fisicos, 0-5, sin SMT
export OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close

# 4. Coleccion real -- notese "-r" para el directorio de resultado
#    (no "-result-dir", esa es la sintaxis larga que algunas versiones de
#    VTune tambien aceptan pero que este proyecto no usa en ningun otro
#    lugar del repo; "-r" es la que ya confirmo pipelinevtune contra esta
#    misma instalacion 2023.0.0)
taskset -c 0-5 vtune -collect uarch-exploration \
  -r "$HOME/vtune_validation/manual/ep_C" \
  -- "$HOME/vtune_selfcheck/bin/ep.C.x"

# 5. Reportes -- texto para leer, CSV para parsear
vtune -report summary -r "$HOME/vtune_validation/manual/ep_C"
vtune -report summary -r "$HOME/vtune_validation/manual/ep_C" -format=csv \
  > ep_C_summary.csv
vtune -report hw-events -r "$HOME/vtune_validation/manual/ep_C" -format=csv \
  > ep_C_hw_events.csv
```

**Verificar, no asumir, antes de confiar en el resto:** correr esto una vez
a mano contra `ep.C.x` (el único kernel con tiempo real ya medido en este
proyecto, 13.6 s bajo `hpc-performance`) y comparar la salida real de
`-report summary` contra los nombres de campo que asume
`Vtune/uarch_parser.py` (sección "TOP_LEVEL_LABELS"/"LEVEL2_LABELS" del
archivo). Si algo no coincide, corregir el parser ahí, no en este documento
primero — el código y el documento deben quedar sincronizados, y el código
es lo que efectivamente corre la campaña.

**Cómo automatiza esto el pipeline:** `Vtune/run_validation.py::process_workload`
ejecuta exactamente esta secuencia (con `taskset`/`OMP_*` ya armados por
`build_env()`), agregando antes un baseline sin VTune (para separar tiempo
de instrumentación del tiempo real del kernel) y after guardando
`summary.txt`/`report.csv`/`raw_hw_events.csv`/`metadata.json` en la
estructura de `results_vtune/` documentada en `Vtune/README.md`, y
alimentando cada resultado a `uarch_parser.py` + `validation_classifier.py`
para producir la fila correspondiente de `consolidated_validation.csv`.

---

## 10. Apertura posterior en VTune GUI

### Qué copiar

El `result_dir` completo de la corrida que interese (columna `result_dir` de
`consolidated_validation.csv`, o `<kernel>.<clase>/rep_NN/result/` dentro de
`results_vtune/job_<JOBID>/`). Es autocontenido — no hace falta el binario
original ni ningún otro archivo del nodo para visualizarlo, solo ese
directorio.

### Cómo llevarlo al equipo local

```bash
# Comprimir en el nodo/login antes de transferir (los result-dir de VTune
# traen muchos archivos pequeños -- un solo tar es mucho mas rapido por SSH
# que copiar el arbol disperso)
tar -czf ep_C_rep01_result.tar.gz -C ~/vtune_validation/results_vtune/job_<JOBID>/ep.C/rep_01 result

# Desde el equipo local (ajustar el alias segun tu ~/.ssh/config, ver
# docs/retoma/pacca/Auditoria_PaccaA100_Unicartagena.md seccion "Acceso")
scp hpc-unicartagena:~/ep_C_rep01_result.tar.gz .
tar -xzf ep_C_rep01_result.tar.gz
```

### Compatibilidad de versión

Abrir con **VTune 2023.x local** (misma línea mayor que la del clúster,
confirmada por `vtune --version` en el log del job) — VTune generalmente
puede abrir resultados de versiones anteriores dentro de la misma serie,
pero no está garantizado en sentido inverso (una versión más vieja abriendo
un resultado más nuevo). Instalar la misma serie evita el problema por
completo en vez de depender de compatibilidad no garantizada.

### Cómo abrir

```bash
vtune-gui result/     # desde la carpeta que contiene el result-dir extraído
# o, con la GUI ya abierta: File → Open Result → seleccionar la carpeta
```

### Qué vistas inspeccionar, y cuáles vale la pena exportar para la tesis

| Vista | Para qué sirve aquí | ¿Exportar como figura? |
|---|---|---|
| **Summary (Top-Down tree)** | La vista central de esta campaña: árbol interactivo Retiring/Front-End/Bad Speculation/Back-End Bound con expansión a Memory Bound/Core Bound y sub-niveles. Confirma visualmente lo que `validation_classifier.py` decidió, con la ventaja de poder expandir cada rama. | **Sí**, una por régimen (un kernel compute-bound claro, uno memory-bound claro) — es la evidencia visual directa de la metodología de la sección E.3. No exportar las 16 corridas, sería redundante con el CSV. |
| **Bottom-up / Caller-Callee** | Útil para depurar *por qué* un kernel específico salió inesperado (p. ej. si EP sale `ambiguous`, ver qué función domina) — herramienta de diagnóstico, no de tesis. | No, salvo para justificar un caso atípico puntual (ver D4, revisión manual de EP/IS). |
| **Timeline por hilo** | Muestra balance de carga entre los 6 hilos OMP y transiciones de fase (relevante para MG/FT, que mezclan regímenes dentro de una corrida, ver sección D). | Posiblemente **sí** para un único kernel que mezcle fases (MG o FT) — ilustra visualmente el fenómeno que el CSV agregado no puede mostrar (un solo número por corrida). |
| **Memory Bound breakdown (histograma L1/L2/L3/DRAM)** | Detalle de qué nivel de la jerarquía domina dentro de `Memory Bound` — más rico que el campo único `dram_bound_pct` del CSV. | Considerar para STREAM (debería mostrar dominancia clara de DRAM Bound) como evidencia de que el ancla se comporta como se espera. |
| **CPU Utilization / Effective Utilization** | Confirma que los 6 hilos realmente se usaron durante toda la corrida (afinidad correcta, sin serialización inesperada). | No, es un chequeo de sanidad, no un resultado de tesis. |
| **Hotspots (función dominante)** | Complementaria, no central en esta campaña (a diferencia del pipeline anterior con `hotspots` como análisis propio) — aquí solo aparece como parte del mismo `result-dir` de `uarch-exploration`. | No, salvo que se necesite justificar un caso atípico. |

**Principio para decidir qué exportar:** una figura por *fenómeno* que el
texto necesite mostrar (un compute-bound claro, un memory-bound claro, un
caso ambiguo/mixto interesante como MG o LU), no una figura por kernel — el
CSV consolidado ya lleva el registro completo y numérico de los 16+
resultados; las figuras de la GUI son para ilustrar el mecanismo, no para
repetir la tabla en forma de imagen.
