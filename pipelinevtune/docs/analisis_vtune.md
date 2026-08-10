# Tipos de análisis de Intel VTune Profiler usados en este pipeline

Documento de respaldo metodológico: qué es cada tipo de análisis de VTune que
toca este proyecto, qué mide, cómo llega a esos números, cómo se usa en la
práctica, y con qué se puede defender su uso como segundo método de
validación de etiquetas frente al orquestador de Hyperion (D2,
`context/02_decisiones.md`). Cierra con la lógica de clasificación vigente
(`classifier.py`) y una nota sobre reemplazar DGEMM por ERT como techo de
cómputo.

VTune Profiler 2023 (Intel oneAPI) ofrece más de una decena de tipos de
análisis (Hotspots, Microarchitecture Exploration, Memory Access, Memory
Consumption, Threading, I/O, HPC Performance Characterization, GPU
Compute/Media, Platform Profiler, etc.). Este proyecto solo toca tres,
confirmados disponibles en Cartagena por `vtune-self-checker.sh`
(`context/04_vtune_selfchecker_resultados.md`):

| Análisis | Estado en este nodo | Rol en el pipeline |
|---|---|---|
| Hotspots (con Hardware Event-Based Sampling) | Disponible | Identifica la función dominante (`dominant_function`) |
| HPC Performance Characterization | Disponible | Fuente de `Memory Bound`, `Cache Bound`, `DRAM Bound`, `DP GFLOPS`, vectorización — lo que alimenta `classifier.py` |
| Microarchitecture Exploration | **No disponible** | Sería la fuente ideal (ver §3) — no se puede usar aquí |

---

## 1. Hotspots (Hardware Event-Based Sampling)

### Qué dice Intel que hace

Identifica qué funciones/líneas de código consumen más tiempo de CPU, para
guiar dónde vale la pena optimizar. Es el análisis más básico de VTune y la
base sobre la que se construyen varios de los demás (HPC Performance
Characterization internamente también recolecta con el mismo mecanismo de
muestreo).

### Cómo mide (el mecanismo, no solo el resultado)

VTune ofrece dos modos de muestreo para Hotspots:

- **User-mode sampling** (software): usa temporizadores del sistema operativo
  para interrumpir periódicamente y capturar el stack. No necesita PMU ni
  permisos especiales, pero tiene más overhead y menos precisión de
  atribución (el punto de interrupción no siempre coincide con la instrucción
  real que causó el costo).
- **Hardware Event-Based Sampling (EBS)** — el modo que usa este pipeline
  (`-knob sampling-mode=hw`): configura un contador de la PMU (por ejemplo,
  ciclos o instrucciones retiradas) para que genere una interrupción cada *N*
  eventos, usando el mecanismo **PEBS** (Precise Event-Based Sampling) del
  procesador. Cuando el contador desborda, la CPU captura el *instruction
  pointer* exacto y el stack en ese instante. Esto da atribución precisa a
  nivel de instrucción con overhead mucho menor que el muestreo por software.

En Linux, EBS se recolecta sin necesitar el driver propio de VTune (`sep`) ni
`sudo`, siempre que `/proc/sys/kernel/perf_event_paranoid` lo permita — VTune
usa el mecanismo estándar del kernel (`perf_event_open`), reportado en las
capturas reales de este nodo como `Collector Type: Driverless Perf
per-process sampling` (ver `tests/unit/fixtures/real_summary_*.txt`, línea
`Collector Type`). Esto es justo lo que hace posible correr este pipeline sin
privilegios de administrador (regla dura del proyecto, ver `CLAUDE.md`).

### Sintaxis

```bash
vtune -collect hotspots -knob sampling-mode=hw -r RESULT_DIR -- BINARIO [args]
vtune -report hotspots -r RESULT_DIR                    # texto, tabla "Top Hotspots"
vtune -report hotspots -r RESULT_DIR -format=csv         # mismo reporte en CSV
```

### Qué genera primero: ¿corrida o reporte?

**Primero la corrida, después el reporte — son dos pasos separados.** `vtune
-collect ...` no imprime nada legible por humanos de forma directa: ejecuta el
binario bajo muestreo y guarda una base de datos cruda de samples + símbolos
en el directorio `-r RESULT_DIR` (una carpeta `RESULT_DIR/` con archivos
internos de VTune, no un CSV). Los reportes (`vtune -report ...`) se generan
**a demanda**, después, leyendo esa base de datos — se puede pedir el mismo
resultado en distintos formatos (texto, csv, xml) tantas veces como se quiera
sin volver a correr el binario.

### Ejemplo de salida (tabla "Top Hotspots")

```
Function          Module      CPU Time  % of CPU Time
binvcrhs           bt.C.x      42.310s        61.2%
matvec_sub          bt.C.x      9.874s        14.3%
...
```

`vtune_parser.parse_hotspots_text()` extrae solo la primera fila (la función
dominante, VTune ya la ordena por CPU Time descendente) — ver
`vtune_parser.py:121-141`.

**Límite honesto:** Hotspots identifica *dónde* se gasta el tiempo, no *por
qué* (no dice si ese tiempo es memoria o cómputo) — esa es la razón por la
que el pipeline no usa Hotspots para clasificar, solo para `dominant_function`
(regla ya establecida, ver `PLAN.md` Fase 3.3).

### Permisos necesarios

Ninguno especial más allá de `perf_event_paranoid` en un valor que permita
muestreo por proceso propio (típicamente ≤2 en la mayoría de distros/kernels
de cluster) — sin `sudo`, sin instalar el driver `sep`, sin tocar
`perf_event_paranoid` (regla dura del proyecto).

---

## 2. HPC Performance Characterization

### Qué dice Intel que hace

Es un análisis pensado específicamente para código HPC: corre el mismo
mecanismo de muestreo EBS que Hotspots, pero además calcula y expone un
conjunto curado de métricas derivadas orientadas a caracterizar el
comportamiento de cómputo/memoria/vectorización de la aplicación en una sola
pasada — sin que el usuario tenga que armar la fórmula a mano contador por
contador.

### Qué mide y cómo llega a esos números

Todos estos campos, con los nombres literales confirmados en este nodo
(`context/04`, addendum Fase 0):

| Campo | Qué es | Cómo se deriva |
|---|---|---|
| `DP GFLOPS` / `SP GFLOPS` / `x87 GFLOPS` | Tasa de operaciones de punto flotante retiradas | Contadores de uops de FP retirados (por tipo: escalar/128/256/512-bit) × FLOPs por uop, dividido entre tiempo transcurrido. **No depende de uncore** — son contadores de ejecución del core. |
| `CPI Rate` | Ciclos por instrucción | `CPU_CLK_UNHALTED / INST_RETIRED`, contadores estándar del core. |
| `Memory Bound` | % de *Pipeline Slots* vacíos atribuidos a la espera de memoria | Metodología **TMAM** (Top-Down Microarchitecture Analysis Method, Intel/Yasin — ver §3). Nivel superior del árbol Top-Down. |
| `Cache Bound` / `DRAM Bound` | Subcategorías de `Memory Bound`, % de *Clockticks* | Contadores de stall por nivel de caché (`CYCLE_ACTIVITY.STALLS_L1D_MISS/L2_MISS/L3_MISS`, `MEM_LOAD_L3_MISS_RETIRED.LOCAL_DRAM/REMOTE_DRAM`, etc.) |
| `NUMA: % of Remote Accesses` | % de accesos servidos desde memoria remota | Contadores de latencia de carga con clasificación de origen NUMA |
| `Vectorization` / `Instruction Mix` | % de operaciones FP empaquetadas (128/256/512-bit) vs escalares | Contadores de uops de FP retirados, desglosados por ancho de empaquetado |

**Sí es TMAM, pero solo el nivel superior.** `Memory Bound` que reporta HPC
Performance Characterization **es** una categoría real del árbol Top-Down de
Intel (no una métrica inventada para este análisis) — pero este tipo de
análisis solo expone esa rama (más sus hijas `Cache Bound`/`DRAM Bound`), no
el árbol completo con las 4 categorías de nivel 1 (`Retiring` / `Bad
Speculation` / `Front-End Bound` / `Back-End Bound`) ni el hermano de
`Memory Bound` dentro de `Back-End Bound` (`Core Bound`). Eso solo lo imprime
Microarchitecture Exploration (§3) — confirmado empíricamente en este nodo,
no asumido de la documentación general (ver `context/04`, "Hallazgo que sí
cambia algo del plan").

### Sintaxis

```bash
vtune -collect hpc-performance -r RESULT_DIR -- BINARIO [args]
vtune -report summary -r RESULT_DIR                      # texto
vtune -report summary -r RESULT_DIR -format=csv           # CSV
vtune -report hw-events -r RESULT_DIR -format=csv          # contadores crudos (TOPDOWN.*, CYCLE_ACTIVITY.*, etc.), por funcion
```

Igual que Hotspots: `-collect` ejecuta y guarda la base de datos cruda;
`-report` genera el texto/CSV a demanda, después, tantas veces como haga
falta.

### Ejemplo real de salida (captura real de este nodo, STREAM)

```
Elapsed Time: 0.687s
    DP GFLOPS: 2.746
    CPI Rate: 5.345
    Average CPU Frequency: 3.416 GHz
Effective Physical Core Utilization: 43.6% (6.970 out of 16)
Memory Bound: 51.9% of Pipeline Slots
    Cache Bound: 19.2% of Clockticks
    DRAM Bound: 67.7% of Clockticks
    NUMA: % of Remote Accesses: 0.0%
Vectorization: 100.0% of Packed FP Operations
    ...
```

(Captura completa en `tests/unit/fixtures/real_summary_stream.txt`.)

### ¿Clasifica por sí solo, sin anclas?

**No en este nodo, y esa es la razón concreta por la que existen las anclas
STREAM/DGEMM.** `HPC Performance Characterization` da un número aislado
(`Memory Bound=51.9%`), pero no imprime junto a él una contraparte de cómputo
comparable de la misma jerarquía (`Core Bound`) para decidir cuál domina. Sin
ese hermano, un solo porcentaje no dice nada por sí mismo — 51.9% podría ser
"alto" o "bajo" según qué arquitectura/carga se use como referencia. Esto es
justo lo que forzó D3-v3 (`context/02_decisiones.md`): comparar el número del
kernel contra dos puntos de referencia conocidos (STREAM=memoria pura,
DGEMM=cómputo puro) medidos con este mismo análisis en este mismo nodo, en
vez de contra una frontera fija sin fundamento. Ver §5.

### Permisos necesarios

Los mismos que Hotspots HW — se construye sobre el mismo mecanismo de
muestreo EBS. Ninguno de los campos usados por este pipeline depende de
contadores uncore (confirmado en `context/04`: `DRAM Bound` sale poblado sin
uncore en este nodo, hallazgo empírico documentado explícitamente porque
contradecía lo anticipado).

---

## 3. Microarchitecture Exploration — la "reina" que no está disponible aquí

### Qué dice Intel que hace

Es la implementación completa en VTune del método **TMAM** (*Top-Down
Microarchitecture Analysis Method*), publicado por Ahmad Yasin (Intel) en
*"A Top-Down Method for Performance Analysis and Its Application to Skylake"*
(IEEE ISPASS 2014) — no es una heurística propia de VTune, es una metodología
de Intel publicada y revisada por pares, ya estándar en la industria de
optimización de rendimiento.

### El árbol completo (lo que este análisis sí imprime y HPC Performance Characterization no)

```
100% de Pipeline Slots
├── Retiring            (trabajo útil retirado)
├── Bad Speculation      (desperdiciado por mispredicción de ramas)
├── Front-End Bound      (vacío: el frontend no alimentó uops a tiempo)
└── Back-End Bound       (vacío: el backend no pudo aceptar más trabajo)
     ├── Core Bound       (stalls por recursos de ejecución: puertos, ALUs, FPU ocupadas)
     └── Memory Bound     (stalls por la jerarquía de memoria)
          ├── L1 Bound / L2 Bound / L3 Bound / DRAM Bound / Store Bound
```

`Core Bound` y `Memory Bound` aparecen **como hermanos explícitos, en el
mismo reporte, de la misma corrida** — con esto, clasificar
compute-bound/memory-bound es una comparación directa de dos campos reales de
Intel, sin necesitar inventar un complemento (`100 - memory_bound_pct`, la
solución que se usó en la versión descartada de D3-native) ni calibrar contra
anclas externas. Por eso es "la reina" que resolvería D3 de forma más limpia.

### Por qué no está disponible en Cartagena

Confirmado empíricamente por el self-checker (`context/04`), no asumido:
`Microarchitecture Exploration` y `Memory Access` no aparecen en la lista de
análisis disponibles en este nodo. La razón técnica más probable, consistente
con el resto de restricciones ya documentadas: este análisis necesita
multiplexar muchos más eventos de PMU en varias pasadas (incluyendo eventos
de nivel más fino/uncore para los niveles inferiores del árbol), lo que en la
práctica en clusters compartidos suele requerir o bien un
`perf_event_paranoid` más permisivo que el disponible aquí, o bien el driver
propio de muestreo de VTune (`sep`), que se instala como módulo de kernel y
necesita privilegios de administrador para cargarse — justo lo que la regla
dura del proyecto prohíbe pedir o asumir (`CLAUDE.md`: "No pidas ni asumas
privilegios de administrador"). No se diagnosticó la causa exacta byte a
byte porque no hace falta: el resultado — no disponible, sin privilegios para
intentarlo — ya es suficiente para descartar esta vía en este nodo.

### Permisos que necesitaría (si se consiguieran en el futuro)

- `perf_event_paranoid` en un nivel más permisivo, y/o
- El driver `sep` de VTune instalado por un administrador del cluster, y/o
- Acceso a contadores uncore (a veces restringido a nivel de BIOS/firmware en
  nodos compartidos por razones de aislamiento entre tenants).

Ninguna de estas rutas está bajo control de este proyecto — queda como
extensión futura documentada (`CLAUDE.md`: "si más adelante se obtienen
permisos... no está bloqueado por ahora, simplemente no es parte de este
trabajo").

---

## 4. Exportar y visualizar localmente

### Exportar datos (lo que ya usa este pipeline)

```bash
vtune -report summary -r RESULT_DIR -format=csv > resumen.csv
vtune -report hotspots -r RESULT_DIR -format=csv > hotspots.csv
vtune -report hw-events -r RESULT_DIR -format=csv > contadores_crudos.csv
```

Formatos soportados: `text` (default), `csv`, `xml`. Es lo que consume
`vtune_parser.py` — el pipeline no necesita GUI para funcionar, todo el
parseo es sobre texto/CSV.

### Ver visualmente en local (GUI)

El directorio `RESULT_DIR/` que deja `vtune -collect` es autocontenido y
portable: contiene la base de datos cruda de samples y el caché de símbolos,
no solo un reporte de texto. Para explorarlo visualmente (vista Bottom-up,
Top-down, línea de código anotada, timeline):

1. Copiar `RESULT_DIR/` completo del nodo remoto a una máquina local con
   VTune instalado (Intel oneAPI Base/HPC Toolkit, gratuito con cuenta
   Intel — no requiere el cluster).
2. Abrirlo con la GUI de escritorio: `vtune-gui`, `File > Open Result...`.
   — o con el servidor web local (más cómodo si no hay entorno gráfico
   pesado disponible): `vtune-backend --data-directory RESULT_DIR`, que
   levanta un servidor local (accesible por navegador, se puede tunelizar
   por SSH desde el nodo de login si se prefiere no copiar el resultado).

Simplificado: **`vtune -collect` en el cluster (headless, por CLI) → copiar
el resultado → `vtune-gui`/`vtune-backend` en una máquina con pantalla**, o
simplemente seguir con el CSV/texto (la vía que usa este pipeline, sin
necesitar GUI en ningún punto).

---

## 5. Cómo defender VTune como segundo método de validación de etiquetas

Puntos concretos para sustentar esto en el TG:

1. **No es una heurística propia del pipeline.** `Memory Bound`, `Cache
   Bound`, `DRAM Bound` son categorías del método TMAM de Intel (Yasin,
   ISPASS 2014), publicado y usado ampliamente en la industria de
   optimización de HPC — el pipeline no inventa cómo se calculan, solo
   decide cómo interpretarlas (§6).
2. **Basado en contadores de hardware reales de esta CPU (Ice Lake-SP), no en
   simulación ni análisis estático.** Cada número refleja lo que realmente
   ocurrió en la ejecución real sobre el nodo real, con las condiciones reales
   de la campaña (mismo dominio de cores que el orquestador, D6).
3. **Independiente por diseño del orquestador que valida (D2).** El
   clasificador de este pipeline no comparte código, heurísticas ni fuente de
   datos con el orquestador de Hyperion — si coinciden, es evidencia real de
   corroboración, no un resultado circular.
4. **Las limitaciones se documentan, no se esconden.** Que
   `HPC Performance Characterization` no imprima `Core Bound` en este nodo, y
   que por eso se calibra contra anclas (D3-v3) en vez de usar el árbol
   completo, es una decisión metodológica registrada con su motivo — no una
   discrepancia oculta si alguien pregunta "¿por qué no usaron directamente
   el veredicto nativo de VTune?".
5. **Las anclas STREAM/DGEMM son referencias con comportamiento conocido por
   construcción** (STREAM Triad es memory-bandwidth-bound por diseño, DGEMM es
   compute-bound por diseño), no valores arbitrarios — calibrar contra ellas
   es análogo a calibrar un instrumento contra patrones de referencia
   conocidos, no a ajustar el resultado para que "salga bien".

---

## 6. Lógica de clasificación vigente (VTune + anclas)

Implementada en `classifier.py`. Dos funciones separadas, nunca fusionadas
(D8).

### 6.1 `clasificar_nativo()` — compara VTune contra VTune

Usa tres métricas del mismo reporte `hpc-performance` (`memory_bound_pct`,
`cache_bound_pct`, `dram_bound_pct_or_na`), medidas en tres corridas: el
kernel a clasificar, y las dos anclas (DGEMM = extremo cómputo, STREAM =
extremo memoria):

```python
posicion = (valor_kernel - valor_dgemm) / (valor_stream - valor_dgemm)
# 0.0 = tan "computo" como DGEMM, 1.0 = tan "memoria" como STREAM
```

Se promedian las tres posiciones disponibles; con `margen=0.15` (declarado en
config, no derivado estadísticamente):

- posición > 0.65 → `memory_bound`
- posición < 0.35 → `compute_bound`
- si no → `ambiguous`

**Ejemplo real** (anclas de `tests/unit/test_classifier_native.py`, medidas
en Fase 0/3 de este nodo):

| | Memory Bound | Cache Bound | DRAM Bound |
|---|---|---|---|
| Ancla DGEMM (pos=0.0) | 8.7% | 6.7% | 2.2% |
| Ancla STREAM (pos=1.0) | 51.9% | 19.2% | 67.7% |
| EP clase C | 6.1% | 11.5% | 0.0% |

```
pos_memory = (6.1-8.7)/(51.9-8.7)  = -0.06
pos_cache  = (11.5-6.7)/(19.2-6.7) =  0.38
pos_dram   = (0.0-2.2)/(67.7-2.2)  = -0.03
promedio = 0.10  →  0.10 < 0.35  →  compute_bound, alta_confianza
```

STREAM evaluado contra sí mismo da posición exacta 1.0 en las tres métricas →
`memory_bound`, alta_confianza (el caso que con la fórmula anterior del
complemento simétrico en 50% salía `ambiguous` — el hallazgo que motivó
D3-v3, ver `context/02_decisiones.md`).

#### Aclaración: ¿no es redundante promediar Memory/Cache/DRAM Bound si Cache y DRAM "son parte de" Memory Bound?

Premisa a corregir primero: en el árbol Top-Down completo (Microarchitecture
Exploration) sí seria así — los hijos de `Memory Bound` se normalizan como %
*del propio* `Memory Bound`, y suman ≤100% del padre. Pero en **HPC
Performance Characterization** (lo único disponible aquí) los tres campos se
imprimen con bases de medición distintas, confirmado literal en la captura
real (`context/04`, addendum): `Memory Bound` es "% of Pipeline Slots";
`Cache Bound` y `DRAM Bound` son "% of Clockticks" — un contador de stalls de
ciclo, no una fracción del mismo total.

Los datos reales lo confirman: si `Cache Bound`/`DRAM Bound` fueran
subconjuntos de `Memory Bound`, nunca podrían superarlo. Pero sí lo superan:

| | Memory Bound | Cache Bound | DRAM Bound |
|---|---|---|---|
| STREAM | 51.9% | 19.2% | **67.7%** (> que el "padre") |
| EP | 6.1% | **11.5%** (> que el "padre") | 0.0% |

Es decir: no son fracciones anidadas de la misma torta en este reporte, son
tres mediciones relacionadas pero calculadas distinto — correlacionadas, no
redundantes en sentido matemático estricto.

Y aun si lo fueran, `_posicion_relativa()` no compara los valores absolutos
entre sí — normaliza cada métrica **independientemente contra su propio par
de anclas** antes de promediar, así que las unidades distintas dejan de
importar. La prueba de que no es la misma señal repetida tres veces está en
el propio EP: `pos_memory=-0.06`, `pos_cache=0.38`, `pos_dram=-0.03` — si
fueran ecos entre sí se moverían juntas; no lo hacen, porque EP tiene un
working set que cabe en caché (stalls de caché moderados) pero casi no toca
DRAM, y eso es información real que solo aporta esa métrica.

**Crítica que sí es válida** (vale la pena tenerla presente si preguntan más
a fondo): las tres métricas son "sabor memoria" — ninguna funciona como
contrapeso de cómputo independiente dentro del promedio, y se les da igual
peso sin ponderar cuál separa mejor en este nodo (empíricamente, `DRAM
Bound` es la que más separa STREAM del resto — 67.7% vs 0.0–2.2% en las
anclas de cómputo — más que `Cache Bound`).

**¿Y comparar Memory Bound contra un "techo de memoria" (como `eje_techos()`
hace con GFLOPS)?** No es posible con el mismo mecanismo, y por una razón de
unidades: `eje_techos()` funciona porque compara GFLOPS contra GFLOPS (mismas
unidades). `Memory Bound` es un % de slots/ciclos con stall, no un ancho de
banda — no se puede dividir contra `stream_bandwidth_mb_s` (MB/s) y obtener
un "% del techo" con sentido físico. Para eso se necesitaría el ancho de
banda real *de cada kernel NPB* (bytes movidos/segundo), lo que requiere
contadores uncore o LIKWID — la misma restricción ya aceptada en D1 y D3
revisada (`context/04`: "no se calcula un % del techo de memoria por kernel
de NPB — requeriría el ancho de banda real de ese kernel, que este nodo no
puede dar sin uncore ni LIKWID"). Es la misma asimetría documentada en
`eje_techos()`: el eje de techos solo compara cómputo, nunca memoria, a
propósito.

### 6.2 `eje_techos()` — compara GFLOPS contra el techo autorreportado

No usa Memory/Cache/DRAM Bound. Compara el `DP GFLOPS` que VTune midió para
el kernel contra el GFLOP/s que el propio binario DGEMM reportó de sí mismo
por software (reloj de pared, sin pasar por VTune, sin depender de uncore —
ver D3 revisada):

```python
pct_del_techo = 100.0 * dp_gflops_kernel / dgemm_gflops_ref
```

Resultado: columna informativa (`roofline_vs_ceilings_pct_compute`), nunca
compite con `classification_vtune_native` (D8).

### 6.3 Consideración pendiente: reemplazar DGEMM por ERT como techo de cómputo

**Nota:** esto reabre parcialmente D1 (`context/02_decisiones.md`), que había
descartado ERT junto con LIKWID para este nodo. Vale la pena separar por qué,
porque el motivo original de D1 no aplica igual de fuerte a este uso
puntual:

- **Lo que D1 descartó:** usar ERT + LIKWID para ubicar la intensidad
  aritmética (AI = FLOP/byte) real de *cada kernel NPB* en un Roofline
  completo. Eso seguía necesitando LIKWID (o uncore) para medir bytes
  movidos por kernel — sin eso, "ERT solo" no permitía completar esa tarea, y
  esa limitación **sigue vigente hoy** (ver D3 revisada / `context/04`: los
  kernels NPB en este nodo no pueden ubicar su posición horizontal completa
  en un Roofline).
- **Lo que se está considerando ahora es más angosto:** usar ERT únicamente
  para obtener un mejor número de **techo de cómputo** (`dgemm_gflops_ref` en
  `eje_techos()`), no para ubicar la AI de los kernels NPB. Para esto, ERT no
  necesita LIKWID ni uncore — es una batería de microkernels que barre varias
  intensidades aritméticas y se autocronometra (mismo mecanismo que ya usan
  STREAM y DGEMM: reloj de pared + tamaño de problema conocido), sin tocar
  contadores de PMU.

**Por qué sería un techo más defendible que DGEMM solo:**
`dgemm_gflops_ref` hoy es "lo que esta implementación de OpenBLAS logró en
esta corrida" — depende de la calidad de esa implementación concreta, no
necesariamente el máximo real que el hardware puede sostener. ERT en cambio
barre el espacio de intensidades y ajusta la asíntota de cómputo (y el punto
de codo/*ridge point*) empíricamente — es una estimación del techo real de la
arquitectura, no del techo de una librería específica. Es la diferencia entre
"lo que un programa logró" y "lo que el hardware permite".

**Qué no cambia si se hace este reemplazo:** `clasificar_nativo()` seguiría
necesitando un `ancla_compute` con `memory_bound_pct`/`cache_bound_pct`/
`dram_bound_pct_or_na` medidos por VTune — eso requiere correr `vtune
-collect hpc-performance` sobre *algún* binario compute-bound (podría seguir
siendo DGEMM, o el microkernel de mayor AI de ERT, con VTune encima). Son dos
decisiones independientes: qué binario da el número de `eje_techos()`
(candidato: ERT) y qué binario da el perfil de VTune para `ancla_compute`
(puede seguir siendo DGEMM sin contradicción). No decidir esto en automático
sin registrarlo — si se adopta, debería quedar como entrada nueva en
`context/02_decisiones.md`, con el mismo criterio de trazabilidad que D3-v3.
