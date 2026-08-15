# advisor — validación cruzada con Intel Advisor Roofline

Carpeta de trabajo local para la validación cruzada con **Intel Advisor
2023.0.0** (Roofline cache-aware). Reemplaza el enfoque anterior basado en
VTune (`Vtune/`, `docs/vtune/`, `pipelinevtune/` — todos quedan congelados
como referencia histórica, sin tocar).

Documentos:
- [`docs/advisor/estudio_intel_advisor_roofline.md`](../docs/advisor/estudio_intel_advisor_roofline.md) — qué es Advisor, cómo mide, citas a Intel.
- [`docs/advisor/pipeline_advisor_diseno_y_arquitectura.md`](../docs/advisor/pipeline_advisor_diseno_y_arquitectura.md) — diseño de este pipeline: qué memory/compute roof se usa y por qué, definición de ridge point, cómo se agregan loops a nivel de kernel, arquitectura de módulos. **Léelo antes de tocar el código.**

## Qué hay aquí

Pipeline completo de caracterización:

| Archivo | Propósito |
|---|---|
| `preflight_advisor.py` | Entorno: módulo/versión de Advisor, host/CPU/NUMA/tamaño de línea de caché (leídos en vivo, nunca asumidos — ver hallazgo del modelo de CPU real en el doc de diseño §0), smoke test real de `--collect=survey`. |
| `kernel_registry.py` | Descubre kernels `<kernel>.<clase>.x`, verifica flags de compilación reales (`-O2/-O3`, `-g`) leyendo `make.def`, calcula checksum, hint (no autoritativo) de precisión desde el fuente. |
| `advisor_report_parser.py` | Parsea los reportes CSV **oficiales** de Advisor (`--report=survey`/`--report=roofs --format=csv`) — 145 columnas reales confirmadas, nunca scraping de la GUI. |
| `classify_roofline.py` | Ridge point (`i_ridge_advisor`), selección de loops calientes (cobertura 80% de tiempo propio), clasificación por loop y agregación a nivel de kernel — todas las constantes declaradas y justificadas en el doc de diseño §2. |
| `run_characterization.py` | Orquestador: por kernel×repetición, corre las 4 pasadas (survey+tripcounts, sin simular y con `--enable-cache-simulation`), genera reportes, clasifica, escribe los dos CSV consolidados. |
| `sbatch_advisor_characterization.sh` | Campaña completa vía Slurm — alcance/clase/timeout controlados por variables de entorno, ver sección "Cómo lanzar una campaña" abajo. |
| `run_roofline_unit_test.sh` | Prueba manual de un solo binario (la que ya se corrió sobre `EP.A`, sigue sirviendo para depuración rápida). |

Validado end-to-end como orquestador completo (no solo piezas sueltas) en
`paccaA100`, clases A y C reales. Todo el despliegue vive en
`~/raperezp/` en el servidor — código en `~/raperezp/advisor/`, árbol NPB
en `~/raperezp/NPB3.4-OMP/`, resultados en `~/raperezp/results/`.

## Flujo del pipeline

```
sbatch sbatch_advisor_characterization.sh   (o srun para un piloto corto)
  ↓
module load devtools/intel/oneapi/2023 && module load advisor/2023.0.0
  ↓
preflight_advisor.py
  - módulo/versión de Advisor, host/CPU/NUMA/línea de caché, frecuencia -- todo leído en vivo, nunca asumido
  - smoke test real: --collect=survey sobre un target trivial, no solo "¿está instalado?"
  ↓ (si algo bloqueante: aborta aquí explícito, preflight_result.json, no toca ningún kernel)
kernel_registry.py
  - descubre <kernel>.<clase>.x en --bin-dir (nunca inventa un nombre)
  - verifica flags reales contra make.def (-O2/-O3, -g), checksum sha256
  ↓
por cada kernel × repetición:
  ↓
  project_nosim/:   collect=survey  →  collect=tripcounts -flop            (SIN simular -- solo referencia)
  ↓
  project_cachesim/: collect=survey  →  collect=tripcounts -flop --enable-cache-simulation  (fuente real del veredicto)
  ↓
  frecuencia muestreada EN VIVO (cada 1s) durante la pasada que clasifica -- freq_mhz_during_run = promedio de esas muestras
  ↓
  reportes OFICIALES: --report=survey/roofs --format=csv  (+ --report=roofline para el HTML -- nunca se parsea la GUI)
  ↓
  advisor_report_parser.py: CSV oficial → filas de loop (dict)
  ↓
  classify_roofline.py:
    i_ridge_advisor = P_peak / BW_peak   (roofs de ESTA corrida, no de Hyperion)
    loops calientes = 80% del tiempo propio acumulado
    veredicto por loop (AI vs. i_ridge, margen ±25%)
    agregación por kernel (ponderada por tiempo, margen de dominancia 15pp)
  ↓
  metadata.json: comando exacto, afinidad, checksum, frecuencia, duración real de cada pasada
  ↓
acumula fila → consolidated_characterization.csv (1 fila por kernel×repetición)
             → consolidated_loops.csv (1 fila por loop caliente, con project_dir de origen)
  (se reescriben después de cada kernel -- un timeout/scancel a mitad de camino no pierde lo ya corrido)
  ↓
inspección: CSV consolidado / roofline_report.html en el navegador / snapshot .advixeexpz en la GUI local
```

## Cómo se decide el veredicto (y por qué esos márgenes específicamente)

`classify_roofline.py` decide en **dos capas separadas**, cada una con su
propio margen — y los dos márgenes NO son el mismo tipo de número, por eso
valen cosas distintas (7pp no es lo mismo que 7%). Detalle completo con la
justificación teórica en
`docs/advisor/pipeline_advisor_diseno_y_arquitectura.md` §2.4; acá el
resumen operativo, con ejemplos reales de la campaña de clase B.

### Capa 1 — por loop: `AMBIGUOUS_AI_LOG_MARGIN = 0.25` (±25%, multiplicativo)

Para cada loop caliente se compara su intensidad aritmética (`AI`, FLOP/byte)
contra `i_ridge_advisor` (el ridge point de ESTA corrida, `P_peak/BW_peak`
de los roofs medidos por Advisor). Si `AI` cae entre `0.75×i_ridge` y
`1.25×i_ridge`, ese loop se marca `ambiguous_loop` — ni compute ni memory.

**Por qué multiplicativo y no en puntos porcentuales:** `AI` e `i_ridge` son
valores de intensidad (FLOP/byte), no fracciones de un mismo total — un
gráfico Roofline es log-log por construcción, así que la distancia que
importa es "cuántas veces más/menos", no "cuántos FLOP/byte de diferencia".
Comparar en puntos porcentuales aquí no tendría sentido dimensional.

**Por qué 25% y no otro número:** no es una derivación estadística — es un
margen declarado para cubrir la imprecisión ya documentada de la simulación
de caché de Advisor (extrapola desde un subconjunto de accesos, no simula
el 100% — ver el estudio, sección 4). Es una decisión de diseño explícita,
no una verdad matemática.

**Dato real, para que quede honesto:** en los 29 loops calientes de la
campaña de clase B, **ninguno cayó dentro de este margen** — el más cercano
(`lu`, `buts_` en `buts.f90:49`, AI=5.21 contra ridge=18.79) sigue 72% por
debajo. Este margen todavía no se ha visto "en acción" contra un caso límite
real — solo sabemos que no dispara falsos ambiguos con los datos que
tenemos hasta ahora.

### Capa 2 — por kernel: `KERNEL_DOMINANCE_MARGIN_PP = 15` (15 puntos porcentuales, aditivo)

Una vez clasificado cada loop caliente, se pondera por su `Self Time` y se
suma el tiempo de cada clase (`compute_bound`/`memory_bound`) entre los
loops que sí dieron un veredicto usable. Si la clase líder no supera a la
segunda por al menos 15pp de ese tiempo caliente clasificable, el **kernel**
completo queda `ambiguous` — aunque cada loop individual haya salido con
confianza alta.

**Por qué en puntos porcentuales y no multiplicativo:** acá sí se están
comparando dos fracciones del mismo total (tiempo caliente clasificable,
que suma 100%) — mismo principio que ya se usó para comparar `Memory Bound`
vs. `Core Bound` en el clasificador de VTune (`raperezp/validation_classifier.py`),
donde ambos valores viven en la misma escala absoluta.

**Por qué 15pp:** declarado con el mismo criterio que el margen de 5pp de
VTune (`MEMORY_VS_CORE_MARGIN_PP`), ajustado hacia arriba a propósito porque
acá se está agregando *entre loops distintos* (con su propio ruido de
medición cada uno), no comparando dos métricas del mismo reporte — un margen
más angosto arriesgaría marcar como "dominante" una diferencia que en
realidad es ruido de agregación.

**Dato real:** `bt.B` es el caso real donde este margen disparó —
`compute_bound=53.6%` vs. `memory_bound=46.4%`, diferencia de solo 7.2pp,
muy por debajo del umbral de 15pp. Importante no confundirlo con la Capa 1:
ningún loop de BT individualmente salió `ambiguous_loop` — varios loops
salieron `compute_bound` con confianza y varios `memory_bound` con
confianza, y es la **mezcla entre loops** (BT tiene fases genuinamente
distintas) la que produce la ambigüedad a nivel de kernel, no una medición
dudosa de un loop puntual.

### Cómo podrían cambiar estos números

Ninguno de los dos sale de una fórmula ni de varianza medida — son juicios
declarados, explícitos, pensados para poder cambiarse con evidencia nueva
sin tocar la lógica del clasificador (son constantes nombradas al principio
de `classify_roofline.py`). Candidatos concretos para revisarlos más
adelante:

- Con más repeticiones por kernel (hoy la mayoría de las corridas son de
  una sola repetición) se podría medir la variabilidad real del `AI` entre
  corridas del mismo kernel/clase, y reemplazar el 25% fijo por algo
  derivado de esa variabilidad — más riguroso que un número elegido a mano.
- Si una campaña futura empieza a marcar `ambiguous_loop` con frecuencia
  sobre casos que claramente no lo son (comparando contra el hint de
  literatura de `expected_behavior`), es señal de que el 25% quedó
  demasiado ancho.
- Si `bt` (u otro kernel con fases mixtas conocidas, ver
  `pipelinevtune/context/03_kernels_notas.md` para FT/LU/MG) deja de salir
  `ambiguous` en clase C con el mismo patrón de loops, vale la pena revisar
  si el 15pp sigue siendo el punto de corte correcto o si el comportamiento
  cambia genuinamente con el tamaño del problema.

Cualquier cambio a estos valores debe quedar documentado acá y en el
docstring de `classify_roofline.py` — mismo criterio que el resto del
proyecto para no cambiar un umbral sin dejar rastro de por qué.

## Afinidad y cómo se corrieron las pruebas

### Dominio de cores: 6 físicos, `0-5`, sin SMT (D6)

Mismo dominio que ya usa el orquestador principal de Hyperion en este nodo
(`delegated_cpus` en `orchestrator/schemas/campaign_pacca_ref.yaml`) y que
ya adoptó `pipelinevtune`/la campaña de VTune — a propósito, para que las
clasificaciones de Advisor sean comparables kernel por kernel contra el
resto del proyecto, no un dominio distinto que mida otra cosa.

### `--exclusive` da el nodo completo — la restricción a `0-5` la hace `taskset`, no Slurm

Esto vale la pena dejarlo explícito porque se comprobó en vivo, no se
asumió: `--exclusive` reserva las **32 CPUs lógicas** del nodo para el job
(confirmado con `preflight_advisor.py::check_affinity()`,
2026-08-12 — `cpus_afinidad_real: 0,1,2,...,31`), no solo las 6 que
declaramos con `--cpus-per-task=8`. Slurm **no** restringe por cgroup la
afinidad del proceso a un subconjunto de cores en este clúster.

El confinamiento real a `0-5` lo hace este pipeline, explícitamente, con
`taskset -c 0-5` antepuesto a cada invocación de `advisor` (`pin_prefix` en
`run_characterization.py`) — nunca se depende de que Slurm lo haga por
nosotros. Es el mismo motivo por el que `OMP_PLACES=cores` **solo** no
alcanza (garantiza usar cores físicos, pero no dice *cuáles* 6 de los 32
disponibles) — de ahí que `taskset` y las variables OMP se combinen
siempre:

```
OMP_NUM_THREADS=6
OMP_PLACES=cores
OMP_PROC_BIND=close
taskset -c 0-5  <-- antepuesto al comando real de advisor, no solo al OMP
```

### Verificado en vivo antes de tocar cualquier kernel, no asumido

`preflight_advisor.py::check_affinity()` confirma, con `sched_getaffinity`
real, que el dominio pedido (`0-5`) está disponible en la afinidad del
proceso **antes** de correr nada — si Slurm alguna vez asignara menos de lo
esperado, esto aborta la campaña ahí mismo en vez de dejar que `taskset`
falle en silencio sobre cores que el job nunca tuvo. Se agregó
específicamente al escribir esta sección (no estaba antes) — el preflight
de la campaña de VTune (`raperezp/preflight_uarch.py`) ya lo tenía, este
pipeline no, ahora sí.

`preflight_advisor.py` también confirma que `0-5` cae **dentro de un solo
nodo NUMA** (`numa_node0_cpu: 0-7,16-23`, medido en vivo) — importante
para que `Memory Bound`/tráfico a DRAM signifiquen algo coherente; si el
dominio cruzara dos nodos NUMA, el tráfico de memoria mediría también
latencia de acceso remoto, otra variable mezclada sin avisar.

### ¿Y el SMT (hyperthreading) de esos 6 cores? — bloqueado, por dos vías, ninguna es `--cpus-per-task`

Verificado con `/sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list`
real (no con el patrón "lógico+16" que `lscpu` sugeriría a simple vista —
**ese patrón no aplica en este nodo**, se comprobó y se descartó):

```
cpu0: 0,24   cpu1: 1,25   cpu2: 2,26   cpu3: 3,27   cpu4: 4,28   cpu5: 5,29
```

Los hilos SMT hermanos de los cores `0-5` son `24-29`, no `16-21`. Quedan
fuera de la corrida por dos mecanismos independientes, no por
`--cpus-per-task=8` (ese valor es solo metadata de *scheduling* — bajo
`--exclusive` no cambia el aislamiento real, que ya lo da `--exclusive` por
sí solo):

1. **`taskset -c 0-5` nunca los incluye** — cada uno de los 6 hilos OMP usa
   un core físico completo, sin compartir unidades de ejecución con su
   hermano SMT.
2. **`--exclusive` reserva el nodo completo**, incluidos `24-29` — ningún
   otro usuario puede meter trabajo ahí mientras corre nuestro job.

Nada de esto deshabilita SMT a nivel de hardware/kernel para el nodo en
general — solo garantiza que ni nosotros ni nadie más lo use en el rango
que estamos midiendo.

### Nota operativa: `--ntasks=1` explícito, siempre, en cualquier `srun` manual

Un `srun` de prueba corrido sin `--ntasks=1` (solo con `--cpus-per-task=8`
y `--exclusive`) disparó **4 copias paralelas** del mismo `preflight_advisor.py`
en este clúster (32 CPUs del nodo ÷ 8 por tarea = 4 tareas, inferido por
Slurm a falta de un `--ntasks` explícito) — no es un bug del pipeline, es
un comportamiento real de este `srun` que hay que evitar a propósito. Las
campañas reales (`sbatch_advisor_characterization.sh`, y todos los `srun`
usados para las corridas de datos) ya llevan `--ntasks=1` explícito y no
mostraron este problema — queda anotado acá para cualquier chequeo manual
suelto que se arme más adelante.

## Datos reales de tiempo por clase (ir actualizando esta tabla, no inventar)

Tiempo de la pasada más cara (`tripcounts+flop --enable-cache-simulation`,
la que domina el total) por kernel/clase, medido de verdad, no estimado:

| Kernel | A | B | C |
|---|---|---|---|
| `ep` | rápido (<20s) | sin medir | 6.0 min |
| `mg` | rápido (<1 min, dio `memory_bound` real) | sin medir | sin medir |
| `cg` | sin medir | sin medir | 23.7 min |
| `bt` | sin medir | sin medir | **>30 min — superó el timeout de 1800s, se mató sin dato** |
| `ft` | sin medir | sin medir | sin medir (interrumpido antes de completar) |
| `lu` | sin medir | sin medir | sin medir (no llegó a correr) |

Job 5024 (2026-08-11, clase C completa) confirmó dos cosas: (1) `cg.C`/`ep.C`
dan veredictos reales coherentes con la literatura (`memory_bound`/
`compute_bound`); (2) el timeout por pasada de 1800s se queda corto para
kernels pesados en clase C — `ADVISOR_TIMEOUT` (ver abajo) ahora es
configurable por eso mismo, default subido a 3600s.

## Anclas (STREAM/DGEMM/ERT) — validación del pipeline contra kernels de referencia conocida

A diferencia de los NPB, las anclas no tienen clase A/B/C — se registran con
`klass="anchor"` en `consolidated_characterization.csv`. Se descubren por
nombre real de archivo (`kernel_registry.ANCHOR_NAME_HINTS`, nunca por
carpeta contenedora) en el directorio que se pase con `--anchor-dir`
(`ADVISOR_ANCHOR_DIR` en el sbatch), y corren por el **mismo pipeline
completo** que cualquier kernel NPB (survey+tripcounts, sin simular y con
`--enable-cache-simulation`, mismo `taskset -c 0-5` +
`OMP_NUM_THREADS=6/OMP_PLACES=cores/OMP_PROC_BIND=close`) — nunca se usa el
GB/s o GFLOP/s que ellas mismas reportan por su cuenta, esa cifra es solo
referencia/sanity-check, no la fuente de la clasificación.

| Ancla | Binario usado | Clasificación real | Confianza | AI dominante | `i_ridge` | Coherente con literatura |
|---|---|---|---|---|---|---|
| STREAM | `~/raperezp/NPB3.4-OMP/bin/STREAM/stream_omp` | `memory_bound` | alta (100% del tiempo caliente) | 0.031 | 18.81 | Sí — STREAM mide ancho de banda por diseño. |
| DGEMM | `~/raperezp/NPB3.4-OMP/bin/openBLAS-dgme/bin/dgemm_bench 4096 5` | `compute_bound` | alta (100% del tiempo caliente) | 48.30 | 18.82 | Sí — GEMM es el ejemplo canónico de compute-bound en Roofline. |
| ERT (`ert_probe`, AVX-512) | `~/raperezp/bin/ert_probe` (job 5118, 2026-08-12) | `memory_bound` | alta (100% del tiempo caliente) | 6.23 | 18.81 | **No** — ver nota abajo. |

Los tres casos dan `confidence=alta` con `hot_loops_considered` bajo (1 a 3
loops) porque son binarios pequeños y de una sola fase — coherente con lo
esperado para microbenchmarks, no un indicio de datos pobres.

### Por qué ERT sale `memory_bound` incluso ya compilado con AVX-512 real — resultado real, no forzado, y por qué no contradice el pipeline

`kernels/ert/ert_probe.c` está documentado en su propio encabezado como
"microbenchmark de FLOPs pico (régimen compute-bound)". La primera corrida
(job 5114, binario de `/home/latorresn/hyperion-kernels/bin/ert_probe`, sin
flags de vectorización explícitos) salió `memory_bound` con `AI=0.229`. Antes
de aceptar ese resultado se recompiló el mismo fuente (`kernels/ert/ert_probe.c`
de este repo, sin modificar) con vectorización AVX-512 real y verificada, no
asumida por flags:

```bash
module load gnu12/12.4.0
gcc -O3 -march=native -mprefer-vector-width=512 -fopenmp \
    -o ~/raperezp/bin/ert_probe kernels_src/ert/ert_probe.c
```

Verificado con `objdump -d ~/raperezp/bin/ert_probe`: 41 instrucciones
`vfmadd132{pd,sd}` reales, 30 de ellas sobre registros `%zmm` (512 bits, 8
doubles por vector — AVX-512, no solo AVX2/`%ymm`) — confirma que el binario
sí ejecuta FMA vectorizado de ancho completo en este CPU (Ice Lake, `avx512f`
confirmado en `/proc/cpuinfo`), no una build escalar.

**Con AVX-512 confirmado, el veredicto de la corrida siguiente (job 5118)
sigue siendo `memory_bound`** — pero con evidencia más fina que antes.
Advisor ahora separa dos loops calientes en vez de uno:

| Loop | AI (FLOP/byte) | `i_ridge` | Veredicto |
|---|---|---|---|
| `[loop in main._omp_fn.1]` (la región `#pragma omp parallel` real, el FMA) | **6.23** | 18.81 | `memory_bound` (3× por debajo del ridge, fuera del margen ±25%) |
| `[loop in main]` (el barrido externo de tamaños/trials) | 0.24 | 18.81 | `memory_bound` |

Antes de recompilar, ambos loops estaban fusionados en una sola entidad con
`AI=0.229`; con AVX-512 Advisor logra aislar la región de cómputo real, y su
AI sube ~27× (`0.229 → 6.23`) — pero sigue 3× por debajo del ridge point.
**Esto es la confirmación empírica de que el problema nunca fue
vectorización:** la intensidad aritmética (FLOP/byte) es una propiedad del
*algoritmo* (cuántos bytes se mueven por FLOP ejecutado), no de qué tan
rápido la CPU ejecuta esos FLOPs. `run_once()` hace `FLOPS_PER_ELEM=16`
operaciones por elemento contra ~16 bytes tocados (1 lectura + 1 escritura de
`double`, reutilizados solo dentro de un `trials` acotado — máximo 8 para
los *working sets* más grandes) — ese ratio es intrínseco al diseño del
kernel y ningún flag de compilador lo cambia; AVX-512 hace la misma
proporción de FLOPs/bytes más rápido, no la mejora.

**Conclusión metodológica (revisada):** el veredicto `memory_bound` de ERT es
real y robusto — se sostiene con o sin vectorización explícita, y ahora con
evidencia de que ni siquiera la región de cómputo aislada (una vez separada
del *overhead* de *setup*/*teardown* del barrido) alcanza el ridge point de
este nodo. Si se quisiera un ancla que sí alcance el régimen compute-bound,
habría que rediseñar el kernel para tener mayor reutilización de datos por
byte cargado (p.ej. un *stencil* con más FLOPs por elemento, o un *working
set* fijo con muchos más `trials` para amortizar mejor la carga inicial) —
cambio no implementado, queda como decisión pendiente si se necesita esa
comparación específica. `ert_probe` sigue siendo válido como lo que
realmente mide: pico de GFLOP/s alcanzable en un barrido, una cifra de
*throughput*, no una garantía de intensidad aritmética alta.

Fila anterior (`AI=0.229`, binario sin flags explícitos) reemplazada, no
acumulada — respaldo en
`~/raperezp/results/clase_B/consolidated_characterization.csv.bak_pre_ert_avx`
y `consolidated_loops.csv.bak_pre_ert_avx`. Metadata completa de la corrida
vigente (comando exacto, checksum del binario AVX-512, muestras de
frecuencia durante la pasada que clasifica) en
`~/raperezp/results/clase_B/ert.anchor_rep01/metadata.json`.

## Prueba unitaria ya ejecutada (evidencia real)

`EP.A`, 2026-08-10, en `paccaA100`:

```
GFLOPS:  3.59
GINTOPS: 2.24
```

Detalle completo (incluido el overhead real medido, ~17× entre `survey` y
`tripcounts`) en `docs/advisor/estudio_intel_advisor_roofline.md` sección 9.

## Cómo correr la prueba unitaria sobre otro binario

```bash
# copiar el script al servidor (una sola vez, o cuando cambie)
scp advisor/run_roofline_unit_test.sh latorresn@hpc.unicartagena.edu.co:~/raperezp/

# en el servidor, dentro de un srun corto (no sbatch -- esto es prototipado, D7)
ssh latorresn@hpc.unicartagena.edu.co
ssh pacca
srun -p GPU -w paccaA100 --exclusive --gres=gpu:1 --cpus-per-task=8 --time=00:10:00 \
  bash ~/raperezp/run_roofline_unit_test.sh \
  $HOME/raperezp/NPB3.4-OMP/bin/mg.C.x
```

Salida: `~/raperezp/manual_tests/<binario>_roofline/{roofline_report.html, <binario>.advixeexpz}`.

## Cómo ver el resultado sin GUI (rápido)

`roofline_report.html` es autocontenido — bájalo y ábrelo en cualquier
navegador, no necesita Advisor instalado:

```bash
scp latorresn@hpc.unicartagena.edu.co:~/raperezp/ep.A.x_roofline/roofline_report.html .
```

(el `scp` es directo al gateway porque `$HOME` es NFS compartido con
`pacca` — no hace falta un segundo salto, ver
`docs/retoma/pacca/Auditoria_PaccaA100_Unicartagena.md`).

## Cómo abrirlo en la GUI de Advisor, en tu equipo local

1. **Instala Intel Advisor 2023.x localmente** — misma serie mayor que el
   nodo (`2023.0.0`), para evitar problemas de compatibilidad de formato de
   resultado. Es parte del oneAPI Base/HPC Toolkit gratuito de Intel
   (`https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html`),
   igual que VTune.
2. **Baja el snapshot empaquetado** (ya incluye fuentes y binarios, gracias
   a `--cache-sources --cache-binaries` en el script — se puede abrir sin
   acceso al nodo):
   ```bash
   scp latorresn@hpc.unicartagena.edu.co:~/raperezp/ep.A.x_roofline/ep.A.advixeexpz .
   ```
3. **Ábrelo:**
   ```bash
   advisor-gui ep.A.advixeexpz
   ```
   o desde la GUI ya abierta: `File → Open → Project/Result`, selecciona el
   archivo `.advixeexpz`.
4. **Qué mirar primero:** la pestaña `Survey & Roofline` — el gráfico
   principal, con un punto por loop. El panel lateral izquierdo permite
   activar/desactivar techos (L1/L2/L3/DRAM Bandwidth, Scalar/Vector/FMA
   Peak) — igual que se ve en las capturas oficiales de Intel citadas en el
   estudio.

## Cómo lanzar una campaña

Primero, sincronizar el código (siempre que cambie algo local):

```bash
scp preflight_advisor.py kernel_registry.py advisor_report_parser.py \
    classify_roofline.py run_characterization.py sbatch_advisor_characterization.sh \
    latorresn@hpc.unicartagena.edu.co:~/raperezp/advisor/
```

Luego, dentro del servidor (`ssh latorresn@hpc.unicartagena.edu.co` → `ssh pacca`
→ `cd ~/raperezp/advisor`), **todo se controla con variables de entorno antes
del `sbatch`** — nunca hay que editar el script para cambiar de clase, de
kernels o de timeout:

| Variable | Default | Para qué |
|---|---|---|
| `ADVISOR_KERNELS` | `ep,cg,mg,ft,lu,bt` | Lista separada por comas. Un solo kernel: `ADVISOR_KERNELS=ep`. |
| `ADVISOR_CLASSES` | `A` | `A`, `B`, `C`, o varias juntas (`A,B`) si algún día se quiere comparar en la misma corrida. |
| `ADVISOR_REPETITIONS` | `2` | Bajar a `1` para pilotos/exploración, como hicimos con C. |
| `ADVISOR_TIMEOUT` | `3600` | Segundos máximo por pasada individual (`survey`/`tripcounts`, con y sin simular). Ver la tabla de tiempos reales arriba antes de bajarlo — `bt.C` ya demostró que 1800s no alcanza para kernels pesados en C. |
| `ADVISOR_OUTPUT_DIR` | `~/raperezp/results/job_<JOBID>` | Cambiarlo si se quiere un nombre más descriptivo que el número de job (lo que se usó para la corrida piloto de C: `~/raperezp/results/fase6_clase_C`). |
| `ADVISOR_NPB_ROOT` | `~/raperezp/NPB3.4-OMP` | Solo si el árbol de NPB se vuelve a mover. |
| `ADVISOR_BIN_DIR` | `$ADVISOR_NPB_ROOT/bin` | Cambiarlo para apuntar a otro árbol de binarios (p.ej. `/home/latorresn/hyperion-kernels/bin`). Si se quiere correr **solo** anclas sin tocar ningún kernel NPB, apuntarlo a un directorio vacío (ver ejemplo de ERT abajo). |
| `ADVISOR_ANCHOR_DIR` | vacío (sin anclas) | Directorio con binarios de anclas (STREAM/DGEMM/ERT, ver `kernel_registry.ANCHOR_NAME_HINTS`) — búsqueda recursiva por nombre real de archivo, nunca por carpeta. |

El `--time` del propio `sbatch` (el límite duro de Slurm, distinto de
`ADVISOR_TIMEOUT`) se pasa aparte, con `sbatch --time=HH:MM:SS ...` — el
`10:00:00` que trae el script por defecto está calculado para clase A (ver
el comentario dentro del archivo); para B o C, ajustarlo con el criterio de
la tabla de tiempos reales de arriba, no reusar el de A.

**Clase A (default, la única con cobertura completa 6/6 hoy):**

```bash
sbatch sbatch_advisor_characterization.sh
```

**Clase B, los 6 kernels, una repetición (la toma de datos que dejamos
corriendo el 2026-08-11 — sin dato real de tiempo todavía, por eso
`--time` generoso y `ADVISOR_TIMEOUT` en el default de 3600s en vez de
recortarlo):**

```bash
ADVISOR_CLASSES=B ADVISOR_REPETITIONS=1 \
ADVISOR_OUTPUT_DIR=$HOME/raperezp/results/clase_B \
sbatch --time=10:00:00 sbatch_advisor_characterization.sh
```

**Clase C, solo los kernels que quedaron pendientes del piloto anterior
(`bt`/`ft`/`lu` invalidados o sin correr — `ep`/`cg` ya tienen dato real,
no hace falta repetirlos), con más margen de timeout que la primera vez:**

```bash
ADVISOR_CLASSES=C ADVISOR_KERNELS=bt,ft,lu,mg ADVISOR_REPETITIONS=1 \
ADVISOR_TIMEOUT=5400 \
ADVISOR_OUTPUT_DIR=$HOME/raperezp/results/clase_C_pendientes \
sbatch --time=14:00:00 sbatch_advisor_characterization.sh
```

**Un solo kernel suelto, cualquier clase (piloto rápido antes de comprometer
una campaña completa — el mismo patrón que ya salvó tiempo con `mg`/`cg`
antes de ir a C entero):**

```bash
ADVISOR_KERNELS=lu ADVISOR_CLASSES=B ADVISOR_REPETITIONS=1 \
sbatch --time=02:00:00 sbatch_advisor_characterization.sh
```

**Solo una ancla (p.ej. ERT), sumándola a un consolidado ya existente —
`ADVISOR_BIN_DIR` apunta a un directorio vacío para no reprocesar ningún
kernel NPB de paso. `ert_probe` se compila desde el fuente propio del repo
(`kernels/ert/ert_probe.c`, nunca desde un binario ajeno) directamente en
`~/raperezp/`, con vectorización AVX-512 real (verificada con `objdump`, no
asumida por flags — este nodo es Ice Lake, soporta `avx512f`):**

```bash
scp kernels/ert/ert_probe.c latorresn@hpc.unicartagena.edu.co:~/raperezp/kernels_src/ert/

ssh latorresn@hpc.unicartagena.edu.co
ssh pacca
module load gnu12/12.4.0
mkdir -p ~/raperezp/bin ~/raperezp/anchors_ert ~/raperezp/empty_bin
gcc -O3 -march=native -mprefer-vector-width=512 -fopenmp \
    -o ~/raperezp/bin/ert_probe ~/raperezp/kernels_src/ert/ert_probe.c
objdump -d ~/raperezp/bin/ert_probe | grep -c '%zmm'   # >0 confirma AVX-512 real, no solo el flag
ln -sf ~/raperezp/bin/ert_probe ~/raperezp/anchors_ert/ert_probe

ADVISOR_BIN_DIR=$HOME/raperezp/empty_bin \
ADVISOR_ANCHOR_DIR=$HOME/raperezp/anchors_ert \
ADVISOR_OUTPUT_DIR=$HOME/raperezp/results/clase_B \
ADVISOR_REPETITIONS=1 \
sbatch --time=01:00:00 sbatch_advisor_characterization.sh
```

Job 5118 (2026-08-12) es la corrida vigente con este binario. `anchors_ert/`
se mantiene como un directorio aislado (no apuntar `--anchor-dir` directo a
un árbol de binarios ajeno de terceros) a propósito: cualquier árbol más
amplio que también tenga binarios con nombre `stream*`/`dgemm*` haría que
`kernel_registry.discover_anchors()` los recoja también, re-corriendo esas
dos anclas con binarios de otra procedencia sin que se pida.

### Revisar progreso y resultados

```bash
squeue -u $USER                                    # estado del job
tail -f ~/raperezp/logs/slurm_<JOBID>.out           # progreso general
tail -f ~/raperezp/logs/slurm_<JOBID>.err           # el log real de avance (INFO/WARNING de Python va aquí, no a .out)
```

Resultados en el `ADVISOR_OUTPUT_DIR` que se haya usado:
`consolidated_characterization.csv` (un veredicto por kernel×repetición) y
`consolidated_loops.csv` (un renglón por loop caliente, con `project_dir`
de origen para trazabilidad completa hasta el resultado real de Advisor).
Se reescriben después de cada kernel — si el job se corta a mitad de
camino (timeout, `scancel`), lo ya corrido no se pierde.

### Si hay que cortar un job propio

```bash
scancel <JOBID>
```

Solo sobre jobs propios, nunca sobre trabajos de otros usuarios — ver la
regla dura al principio de `sbatch_advisor_characterization.sh`.
