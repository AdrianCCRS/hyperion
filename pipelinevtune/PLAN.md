# Plan de implementación — Pipeline VTune para el nodo Cartagena (paccaA100)

Antes de leer este plan, asegúrate de haber leído `CLAUDE.md` y todo `context/`.
Este documento asume ese contexto y no lo repite salvo cuando es necesario para
justificar una decisión de implementación puntual.

## Alcance de esta tarea (una sola frase)

Construir un pipeline en Python 3 que ejecute VTune 2023 (Hotspots HW +
HPC Performance Characterization) sobre binarios NPB-OMP en Cartagena, y clasifique
cada kernel según el propio desglose Top-Down que VTune reporta en una sola
corrida — sin calibración externa para ese veredicto. En paralelo, mide STREAM y
DGEMM en el mismo nodo como un **segundo eje de comparación**, independiente del
veredicto de VTune, para que el usuario lo cruce contra su propio modelo externo
de Roofline. Ver `context/02_decisiones.md` D3 (revisada) y D3-native.

---

## Fase 0 — Verificación de premisas

**Ya resuelta en parte:** `vtune-self-checker.sh` ya se corrió en Cartagena y su
resultado está registrado en `context/04_vtune_selfchecker_resultados.md`. Hotspots
HW y HPC Performance Characterization están confirmados disponibles. No repetir el
self-checker completo — lo que falta es la verificación puntual de contenido real
de las métricas, no de disponibilidad del análisis.

```bash
module purge && module load vtune/2023
which vtune && vtune --version
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
sinfo -o "%P %G"                # buscar particion sin GPU si existe
```

### Prueba de humo — ahora enfocada en CONTENIDO de las métricas, no disponibilidad

```bash
vtune -collect hotspots -knob sampling-mode=hw -r /tmp/smoke_hs -- ./bin/ep.C.x
vtune -collect hpc-performance -r /tmp/smoke_hpc -- ./bin/ep.C.x
vtune -report summary -r /tmp/smoke_hpc
```

**Qué revisar específicamente en la salida (esto es lo nuevo, no estaba en la
versión anterior del plan):**

1. Nombres exactos de los campos del desglose Top-Down (`Memory Bound`, y cómo se
   llama la contraparte de cómputo en esta versión 2023 — no asumir el nombre,
   copiarlo literal de la salida real).
2. Si `Memory Bound` de nivel superior sale con un número real (no `NA`) pese a la
   falta de uncore — es la hipótesis de `04_vtune_selfchecker_resultados.md`, hay
   que confirmarla aquí, no darla por sentada.
3. Si `DRAM Bound` (nivel más fino) sale `NA` como se anticipa, o si sorprende y
   sale poblado — cualquiera de los dos resultados es válido, solo hay que
   registrar cuál pasó realmente.
4. Si `DP GFLOPS` sale con un número plausible (no cero, no `NA`) — este es el que
   alimenta el eje de comparación con DGEMM.

**Criterio de paso:** al menos `Memory Bound` (o su equivalente de nivel superior)
y `DP GFLOPS` salen poblados con números reales. Si ambos salen `NA`, el plan
completo cambia — hay que diagnosticar por qué antes de seguir, no forzar el resto
pipeline sobre una base que no funciona. No asumir que "debería funcionar" es
suficiente sin esta prueba.

---

## Fase 1 — Preparar los kernels: NPB + kernels ancla

### 1.1 NPB-OMP

Igual que en el trabajo previo del proyecto: descargar NPB 3.4.4, compilar con GCC
(`-O3 -march=icelake-server -mtune=icelake-server -fopenmp -g`, ajustar el flag de
arquitectura si el compilador disponible no reconoce `icelake-server` — verificar
con `gcc -march=native -Q --help=target | grep march` en el propio nodo en vez de
asumir el nombre del flag).

Kernels: `ep, cg, mg, ft, lu, bt` como mínimo, clases exploratorias `A/B/C`.

### 1.2 Kernels ancla para calibración (decisión D3)

- **STREAM** (memory-bound): compilar con `gcc -O3 -fopenmp -march=icelake-server`,
  sin dependencias adicionales. No necesita LIKWID ni permisos especiales.
- **DGEMM/OpenBLAS** (compute-bound): si `libopenblas-dev` está disponible sin
  `sudo` (módulo del clúster o ya instalado), usarlo. Si no, usar `bt.C.x` o
  `sp.C.x` de NPB como ancla compute-bound sustituta (ver `context/03`, son
  aritmética densa sin el problema de EP).

Verificar funcionalmente cada binario (`VERIFICATION SUCCESSFUL` para NPB; para
STREAM/DGEMM, que corran sin error) antes de usarlos como ancla — un ancla que falla
silenciosamente invalida todos los umbrales calculados a partir de ella.

---

## Fase 2 — `check_vtune.py` (preflight)

Mantener las verificaciones ya especificadas por el usuario, sin recortar ninguna:

1. `vtune` en PATH
2. Versión (`vtune --version`)
3. Que la versión pueda ejecutar los análisis pedidos
4. Disponibilidad de `hotspots`, `hpc-performance`, `sampling-mode=hw`
5. **EBS funcional de verdad** (no solo que la opción exista) — reutilizar la
   prueba de humo de la Fase 0 como parte de este chequeo automatizado
6. Que VTune pueda crear, finalizar y reportar un resultado de prueba
7. Que `bin/` exista
8. Que los binarios tengan permiso de ejecución
9. Que el nombre de archivo permita extraer `kernel` y `clase`
10. Que `OMP_NUM_THREADS`, `OMP_PLACES`, `OMP_PROC_BIND` estén configurados o se
    puedan configurar — **y que además coincidan con la decisión D6** (un socket,
    8 físicos, sin SMT) o que la desviación esté documentada explícitamente
11. Contexto Slurm: `SLURM_JOB_ID`, nodo, CPUs asignadas, tareas, CPUs/tarea
12. Espacio en disco suficiente

Añadido respecto a la especificación original: verificar que existan (o se puedan
generar) los binarios ancla de STREAM y DGEMM/BT-SP — sin ellos, la Fase 4
(calibración de umbrales) no tiene con qué trabajar.

Salida: el mismo resumen que ya se especificó (`VTune disponible`, `Versión`,
`Hotspots HW disponible`, `HPC Performance disponible`, `EBS funcional`, `Slurm
detectado`, `Binarios encontrados`, `Errores bloqueantes`, `Advertencias`), con una
línea añadida: `Kernels ancla disponibles: sí/no (cuáles)`.

Código de salida distinto de cero si hay error bloqueante — sin cambios respecto a
lo pedido.

---

## Fase 3 — `run_vtune_pipeline.py`

CLI y parámetros: igual a lo especificado (`--bin-dir`, `--output-dir`, `--kernels`,
`--threads`, `--repetitions`, `--overwrite`, `--timeout`, `--skip-hotspots`,
`--skip-hpc`), con un parámetro adicional:

```
--anchor-dir       ruta a los binarios ancla (STREAM, DGEMM/BT-SP), separado
                    de --bin-dir para no mezclarlos con los kernels a clasificar
--skip-calibration flag para desarrollo rápido, usa umbrales por defecto
                    documentados como "no calibrados" en el reporte si se usa
```

Descubrimiento automático de kernels: igual a lo especificado
(`<kernel>.<clase>.x`).

### 3.1 Baseline (sin cambios respecto a la especificación)

Cada binario corre primero sin VTune. Validar: código de salida 0, presencia de
`VERIFICATION SUCCESSFUL` en la salida (para binarios NPB — los kernels ancla no
tienen esa cadena, validar por código de salida y ausencia de errores en stderr en
su caso), no exceder el timeout, sin procesos hijos residuales, stdout/stderr
completos guardados.

**Gotcha confirmado en Fase 1 real:** no todos los kernels imprimen la cadena
literal `VERIFICATION SUCCESSFUL` en mayúsculas contiguas — CG y MG sí (línea
extra propia de su `verify.f90`), pero BT/EP/FT/LU solo imprimen la línea común de
`common/print_results.f90`: ` Verification    =               SUCCESSFUL` (con
espacios/`=` en medio, "Verification" no en mayúsculas). El chequeo en
`run_vtune_pipeline.py` debe usar un patrón case-insensitive tipo
`re.search(r"verification.*successful", output, re.IGNORECASE)`, no un
`in` literal con `"VERIFICATION SUCCESSFUL"` — ese patrón literal falla
silenciosamente para 4 de los 6 kernels NPB en este nodo.

Una corrida que no cumpla se marca inválida y no entra a clasificación — sin
excepciones, incluidos los kernels ancla (un ancla inválida no calibra nada).

### 3.2 Estructura de resultados

La misma que ya se especificó, con dos carpetas nuevas al mismo nivel de `MG/`,
`EP/`, etc.:

```
vtune_results/
├── campaign_metadata.json
├── consolidated_results.csv       <- ver Fase 5 para el esquema recortado
├── consolidated_by_kernel.csv
├── vectorization_detail.csv       <- métricas secundarias, ver Fase 5
├── calibration/
│   ├── stream_hpc/
│   ├── dgemm_hpc/                 <- o bt_hpc/ / sp_hpc/ si se usa NPB como ancla
│   └── calibration_summary.json   <- los dos anclas + umbrales derivados
├── classification_summary.md
├── logs/
│   └── pipeline.log
├── MG/class_C/rep_01/...          <- estructura ya especificada, sin cambios
```

No sobrescribir resultados existentes silenciosamente — mismo comportamiento ya
pedido (`--overwrite` explícito o carpeta versionada nueva).

### 3.3 Recolección Hotspots HW y HPC Performance Characterization

Comandos y reportes exactamente como se especificó originalmente — no hay cambios
técnicos aquí, la simplificación del plan no afecta cómo se invoca VTune:

```bash
vtune -collect hotspots -knob sampling-mode=hw -r RESULT_DIR -- BINARIO
vtune -report hotspots -r RESULT_DIR
vtune -report hotspots -format=csv -r RESULT_DIR

vtune -collect hpc-performance -r RESULT_DIR -- BINARIO
vtune -report summary -r RESULT_DIR
vtune -report summary -format=csv -r RESULT_DIR
vtune -report hw-events -format=csv -r RESULT_DIR   # cuando esté disponible
```

Hotspots identifica la función dominante, no decide memory/compute-bound por sí
solo — regla ya establecida en la especificación original, se mantiene igual.

---

## Fase 4 — `vtune_parser.py`, clasificación nativa y eje de techos (dos cosas separadas)

### 4.1 Parser

Tolerante a métricas ausentes (`NA` en vez de excepción), igual a lo especificado.
Construir y probar este módulo primero, antes que el resto del pipeline, porque
todo depende de que el parseo sea confiable. Debe extraer, cuando existan, los
nombres de campo confirmados en la Fase 0 (Top-Down: Memory Bound / contraparte de
cómputo; y `DP GFLOPS`).

### 4.2 Clasificación (decisión D3-v3, 2026-08-07 — reabre D3-native)

**Historial breve (ver `context/02_decisiones.md` para el detalle completo):**
Fase 0 confirmó que `hpc-performance` no imprime una "Core Bound" junto a
`Memory Bound` en este nodo/versión (el Top-Down de 4 categorías vive en
Microarchitecture Exploration, no disponible aquí). La primera solución
(complemento simple `100 - memory_bound_pct`, "D3-native") mantenía el
principio de "una sola corrida, sin calibración", pero producía un resultado
contraintuitivo: STREAM (memory-bound por construcción) clasificaba
`ambiguous` porque la fórmula fija la frontera de decisión en 50% exacto, un
punto sin relación con dónde se separan realmente los datos de este nodo.

**Decisión vigente (D3-v3), tomada explícitamente con el usuario:**
`classification_vtune_native` **sí depende de las anclas STREAM/DGEMM** — esto
reabre D3 (primera versión), que se había descartado antes. Se acepta el
riesgo porque los anclas muestran una separación enorme (`Memory Bound`:
DGEMM=8.7%, EP=6.1% vs STREAM=51.9%; `DRAM Bound`: 2.2%/0.0% vs 67.7%),
suficiente para justificar calibrar contra ellas en vez de usar una frontera
arbitraria sin referencia.

```python
METRICAS_CALIBRACION = ("memory_bound_pct", "cache_bound_pct", "dram_bound_pct_or_na")

def _posicion_relativa(valor, ancla_compute, ancla_memoria):
    # 0.0 = tan "computo" como DGEMM, 1.0 = tan "memoria" como STREAM
    span = ancla_memoria - ancla_compute
    if valor is None or ancla_compute is None or ancla_memoria is None or abs(span) < 1e-9:
        return None
    return (valor - ancla_compute) / span

def clasificar_nativo(reporte_hpc, ancla_compute, ancla_memoria, margen=0.15):
    if reporte_hpc.get("memory_bound_pct") is None:
        return "invalid", "NA", "Memory Bound no disponible en el reporte del kernel"
    if not ancla_compute or not ancla_memoria:
        return "invalid", "NA", "Anclas DGEMM/STREAM no disponibles -- no se puede calibrar"

    posiciones = [p for p in (
        _posicion_relativa(reporte_hpc.get(m), ancla_compute.get(m), ancla_memoria.get(m))
        for m in METRICAS_CALIBRACION
    ) if p is not None]
    if not posiciones:
        return "invalid", "NA", "Ninguna metrica se pudo calibrar contra las anclas"

    posicion = sum(posiciones) / len(posiciones)
    if posicion > 0.5 + margen:
        return "memory_bound", "alta_confianza", f"Posicion relativa={posicion:.2f} -> cerca del ancla de memoria"
    if posicion < 0.5 - margen:
        return "compute_bound", "alta_confianza", f"Posicion relativa={posicion:.2f} -> cerca del ancla de computo"
    return "ambiguous", "zona_intermedia", f"Posicion relativa={posicion:.2f}, dentro de la zona ambigua"
```

Implementación real (con justificación completa por métrica, no la versión
resumida de arriba) en `classifier.py`. `margen` sigue siendo un criterio
declarado en config, no derivado estadísticamente — misma tradición que el
`margen_pp` anterior, ahora aplicado a una escala anclada en vez de a un 50%
sin referencia.

**Consecuencia operativa que hay que respetar en Fase 3/5:** `calibration/`
(STREAM + DGEMM, Fase 3.2) debe existir y ser válida antes de poder llenar
`classification_vtune_native` para cualquier kernel — ya no es opcional para
la clasificación en sí (`--skip-calibration` en `run_vtune_pipeline.py` deja
todos los kernels en `invalid` para esta columna, no solo sin el eje de
techos de la Fase 4.3).

**Nota de nomenclatura:** el nombre `classification_vtune_native` se mantiene
por continuidad con el resto de este documento y el esquema del CSV (Fase 5),
aunque ya no sea "nativo" en el sentido de "sin calibración externa" con el
que se introdujo originalmente. Quien lea `CLAUDE.md` debe saber que la regla
dura que decía "no depende de STREAM ni de DGEMM para calcularse" quedó
corregida — ver la nota en ese archivo.

Esta función sola es la que llena `classification_vtune_native` en el CSV.
Todo lo que sigue en esta fase (4.3, eje de techos) es adicional, no
reemplaza esto, y sigue sin fusionarse con el veredicto nativo (D8).

### 4.3 Eje de techos STREAM/DGEMM (decisión D3 revisada — comparación aparte, no clasificador)

Correr los dos binarios ancla, extrayendo sus propios números autorreportados por
software (no contadores de VTune, ver `context/04_vtune_selfchecker_resultados.md`
sobre por qué):

```python
# ceilings_summary.json (esqueleto)
{
  "stream_bandwidth_mb_s": 45210.3,     // del propio stdout de STREAM (Triad), no de VTune
  "dgemm_gflops": 612.4,                // del propio stdout del driver DGEMM, no de VTune
  "node": "paccaA100",
  "domain": "single_socket_8cores_noSMT",
  "source": "self-timed (wall clock), sin dependencia de contadores uncore",
  "timestamp": "..."
}
```

Con esto, por cada kernel de NPB se agrega una columna informativa (no una clase
categórica que compita con `classification_vtune_native`):

```python
def eje_techos(dp_gflops_kernel, dgemm_gflops_ref):
    if dp_gflops_kernel is None or dgemm_gflops_ref in (None, 0):
        return "NA — DP GFLOPS no disponible"
    pct_del_techo = 100.0 * dp_gflops_kernel / dgemm_gflops_ref
    return f"{pct_del_techo:.1f}% del techo de computo (DGEMM={dgemm_gflops_ref:.1f} GFLOP/s)"
```

**No se calcula un "% del techo de memoria" por kernel de NPB** — requeriría el
ancho de banda real de ese kernel, que este nodo no puede dar sin uncore ni LIKWID
(limitación aceptada en `context/04`). El eje de techos queda entonces asimétrico
a propósito: sí compara cómputo, no compara memoria. Esto debe quedar explícito en
`classification_summary.md`, no disimulado.

**Nota importante:** no usar EP como ancla de DGEMM, por el riesgo de subconteo de
FLOPs documentado en `context/03_kernels_notas.md` (decisión D4).

**Uso previsto de `ceilings_summary.json` y de la columna `roofline_vs_ceilings`:**
son el punto de entrada para que el usuario los compare, fuera de este pipeline,
contra su propio modelo de Roofline externo. Este pipeline no intenta reproducir
ese modelo ni decidir en su nombre.

---

## Fase 5 — Consolidado CSV (esquema recortado)

Este es el punto donde la simplificación del plan cambia algo concreto respecto a
la especificación original: se recortan columnas pensadas para entrenar un modelo,
porque el objetivo es solo clasificar y comparar, no construir un dataset de
features.

### `consolidated_results.csv` — columnas que se quedan

```
campaign_id, timestamp, hostname, slurm_job_id, kernel, class, binary_path,
binary_checksum, repetition, threads, domain_config,
baseline_valid, verification_successful, baseline_elapsed_seconds,
hotspots_valid, hpc_valid,
dominant_function, dominant_function_percentage,
cpi, ipc_estimated, dp_gflops,
memory_bound_pct, dram_bound_pct_or_na, cache_bound_pct,
average_frequency_ghz, physical_core_utilization_pct, numa_remote_access_pct,
classification_vtune_native, classification_confidence, classification_justification,
roofline_vs_ceilings_pct_compute, ceilings_source,
quality_status, error_message
```

Nota sobre el cambio de nombres respecto a la versión anterior de este documento:

- `classification` se renombra a `classification_vtune_native` para dejar
  explícito que sale solo de la corrida individual (D3-native), sin techos.
- `dram_bound_pct_or_na` deja explícito en el propio nombre de columna que este
  campo puede venir vacío por la falta de uncore — evita que alguien lo lea después
  como un cero real.
- `roofline_vs_ceilings_pct_compute` y `ceilings_source` son las columnas nuevas
  del eje de techos (Fase 4.3) — texto informativo, no una clase categórica.

`numa_remote_access_pct` se queda pero como **bandera de calidad**, no como señal de
clasificación — si sale alto, debe reflejarse en `quality_status`, no interpretarse
como evidencia adicional de memory-bound.

### Columnas que se mueven a `vectorization_detail.csv` (no se pierden, solo no
saturan la tabla principal)

```
sp_gflops, vectorization_pct, packed_128_pct, packed_256_pct, packed_512_pct,
fp_uops_pct, non_fp_uops_pct, fp_arith_mem_read_ratio, fp_arith_mem_write_ratio,
instructions_retired, dominant_function_cpu_time
```

Útiles para explicar un caso atípico en la discusión del TG (por ejemplo, si EP
sale mal clasificado, revisar aquí el desglose de vectorización antes de descartar
el resultado), pero no forman parte de la decisión de clasificación en sí.

### `consolidated_by_kernel.csv`

Sin cambios respecto a la especificación original: por kernel y clase, número de
repeticiones válidas, media, mediana, desviación estándar, mínimo, máximo,
coeficiente de variación de las métricas principales.

### Columna opcional para comparación futura contra el orquestador

```
orchestrator_label   -- NA si no hay log del orquestador disponible todavía;
                         se llena via join externo cuando exista, no es parte
                         del pipeline de VTune en sí
```

No construir el mecanismo de *join* ahora si no hay de dónde sacar
`orchestrator_label` — dejar la columna presente y vacía es suficiente para que el
CSV ya tenga el lugar preparado.

---

## Fase 6 — `classifier.py`

Dos funciones separadas, ninguna llama a la otra (D8):

- `clasificar_nativo()` (Fase 4.2) escribe `classification_vtune_native`,
  `classification_confidence`, `classification_justification`. Justificación con
  números reales, no frase genérica:

  ```
  Memory Bound=61.2%, contraparte de computo=22.4%. Diferencia=38.8pp, supera el
  margen de 10pp -> memory_bound, alta_confianza. Consistente en 3/3 repeticiones
  (CV=4.1%). DRAM Bound no disponible en este nodo (sin uncore, ver context/04).
  ```

- `eje_techos()` (Fase 4.3) escribe `roofline_vs_ceilings_pct_compute` y
  `ceilings_source`, de forma independiente — no participa en la clasificación.

Clases de `classification_vtune_native`: `compute_bound`, `memory_bound`,
`ambiguous`, `invalid`.

---

## Fase 7 — `classification_summary.md` y `README.md`

Contenido igual a lo ya especificado por el usuario (16 puntos para el resumen,
lista de contenidos del README), con dos añadidos:

- En "Restricciones conocidas" del resumen: mencionar explícitamente que no hay
  LIKWID/ERT en este nodo y por qué (D1), y que la calibración es autorreferencial
  (D3) — para que quien lea el reporte no asuma que hay un techo externo validado.
- En el README, sección de reserva Slurm: usar el ejemplo real de Cartagena
  (`context/01_nodo_cartagena.md`) con la nota sobre `--gres=gpu:1` y sobre
  `OMP_PLACES=cores` explícito.

---

## Fase 8 — Campaña completa

Igual que se estableció en la conversación de planeación: prototipar y depurar en
sesión interactiva corta, lanzar la campaña completa por `sbatch`, desacoplada de
la sesión de Claude Code (decisión D7).

```bash
#!/usr/bin/env bash
#SBATCH --job-name=hyperion-vtune-cartagena
#SBATCH -p GPU
#SBATCH -w paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      # exclusivo: se pide el nodo completo, el dominio real (8/socket, sin SMT) lo fija OMP_* abajo -- ver D6
#SBATCH --gres=gpu:1            # obligatorio en la particion GPU de este cluster, no se usa la GPU
#SBATCH --exclusive
#SBATCH --hint=nomultithread    # mismo criterio de evitar SMT que usa el resto del proyecto Hyperion (ver AGENTS.md, scripts/felix/)
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

module purge
module load devtools/intel/oneapi/2023   # vtune es un modulo jerarquico, no aparece sin este padre (ver Fase 0)
module load vtune/2023.0.0

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close

python3 run_vtune_pipeline.py \
    --bin-dir NPB3.4-OMP/bin \
    --anchor-dir NPB3.4-OMP/bin \
    --output-dir ./vtune_results \
    --threads 8 \
    --repetitions 3
```

---

## Fase 9 — Tests

Dos suites, con propósitos distintos. Ver `TESTS.md` para el detalle completo; aquí
solo el resumen de qué cubre cada una y cuándo correrlas.

### 9.1 Tests unitarios (`tests/unit/`) — corren en cualquier parte, sin VTune

Contra `vtune_parser.py` y `classifier.py`, usando archivos de reporte capturados
como *fixtures*. **No usar reportes inventados a mano como fixture definitivo** —
la primera vez que corra la Fase 0 en el nodo real, guardar una copia real de
`vtune -report summary` (texto y CSV) de al menos un kernel memory-bound conocido
(ej. STREAM o CG) y uno compute-bound conocido (DGEMM), y usar esas capturas reales
como fixtures. Mientras tanto, las plantillas en `tests/unit/fixtures/` son
ilustrativas del formato esperado, no una garantía de que VTune 2023 en este nodo
imprime exactamente así — el propio parser debe fallar de forma visible (no
silenciosa) si el formato real difiere, para detectarlo temprano.

Casos obligatorios a cubrir:

- Reporte con todas las métricas pobladas → clasificación correcta en los dos ejes.
- Reporte con `DRAM Bound` en `NA` (el caso esperado en este nodo) → el parser no
  falla, y `dram_bound_pct_or_na` queda vacío sin romper el resto de la fila.
- Reporte con `Memory Bound` también ausente → `classification_vtune_native` debe
  salir `invalid`, no un valor adivinado.
- Diferencia entre categorías dentro del margen (`ambiguous`) y fuera de margen en
  ambos sentidos (`memory_bound`, `compute_bound`).
- Caso EP simulado (GFLOPS bajo, Memory Bound intermedio) → confirma que sale
  marcado para revisión manual según D4, no aceptado sin más.

### 9.2 Tests de integración (`tests/integration/`) — requieren el nodo real

Formalizan como scripts ejecutables los criterios de paso que ya estaban descritos
como prosa en cada fase de este plan (Fase 0 a 4). Cada uno termina con código de
salida 0/1, pensados para correrse en orden dentro de una reserva Slurm corta antes
de lanzar la campaña completa por `sbatch` (D7):

```
tests/integration/
├── 00_test_module_and_smoke.sh     -- Fase 0: modulo carga, smoke test, campos reales
├── 01_test_preflight.sh            -- Fase 2: corre check_vtune.py, exige exit 0
├── 02_test_baseline_ep.sh          -- Fase 3.1: baseline de un kernel corto, VERIFICATION SUCCESSFUL
├── 03_test_anchors.sh              -- Fase 1.2/4.3: STREAM y DGEMM compilan y corren
├── 04_test_single_kernel_full.sh   -- Fase 3-6 end-to-end sobre un solo kernel (ep.C.x)
└── run_all.sh                      -- corre los anteriores en orden, para antes de sbatch
```

**No correr `tests/integration/` dentro de la sesión de Claude Code sin una
reserva activa** — son los mismos comandos que requieren el nodo, no simulacros.

### 9.3 Cuándo correr qué

| Momento | Suite |
|---|---|
| Mientras se escribe/edita `vtune_parser.py` o `classifier.py` | `tests/unit/` (rápido, sin nodo) |
| Antes de dar por buena la Fase 0 | `tests/integration/00_test_module_and_smoke.sh` |
| Antes de construir la Fase 3 completa | `01` y `02` |
| Antes de la Fase 4.3 (techos) | `03` |
| Antes de lanzar la campaña completa por `sbatch` | `run_all.sh` completo, sin fallos |

---

## Entregables (mismos archivos pedidos originalmente, sin recortes en el código)

```
check_vtune.py
run_vtune_pipeline.py
vtune_parser.py
classifier.py
config.example.yaml     -- incluye margen de clasificacion nativa (4.2) y
                            configuracion del eje de techos (4.3), separados
README.md
requirements.txt
TESTS.md
tests/unit/...
tests/integration/...
```

Más lo ya generado en este documento como contexto de arranque:
`CLAUDE.md`, `context/00..04`, este mismo `PLAN.md`.

---

## Checklist final

- ☐ Nombres reales de los campos Top-Down confirmados en la salida de VTune 2023
  en este nodo (Fase 0) — no asumidos de la documentación general.
- ☐ Confirmado empíricamente si `Memory Bound` sobrevive sin uncore y si `DRAM
  Bound` sale `NA` como se anticipa — cualquiera de los dos resultados es válido,
  pero debe quedar registrado, no supuesto.
- ☐ Dominio de cores fijado explícitamente (`OMP_PLACES=cores`, D6) y coherente en
  todos los scripts.
- ☐ `classification_vtune_native` sale de una sola corrida (D3-native), sin
  depender de STREAM/DGEMM para calcularse.
- ☐ Kernels ancla (STREAM + DGEMM/BT-SP) compilan y usan sus propios números
  autorreportados por software, no contadores de VTune (D3 revisada).
- ☐ `ceilings_summary.json` generado con números reales de este nodo.
- ☐ `roofline_vs_ceilings_pct_compute` presente como columna informativa aparte,
  nunca fusionada con `classification_vtune_native` (D8).
- ☐ EP marcado con bandera de revisión si sale `memory_bound`/`ambiguous`, y
  excluido como ancla de DGEMM (D4).
- ☐ CSV consolidado con el esquema de la Fase 5 (nombres actualizados), no la
  lista completa original.
- ☐ `orchestrator_label` presente como columna vacía, sin mecanismo de join
  forzado si no hay datos todavía.
- ☐ Tests unitarios (`tests/unit/`) pasan contra al menos una captura real de
  reporte de VTune de este nodo, no solo contra las plantillas ilustrativas.
- ☐ Tests de integración (`tests/integration/run_all.sh`) pasan antes de lanzar la
  campaña completa por `sbatch`.
- ☐ Campaña completa lanzada por `sbatch`, no sostenida en la sesión de Claude Code.
- ☐ `--gres=gpu:1` solo si se confirmó que no existe partición sin GPU.
