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
  lectura de frecuencia antes/después de la pasada que clasifica (governor, scaling_cur_freq)
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
