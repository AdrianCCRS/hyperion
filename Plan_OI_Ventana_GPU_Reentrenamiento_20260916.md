# Plan: OI por ventana para GPU, rebalanceo de familias y reentrenamiento

Fecha: 2026-09-16. Escrito para que lo ejecute otro agente (Codex) sin
contexto previo de la conversación que lo originó. Cada fase cita
archivos y decisiones reales verificadas contra el repo en esa fecha —
**verificar de nuevo contra el estado actual antes de asumir que algo
sigue igual**, puede haber cambiado.

## Estado de implementación (2026-09-16)

- Implementada la infraestructura genérica: preload CUPTI Activity en el
  proceso CUDA hijo, trazas por corrida, unión con modelos analíticos y
  construcción de filas NVML por ventanas de 120 ms.
- Implementados y validados contra trazas CUPTI de ELF directos los modelos
  de `rodinia_gaussian`, Dual-AXPY (las dos escalas finales), Stencil 2D
  (las dos escalas finales), Dual-SpMV (las dos escalas finales),
  Dual-Cholesky, Dual-FFT y `rodinia_kmeans`; no usan la OI histórica de
  `ncu` como fallback.
- Los wrappers Bash de varios Dual-* no pueden recibir el preload de forma
  fiable. El catálogo ahora admite, y verifica por checksum, un ELF CUDA
  directo exclusivo para Activity; esto preserva el wrapper histórico para
  la ejecución normal.
- Implementado un gate *fail-closed*: si `gpu.activity_trace.enabled` está
  activo, `run_campaign()` no toca el nodo mientras falte el modelo o los
  parámetros analíticos de algún kernel GPU declarado.
- Implementados pliegues de validación con familias mixtas, dejando LOFO
  disponible como comparación histórica; el entrenamiento nuevo rechaza
  mezclar filas históricas por corrida con las nuevas por ventana.
- La traza ya incorpora H2D/D2H y el pipeline agrupa eventos internos antes
  de asignar una única fórmula por operación. SpMV, Cholesky y FFT validaron
  respectivamente grupos de 7, 5 y 5 eventos (incluidas copias), y Dual-*
  declara su región warm con timestamps `CLOCK_MONOTONIC` para excluir setup,
  frío y verificación. Pendiente antes de relanzar: completar y revisar las
  fórmulas para las demás familias elegibles; por diseño el gate bloquea la
  campaña hasta entonces. `rodinia_lavamd` y `rodinia_dwt2d` siguen fuera del
  conjunto equilibrado acordado.

## 0. Contexto (resumen para quien ejecuta esto sin haber visto la discusión)

El director pidió reemplazar `ncu` como método de medición de intensidad
operacional (OI) de GPU por algo "online", análogo a como ya funciona
CPU. Investigación real (no hipótesis) encontró:

- **CPU** ya entrena por ventana/intervalo de uncore (~10ms), no por
  corrida agregada (`fase2_clasificador/training/train_phase.py:85-148`).
- **GPU** hoy entrena con **una fila = una corrida completa**, agregada
  por mediana (`fase2_clasificador/training/train_phase_gpu.py`), y la OI
  de cada corrida es una **constante** medida una sola vez offline con
  `ncu` e inyectada igual en todas sus ventanas
  (`fase1_telemetria/postprocess.py:906`, comentario ARC-80).
- Esa constante es **correcta** para kernels homogéneos (mismo trabajo
  aritmético en toda la corrida: `miniBUDE`, `lavamd`, `dwt2d`,
  `gpu_cutlass_simt_dgemm_n4096`, la mayoría de RAJAPerf) — no hay nada
  que arreglar ahí, y así se confirma con la composición de clase 100%
  homogénea de 13 de 16 kernels reportada en
  `docs/libro/secciones/03_resultados.tex:208`.
- Es **incorrecta** para kernels donde el trabajo por lanzamiento cambia
  a lo largo de la corrida. El caso confirmado y prioritario es
  `rodinia_gaussian`: ~8190 lanzamientos (`Fan1`/`Fan2`, n-1 pasos de
  eliminación gaussiana), cuya OI declarada (0.2814 FLOP/byte) se midió
  **solo sobre los primeros 20 lanzamientos** de esos 8190
  (`fase1_telemetria/catalog/catalog.yaml:543-578`, comentario ARC-110) —
  el trabajo por paso decrece porque la submatriz de trabajo se encoge, y
  eso nunca se ha medido a lo largo de toda la corrida.
- `rodinia_gaussian` es además la única falla del clasificador GPU sin
  explicación en el libro: familia memory_bound más alejada del ridge
  ($\log_2=-5.73$) y aun así falla 15/18 corridas en la dirección
  contraria (`docs/libro/secciones/03_resultados.tex:353`). Ni cercanía
  al ridge ni error de OI lo explican (el margen de OI es >10x). El
  mecanismo más probable, sin confirmar todavía, es exactamente esta OI
  no-constante intra-corrida.
- Aparte de esto, el catálogo GPU tiene una **escasez estructural de
  familias compute_bound** (4 de 11 en el libro) que produce un techo
  matemático de F1 macro ~0.5 bajo LOFO (Leave-One-Familia-Out): cuando
  la familia que se deja fuera para probar es 100% de una clase, esa
  clase queda con soporte cero en ese pliegue y F1=0 automático para
  ella, sin importar qué tan bueno sea el modelo
  (`docs/libro/secciones/03_resultados.tex:359`, "8 de los 11 pliegues
  quedan con soporte cero para una de las dos clases"). Este es el mismo
  mecanismo ya documentado como negativo para CPU en la memoria de
  proyecto `intra-kernel-phase-hunt-negative` (techo F1~0.5).
- Hay 3 kernels compute_bound ya medidos y con campaña propia cerrada,
  agregados DESPUÉS del snapshot del libro, pendientes de fusionar al
  dataset de entrenamiento oficial:
  - `rodinia_kmeans_gpu` — OI=68.9495, commit `de920e4`.
  - `miniBUDE` — OI=25582.5597, commit `1f9be2b`.
  - `gpu_cutlass_simt_conv2d` — OI=438.06, commit `54e6b10`.
  Con estos tres, el catálogo pasa de 4:7 a **7:7** familias
  compute/memory_bound (14 familias totales, excluyendo `lavamd`/`dwt2d`
  que siguen sin datos utilizables — ver §5).

## 1. Alcance explícito

**SÍ incluye:**
- Fusionar kmeans/miniBUDE/conv2d al dataset de entrenamiento GPU.
- Rediseñar el esquema de validación de LOFO puro a pliegues mixtos
  (agrupando varias familias por pliegue, balanceando clase).
- Construir un mecanismo de OI por ventana/lanzamiento para **todo el
  catálogo GPU** (registro analítico fail-closed + CUPTI Activity API para
  ubicar cada lanzamiento en el tiempo real). `rodinia_gaussian` es el
  primer caso de validación, no una excepción de implementación.
- Reentrenar el clasificador GPU completo con el dataset resultante.
- Relanzar/reprocesar lo necesario para tener datos limpios de punta a
  punta.

**NO incluye (fuera de alcance de este plan, decisión explícita):**
- `rodinia_lavamd` y `rodinia_dwt2d`: ambos siguen sin datos utilizables
  (`lavamd`: intento de extender `boxes1d` 70→100 dio NEGATIVO, el
  binario de Rodinia solo acepta `-boxes1d`, sin bucle de iteraciones,
  verificado contra la fuente real en GitHub — arreglarlo exige parchear
  el binario, hay precedente en
  `old/scripts/pacca/patches/rodinia/{kmeans_cuda,needle_gpu}_patched.cu`,
  pero es trabajo nuevo no hecho. `dwt2d`: decisión abierta documentada en
  Anexo G de `docs/general/PLAN_MAESTRO_FASE2.md` — **ese archivo solo
  existe en la rama `origin/fase-02`, no en `main`**, y su ítem G.6.3
  pide un criterio de energía total integrada nunca implementado).
  Agregar solo uno de los dos desbalancearía el 7:7 ya logrado (`lavamd`
  es compute_bound, `dwt2d` es memory_bound). Decisión: dejarlos fuera de
  esta ronda, documentados como trabajo futuro.
- `cfd`/euler3d: no existe en el catálogo actual (nunca se agregó,
  quedó pausado por decisión explícita del usuario por mezcla de 4
  kernels/iteración sin filtrar en `ncu_convergence.py`). Reabrir esto es
  una decisión de alcance nueva, no implícita en este plan.
- Polybench 2MM/3MM: hay una decisión previa explícita de NO invertir más
  en esta familia (`Nota_Candidatos_GPU_Compute_Bound_20260914.md`,
  sección 4, punto 6 / recomendación inicial).
- El libro (`docs/libro/`) va sistemáticamente por detrás del trabajo
  real — **no usarlo como fuente de "qué ya está hecho"**, solo como
  snapshot histórico. Confirmar todo contra código/commits/datos reales.

## 2. Fase 0 — Fusión de datos existentes (barata, sin campaña nueva)

Objetivo: pasar de 11 a 14 familias sin generar ni un solo dato nuevo,
solo uniendo lo que ya existe.

1. Localizar los 3 CSV de campaña cerrada (`gpu_kmeans_extra`,
   `gpu_minibude_extra` — nombre exacto a confirmar, buscar en
   `~/hyperion-results/final/campaigns/` en pacca — y `gpu_conv2d_extra`).
2. Confirmar el bloqueo `CAM-09` sigue vigente (fingerprint de protocolo
   cambia si se agregan kernel_refs al mismo `output_dir`) antes de
   intentar un resume normal — si ya se resolvió de otra forma, ajustar
   este paso.
3. Fusionar a nivel de **`training_gpu_phases.csv`** (Fase 2, no Fase 1):
   escribir/adaptar un script que concatene las filas de las 3 campañas
   extra con las 540 filas oficiales, verificando que el esquema de
   columnas (features de mediana: `gpu_util_pct_median`,
   `gpu_mem_util_pct_median`, `gpu_power_mw_median`,
   `gpu_sm_clock_mhz_median`, más `phase_label_train`,
   `kernel_ref`/familia) sea idéntico entre campañas.
4. Verificar el nuevo conteo: 14 familias, 7 compute_bound / 7
   memory_bound. Si algún número no cuadra, DETENERSE y reportar antes de
   seguir — no ajustar a mano para que cuadre.
5. Correr `fase2_clasificador/training/train_phase_gpu.py` tal cual
   (con LOFO viejo) sobre el dataset fusionado, solo para tener un
   número de referencia ANTES del cambio de validación — sirve para
   aislar cuánto mejora por rebalanceo de familias vs. por el cambio de
   esquema de validación de la Fase 1.

## 3. Fase 1 — Esquema de validación: pliegues mixtos en vez de LOFO puro

Objetivo: eliminar el arrastre mecánico de F1=0 por clase ausente.

1. En `fase2_clasificador/training/train_phase_gpu.py`, localizar dónde
   se arma el split `leave_one_familia_out` (buscar
   `leave_one_familia_out` / `LeaveOneGroupOut` en el archivo y en
   `train_phase.py` si comparten utilidad).
2. Reemplazar por un agrupamiento de familias en pliegues mixtos, cada
   uno con al menos una familia compute_bound y una memory_bound. Con 14
   familias 7:7, un diseño razonable es ~5-7 pliegues de 2 familias cada
   uno (1 compute + 1 memory), documentando el criterio de emparejamiento
   (ideal: aleatorio con semilla fija, o balanceado también por número de
   ventanas/corridas por familia para no dejar un pliegue con muy pocas
   filas). Usar `sklearn.model_selection.StratifiedGroupKFold` si encaja
   con la estructura de datos (grupo=familia, estrato=clase); si no
   encaja limpio, documentar el agrupamiento manual explícitamente en
   comentario de código, no dejarlo implícito.
3. **No borrar el código de LOFO puro** — dejarlo disponible detrás de un
   flag, para poder reportar ambos números en la tesis (el viejo esquema
   con su techo de 0.5, el nuevo con pliegues mixtos) como parte de la
   justificación metodológica del cambio.
4. Correr el entrenamiento con el dataset de la Fase 0 y este nuevo
   esquema. Registrar F1 macro/por clase, y cuántos pliegues quedaron con
   soporte cero para alguna clase (debería ser 0, o muy pocos).

## 4. Fase 2 — Mecanismo genérico de OI por ventana (todo el catálogo GPU)

**Corrección de alcance (2026-09-16, después de escribir la primera
versión de este plan): el mecanismo NO es exclusivo de `rodinia_gaussian`.
Se construye como un componente genérico del pipeline, aplicable a los
19+ kernels del catálogo GPU, no como código especial para un solo
kernel.** La diferencia entre kernels no está en si se les aplica el
mecanismo, sino en qué arroja al aplicarlo:

- Para kernels **homogéneos** (miniBUDE, lavamd, `dwt2d`, GEMM/cutlass, la
  mayoría de RAJAPerf — el trabajo por lanzamiento no cambia a lo largo
  de la corrida), la fórmula analítica debe dar una OI(t) prácticamente
  constante, lo cual simplemente **confirma** la OI ya validada con
  `ncu` — no cambia nada en el dataset, pero tampoco cuesta nada extra
  una vez que el mecanismo existe.
- Para kernels **heterogéneos** (`rodinia_gaussian` es el único
  confirmado hasta ahora, ver más abajo; posiblemente otros si al
  auditar el catálogo completo aparece un patrón similar — ver 4.0), la
  OI(t) sí cambia el dataset de forma material.

Diseñar esto como un **registro/tabla de fórmulas por kernel** (una
función pura por `kernel_ref` que devuelve `FLOP(t)`/`Bytes(t)` dado el
índice de lanzamiento y los parámetros de tamaño de esa corrida). La
campaña final debe operar en modo **fail-closed**: todo kernel de dataset
GPU necesita un modelo analítico registrado y trazable. La OI histórica
medida con `ncu` puede conservarse únicamente para validación comparativa;
no puede ser el fallback de la nueva etiqueta porque eso no reemplazaría
realmente a `ncu`.

La unidad ML tampoco será una lectura NVML aislada ni necesariamente un
lanzamiento CUDA aislado. La resolución efectiva observada de NVML es de
aproximadamente 105--120 ms y muchas lecturas a 5 ms repiten el mismo valor.
Se construirán **ventanas temporales de al menos 120 ms** (parámetro
calibrable, registrado en el contrato). Para cada ventana, CUPTI aporta los
intervalos exactos de kernels que la solapan y el registro analítico aporta
sus FLOPs/bytes. La verdad de la ventana se calcula como
`sum(FLOP_solapados) / sum(Bytes_solapados)`; un lanzamiento que cruza el
límite se prorratea por fracción temporal, dejando explícito el supuesto de
tasa uniforme dentro de ese lanzamiento. Las features NVML se agregan sobre
las muestras de esa misma ventana. Ventanas sin actividad CUDA suficiente
quedan fuera del entrenamiento, nunca reciben una etiqueta inventada.

### 4.0 Auditoría previa: qué otros kernels necesitan fórmula propia

Antes de escribir solo la fórmula de `gaussian`, revisar el catálogo GPU
completo (`fase1_telemetria/catalog/catalog.yaml`, entradas `device: gpu,
role: dataset`) buscando el mismo patrón que delató a `gaussian`: OI
declarada como medida sobre una MUESTRA PARCIAL de lanzamientos (buscar
comentarios tipo "primeros N lanzamientos de un total de M", "no es un
promedio de la corrida completa") combinado con un algoritmo cuyo trabajo
por paso varíe analíticamente de forma conocida (reducción de tamaño de
problema, número de elementos restantes, etc. — no basta con "tiene
muchos lanzamientos", como ya se descartó para `kmeans`, cuyo trabajo por
lanzamiento es idéntico cada vez). Documentar cada candidato encontrado
con su propia fórmula antes de pasar a producción, no limitarse a
`gaussian` si aparece algo más.

### 4.1 Fórmula analítica para `rodinia_gaussian` (primer caso, sin overhead, sin CUPTI todavía)

1. Confirmar contra la fuente real de Rodinia CUDA `gaussianElim`
   (`kernel_gpu_cuda.cu`/`gaussianElim.cu` del repo upstream, mismo
   patrón de verificación que se usó para `lavaMD` — no asumir, leer la
   fuente) la relación exacta entre el índice de paso `t` (0..N-2, N=4096
   en este catálogo) y las dimensiones de la submatriz de trabajo que
   procesan `Fan1` y `Fan2` en ese paso.
2. Derivar `FLOP(t)` y `Bytes(t)` en función de `t` y `N` para ambos
   kernels (`Fan1`: O(N-t) trabajo; `Fan2`: O((N-t)^2) trabajo — confirmar
   exponentes reales contra el código, no asumir).
3. Escribir esto como una función pura (sin dependencias de hardware) en
   algún módulo de `fase1_telemetria/` (ubicar el lugar natural, p.ej.
   junto a `gpu_phases.py`), con tests unitarios que verifiquen: (a) que
   sumando `FLOP(t)`/`Bytes(t)` sobre todo `t` da un total consistente
   con el orden de magnitud ya medido con `ncu` en los primeros 20
   lanzamientos, (b) que la OI(t) decrece/varía de forma monótona o al
   menos no constante conforme avanza `t`.

### 4.2 CUPTI Activity API (ubicar cada lanzamiento en el tiempo real)

**Importante, no confundir**: CUPTI Activity API da timestamps de inicio
y fin de cada `cudaLaunchKernel`, nombre del kernel y metadatos de grid —
**nunca** cuenta FLOPs ni bytes reales. Esa parte la da la fórmula
analítica de 4.1. Aquí solo se usa para saber CUÁNDO ocurrió el
lanzamiento `t`, para poder alinear la OI(t) analítica con las ventanas
NVML/telemetría reales de la corrida.

1. CUPTI no existe hoy en el repo (`common/telemetry/` solo tiene
   `perf_reader.cpp`/`uncore_reader.cpp` para CPU y `nvml_reader.cpp` para
   GPU). No agregarlo como lector de `collector.cpp`: el colector vive en
   el proceso padre y el binario CUDA medido se ejecuta tras `fork+exec` en
   otro proceso. Implementar una biblioteca compartida de trazado cargada
   **solo en el proceso objetivo** (inyección desde el launcher justo antes
   de `execv`, no `LD_PRELOAD` global sobre el launcher), usando
   `cuptiActivityRegisterCallbacks`/`CUpti_ActivityKernel` (API de
   Activity, no Profiling) para capturar, por cada lanzamiento: nombre de
   kernel, timestamp de inicio, timestamp de fin. Registrar el callback de
   timestamps CUPTI con `CLOCK_MONOTONIC`, el mismo dominio temporal del
   harness, para evitar una alineación post-hoc entre relojes distintos.
2. Verificar disponibilidad de CUPTI en el entorno de compilación de
   pacca (headers/lib de CUDA Toolkit, `libcupti.so`) antes de escribir
   código contra una API que no se pueda enlazar — hacer esta
   verificación ANTES de escribir el lector completo (preflight barato:
   compilar un `.cu` mínimo que solo linkee CUPTI y corra
   `cuptiGetVersion()`).
3. Wirear el nuevo lector al binario de telemetría (`collector.cpp`) de
   forma **genérica** (cualquier kernel `device: gpu` puede activarlo),
   pero **habilitarlo primero solo para `rodinia_gaussian`** como
   validación inicial — no activarlo en campaña completa hasta confirmar
   que funciona y no añade overhead perceptible. Una vez validado, es el
   mismo lector para todo el catálogo, no algo que reescribir por kernel.
4. Con los timestamps reales + la fórmula correspondiente del registro,
   construir ventanas temporales de ≥120 ms y calcular su OI por suma de
   FLOPs/bytes solapados. Comparar esa OI contra el ridge del nivel y
   precisión de la corrida. Para kernels homogéneos, verificar que las
   etiquetas resultantes coinciden con la clasificación histórica; para
   kernels sin fórmula registrada, detener el postproceso/campaña final con
   un error trazable.

### 4.3 Validación antes de confiar en esto para entrenar

1. Correr una corrida de diagnóstico (no campaña completa) de
   `rodinia_gaussian` con el nuevo mecanismo, y confirmar EMPÍRICAMENTE
   que la OI(t) calculada varía de forma no trivial a lo largo de la
   corrida (graficar OI(t) vs. tiempo). Si resulta que la OI(t) es
   esencialmente constante en la práctica (posible: los primeros pasos
   dominan el tiempo total, los últimos son rapidísimos y aportan pocas
   ventanas), documentar ese resultado honestamente — puede que el
   esfuerzo no cambie el dataset de forma material, y eso también es un
   resultado válido para la tesis (dejar constancia, no forzar que
   "funcione").
2. Solo si la variación es real y material, incorporar `rodinia_gaussian`
   reprocesado al dataset de entrenamiento de la Fase 0/1.
3. Repetir 4.0-4.3 para cualquier otro candidato que haya salido de la
   auditoría de 4.0. El mecanismo (registro de fórmulas + lector CUPTI +
   wiring en `postprocess.py`) ya queda construido de forma genérica
   desde el primer caso — lo que se repite por kernel adicional es solo
   derivar su fórmula y validar su variación real, no reescribir el
   pipeline.
4. Una vez validados todos los candidatos reales, activar el lector
   CUPTI para el catálogo GPU completo (no solo los kernels con fórmula
   no-constante), para que quede como el mecanismo de medición vigente
   de aquí en adelante, reemplazando a `ncu` de forma consistente en todo
   el catálogo — no solo en los kernels donde hizo diferencia.

## 5. Fase 3 — Reentrenamiento final y relanzamiento de campaña

1. Con el dataset fusionado (Fase 0) + esquema de validación nuevo (Fase
   1) + todo el catálogo GPU reprocesado con cobertura analítica completa
   (Fase 2), correr el
   entrenamiento completo de `train_phase_gpu.py` (los 7 modelos que ya
   compara `docs/libro/secciones/03_resultados.tex` tabla
   `tab:gpu-clasificador-tres-etapas`, o el conjunto vigente).
2. Si algo requiere **datos nuevos de hardware** (no solo reprocesar lo
   que ya existe), lanzar en pacca siguiendo las reglas de `AGENTS.md`:
   - Nunca `--time` en `sbatch`/`srun`.
   - Todo `--job-name` empieza con `hyp_`.
   - Usar el patrón de job contenedor (`hyp_final_holder.sbatch`) si la
     corrida es larga, nunca jobs encadenados con `--dependency`.
   - Turbo SIEMPRE desactivado para corridas de frecuencia fija
     (envolver con el script de turbo-disable existente).
   - Verificar `squeue`/estado antes de cualquier `--overlap` — no asumir
     que el nodo está libre.
   - Preflight de ~20 min (leer código, correr algo corto) antes de
     encolar cualquier campaña de horas.
3. Documentar en el libro (`docs/libro/secciones/03_resultados.tex` y/o
   una sección nueva de metodología) el cambio de método como una
   decisión metodológica final: mencionar que `ncu` offline-constante fue
   el camino inicial, que se abandonó por overhead y por no capturar
   variación intra-corrida, y que el método final es OI analítica +
   CUPTI Activity API para los kernels heterogéneos, con OI constante
   (ya validada) para los homogéneos. **No narrar el proceso de
   descubrimiento/depuración** — solo la metodología final y el
   resultado final, según la convención ya establecida del libro (ni
   texto, ni tablas, ni figuras "antes/después" del propio proceso de
   arreglo).

## 6. Checklist final antes de dar esto por cerrado

- [ ] 14 familias confirmadas en el dataset fusionado (7:7).
- [ ] Ningún pliegue de validación con soporte cero para una clase (o,
      si alguno queda, documentado por qué es inevitable con este
      catálogo).
- [ ] Todos los kernels GPU entrenables tienen modelo analítico registrado;
      cero fallback a la OI histórica de `ncu`.
- [ ] Catálogo GPU reprocesado con OI por ventana; diagnóstico de variación
      real por kernel documentado (positivo o negativo).
- [ ] F1 macro reportado en al menos 3 configuraciones para poder
      justificar el cambio en la tesis: (a) LOFO puro + dataset viejo
      (11 familias, ya lo tiene el libro), (b) LOFO puro + dataset
      fusionado (14 familias, aísla el efecto de rebalanceo), (c)
      pliegues mixtos + dataset fusionado (+ gaussian reprocesado si
      aplica) — el número final a reportar como resultado.
- [ ] `lavamd`/`dwt2d`/`cfd`/2MM-3MM explícitamente fuera de esta ronda,
      con nota de trabajo futuro (no silenciados, no perseguidos).
- [ ] Todo cambio de catálogo/entrenamiento commiteado con mensajes
      claros, sin `--no-verify`, sin amend de commits ya empujados.
