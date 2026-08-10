# 02 — Decisiones ya tomadas (no reabrir sin razón nueva)

Registro corto, en orden. Si algo aquí parece contradecir lo que estás a punto de
hacer, para y confirma con el usuario en vez de asumir que el registro está
desactualizado.

## D1 — Se abandona LIKWID + ERT para este nodo

Motivo: no hay permisos todavía (LIKWID necesita acceso a `perf_event` o a un
daemon con privilegios; ERT en sí no necesita permisos especiales, pero sin LIKWID
para medir el punto real del kernel, ERT solo sin contraparte de medición pierde
utilidad para este pipeline específico). Se retoma como posible extensión futura,
no como bloqueo actual.

**Consecuencia práctica:** no hay un "ridge point" externo calculado con
microkernels sintéticos. Ver D3 para cómo se reemplaza.

## D2 — La validación cruzada de este nodo es VTune, solo VTune

La tarea completa del pipeline es: correr VTune sobre cada kernel/clase/repetición,
leer sus métricas, y clasificar según esas métricas. No se mezcla con ninguna otra
fuente de medición en este nodo. Si más adelante hay que comparar contra el
orquestador de Hyperion en producción, eso es un *join* posterior sobre el CSV que
este pipeline ya produce — no cambia nada de lo que el pipeline hace internamente.

## D3 (revisada) — Doble eje de clasificación: VTune nativo + techos STREAM/DGEMM,
sin que uno reemplace al otro

Primera versión de esta decisión (abandonada): usar STREAM/DGEMM para calibrar los
umbrales internos del clasificador de VTune. Se revirtió porque eso equivale a
construir un segundo orquestador encima de VTune, que es exactamente lo que este
pipeline no debe ser — la tarea es "clasificar según lo que VTune mide", punto.

**Versión vigente:** VTune ya trae su propio veredicto, sin calibración externa
(ver D3-native abajo). STREAM y DGEMM se reinstauran, pero como un **segundo eje de
comparación independiente**, no como insumo del clasificador de VTune. Su función:

1. Dar un techo de cómputo (DGEMM) y un techo de memoria (STREAM) medidos en este
   nodo, para que el usuario los compare directamente contra su propio modelo de
   Roofline externo — ese es el motivo explícito por el que se reinstauran.
2. Servir de referencia de sanidad: si `DP GFLOPS` de un kernel NPB está muy cerca
   del techo de DGEMM, es una señal adicional de que el veredicto nativo
   "compute_bound" de VTune es consistente con algo medible por fuera.

**No calcular el ancho de banda de STREAM/DGEMM a partir de contadores de VTune.**
El nodo no tiene eventos uncore (ver `04_vtune_selfchecker_resultados.md`) — usar
en su lugar el ancho de banda y GFLOP/s que STREAM y el driver DGEMM ya calculan y
reportan por software (reloj de pared + tamaño de problema conocido). Ninguno de
los dos depende de uncore.

**Limitación que se acepta, no se oculta:** para los kernels de NPB (a diferencia
de STREAM/DGEMM) no hay forma de obtener su AI real (FLOP/byte) en este nodo sin
uncore y sin LIKWID. El eje "techos" para NPB se reduce a comparar `DP GFLOPS`
contra el techo de cómputo — no se puede ubicar su posición horizontal completa en
un Roofline. Esto queda como columna separada y explícitamente rotulada, nunca
mezclado con el veredicto nativo de VTune como si fuera el mismo cálculo.

### D3-native — El veredicto principal sale de una sola corrida, sin calibración

**SUPERADA por D3-v3 (más abajo en este documento), 2026-08-07.** Se deja el
texto original completo por trazabilidad — no borrar la historia de por qué se
llegó hasta acá — pero `clasificar_nativo()` ya NO funciona como se describe en
esta sección: ahora sí depende de las anclas STREAM/DGEMM. Ir a D3-v3 para la
versión vigente.

`hpc-performance` ya reporta el desglose Top-Down (TMAM) de esa corrida específica:
Retiring / Front-End Bound / Bad Speculation / Back-End Bound, y dentro de este
último, Memory Bound vs. la parte de cómputo. El clasificador compara estas
categorías **entre sí, dentro del mismo reporte** — no contra una corrida externa.

```
dominante = la categoria de mayor porcentaje en el reporte de ESE kernel
si dominante es "memory bound" y supera a la siguiente por > margen -> memory_bound
si dominante es la parte de computo y supera a "memory bound" por > margen -> compute_bound
si no hay diferencia clara -> ambiguous
```

`margen` es un criterio de desempate declarado en config, no un umbral calibrado
externamente — sigue cumpliendo lo que pedía la especificación original
("umbrales no codificados de forma rígida"), sin necesitar STREAM/DGEMM para
funcionar. STREAM/DGEMM entran después, como columnas adicionales de comparación,
no como parte de esta fórmula.

**Verificar antes de confiar en esto:** confirmar en la primera corrida real qué
nombres de campo usa exactamente `vtune -report summary` en la versión 2023 para
estas categorías — no asumir los nombres de memoria (ver `PLAN.md` Fase 0/2).

**Verificado en Fase 0 real (2026-08-07) — ajuste a la fórmula, no a la decisión:**
el reporte real no trae una "Core Bound" aislada junto a `Memory Bound` (esa
descomposición de 4 categorías vive en Microarchitecture Exploration, no
disponible en este nodo). Se decidió con el usuario usar el complemento simple
`compute_equiv_pct = 100 - memory_bound_pct` dentro de Pipeline Slots, documentado
explícitamente como "resto del pipeline" (no Core Bound puro). Sigue siendo una
sola corrida, sin STREAM/DGEMM. Detalle completo en `PLAN.md` §4.2 y
`context/04_vtune_selfchecker_resultados.md` (addendum).

**Hallazgo de Fase 4 (2026-08-07) — STREAM salía `ambiguous` con el margen
default:** con `margen_pp=10.0`, la captura real de STREAM (`Memory
Bound=51.9%`, complemento=48.1%, diferencia=3.8pp) caía dentro del margen y
clasificaba `ambiguous`, no `memory_bound`, pese a ser el ancla memory-bound
por construcción. Primera reacción (descartada, ver D3-v3 abajo): mantener el
margen tal cual y aceptar el caso como documentado. Al revisar los números
reales de las tres anclas junto con el usuario, quedó claro que el problema no
era solo el valor del margen sino la fórmula misma: comparar `Memory Bound`
contra su propio complemento fija la frontera de decisión en 50% exacto, un
punto sin ninguna relación con dónde se separan realmente los datos de este
nodo (`Memory Bound`: DGEMM=8.7%, EP=6.1% vs STREAM=51.9%; `DRAM Bound`:
2.2%/0.0% vs 67.7% — brechas enormes, muy lejos de cualquier frontera fija en
50%).

## D3-v3 (2026-08-07) — Se reabre D3: `clasificar_nativo()` sí se calibra con
STREAM/DGEMM, combinando Memory Bound + Cache Bound + DRAM Bound

**Esto reabre explícitamente D3 (primera versión), que se había abandonado
por la razón documentada arriba ("construir un segundo orquestador encima de
VTune"). Se reabre con el usuario, con evidencia nueva (el caso STREAM) y de
forma deliberada — no es un descuido ni una reinterpretación silenciosa.**

Decisión tomada: `clasificar_nativo(reporte_hpc, ancla_compute, ancla_memoria,
margen=0.15)` ya no evalúa un kernel de forma aislada. Para cada una de las
tres métricas del mismo reporte de `hpc-performance` (`memory_bound_pct`,
`cache_bound_pct`, `dram_bound_pct_or_na`), se calcula la posición relativa del
kernel entre el ancla de cómputo (DGEMM) y el ancla de memoria (STREAM):

```
posicion = (valor_kernel - valor_dgemm) / (valor_stream - valor_dgemm)
```

`0.0` = tan "cómputo" como DGEMM, `1.0` = tan "memoria" como STREAM (puede
salir fuera de `[0,1]` si el kernel es más extremo que cualquiera de las dos
anclas). Se promedian las tres posiciones disponibles; `margen` (declarado, no
derivado — igual tradición que el `margen_pp` anterior, pero ahora aplicado a
una escala anclada empíricamente, no a un 50% sin referencia) define el ancho
de la zona ambigua alrededor de 0.5.

**Consecuencia que hay que asumir, no esconder:** `classification_vtune_native`
ya **no** sale de "una sola corrida, sin calibración externa" — depende de que
`calibration/` (STREAM + DGEMM) exista y sea válida. Si `--skip-calibration` se
usa en `run_vtune_pipeline.py`, `clasificar_nativo()` no puede evaluar ningún
kernel (`invalid`, "anclas no disponibles"). El nombre de la columna/función se
mantiene por continuidad con el resto del proyecto y el esquema del CSV, pero
ya no es "nativo" en el sentido original con el que se documentó en D3-native
— quien lea `CLAUDE.md`/`PLAN.md` debe saber que esa frase quedó desactualizada
y se corrigió aquí.

**Por qué se acepta el riesgo metodológico:** los tres anclas muestran una
separación tan grande (varios órdenes de magnitud en `DRAM Bound`) que
calibrar contra ellas no es un ajuste fino cuestionable — es la diferencia
entre poner la frontera en un punto arbitrario (50%) o en un punto informado
por cómo se comporta esta arquitectura específica. Sigue siendo honesto en el
sentido de que no se inventan umbrales por kernel ni por clase — las mismas
dos anclas (una corrida de DGEMM, una de STREAM) valen para toda la campaña.

Ver `classifier.py` (docstring del módulo) y
`tests/unit/test_classifier_native.py` para la implementación y los casos de
prueba con números reales.

**Por qué no EP como ancla compute-bound (para el eje de techos):** ver D4, EP
tiene un problema de conteo de FLOPs que lo hace poco confiable como referencia.

## D4 — EP se mantiene en el pipeline, pero con una bandera de riesgo, no se descarta

Se decidió no eliminar EP ni IS del proyecto por el problema de conteo de FLOPs
(ver `03_kernels_notas.md`), sino:

- No usar EP como kernel ancla de calibración (D3) — su GFLOPS medido puede estar
  artificialmente bajo.
- Marcar explícitamente en el reporte cuando la clasificación de EP resulte
  `memory_bound` o `ambiguous`, como un caso que requiere revisión manual antes de
  aceptarse, en vez de un resultado más del lote.
- IS, si se retoma en este nodo, se trata igual: fuera del cálculo de intensidad
  basado en FLOPs, con una nota explícita en vez de forzarlo a una clase.


## D6 — Dominio de ejecución: 6 cores físicos (0-5), sin SMT (default, ACTUALIZADA 2026-08-07)

Ver justificación completa en `01_nodo_cartagena.md`. Cualquier corrida con otro
dominio (dos sockets, SMT activo, otro rango de cores) debe declararse
explícitamente como tal en los metadatos de la campaña, no mezclarse
silenciosamente con las corridas de dominio estándar.

**Cambio respecto al default original (8 cores, todo el socket 0):** al
integrar `pipelinevtune/` al repo principal de Hyperion se encontró que el
orquestador (la pieza que este pipeline valida de forma independiente, ver
`context/00`) ya corre campañas reales contra este mismo nodo con
`delegated_cpus=0-5` + `collector_cpu=6` + `consumer_cpu=7`
(`orchestrator/schemas/campaign_pacca_ref.yaml`, ver también
`docs/retoma/pacca/Auditoria_PaccaA100_Unicartagena.md` en la raíz del repo).
Se decidió con el usuario alinear el dominio de VTune a exactamente los mismos
6 cores (0-5) que usa el orquestador para el kernel — no los 8 originales —
para que ambos midan el mismo kernel bajo las mismas condiciones y sus
veredictos sean comparables kernel-por-kernel. `run_vtune_pipeline.py` fija
esto con `taskset -c 0-5` (no solo `OMP_PLACES=cores`, que por sí solo no
garantiza cuáles cores específicos se usan) — ver `--core-range` en su CLI.

## D8 — Doble salida de clasificación, nunca fusionadas en una sola columna

`consolidated_results.csv` lleva dos columnas de veredicto, no una:

- `classification_vtune_native` — sale de D3-native, una sola corrida, sin techos.
- `roofline_vs_ceilings_signal` — sale de comparar `DP GFLOPS` del kernel contra el
  techo de DGEMM (columna informativa, no una tercera clase categórica obligatoria;
  puede quedar como texto libre tipo "cerca del techo de cómputo (92%)" o
  "muy por debajo del techo de cómputo (18%)").

Nunca promediar o combinar ambas en un solo veredicto dentro de este pipeline — la
comparación entre las dos, y contra el modelo de Roofline externo del usuario, se
hace después, fuera de este código.

## D7 — Claude Code no sostiene la campaña larga en su propia sesión

Prototipado y depuración: sesión interactiva corta (`srun --pty`, minutos). Campaña
completa (todos los kernels × clases × repeticiones × dos análisis de VTune, puede
tomar horas): `sbatch`, desacoplada de la sesión de edición.
