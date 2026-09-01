# Proyecto Hyperion — pipeline VTune para el nodo Cartagena

Este archivo es el punto de entrada. Léelo primero, después lee todo `context/` antes de tocar código.

**Ubicación en el repo:** este directorio (`pipelinevtune/`, nivel superior del repo
`hyperion`, rama `hpc-startup-diagnostic`) es la fuente de verdad local. Cuando el
texto de este documento o de `context/`/`PLAN.md` menciona `vtune_selfcheck/`, se
refiere al directorio de trabajo remoto en el nodo
(`/home/latorresn/vtune_selfcheck/` en `paccaA100`) — es una convención de nombres
para que el mapeo local↔remoto sea obvio al desplegar, no el nombre de esta carpeta
en el repo. Ver `AGENTS.md` (raíz del repo) para las convenciones generales del
proyecto Hyperion — este pipeline es una fuente de validación independiente para el
orquestador que se construye ahí, no parte del orquestador en sí (ver `context/00`).

## Qué es esto en una frase

Un pipeline en Python 3 que ejecuta Intel VTune Profiler 2023 sobre binarios de NAS
Parallel Benchmarks (NPB-OMP) en el nodo `paccaA100` (alias interno: **Cartagena**), y
clasifica cada kernel como `compute_bound`, `memory_bound` o `ambiguous` usando
únicamente las métricas que VTune reporta. No hay LIKWID ni ERT en este nodo — ver
`context/02_decisiones.md` para por qué.

## Antes de escribir código, lee en este orden

1. `context/00_overview_hyperion.md` — qué es Hyperion y para qué sirve este pipeline
2. `context/01_nodo_cartagena.md` — specs del nodo, qué SÍ funciona aquí y qué hay que decidir
3. `context/02_decisiones.md` — decisiones ya tomadas, no las reabras sin razón nueva
4. `context/03_kernels_notas.md` — advertencias específicas por kernel (EP e IS, sobre todo)
5. `context/04_vtune_selfchecker_resultados.md` — evidencia real del nodo: qué análisis
   de VTune están confirmados, y la restricción de uncore (afecta DRAM Bound)
6. `PLAN.md` — el plan de implementación fase por fase
7. `TESTS.md` — estrategia de pruebas; `tests/unit/` corre y pasa (19/19) contra
   `vtune_parser.py` y `classifier.py` definitivos (ya construidos, Fase 4), usando
   capturas reales del nodo como fixtures (`tests/unit/fixtures/real_*`)

## Reglas duras que no se negocian sin decírselo al usuario primero

- **No asumas LIKWID ni ERT disponibles.** Si en algún momento parece que hacen falta,
  para y pregunta — no los instales ni los actives por tu cuenta.
- **No pidas ni asumas privilegios de administrador.** Nada de `sudo`, nada de tocar
  `perf_event_paranoid`, nada de instalar drivers.

- **Fija el dominio de cores explícitamente: 6 cores físicos, `0-5`, sin SMT**
  (`taskset -c 0-5` + `OMP_PLACES=cores`), salvo decisión contraria documentada.
  Alineado a propósito con `delegated_cpus` del orquestador principal en este
  nodo (`orchestrator/schemas/campaign_pacca_ref.yaml`) para que las
  clasificaciones de VTune sean comparables kernel-por-kernel contra las
  suyas — no eran 8 cores/todo el socket como se documentó originalmente. Ver
  `context/01_nodo_cartagena.md` y `context/02_decisiones.md` D6.
- **ACTUALIZADO 2026-08-07 (D3-v3, reabre D3-native):** `classification_vtune_native`
  **sí depende de STREAM/DGEMM** — se calibra la posición del kernel entre ambas
  anclas usando Memory Bound + Cache Bound + DRAM Bound del mismo reporte de
  `hpc-performance`. Esto reemplaza la regla anterior ("no depende de STREAM ni de
  DGEMM"), que quedó desactualizada tras un hallazgo real (STREAM salía `ambiguous`
  con la fórmula del complemento simétrico en 50%). Ver `PLAN.md` §4.2 y
  `context/02_decisiones.md` D3-v3 para el detalle y el porqué. `roofline_vs_ceilings_pct_compute`
  (Fase 4.3) sigue siendo un eje **separado**, nunca fusionado con el veredicto
  nativo en una sola columna (D8) — eso no cambió, solo cambió que el veredicto
  nativo ahora sí usa las anclas internamente.
- **El nodo no tiene eventos uncore.** `DRAM Bound` puede salir `NA` — es esperado,
  no un bug del parser. Verificar en la Fase 0 qué sobrevive realmente antes de
  asumir nombres de campo. Ver `context/04`.
- **No corras la campaña completa dentro de la sesión interactiva de Claude Code.**
  Prototipa y depura con `srun --pty` de corta duración; la campaña larga va por
  `sbatch`, desacoplada de esta sesión.

## Estado actual del proyecto (resumen de una línea cada uno)

- Nodo Westmere ("Félix" en conversaciones anteriores) queda fuera de alcance para
  este trabajo. No reutilizar sus umbrales ni su ridge point — otra arquitectura.
- VTune 2023 con Hardware Event-Based Sampling debería funcionar en Cartagena porque
  es Ice Lake-SP (VTune soporta Ice Lake y posteriores para servidor). Confirmar con
  la prueba de humo antes de construir nada (`PLAN.md` Fase 0).
- La validación cruzada de este nodo es VTune solo. Si más adelante se obtienen
  permisos para LIKWID, se añade como fuente adicional — no está bloqueado por ahora,
  simplemente no es parte de este trabajo.
