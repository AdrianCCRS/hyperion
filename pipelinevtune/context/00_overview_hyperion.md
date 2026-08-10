# 00 — Qué es Hyperion y qué rol juega este pipeline

## El proyecto en una frase

Hyperion es un trabajo de grado que construye un **orquestador** capaz de clasificar
cargas de trabajo científicas (kernels de NAS Parallel Benchmarks, NPB-OMP) como
`compute_bound`, `memory_bound` o `ambiguous`, usando contadores de hardware leídos
en tiempo de ejecución.

## Por qué existe este pipeline de VTune

Un orquestador que clasifica solo necesita apoyarse en *algo* que valide si sus
decisiones son correctas. Ese "algo" es una fuente de referencia independiente,
medida con una herramienta distinta a la que usa el orquestador en producción.

En el nodo anterior (Westmere-EP, apodado "Félix" en conversaciones previas del
proyecto) esa fuente de referencia se construyó combinando LIKWID (mide el punto real
del kernel) y ERT (mide los techos de la máquina). Ese trabajo **no se continúa en
este nodo** — ver `02_decisiones.md`.

En **Cartagena** (`paccaA100`, Xeon Gold 5315Y, Ice Lake-SP), la fuente de referencia
es **VTune Profiler 2023 por sí solo**. La tarea de este pipeline es exactamente:

> Según los resultados que da VTune (Hotspots HW + HPC Performance Characterization),
> clasificar cada kernel de NPB como compute_bound, memory_bound o ambiguous.

Nada más que eso. No es un pipeline de entrenamiento de modelo, no genera un dataset
con decenas de columnas — genera una clasificación por kernel/clase/repetición, con
su justificación, lista para compararse después contra lo que produzca el
orquestador en producción (esa comparación puede o no hacerse en este mismo momento
del proyecto; el pipeline no depende de que exista ya un log del orquestador).

## Los kernels involucrados

`ep, cg, mg, ft, lu, bt` (y potencialmente `sp`, `is` si se retoman más adelante).
Nombre de archivo: `<kernel>.<clase>.x`, ej. `mg.C.x`, `ep.D.x`.

## Advertencia heredada de trabajo previo (relevante aquí también)

En el nodo anterior se detectó que **EP e IS rompen cualquier clasificador basado en
FLOP/byte**, por razones distintas:

- **IS** no tiene FLOPs (ordena enteros). Cualquier métrica de VTune basada en
  GFLOPS para IS será cero o cercana a cero — eso no significa "extremadamente
  memory-bound", significa que la pregunta no aplica a ese kernel.
- **EP** sí es compute-bound por naturaleza, pero su trabajo dominante son raíces
  cuadradas, logaritmos y un generador congruencial — operaciones que algunos
  contadores de "FLOPs" estándar no capturan completo (típicamente las
  suma/multiplicación SSE/AVX sí se cuentan, pero la unidad de división/raíz
  cuadrada puede quedar fuera del contador que arma el "DP GFLOPS" reportado). Esto
  puede hacer que EP salga con GFLOPS artificialmente bajos y termine mal
  clasificado como memory-bound o ambiguous sin serlo.

Ver `03_kernels_notas.md` para el detalle y qué hacer al respecto en este pipeline.
