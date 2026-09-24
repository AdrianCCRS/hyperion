# Plan de la Fase 4 (evaluación experimental del daemon frente a REF)

Fecha: 2026-09-24 (diseño acordado con el usuario el 2026-09-23). Complementa
`Plan_Detallado_Realineacion_Hyperion.md` §5; el módulo `fase4_evaluacion/` anterior
(`run_evaluation.py`, gobernadores) quedó como andamiaje previo y no se usa en esta matriz.

## Diseño

Ejes: **alcance** (solo CPU, solo GPU, ambos juntos) x **kernels** (vistos `known` frente a
inéditos `unseen`) x **brazo**.

| Alcance | Aplicación (vistos / inéditos) | Brazos | Repeticiones |
|---|---|---|---|
| cpu | dgemm_n2048 + npb_cg / xsbench + rsbench | base, sombra, activo, activo_f0 | 3 |
| gpu | cutlass_dgemm + stream_triad / BabelStream + lavamd escalado | base, sombra, activo | 3 |
| joint | alterna 2 fases de CPU y 2 de GPU, ambos daemons a la vez | base, sombra, activo (piso de CPU 3.2 GHz), activo_nofloor | 5 |

- **base = REF**: gobernador nativo con turbo, sin daemon. Es contra quien se compara todo.
- **sombra**: el daemon decide y no escribe; sombra menos base es la sobrecarga del daemon.
- **activo**: CPU F0 3.2 GHz (turbo apagado) como base y F1 2.9 GHz en memory (experimental);
  GPU 1260 MHz en memory, libera al nativo en compute. **activo_f0**: F0 también en memory.
  **activo_nofloor**: sin piso de CPU con la GPU activa.
- 82 celdas, 2 ciclos por aplicación, 1 s de reposo entre fases (sin él la actividad de GPU de fases
  pegadas no baja del umbral y el daemon no abre fase nueva, job 7637; con él las 4 fases de
  la B de GPU reciben decisión, job 7638). Orden aleatorizado (semilla 20260924), reanudable.
- Métricas por celda: duración, energía de CPU (RAPL, ambos paquetes) y de GPU (NVML) **por
  separado** (el EDP del nodo = (E_cpu + E_gpu) x T se calcula después y se puede desglosar),
  fronteras de fase reales y decisiones de cada daemon (clasificación, abstención).
- Expectativa: la medición de Fase 2 dice "no actuar" para CPU; el activo de CPU es experimental
  y puede quedar igual o peor que REF. Se corre igual, por decisión del usuario.

## Ejecución y análisis

- `scripts/pacca/hyp_fase4_matrix.sbatch` (variables ONLY_SCOPES, ONLY_SETS, ONLY_ARMS, REPS_*,
  CYCLES, GAP_S, OUT). Compuesta conjunta: `fase3_daemon/composite_apps/composite_joint.py`.
- `fase4_evaluacion/analyze_matrix.py RESULTADOS_DIR`: razones frente a REF por (alcance, kernels,
  brazo) y puntuación de clasificación contra las fronteras reales.
- Corrida de humo: job 7638 (gpu y joint inéditos, sombra y activo). Corrida completa: job 7639,
  resultados en `~/hyperion-results/final/fase4_matrix_main/`.
