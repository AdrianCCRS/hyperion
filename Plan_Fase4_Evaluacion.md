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
| joint | alterna 2 fases de CPU y 2 de GPU, ambos daemons a la vez | base, sombra, activo (piso de CPU 3.2 GHz), activo_nofloor | 3 (5 en la corrida A original, job 7639) |

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

## Resultados del escenario A y corridas de diagnóstico (2026-09-24)

Escenario A = la matriz de arriba, tal cual, con muestreo de 1 ms (job 7639), más las repeticiones a 10 ms
(jobs 7642 y 7643). Se reporta completo, sin filtrar, sea cual sea el resultado.

- Clasificación: CPU 96 % (vistos) y 99.9 % (inéditos); GPU 100 % en lo decidido, con abstenciones.
- EDP frente a REF: activo de CPU 1.7 a 2.0; conjunta 1.3 a 1.6; GPU 1.00. Casi todo el exceso ya aparece en
  `sombra` (+15 a 20 % de tiempo, +60 % de energía de CPU): es costo de tener el daemon de CPU, no de actuar.
- El muestreo (1 ms frente a 10 ms) no explica ese costo. Diagnóstico en curso (job 7645): consumidor sin fijar
  (`--consumer-cpu` se leía y no se aplicaba) e hilos de ONNX Runtime sin acotar (`--pin-consumer`, `--ort-threads`).

## Escenarios adicionales (declarados antes de correrlos)

Regla anti-selección: los escenarios se fijan aquí antes de correr, los resultados de A siguen siendo la referencia y
todos se reportan, salgan como salgan. Repeticiones: 3 por celda en todos los alcances (con 3 contra 3, el p exacto
mínimo de la prueba de permutaciones es 0.05, no 0.0079 como con 5 contra 5; las conclusiones se apoyan en la razón y su
intervalo, no solo en el p).

| Escenario | Qué cambia | Por qué |
|---|---|---|
| B: GPU dominada por memoria | mezcla con mayoría de fases memory_bound de GPU (stream_triad, BabelStream) y fases largas | el techo medido en Fase 2 para GPU memory es ~9 % de EDP (F1 = 1260 MHz, IC95 3.0 a 13.3 %); el efecto total escala con la fracción de tiempo en memoria |
| C: daemon de CPU corregido | repite A (cpu y conjunta) con el daemon sin el costo fijo, si 7645 lo confirma | separa el efecto de actuar del costo del daemon |
| D: aplicación real de terceros | una aplicación HPC no escrita por el autor, con fases naturales (candidatas: LAMMPS, GROMACS, CloverLeaf); pendiente de elección | validez externa: fases y duraciones que no diseñamos nosotros |

Nota sobre niveles de GPU: bajar más que F1 no ayuda. Con los datos de Fase 2, F3 en adelante empeora el EDP en fases
de memoria (F8 llega a -76 %), así que el escenario B mantiene F1 y no explora relojes más bajos.
