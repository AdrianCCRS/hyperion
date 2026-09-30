# Plan Fase 4, escenario E-C: fase de cómputo sensible al reloj (declarado el 2026-09-30, antes de medir)

## Por qué

La revisión del director (C1) mostró que en E-A fijar 1260 MHz durante toda la aplicación da el mismo EDP del nodo
que el agente (agente/F1 fijo 0.998, IC95 0.990 a 1.006). E-A no puede distinguir las dos estrategias porque su
única fase de cómputo (DGEMM de CUTLASS) ya corre por debajo de 1260 MHz por el límite de potencia de la A100. El
valor que el diseño atribuye a la clasificación por fase, no bajar el reloj en fases de cómputo cuyo tiempo depende
de él, no se puso a prueba en ninguna aplicación. E-C lo pone a prueba.

## Criterio de selección (fijado antes de medir)

"El kernel `compute_bound` con mayor pérdida de EDP de GPU en F1 en la Fase 2, que no esté limitado por potencia en la
referencia". Resultado: `rodinia_heartwall` (log-razón F1 = +0.193, unos +21% de EDP; reloj de 1410 MHz en la
referencia; tiempo +26% en F1). Los demás de cómputo no pierden de forma apreciable con F1 (cholesky +2.7%, conv2d y
kmeans mejoran, DGEMM limitado por potencia). Es un escenario favorable a la clasificación, espejo de E-A, que fue
favorable a la política; se reporta como tal.

## Aplicación

`composite_gpu_memdom.py --set sensitive`: `dual_stencil_gpu_N36864` (memoria, ~57 s), `rodinia_heartwall`
(cómputo, repetido `HEARTWALL_REPEATS` veces con `HEARTWALL_FRAMES` cuadros, objetivo ~30 s en total),
`dual_spmv_gpu_N200000000` (memoria, ~46 s). Un ciclo, 1 s de hueco entre fases. Las fases de memoria son las
mismas de E-A.

## Protocolo (job `hyp_fase4_EC`, `scripts/pacca/hyp_fase4_EC.sbatch`)

1. Estado de CPU igual al de la base del libro (job 7696): gobernador `performance`, turbo desactivado, 3.2 GHz
   (min = max) en los 12 procesadores delegados; se restaura el estado original al terminar.
2. Sondeo de heartwall (sin agente): 2000 y 6000 cuadros con reloj nativo y 6000 con F1, muestreando utilización y
   reloj cada 250 ms. Compuerta automática (si falla, el job se pausa sin medir):
   - las cinco corridas terminan con código 0;
   - tiempo con F1 / tiempo nativo (6000 cuadros) >= 1.05;
   - mediana del reloj bajo carga con reloj nativo >= 1350 MHz (no limitado por potencia);
   - con F1, todas las muestras bajo carga entre 1230 y 1290 MHz;
   - al menos 90% de las muestras entre la primera y la última con actividad tienen utilización > 5%.
   Los cuadros y repeticiones se calculan del sondeo para ~30 s de heartwall (<= 20000 cuadros, <= 4 repeticiones).
3. Pre-vuelo: una celda por brazo; compuerta técnica (código de aplicación y de daemon, restauración de estado, reloj
   de F1 fijo dentro de tolerancia en todas las fases). No se filtra por la decisión del clasificador.
4. Confirmatorio: brazos `base`, `activo_gpu` y `fijo_gpu_f1`; **6 bloques** aleatorizados (`balanced_blocks`,
   semilla 20261001). Con 6 bloques la prueba de signos exacta bilateral puede llegar a p = 0.031.

## Análisis declarado

- Principal: prueba de signos exacta bilateral por bloques sobre el EDP del nodo, agente frente a F1 fijo.
- Secundario: la misma prueba agente frente a base y F1 fijo frente a base; media geométrica de las razones por bloque
  con IC95 por t pareada sobre el logaritmo; razones de duración, energía de CPU y energía de GPU.
- Clasificación: decisión del agente en cada fase (clase, confianza, instante), contra la verdad declarada.

## Predicción

Si el agente clasifica heartwall como `compute_bound` o se abstiene (ambas liberan el reloj), F1 fijo alarga la fase
de heartwall y el EDP del nodo del agente queda por debajo del de F1 fijo en los 6 bloques. Si el agente la clasifica
como `memory_bound`, aplica F1 y ambos brazos coinciden: la clasificación no aporta en este caso. Cualquiera de los
dos resultados se reporta.

## Réplica bajo powersave (declarada el 2026-09-30, antes de medir)

Mismo diseño (base, activo_gpu, fijo_gpu_f1; 6 bloques; semilla 20261002) con los 12 procesadores delegados en
`powersave`, EPP `default`, turbo desactivado y 0.8 a 3.2 GHz, el estado de la serie powersave del job 7789. Se
reutiliza la calibración de heartwall del job 7815 (10097 cuadros, 1 repetición) y se omite el sondeo. Mismo análisis y
misma predicción; cada brazo se compara con la REF de su propio gobernador.

## Resultado de la réplica (job 7818, 2026-09-30)

18/18 celdas válidas; F1 fijo en 1260 MHz en todas las muestras bajo carga. Agente/F1 fijo 0.978 de EDP del nodo (IC95
0.970 a 0.987), 6/6 bloques, p = 0.031; agente/base 0.979 (6/6, p = 0.031); F1 fijo/base 1.001. Heartwall: 30.0 s
(base), 30.3 s (agente), 33.6 s (F1 fijo). El agente clasificó heartwall como compute_bound en 5 bloques y se abstuvo
en 1 (confianza 0.85); ambas liberan el reloj. REF powersave / REF performance = 0.998 de EDP. Datos en
docs/libro/datos/fase4_20260930/EC_7818_powersave/.
