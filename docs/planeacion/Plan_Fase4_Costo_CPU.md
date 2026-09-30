# Plan Fase 4: atribución del costo de observación del agente de CPU (declarado el 2026-09-30, antes de medir)

## Por qué

Revisión del director, C6: el brazo sombra del agente de CPU cuesta entre 5 y 9 % de energía de CPU sin que la inferencia
(16 µs) lo explique. Con los datos existentes (matriz inicial 7639/7696, Escenario C, corrida con ORT de un hilo) se
estableció, sin medición nueva, que:

- el costo es potencia y no tiempo: entre 80 y 93 % de la energía extra corresponde a mayor potencia a igual duración;
- la potencia extra es reproducible entre campañas: +12.3 a +13.7 W con la aplicación de kernels vistos y +6.1 a +6.2 W
  con la de inéditos;
- el agente de GPU, con el mismo ONNX Runtime, no tiene costo medible (−0.1 a +0.7 W).

Hipótesis: el costo lo producen los despertares del agente de CPU (colector a 1 ms que lee 9 contadores en 6 núcleos,
consumidor que revisa la cola cada 100 µs), que mantienen activos núcleos que de otro modo dormirían y, en las lecturas
de contadores de otros núcleos, interrumpen a los de la aplicación. El agente de GPU despierta cada 50 ms.

## Diseño (job `hyp_fase4_costo_cpu`, `scripts/pacca/hyp_fase4_costo_cpu.sbatch`)

Estado de CPU de la base del libro: `performance`, turbo desactivado, 3.2 GHz fijos en los 12 procesadores delegados.

**Parte A, aplicación de kernels vistos** (alcance `cpu`, set `known`, la de mayor costo), 6 bloques aleatorizados
(`balanced_blocks`, semilla 20261003), con cinco brazos:

| Brazo | Cambio respecto de sombra | Qué prueba |
|---|---|---|
| `base` | sin agente | referencia |
| `sombra` | configuración del libro (colector 1 ms, consumidor 100 µs, consumidor sin fijar) | el costo actual |
| `sombra_10ms` | `--interval-ns 10000000` | la cadencia del colector |
| `sombra_idle1ms` | `--consumer-idle-us 1000` | el sondeo del consumidor |
| `sombra_pin` | `--pin-consumer` (consumidor y ORT en el núcleo 7) | si el consumidor cae en los núcleos de la aplicación |

**Parte B, nodo sin aplicación**: objetivo `sleep` en 0-5; 60 s sin agente y 60 s con el agente en sombra, 3
repeticiones alternadas. Separa el consumo propio del agente de su interacción con la aplicación.

En cada celda (`DIAG_SNAPSHOT=1`) se guardan, al inicio y al final de la ventana de energía: `/proc/stat` por CPU,
interrupciones LOC/RES/CAL/TLB por CPU, residencia y entradas de cada estado de reposo (`cpuidle`) de las CPU 0-7 y
16-23, y `utime`/`stime`/último procesador de cada hilo de `cpu_loop_main`.

## Análisis declarado

- Principal: potencia media de CPU (RAPL de ambos paquetes / duración) de cada brazo sombra frente a la base del mismo
  bloque; prueba de signos exacta bilateral por bloques (6 bloques, p mínimo 0.031) de `sombra_10ms` y
  `sombra_idle1ms` frente a `sombra`.
- Secundario: razones de duración y energía de CPU; tiempo de CPU por hilo del agente; residencia en C6 y tasa de
  interrupciones CAL por CPU, por brazo; potencia en reposo con y sin agente.

## Predicción

Si la hipótesis es cierta, `sombra_10ms` y/o `sombra_idle1ms` reducen la potencia extra frente a `sombra` en los seis
bloques, los núcleos 6 y 7 pierden residencia en C6 con `sombra`, y en reposo el agente añade una potencia del orden de
la medida con la aplicación de inéditos. Si ninguna variante reduce la potencia, la cadencia queda descartada como causa.
Cualquiera de los resultados se reporta.
