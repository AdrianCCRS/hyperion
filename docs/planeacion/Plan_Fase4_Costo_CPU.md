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

## Resultado (job 7824, 2026-09-30)

30 de 30 celdas válidas. Potencia extra de CPU frente a la base del bloque (base de 175.7 W): sombra +12.6 W; sombra a 10 ms
+13.5 W; consumidor con espera de 1 ms +12.1 W; consumidor fijado +11.7 W. Ninguna variante difiere de la sombra en más de
0.8 W (prueba de signos, 6 bloques: p = 0.69, 0.22 y 0.69). Con 10 ms las interrupciones entre núcleos bajan de 1.10 M a
0.12 M por celda y el núcleo del colector vuelve a C6 (23.3 s frente a 4.3 s), sin cambio en la potencia. En reposo el agente
solo sube la potencia de CPU de 75.6 a 146.3 W (3/3 repeticiones), con cerca de 1 s de CPU consumido en 60 s.
**La hipótesis de la cadencia queda refutada**; la causa sigue sin identificarse. Siguiente prueba posible: agente en
reposo sin leer contadores y a 10 ms. Datos: docs/libro/datos/fase4_20260930/costo_cpu_7824/; script
docs/libro/scripts/analisis_diag_7823_7824_20260930.py.

## Resultado de la réplica de E-A con EPP=power (job 7823)

Diagnóstico con carga sostenida: 113.3 W (power) frente a 112.8 W (balance_performance), ambos a 3.2 GHz. Confirmatorio de 5
bloques: agente/base 0.981 de EDP del nodo (IC95 0.974 a 0.988, 5/5, p = 0.0625); agente/F1 fijo 0.994 (0.987 a 1.001, 4/5);
muestras de F1 fijo fuera de rango solo en DGEMM, como en E-A. Datos: .../epppower_7823/.

## Segunda parte: descomposición del costo en reposo (declarada el 2026-09-30, antes de medir)

El job 7824 refutó la cadencia como causa en la aplicación de kernels vistos, y mostró que con el nodo en reposo el agente
solo eleva la potencia de CPU de 75.6 a 146.3 W. El reposo es el lugar donde descomponerlo, porque el efecto es seis veces
mayor y no tiene ruido de aplicación.

Diseño (`scripts/pacca/hyp_fase4_costo_reposo.sbatch`): `sleep` en los núcleos 0-5 como objetivo, 60 s por celda, 3 bloques
aleatorizados (semilla 20261004), RAPL de ambos paquetes y los mismos registros de reposo, interrupciones y tiempo por hilo.
Brazos: `reposo`; `sombra_1ms` (referencia); `sombra_10ms`; `sombra_100ms`; `sombra_noperf` (agente completo, colector sin
leer contadores, `--no-perf`); `despertador_1ms` (sin agente: proceso mínimo en el núcleo 6 que duerme 1 ms en bucle).

Lectura de los resultados, fijada de antemano:
- si `despertador_1ms` y `sombra_noperf` ya elevan la potencia como `sombra_1ms`: el costo es el despertar a 1 kHz;
- si solo `sombra_1ms` y `sombra_10ms` la elevan: es la lectura de contadores o el agente cargado, no el despertar;
- si `sombra_100ms` y `sombra_10ms` bajan en proporción a la cadencia: hay una dosis-respuesta en reposo que la aplicación
  enmascara;
- si ninguna variante baja de `sombra_1ms`: lo produce la mera presencia del proceso (modelo cargado, hilos), a descomponer
  en una tercera parte.

Nodos: paccaA100 (plataforma de la tesis, estado de la base del libro) y un nodo de la partición normal (pacca01, Xeon Gold
5320, sin permisos de frecuencia, turbo activo, `performance`) como medición inmediata sin cola GPU. Los dos SKU son
Ice Lake-SP pero distintos: del nodo normal solo se interpreta qué componente produce el costo, no su magnitud en vatios.
No se escribe nada en el libro hasta tener ambas.

### Resultado en pacca01 (job 7829, 2026-09-30)

18 de 18 celdas válidas (Gold 5320, turbo activo, `performance`; solo se interpreta el componente, no los vatios). Potencia
media de CPU por brazo (3 bloques): reposo 75.0 W (70.5 a 79.2); despertador a 1 ms 126.6 W; sombra a 1 ms 161.7 W; a 10 ms
159.5 W; a 100 ms 160.5 W; sin leer contadores 162.1 W. Ni la cadencia del colector (1 a 100 ms) ni la lectura de contadores
cambian la potencia, aunque el tiempo de CPU del agente baja de 1.08 a 0.34 s por celda. Un proceso mínimo que solo
despierta a 1 kHz ya añade +52 W. La residencia en C6 de cada núcleo apenas cambia (0.92 a 0.98 frente a 1.0 en reposo).

Lectura: el costo no está en el trabajo del agente sino en despertar el paquete. Lo coherente con los datos es que los
despertares periódicos impiden el estado de reposo del paquete (C-state de paquete), que sí apaga el *uncore*, aunque cada
núcleo pase más de 90 % del tiempo en C6. Las interrupciones de temporizador lo apoyan: 0.2 k/s en reposo, 1.8 k/s con el
despertador y 7.5 a 9.6 k/s en todos los brazos del agente, cuyo consumidor despierta cada 100 µs en todos ellos. Este
diseño no varió esa espera del consumidor, así que la regla declarada de «mera presencia» no se puede aplicar: la variable
que quedó constante es justamente la de mayor frecuencia de despertar. Falta: residencia de C-state de paquete
(`cstate_pkg`, disponible en el nodo) y un brazo con `--consumer-idle-us` grande. Con aplicación (job 7824) el paquete ya
está activo, por lo que este mecanismo no explica por sí solo los +12.6 W de entonces.
