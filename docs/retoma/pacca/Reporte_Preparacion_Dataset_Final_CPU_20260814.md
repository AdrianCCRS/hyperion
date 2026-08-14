# Preparación del primer intento de dataset final CPU — 2026-08-14

## Alcance y estado

Este reporte separa tres fuentes de evidencia que no deben mezclarse como si
fueran el mismo protocolo:

1. `pacca_dvfs_full_20260813_clean`: campaña histórica de 126 corridas. Es
   útil para descubrir anomalías y estimar volúmenes, pero no es dataset final
   porque mantuvo Turbo activo y no muestreó la frecuencia por ventana.
2. `pacca_candidate_lavamd_ref_20260814`: caracterización reducida REF de
   LavaMD, 10 corridas.
3. `pacca_candidate_3mm_ref_20260814`: caracterización reducida REF de 3MM,
   10 corridas.

Antes de los cambios se dejó el commit de restauración `0c9f7af`. No se lanzó
la campaña final completa.

## Repeticiones

La evidencia de `docs/justifications/report/sections/repetitions.tex` no
demuestra que tres repeticiones sean suficientes en sentido estadístico. Tres
es el mínimo operativo y su resultado depende de qué terna se ejecute: la
anomalía tardía de `rodinia_lud` apareció en la repetición 7 y el reanálisis de
las 120 ternas posibles mostró dispersión dependiente de la selección. Como el
usuario autorizó el costo adicional, el manifiesto final usa **10 repeticiones
por combinación**, el mayor valor efectivamente ensayado en ese barrido. No se
presenta 10 como óptimo universal.

## Caracterización de los candidatos

### `rodinia_lavamd_omp`

- Fuente: Rodinia, commit
  `9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`, implementación
  `openmp/lavaMD`.
- Precisión: `main.h` define `fp` como `double`; se compila con GCC 12.4.0,
  OpenMP, `-O3 -march=native`, sin cambiar el algoritmo. El builder fija la
  ruta del compilador y verifica el hash antes de instalar: una reconstrucción
  con el `gcc` 8.5.0 implícito produjo un hash distinto y fue
  descartada; la reconstrucción con GCC 12.4.0 reprodujo exactamente el hash
  catalogado.
- Comando del catálogo: `-cores 6 -boxes1d 24`.
- SHA-256 del binario: `92cb259bd2ca1d0a67a4d52490b76cdb2eb844568c7e73864023c52d7abb2648`.
- Criterio de éxito: confirma la configuración exacta y alcanza `Total time`.
  El programa externo no incluye verificador numérico; esta limitación se
  conserva explícita y no se describe como checksum del resultado.

Resultado de 10 corridas REF:

| Métrica | Resultado |
|---|---:|
| Corridas aceptadas/rechazadas | 10 / 0 |
| Ventanas `ok` | 65.705 |
| `compute_bound` | 65.644 (99,907 %) |
| `memory_bound` | 61 (0,093 %) |
| `intensity_undefined` | 718 |
| Rango de duración instrumentada | 7,44–7,54 s |
| `perf_running_ratio_min` | 1,0 en 10/10 |
| `push_retries` | 0 en 10/10 |
| Intensidad, mín./mediana/máx. | 4,44 / 257,40 / 496,82 FLOP/byte |
| Margen relativo al ridge, mín./mediana/máx. | -48,7 % / +2.876 % / +5.645 % |

Los cuatro campos FP64 estuvieron presentes de forma atómica. Solo el evento
escalar tuvo cuentas no nulas; no se inventa cobertura vectorial que el kernel
no produjo. Nueve repeticiones fueron 100 % `compute_bound`; la décima tuvo 61
ventanas `memory_bound` y 6.503 `compute_bound`. La contribución es útil y
estable como extremo aritmético escalar, aunque no sustituye diversidad
vectorial.

### `rajaperf_polybench_3mm_omp`

- Fuente: RAJAPerf v2025.12.1, commit
  `e3c6197dfa8f1c9ac61635c26775c333411bdcd5`; submódulo RAJA
  `eca7c5015a5cf8bf7cc8ad1829fd36d3276ab274`.
- Precisión: `Real_type=double` mediante `RP_USE_DOUBLE`; variante
  `Polybench_3MM/Base_OpenMP`.
- Comando del catálogo: `--repfact 10 --sizefact 1`.
- SHA-256 del ELF real:
  `7f5251ac4c8f4bfd854441b7873f120080affec7bc77abfce1cc0fb9ec165ebb`.
- SHA-256 del adaptador catalogado:
  `386fd7a0db562f12a872891268270171e289be744cebc86c94595d61239c467d`.
- Criterio de éxito: el adaptador verifica el hash del ELF y exige el
  `PASSED` del checksum numérico nativo de RAJAPerf en cada ejecución.
- Reproducibilidad: el árbol fuente quedó conservado en
  `~/hyperion-kernels/src/RAJAPerf-v2025.12.1`; una reconstrucción limpia bajo
  Slurm reprodujo exactamente los hashes del ELF y del adaptador.

Resultado de 10 corridas REF:

| Métrica | Resultado |
|---|---:|
| Corridas aceptadas/rechazadas | 10 / 0 |
| Ventanas `ok` | 119.240 |
| `compute_bound` | 112.096 (94,009 %) |
| `memory_bound` | 7.144 (5,991 %) |
| `intensity_undefined` | 705 |
| Ventanas `pmu_degraded` | 2 |
| Rango de duración instrumentada | 12,65–12,94 s |
| CV muestral de duración | 0,747 % |
| `perf_running_ratio_min` | 1,0 en 10/10 |
| `push_retries` | 0 en 10/10 |
| Intensidad, mín./mediana/máx. | 3,53 / 46,99 / 110,17 FLOP/byte |
| Margen relativo al ridge, mín./mediana/máx. | -57,4 % / +467,9 % / +1.231 % |

La proporción `compute_bound` por repetición estuvo entre 93,39 % y 95,09 %.
Los cuatro campos FP64 estuvieron presentes; los eventos escalar y 128 bits
tuvieron cuentas no nulas, y 256/512 bits quedaron en cero. Las dos ventanas
`pmu_degraded` corresponden a deltas `time_enabled=time_running=0`, una en
rep09 y otra en rep10; fueron excluidas, representan 0,0017 % de las ventanas
posteriores al calentamiento y no hubo multiplexación (`running_ratio=1` en
las ventanas activas). No son evidencia de degradación problemática de la
corrida, pero se reportan en lugar de ocultarlas.

## Auditoría de la campaña histórica 2026-08-13

Se verificaron las 126 combinaciones esperadas: 7 kernels × 6 niveles × 3
repeticiones, sin faltantes, duplicados ni artefactos ausentes. Las 126 tienen
veredicto aceptado, `perf_running_ratio_min=1`, `push_retries=0`, afinidad
`0-5` para carga/PMU, collector 6, consumer 7, un checksum estable por kernel,
stderr vacío y cero ventanas `pmu_degraded`.

La campaña contiene 1.292.556 ventanas `ok`: 47.719 `compute_bound` (3,692 %)
y 1.244.837 `memory_bound` (96,308 %). También contiene 9.570
`intensity_undefined`; aparecen en las 126 corridas, con un máximo de 88 en
una sola corrida, y no fueron usadas como etiquetas.

Además del bug conocido de frecuencia se encontró una anomalía concreta:
`npb_ft/F1/rep03` duró 5,274 s frente a 4,538 y 4,526 s en rep01/rep02. La
relación máximo/mínimo es 1,165 y el CV muestral del trío es 8,96 %. La salida
fue correcta y no hubo pérdida de contadores, pero la campaña no conservó
temperatura ni carga externa por corrida, así que no existe evidencia para
atribuir la causa. Las otras 41 combinaciones kernel/frecuencia tuvieron CV
menor o igual a 1,49 %.

La cabecera de los 126 `samples.csv` no contiene
`scaling_cur_freq_khz`. `windows.csv` repite una sola lectura post-hoc por
corrida; por ejemplo, F3 pidió/aplicó 1,5 GHz pero registró 1,569623 GHz en
todas las ventanas. En F0--F3 muchas lecturas post-hoc cayeron cerca de 0,8
GHz. Por eso ni las etiquetas por frecuencia ni el aparente equilibrio por
nivel de esa campaña validan el protocolo final.

## Distribución: medición y proyección limitada

Sumar los datos históricos inválidos con las dos caracterizaciones REF daría
15,26 % `compute_bound`, pero esa cifra **no es una estimación del dataset
final** porque mezcla protocolos y frecuencias no controladas.

Como cálculo de capacidad, si se escala la campaña histórica de n=3 a n=10 y
se supone —solo para dimensionar— que cada candidato produce en los seis
niveles el mismo número y proporción de ventanas observados en REF, el total
sería aproximadamente 5.418.190 ventanas `ok`, de las cuales 1.225.503
(22,62 %) serían `compute_bound`. Bajo esa hipótesis, 3MM aportaría 54,88 %
de las ventanas compute, LavaMD 32,14 % y los kernels anteriores 12,98 %.
Esta proyección no reemplaza la medición multifrecuencia: cambiar frecuencia
cambia duración, ridge point y cantidad de ventanas.

El pre-vuelo preparado mide `npb_mg`, LavaMD y 3MM tres veces en REF y en
nueve puntos fijos separados por 12,5 % del rango (90 combinaciones). Además
de medir clases por frecuencia y repetición, este reconocimiento debe comparar
las curvas obtenidas en los nueve puntos con el subconjunto F0--F4 y decidir,
antes de la campaña final, si esos cinco niveles capturan la forma observada o
si el manifiesto final debe conservar mayor resolución. No se fija cuota 50/50.
Tampoco se eliminan ventanas `memory_bound` ni se duplican ventanas
`compute_bound`; cualquier ponderación o muestreo posterior debe ocurrir solo
en entrenamiento, después de separar por corrida.

Riesgos de dominancia que deben revisarse tras el pre-vuelo:

- 3MM aporta más ventanas por corrida que LavaMD y podría dominar la clase
  `compute_bound`.
- los niveles bajos pueden producir más ventanas por corrida al alargar el
  tiempo de ejecución, aun con igual número de repeticiones;
- un solo kernel no debe convertirse en atajo para predecir la clase; la
  partición de Fase 2 debe agrupar por corrida y evaluar generalización por
  carga.

## Estado del permiso Turbo y bloqueos restantes

El helper `/usr/local/bin/set_turbo_state` existe y escribe
`intel_pstate/no_turbo`. Sin embargo, dos verificaciones reales del comando
no interactivo —una por SSH y otra dentro de `srun --exclusive`— devolvieron
`sudo: a password is required`. En ambas, `no_turbo` permaneció en 0; la
prueba dentro de Slurm confirmó además `RESTORED=0`. Por tanto, el permiso
administrativo todavía no es operativo para la cuenta `latorresn`, aunque el
helper ya esté instalado.

Antes de la campaña final faltan, en este orden:

1. que `sudo -n set_turbo_state 1` funcione dentro de la asignación exclusiva;
2. sincronizar el commit final de código y scripts a pacca y ejecutar allí la
   suite;
3. ejecutar el pre-vuelo de 90 combinaciones, no la campaña completa;
4. confirmar en sus artefactos Turbo deshabilitado, los nueve objetivos
   nominales 3,2/2,9/2,6/2,3/2,0/1,7/1,4/1,1/0,8 GHz, traza por ventana
   dentro de ±5 %, temperatura y carga registradas, calibraciones válidas y
   distribución de ambos candidatos por frecuencia/repetición;
5. comparar la curva de nueve puntos con F0--F4 y cerrar explícitamente el
   número de niveles del manifiesto final;
6. solo entonces lanzar el manifiesto final (hoy dimensionado en 540
   combinaciones si se conservan los cinco puntos fijos más REF).

El rango térmico E02 de 0–90 °C sigue siendo un guardarraíl operativo del
proyecto, no un límite demostrado a partir del `Tcase` del SKU. La lectura ya
es real y auditable; la fuerza de la justificación del umbral no se
sobredimensiona.

Los cuatro scripts de campaña preparados o usados en esta caracterización
propagan ahora el código de salida de `srun`: un fallo ya no puede quedar
oculto por el `echo` final del script.
