# Resultados iniciales R1 — reformulación por tamaño y amortización

**Fecha de ejecución:** 2026-08-30  
**Dataset de entrada:** `selector_final_20260830`  
**Corridas de origen:** 8.160 aceptadas  
**Estado:** resultado reproducible inicial; no sustituye la evaluación del
agente ni se ha incorporado al libro

## 1. Comando y artefactos

```bash
python -m classifier.selector r1 \
  --dataset-dir ~/hyperion-results/analysis/selector_final_20260830 \
  --output-dir ~/hyperion-results/analysis/selector_final_20260830/r1_canonical_v2_20260830
```

El análisis produce:

- `compact_static.csv` y `compact_with_probe.csv`;
- `amortization_map.csv`;
- `size_folds.csv`;
- `interpolation_baselines.csv` y `extrapolation_baselines.csv`;
- `baseline_metrics.csv`;
- `baseline_oracle_headroom.csv`;
- `dvfs_headroom.csv`;
- `datacard.json` y `datacard.md`;
- `r1_summary.json`.

La unidad independiente continúa siendo `config_id`: 68 configuraciones. Las
204 filas del dataset compacto son tres vistas correlacionadas de esas mismas
configuraciones (`none_ready`, `cpu_ready` y `gpu_ready`), no 204 experimentos
independientes.

## 2. Sanidad de la decisión de dispositivo a REF

| Estado | CPU óptima | GPU óptima | Separadas | Inciertas |
|---|---:|---:|---:|---:|
| `none_ready` | 68 | 0 | 68 | 0 |
| `cpu_ready` | 68 | 0 | 68 | 0 |
| `gpu_ready` | 12 | 56 | 65 | 3 |

La decisión de dispositivo obtenida comparando CPU REF contra GPU REF coincide
en los 204 contextos con el dispositivo del oráculo que explora todas las
frecuencias. Esto respalda separar primero dispositivo y frecuencia: el
producto cartesiano completo no cambió ninguna etiqueta de dispositivo en el
conjunto actual.

## 3. Mapa de amortización

`K_break_even` se calcula con energía y tiempo acumulados:

```text
E_total(d, K) = E_cold(d) + (K - 1) * E_warm(d)
T_total(d, K) = T_cold(d) + (K - 1) * T_warm(d)
EDP_total(d, K) = E_total(d, K) * T_total(d, K)
```

No se suma el EDP de despachos independientes. El resultado central sobre las
medias es:

| Operación | Configuraciones | K finito | K mínimo | K mediano finito | K máximo |
|---|---:|---:|---:|---:|---:|
| AXPY | 8 | 0 | — | — | — |
| Cholesky | 13 | 8 | 2 | 49 | 2.497 |
| FFT | 13 | 8 | 4 | 81,5 | 12.294 |
| GEMM | 13 | 6 | 3 | 37,5 | 1.075 |
| SpMV | 8 | 0 | — | — | — |
| Stencil | 13 | 0 | — | — | — |

En total, 22 de 68 configuraciones presentan un cruce finito. En las otras 46
la GPU no supera asintóticamente a CPU bajo el contrato de transferencias
medido. Por tanto, no es correcto afirmar que una operación completa amortiza
GPU a todos sus tamaños solo porque sus tamaños grandes lo hagan.

Los primeros tamaños con cruce observado son:

- GEMM: N=768, con `K_break_even=1075`;
- FFT: N=192, con `K_break_even=12294`;
- Cholesky: N=384, con `K_break_even=2497`.

La presencia del cruce no se asumirá monótona sin contrastarla: FFT presenta
un cruce aislado en N=192, vuelve a favorecer CPU en N=256 y N=384, y recupera
cruces desde N=512. Este comportamiento es precisamente una razón para usar
los tamaños como eje experimental en vez de imponer una frontera teórica
suavizada.

En los extremos actuales:

- GEMM N=4096: `K_break_even=3`;
- FFT N=4096: `K_break_even=4`;
- Cholesky N=4096: `K_break_even=2`.

La sensibilidad se calculó combinando los extremos marginales observados de
energía y tiempo en sentidos favorables y desfavorables para GPU. Es una
envolvente rectangular conservadora: los mínimos de energía y tiempo no
necesariamente pertenecen a la misma repetición, por lo cual no constituye un
intervalo probabilístico ni un remuestreo pareado.

## 4. Resolución temporal de la región fría

Cinco configuraciones contienen al menos una repetición CPU fría marcada de
baja resolución. No se encontraron regiones GPU REF frías de baja resolución
en este conjunto compacto.

El CSV conserva por candidato la fracción de repeticiones de baja resolución.
Esa información debe utilizarse al interpretar la sensibilidad de
`K_break_even`; el valor central no elimina la limitación instrumental.

## 5. Headroom de frecuencia

Comparar la mejor frecuencia del dispositivo ganador contra REF produce la
siguiente cota superior de mejora, antes de descontar el costo de actuación:

| Estado | Casos sobre 3,11 % | Mejora mediana | Mejora p95 | Máxima |
|---|---:|---:|---:|---:|
| `none_ready` | 38/68 | 4,98 % | 50,70 % | 57,96 % |
| `cpu_ready` | 3/68 | 0,25 % | 2,77 % | 8,54 % |
| `gpu_ready` | 56/68 | 9,45 % | 45,25 % | 68,47 % |

El margen de 1,03 % observado anteriormente entre la mejor y la segunda mejor
frecuencia en C indica que el `argmin` fino es inestable. No implica que REF
sea necesariamente equivalente al conjunto de mejores frecuencias. En
`gpu_ready`, 56 de 68 configuraciones conservan una mejora potencial sobre
REF mayor que la referencia de 3,11 %. Por ello, la siguiente etapa debe
buscar una política DVFS robusta o un conjunto de frecuencias equivalentes,
descontando luego el costo real de actuación.

## 6. Baselines por tamaño: resultado que condiciona R2

En `none_ready` y `cpu_ready`, `always_cpu_ref` coincide con el oráculo de
dispositivo en todos los pliegues actuales. La única rama no trivial es
`gpu_ready`:

| Régimen | Mejor baseline | Balanced accuracy | Razón EDP/oráculo | Brecha recuperable |
|---|---|---:|---:|---:|
| Interpolación | umbral de tamaño | 0,778 | 1,0029 | 0,285 % |
| Extrapolación | tabla de cruce por operación | 0,867 | 1,0130 | 1,283 % |

La brecha es pequeña y, bajo la regla bloqueante del protocolo congelado, ya
produce un resultado negativo para ML de dispositivo: ni siquiera el oráculo
puede mejorar la mejor baseline en 3,11 %, así que ningún modelo puede superar
ese umbral en estos pliegues. R2 sigue siendo necesario para completar la
comparación predeclarada y cuantificar el comportamiento, pero no puede
convertir esta cota superior en una mejora mayor.

Esta es una **conclusión contractual**, no una prueba estadística de ausencia
de señal: 3,11 % es el CV mediano de acciones individuales, no la
incertidumbre de la diferencia agregada entre políticas. Ambas afirmaciones
deben conservarse separadas al escribir el resultado.

## 7. Consecuencias para las siguientes fases

1. No se justifica lanzar todavía la matriz `big` completa.
2. Los tamaños nuevos siguen siendo útiles como prueba confirmatoria de
   extrapolación de `K_break_even` y de la razón CPU/GPU.
3. La adquisición inicial debe limitarse a CPU REF y GPU REF.
4. La selección de frecuencia debe estudiarse por estado y dispositivo; el
   promedio global ocultaba headroom en `gpu_ready`.
5. AXPY, SpMV y stencil no requieren tamaños grandes para intentar fabricar
   un cruce que los datos calientes actuales no respaldan.
6. R2 debe comparar modelos simples contra umbrales por operación/tamaño y
   contra políticas constantes, usando particiones de interpolación y
   extrapolación.

Este resultado no demuestra todavía utilidad de ML ni ganancia neta del
agente. Demuestra que la reformulación produce una pregunta medible, identifica
qué configuraciones pueden amortizar GPU y evita gastar horas de clúster en
una matriz completa antes de conocer el headroom pertinente.
