# Informe de calidad del clasificador CPU compute/memory (2026-09-18)

Documento vivo para alimentar el libro. Toda cifra tiene su artefacto en `docs/libro/datos/cpu_calidad_20260918/` y se regenera con:

```
python3 -m fase2_clasificador.analysis.cpu_quality_report \
  --source <feature_contract_source.csv> --out <dir> --seeds 5 --boot 2000
```

(`fase2_clasificador/analysis/cpu_quality_report.py`, etapas: inventory, final_model, matrix, diagnostics, protocols, learning_curve, smoothing, nested_selective, nested_optuna, regularization, importance, knn_ceiling, twins, temporal, adaptation, adaptation_phases, oi_proxy, cap_sensitivity, threshold_nested, latency). La fuente es `cpu_feature_audit_20260918/feature_contract_source.csv` (sha256 `025e9dde…a45e68`, 1 169 833 intervalos elegibles, 24 familias, 43 kernels).

Las cifras de esta sección **reemplazan como cifras de calidad** a las del cribado (0.653) y de la selección anidada (0.664) de F2-CPU-001, por las razones de la sección 2.

## 1. Resumen ejecutivo

1. **Calidad del candidato (XGBoost, 6 variables PMU base más 6 interacciones, pesos familia×clase), validación por familia completa retenida, test sobre todos los intervalos de la familia:** exactitud balanceada por celda familia×clase **0.709** (IC95 por bootstrap de familias 0.633 a 0.780; desviación entre 5 semillas de muestreo 0.004). F1 macro agrupado 0.709; recall compute 0.59, recall memory 0.82.
2. **Todas las variantes razonables empatan.** XGBoost base, con interacciones, sin frecuencia, sin ponderación por familia, monótono, somero, Random Forest, Extra Trees y regresión logística están entre 0.68 y 0.71, con diferencias pareadas cuyo IC95 contiene el cero. Solo el árbol de profundidad 1 (0.617), la mayoritaria (0.523) y "solo frecuencia" (0.569) quedan claramente por debajo. Las interacciones PMU y la frecuencia no aportan mejora medible (delta interacciones: +0.010 [-0.003, +0.024]; sin frecuencia: 0.000 [-0.014, +0.012]).
3. **El techo es de información, no de modelo.** Un k-vecinos sin entrenamiento (k=25) en el espacio PMU llega a 0.683 (0.698 con frecuencia). Ni la profundidad del árbol, ni el ajuste anidado de complejidad (0.712), ni el umbral de decisión anidado (0.704), ni un regresor de OI con decisión contra el ridge (0.687), ni variables de historia reciente (+0.012, dentro del ruido) rompen el 0.70 a 0.72.
4. **Causa identificada con evidencia directa:** hay perfiles PMU casi idénticos con etiquetas opuestas en familias distintas (tabla 6). El caso más claro: la fase compute de `phasic` (mpki 1.1, IPC 0.96, tasa de fallos 0.735) es casi indistinguible de `gap_pr`, etiquetada memory (mpki 0.8, IPC 1.1, tasa 0.76). Dentro de `phasic` las dos fases se separan con 0.999 de exactitud balanceada; entre familias no.
5. **La fuga entre intervalos infla 0.18.** Partición aleatoria por intervalo: 0.894. Leave-one-family-out: 0.709.
6. **Más familias ayudan, despacio:** de 3 a 23 familias de entrenamiento la métrica sube de 0.616 a 0.713; pendiente 0.049 por unidad de ln(k), es decir, **cerca de +0.034 por duplicar el catálogo** (extrapolación, IC amplio). Pasar de 24 a 27 familias esperaría del orden de +0.006.
7. **La abstención por confianza aporta ganancia real pero modesta y la confianza está mal calibrada** (ECE 0.161; en el rango de confianza 0.95 a 1.0 la exactitud es 0.839). Con umbral 0.85: cobertura 73.6 %, exactitud balanceada 0.739 (contra 0.708 sin abstener; una abstención aleatoria a la misma cobertura no cambia nada: 0.709). Hay familias con errores de alta confianza que ningún umbral evita (`npb_ft` 0.32 de exactitud con confianza ≥ 0.9, `phasic` 0.35, `gap_pr` 0.59).
8. **Calibración en línea (no es validación externa a otra carga):** con etiquetas *uncore* de solo 2.6 % de los intervalos de la primera mitad de una corrida, el modelo adaptado clasifica la segunda mitad con exactitud balanceada 0.85 y F1 agrupado 0.92 en bloques con cambios de fase (0.62 y 0.62 sin adaptar). Las referencias triviales de repetir la última etiqueta quedan en azar (0.51 a 0.52). Es evidencia directa a favor de la rama "microcaracterización" de la política `revisar` (sección 4.13).
9. **Latencia** de una fila en pacca01 (1 hilo, `predict_proba`): XGBoost 227 µs p50, 382 µs p99; regresión logística 266 µs / 275 µs; árbol prof. 1 99 µs / 103 µs; Random Forest y Extra Trees 5.0 ms / 5.2 ms. El presupuesto de latencia del plan sigue sin cuantificarse en el propio plan.

## 2. Por qué las cifras anteriores (0.600, 0.653, 0.664, 0.871) no son una línea de calidad

| Cifra | Qué es | Problema |
|---|---|---|
| 0.600 | XGBoost con Optuna, representación anterior con `running_ratio`, sin pesos por familia | Experimento distinto; no comparable como "antes/después" |
| 0.653 | Mejor celda de una rejilla de cribado (modelos × variantes × pesos), una muestra | Máximo de varias combinaciones; posible doble balanceo en el evaluador `7455` (no reejecutado) |
| 0.664 | Media de F1 macro por familia, selección anidada de variante y umbral | Métrica degenerada (abajo); rejilla de umbral truncada en 0.90 (21 de 24 pliegues la eligieron) |
| 0.860 / 0.871 | F1 sobre decisiones aceptadas | Condicionada a cobertura 69 % / 63 %; no comparable con métricas de cobertura total |

**Defecto de la métrica "F1 macro medio por familia".** Con `labels=[False, True]` y `zero_division=0` una familia pura solo puede obtener 0.5 aunque se clasifique perfecto, y una familia con 0.05 % de filas minoritarias (por ejemplo `cpu_lulesh`) obtiene 0.47 con 87 % de exactitud. Reproducida con la nueva corrida, esa métrica da 0.635 (test capado) y 0.603 (test completo) para el candidato, con desviación entre familias 0.22 a 0.24. La cifra de 0.664 del libro no es reproducible como mejora; es un máximo entre variantes evaluado con una métrica inestable.

**Métrica adoptada:** exactitud balanceada por celda familia×clase (media del recall de cada celda observada). Está definida para familias puras, no depende del volumen de cada familia ni de la prevalencia. Se reporta junto con F1 macro agrupado, recall por clase y exactitud media por familia, siempre con IC95 por bootstrap de conglomerados (familias, 2000 remuestreos) y desviación entre 5 semillas de muestreo.

## 3. Datos e inventario (`inventory.json`, `inventory_by_family.csv`)

- 1 169 833 intervalos elegibles; se retiran 4 451 (0.4 %) con `stalls_mem_per_cache_miss` indefinido (cero fallos de caché): 1 165 382 filas de análisis. Son intervalos casi seguramente compute; el efecto sobre el resultado no se cuantificó.
- 24 familias, 43 kernels. Memory 67.2 %, compute 32.8 %.
- **12 familias casi puras** (una clase >98 %), 11 mixtas (10 a 90 % memory) y 1 intermedia (`npb_ft`, 96 % memory). 20 de 24 tienen ambas clases.
- Por combinación kernel×frecuencia: 430 celdas, 75.6 % casi puras (<2 % o >98 %), solo 59 mezclas 20 a 80 %.
- Proporción memory por nivel de frecuencia: de 0.751 (F0) a 0.615 (F8), 0.773 en REF (`inventory.json`).
- AUC univariado agrupado: `mpki` 0.878, `cache_misses_per_cycle` 0.878, `cache_miss_rate` 0.782; `ipc` 0.474, `ips` 0.480, `freq_khz_observed` 0.572. El modelo es en la práctica un detector de tasa de fallos de caché (importancia por permutación en la familia retenida: `cache_miss_rate` 0.060, `mpki` 0.029; varias variables tienen importancia negativa).
- El ridge por nivel es una constante de calibración (8.63, 8.20, 7.31, 6.58, 5.74, 5.03, 4.33, 3.64, 2.92 FLOP/byte para F0 a F8; 8.69 en REF).

## 4. Resultados por prueba

### 4.1 Matriz de modelos y variantes (`matrix.csv`, `matrix.json`, fig. `fig_cpu_calidad_matriz_modelos_20260918.png`)

Hiperparámetros fijos (sin búsqueda), 5 semillas de muestreo (tope 1000 filas por familia×clase para entrenar), test sobre todas las filas de la familia retenida.

| Configuración | Exactitud bal. | IC95 familias | F1 macro agr. | Recall compute | Recall memory |
|---|---:|---|---:|---:|---:|
| Mayoritaria | 0.523 | 0.478 a 0.571 | 0.402 | 0.000 | 1.000 |
| Árbol prof. 1 (base) | 0.617 | 0.528 a 0.708 | 0.716 | 0.578 | 0.844 |
| Logística (base) | 0.680 | 0.611 a 0.747 | 0.718 | 0.648 | 0.796 |
| Logística (+interacciones) | 0.689 | 0.629 a 0.751 | 0.765 | 0.675 | 0.852 |
| XGBoost (base) | 0.698 | 0.622 a 0.774 | 0.685 | 0.541 | 0.821 |
| **XGBoost (+interacciones), candidato** | **0.709** | 0.633 a 0.780 | 0.709 | 0.592 | 0.823 |
| XGBoost sin frecuencia | 0.709 | 0.632 a 0.786 | 0.705 | 0.584 | 0.821 |
| Solo frecuencia | 0.569 | 0.516 a 0.616 | 0.497 | 0.625 | 0.442 |
| XGBoost con `running_ratio` (legado) | 0.698 | idéntico a base | 0.685 | 0.541 | 0.821 |
| XGBoost solo balance de clase | 0.695 | 0.621 a 0.766 | 0.712 | 0.556 | 0.853 |
| XGBoost monótono (a priori) | 0.698 | 0.634 a 0.762 | 0.718 | 0.636 | 0.805 |
| XGBoost somero (prof. 2, 50 árboles) | 0.701 | 0.630 a 0.773 | 0.585 | 0.549 | 0.642 |
| Random Forest | 0.703 | 0.618 a 0.787 | 0.730 | 0.626 | 0.830 |
| Extra Trees | 0.699 | 0.616 a 0.783 | 0.652 | 0.516 | 0.785 |

Lecturas: (a) `running_ratio` no cambia nada (es casi constante); (b) "solo frecuencia" en 0.569 descarta que la frecuencia sea un atajo para adivinar la etiqueta; (c) las diferencias pareadas frente al candidato están todas dentro de ±0.03 con IC que contienen el cero, salvo mayoritaria, árbol prof. 1 y solo frecuencia.

### 4.2 Protocolos de validación (`protocols.json`, fig. `fig_cpu_calidad_protocolos_20260918.png`)

| Protocolo | Exactitud bal. (5 semillas) |
|---|---:|
| Aleatorio por intervalo (80/20 estratificado) | 0.894 ± 0.015 |
| Leave-one-kernel-out (43 kernels) | 0.723 ± 0.004 |
| Leave-one-family-out (24 familias) | 0.709 ± 0.004 |

Solo 8 familias tienen más de un kernel (los `dual_*`, `phasic`, `rajaperf_*`), por eso kernel-out y family-out casi coinciden. La brecha de 0.18 entre aleatorio y familia es memorización de la carga, no capacidad de clasificar régimen.

### 4.3 Sensibilidad al tope de muestreo (`cap_sensitivity.csv`)

Entrenar con tope por familia×clase de 100, 300, 1000, 3000, 10000 y 30000 (de 4 mil a 594 mil filas) y probar siempre con todo: 0.692, 0.710, 0.707, 0.703, 0.700, 0.703. **El tope de 1000 no pierde información medible**; 36 mil filas de 1.17 millones no son una limitación. (Respuesta a la duda de "solo 36 mil intervalos".)

### 4.4 Curva de aprendizaje por número de familias (`learning_curve.csv`, fig. `fig_cpu_calidad_curva_aprendizaje_20260918.png`)

| Familias de entrenamiento | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 23 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exactitud bal. media | 0.616 | 0.635 | 0.683 | 0.682 | 0.684 | 0.704 | 0.711 | 0.713 |

Ajuste lineal en ln(k): pendiente 0.049, intercepto 0.560. IC95 entre familias ± 0.05 a 0.08 por punto. Duplicar el catálogo daría del orden de +0.034; llegar a 0.80 exigiría unas 2.6 duplicaciones (del orden de 140 familias) bajo extrapolación log-lineal, que no debe tomarse como predicción. **Es la evidencia cuantitativa de que la generalización a cualquier kernel queda fuera de alcance de un trabajo de grado.**

### 4.5 Diagnóstico por familia (`candidate_by_family_full.csv`, fig. `fig_cpu_calidad_por_familia_20260918.png`)

Recall medio de las clases presentes (test completo). Bien (≥ 0.85): `dgemm`, `dual_spmv`, `dual_stencil`, `dual_fft`, `npb_cg`, `dual_cholesky`, `dual_gemm`, `rajaperf_polybench`. En azar o peor (≤ 0.55): `cpu_lulesh` 0.44, `gap_pr` 0.44, `rodinia_lavamd` 0.45, `cpu_hpcg` 0.50, `npb_lu` 0.53, `phasic` 0.53, `npb_mg` 0.54, `npb_ft` 0.54, `rajaperf_basic` 0.54. Siete familias mixtas rinden 0.75 o más; las familias casi puras son las que fallan en bloque (todas etiquetadas como una clase y clasificadas como la otra).

### 4.6 Perfiles gemelos entre familias (`twin_cells.csv`, `median_pmu_by_family_class.csv`)

Distancia (medianas PMU estandarizadas, mpki en log1p) entre una celda familia×clase y la celda más cercana de **otra familia con etiqueta opuesta**:

| Celda | Gemela opuesta | Dist. opuesta | Dist. misma etiqueta más cercana |
|---|---|---:|---:|
| `gap_pr` memory | `phasic` compute | 0.371 | 0.779 |
| `phasic` compute | `gap_pr` memory | 0.371 | 0.920 |
| `cpu_hpcg` compute | `npb_cg` memory | 0.422 | 1.307 |
| `rajaperf_lcals` memory | `rajaperf_stream` compute | 0.380 | 0.417 |
| `rajaperf_polybench` memory | `rajaperf_basic` compute | 0.461 | 0.817 |

En cuatro de estos casos el vecino más parecido en PMU pertenece a la clase contraria. Medianas de `phasic`: compute (mpki 1.11, IPC 0.96, tasa de fallos 0.735, stall 0.22); memory (mpki 324, IPC 0.02, tasa 0.98, stall 0.99). Medianas de `gap_pr` (memory): mpki 0.82, IPC 1.11, tasa 0.76, stall 0.16. En `npb_ft`, dentro de la propia familia, los intervalos memory y compute tienen mpki 2.46 y 2.44: no hay señal PMU que los separe (0.499 de exactitud balanceada validando por nivel de frecuencia). **Hipótesis a probar** (no probada): la etiqueta depende de FLOPs y bytes uncore de paquete, mientras las señales PMU son del núcleo; misma firma de fallos puede corresponder a OI distinta según el prefetch y la fracción de tráfico no atribuible.

### 4.7 Techo independiente del modelo (`knn_ceiling.csv`)

k-vecinos (k=25) sobre `ipc, log1p(mpki), tasa de fallos, stall`, con remuestreo por celda para equilibrar: 0.683 (solo PMU) y 0.698 (con frecuencia). XGBoost: 0.709. Un método sin ajuste llega a menos de 0.03 del candidato.

### 4.8 Distancia al ridge y calibración (`error_by_ridge_margin.csv`, `diagnostics.json`, `calibration_bins.csv`, figuras `margen_ridge` y `calibracion_abstencion`)

- Intervalos a menos de un factor 2 del ridge (|log2(OI/ridge)| < 1): 14.0 % del total, exactitud 0.522 contra 0.771 lejos; concentran 25.4 % de los errores. Por tanto la cercanía al ridge explica **una parte** (1.8 veces sobrerrepresentada), no la mayoría: 75 % de los errores ocurren lejos del ridge.
- **Asimetría:** intervalos memory a cualquier distancia: 0.75 a 0.85 de exactitud. Intervalos compute con OI entre 1 y 8 veces sobre el ridge o más: 0.59 a 0.64; justo sobre el ridge (0 a 1 en log2): 0.24 a 0.34. El punto débil es reconocer compute (recall 0.59), incluso lejos del ridge.
- Calibración: ECE 0.161, Brier (memory) 0.202, confianza media 0.897 contra exactitud 0.736. El modelo está sobreconfiado fuera de familia.

### 4.9 Abstención (`risk_coverage.csv`, `random_abstention_control.csv`, `nested_selective.json`, `high_confidence_by_family.csv`)

Umbral global sobre confianza, diagnóstico sobre las predicciones fuera de familia:

| Umbral | Cobertura | Exactitud bal. | F1 macro agr. | Cobertura mínima de una celda |
|---:|---:|---:|---:|---:|
| 0.50 | 1.000 | 0.708 | 0.691 | 1.00 |
| 0.70 | 0.871 | 0.725 | 0.731 | 0.37 |
| 0.85 | 0.736 | 0.739 | 0.772 | 0.22 |
| 0.90 | 0.664 | 0.731 | 0.790 | 0.08 |
| 0.95 | 0.559 | 0.731 | 0.803 | 0.00 |

Control: abstención aleatoria a coberturas 0.87, 0.74, 0.56 deja 0.706 a 0.709. Selección anidada del umbral con rejilla ampliada (0.50 a 0.98, sin quedar pegada al borde): cobertura media por familia 72.1 %, exactitud balanceada sobre lo decidido 0.737, F1 macro agrupado 0.780. **La ganancia (+0.03 en exactitud balanceada, +0.08 en F1 agrupado) es real pero pequeña**; parte del aumento de F1 agrupado sale de la caída de cobertura de la clase compute. Familias con confianza ≥ 0.9 y baja exactitud: `npb_ft` 0.32 (66 % de sus intervalos), `phasic` 0.35 (92 %), `gap_pr` 0.59 (68 %), `npb_bt` 0.63.

### 4.10 Otras palancas probadas (todas sin mejora medible)

| Palanca | Resultado | Archivo |
|---|---|---|
| Complejidad de XGBoost (prof. 2 a 6), selección anidada | 0.689 a 0.717 fija; anidada 0.712 (elige distinto en cada familia) | `regularization.json` |
| Umbral de decisión sobre P(memory) elegido por LOFO interno | 0.704 contra 0.708 fijo en 0.5 | `threshold_nested.json` |
| Regresor de log2 OI y decisión contra el ridge del nivel | 0.687 (recall compute 0.48) | `oi_proxy.json` |
| Suavizado causal de P(memory) sobre k intervalos previos | 0.708 (k=1) a 0.687 (k=100); F1 agr. sube 0.69 a 0.71 por sesgo hacia memory | `smoothing_window.csv` |
| Variables de historia reciente (media y desv. móvil de 4 señales, k=5) | 0.720 (+0.012, dentro del ruido; k=20: 0.704) | `temporal_history.csv` |
| Restricción monótona (fallos y stalls suben P(memory)) | 0.698 | matriz |

El suavizado no ayuda porque el error es sesgo por familia, no ruido intervalo a intervalo (autocorrelación lag-1 dentro del bloque 0.376 contra -0.003 barajado, así que el orden de filas sí es temporal).

### 4.11 Desempeño por nivel de frecuencia (`by_frequency_level.csv`)

Exactitud balanceada por nivel: F0 0.759, F1 0.801, F2 0.809, F3 0.776, F4 0.706, F5 0.696, F6 0.678, F7 0.667, F8 0.651, REF 0.803. Baja a frecuencias bajas. El recall compute sube de 0.35 (F0) a 0.75 (F8): el umbral de 0.5 no está calibrado entre niveles.

### 4.12 Latencia de inferencia (`latency_pacca01.log`, `latency_pop-os.json`)

Una fila, `predict_proba`, 1 hilo, 2000 repeticiones tras calentamiento. p50 / p95 / p99 en pacca01 (µs): árbol prof. 1 99 / 102 / 103; logística 266 / 271 / 275; XGBoost 227 / 249 / 382; XGBoost somero 200 / 218 / 357; Random Forest 5016 / 5101 / 5193; Extra Trees 5095 / 5160 / 5226. Incluye sobrecarga de Python; la cifra del daemon C++ debe medirse aparte. La cifra anterior de Extra Trees (27.3 ms p99) correspondía a la configuración larga de Optuna, no a esta.

### 4.13 Calibración en línea (`adaptation_online.csv`, `adaptation_phases.csv`, `adaptation_phases_meta.json`)

**Prueba 1, cota superior (mismo kernel y frecuencia).** Se agregan al entrenamiento los primeros n intervalos etiquetados de cada bloque kernel×frecuencia de la familia retenida y se prueba en las filas restantes (posición ≥ 500 del bloque, mismo conjunto para todo n):

| n por bloque | Filas etiquetadas por familia (media) | Exactitud bal. | F1 macro agr. | Recall compute | Recall memory |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.710 | 0.704 | 0.556 | 0.842 |
| 5 | 88 | 0.765 | 0.893 | 0.845 | 0.937 |
| 20 | 350 | 0.771 | 0.923 | 0.896 | 0.950 |
| 100 | 1 728 | 0.800 | 0.939 | 0.922 | 0.958 |
| 500 | 7 330 | 0.807 | 0.954 | 0.949 | 0.965 |

Esta prueba es optimista: gran parte de la ganancia es aprender la etiqueta de un bloque casi puro.

**Prueba 2, causal y con cambios de fase.** Para cada bloque kernel×frecuencia (longitud ≥ 400 intervalos) solo se etiquetan intervalos en la **primera mitad** y se evalúa en la **segunda mitad**, que incluye cambios de fase posteriores a la calibración. Esquemas: `base` (sin adaptar), `adapt_start` (5 primeros intervalos etiquetados; 0.25 % de las filas de evaluación), `adapt_periodic` (5 de cada 200 intervalos de la primera mitad; 2.6 % de las filas de evaluación, un modelo entrenado con esas filas más las de las otras 23 familias), y dos referencias triviales que repiten la última etiqueta conocida (`sticky_start`, `sticky_periodic`). Se separan **bloques mixtos** (proporción memory entre 20 y 80 %: 50 bloques, 157 777 filas de evaluación, de las cuales 41.7 % tienen etiqueta distinta a la del inicio del bloque) y bloques puros. Tres semillas.

| Alcance | Método | Exactitud bal. | F1 macro agr. | Recall compute | Recall memory |
|---|---|---:|---:|---:|---:|
| Todos | base | 0.715 | 0.711 | 0.575 | 0.838 |
| Todos | adapt_start | 0.771 | 0.899 | 0.856 | 0.938 |
| Todos | adapt_periodic | **0.801** | **0.951** | 0.935 | 0.967 |
| Todos | sticky_start | 0.689 | 0.849 | 0.779 | 0.911 |
| Todos | sticky_periodic | 0.692 | 0.851 | 0.720 | 0.952 |
| Bloques mixtos | base | 0.623 | 0.624 | 0.366 | 0.905 |
| Bloques mixtos | adapt_start | 0.782 | 0.823 | 0.697 | 0.936 |
| Bloques mixtos | adapt_periodic | **0.851** | **0.918** | 0.916 | 0.922 |
| Bloques mixtos | sticky_start | 0.506 | 0.565 | 0.440 | 0.691 |
| Bloques mixtos | sticky_periodic | 0.521 | 0.556 | 0.292 | 0.856 |
| Bloques puros | base | 0.704 | 0.745 | 0.696 | 0.817 |
| Bloques puros | adapt_periodic | 0.780 | 0.965 | 0.946 | 0.981 |
| Bloques puros | sticky_periodic | 0.686 | 0.972 | 0.966 | 0.982 |

Lecturas:
- **En bloques mixtos la ganancia no es persistencia de etiqueta:** repetir la última etiqueta conocida queda en azar (0.51 a 0.52), mientras el modelo adaptado con 2.6 % de intervalos etiquetados llega a 0.851 de exactitud balanceada (0.623 sin adaptar) y a 0.918 de F1 agrupado. El modelo aprende la relación PMU-etiqueta propia de esa carga, y esa relación se sostiene después de los cambios de fase de la segunda mitad.
- **En bloques puros la referencia trivial ya iguala al modelo** (F1 agrupado 0.972 contra 0.965): allí no hay nada que aprender más allá de la etiqueta del bloque.
- **Con solo 5 intervalos al inicio** (`adapt_start`) el F1 agrupado sube de 0.711 a 0.899 en total y de 0.624 a 0.823 en bloques mixtos.
- **Alcance de la validez:** la calibración y la prueba pertenecen al mismo kernel, la misma frecuencia y la misma corrida (con repeticiones concatenadas, así que las fases se repiten); no mide un cambio a un kernel distinto. La métrica "exactitud media por familia" del CSV en bloques mixtos no es interpretable (promedia también familias sin bloques mixtos) y no debe citarse.
- Costo: cada intervalo etiquetado requiere medir *uncore* durante ese intervalo (≥ 10 ms) y FLOPs; 2.6 % de los intervalos es del orden de un intervalo cada 380 ms en una carga de 1 ms por ventana.

## 5. Estrategias recomendadas dentro del alcance de un trabajo de grado

Ordenadas por relación beneficio/costo. Ninguna requiere cientos de familias.

1. **Fijar el reporte de calidad con esta línea base** (sección 1 y 4.1): exactitud balanceada 0.71 (IC95 0.63 a 0.78) con F1 agrupado y recall por clase; mostrar el empate entre variantes como resultado (el modelo no depende de la elección de algoritmo) y el techo de información (k-vecinos 0.68 a 0.70).
2. **Declarar el alcance de generalización con datos, no con adjetivos:** curva de aprendizaje (+0.034 por duplicar el catálogo) y brecha aleatorio contra familia (0.89 contra 0.71). Es una conclusión defendible: el clasificador transfiere a algoritmos nuevos con exactitud balanceada de 0.71 y está limitado por la ambigüedad PMU-etiqueta entre familias, no por el modelo ni por el volumen de intervalos.
3. **Política de decisión con abstención y microcaracterización, evaluada como sistema:** la ganancia de la abstención sola es pequeña (+0.03), pero combinada con microcaracterización uncore de pocos intervalos la cota llega a F1 agregado 0.89 con 5 intervalos por bloque (sección 4.13). La sección 4.13 lo probó de forma causal sobre bloques con cambios de fase reales: 0.85 de exactitud balanceada con 2.6 % de intervalos etiquetados. Falta definir el criterio de disparo real (cuándo pedir la caracterización) y probarlo sobre una carga distinta de la usada para calibrar; requiere el mismo dispositivo y no medir kernels nuevos.
4. **Decisión de variante por parsimonia y costo, no por el máximo:** como interacciones y frecuencia no aportan diferencia medible, la variante con menos dependencias (XGBoost base de 6 variables, o la logística por su latencia estable) es tan defendible como el candidato con 12 variables. La decisión final debe apoyarse en la latencia medida en el daemon y no en una décima de F1.
5. **Ampliar datos solo con criterio:** cada familia nueva debe elegirse por su distancia a los gemelos actuales (sección 4.6), no por volumen. Las tres familias Rodinia de la campaña `pacca_cpu_new_families_20260918_r2` esperan un aumento del orden de +0.006 en conjunto (extrapolación de la curva de aprendizaje) y son casi puramente memory (sección 9), así que no atacan la clase débil (compute).
6. **No perseguir:** más búsqueda de hiperparámetros, más interacciones PMU, suavizado, regresión de OI, ajuste de umbral. Todo probado con resultado nulo.

## 6. Correcciones necesarias en el libro y en el seguimiento

- Reemplazar 0.664 (F1 medio por familia) como resultado principal por la exactitud balanceada por celda con IC95, F1 agrupado y recall por clase (sección 2). Mantener 0.860/0.871 solo como métrica condicionada a cobertura.
- Retirar o matizar las afirmaciones que esta batería no sostiene: "no es ambigüedad cerca del ridge (5.1 %)" (medido: 14 % de los intervalos a menos de un factor 2, 25 % de los errores); "el split aleatorio confirma memorización" (ahora medido: +0.18); "Extra Trees 27 ms de latencia" (5.2 ms en la configuración fija).
- Añadir figuras y tablas de esta batería (seis figuras ya generadas con `docs/libro/scripts/generar_figuras_calidad_cpu.py`: matriz de modelos, protocolos, por familia, curva de aprendizaje, calibración y riesgo-cobertura, margen al ridge).
- Antes de citar el 0.653 en cualquier parte, reejecutar `7455` con `scale_pos_weight=1` confirmado o descartar la cifra.
- Estilo: `04_discusion.tex` contiene 7 usos de `---` y `05_conclusiones.tex` 1; el resto del libro pide no usar el guion largo en prosa.

## 7. Auditoría de contenido perdido en el libro (git HEAD `bb2cbf7` contra el árbol de trabajo)

`docs/libro/secciones/03_resultados.tex` pasó de 395 a 500 líneas con 613 líneas cambiadas (+346, -225), y **nada de esto está confirmado en git**. Se conserva en `HEAD` (recuperable con `git show HEAD:docs/libro/secciones/03_resultados.tex`). Contra `HEAD`, estas secciones y elementos ya no existen en ningún archivo de `secciones/`:

Secciones: Descripción del conjunto de datos experimental; Validación de la actuación DVFS en CPU (queda una línea de resumen); Caracterización de precisión aritmética del catálogo GPU; Estado del control de frecuencia en GPU; Medición directa de bytes movidos (*uncore*) y su validación de esfuerzo y aislamiento; Validación cruzada de los techos Roofline con herramienta independiente; Evolución del catálogo GPU (DGEMM a reemplazo verificable); Barrido de memoria real y calibración de warmup; Techos Roofline de referencia GPU; Análisis exploratorio y clasificador diagnóstico GPU (contrato de features, balanceo, selección de modelo e hiperparámetros, matriz de confusión); Ampliación del catálogo con kernels de acceso irregular.

Etiquetas de tablas y figuras desaparecidas (14): `tab:gpu-memoria-barrido`, `fig:gpu-memoria-barrido`, `tab:gpu-warmup-calibrado`, `tab:gpu-catalogo-final-18`, `fig:gpu-roofline-18kernels`, `fig:gpu-composicion-clase-por-kernel`, `fig:gpu-matriz-correlacion`, `tab:gpu-clasificador-tres-etapas`, `fig:gpu-hiperparam-f1`, `fig:gpu-pareto-calidad-latencia`, `fig:gpu-matriz-confusion`, `tab:gpu-metricas-tradicionales`, `tab:gpu-lofo-por-familia`, `tab:kernels-irregulares-cpu`.

Parte de este material describe el clasificador GPU anterior (334 corridas, 11 familias), que el reanálisis con 394 corridas reemplaza; el resto (catálogo, precisión, uncore, DVFS, warmup, techos, kernels irregulares) es metodología y evidencia de instrumento que no debería haberse eliminado. Lo que ya no está también deja referencias `\ref` colgantes potenciales en discusión y conclusiones. **Acción propuesta:** restaurar esas secciones desde `HEAD` en el orden original y anteponer las de CPU y GPU nuevas, o moverlas a un capítulo de metodología del instrumento. **Estado:** las partes CPU coherentes con lo vigente ya se restauraron (sección 10); lo GPU queda pendiente.

## 8. Limitaciones de este informe

- Una sola fuente de datos (campaña CPU final, 24 familias); las 3 familias nuevas de Rodinia no están incorporadas.
- Muestreo de entrenamiento con tope por celda y 5 semillas; test sobre todos los intervalos de la familia. No hay conjunto externo sellado.
- El IC por bootstrap de familias asume que las 24 familias son exhaustivas e intercambiables; con 24 familias es amplio (±0.07).
- Las etiquetas provienen de FLOPs y bytes uncore de paquete; no se auditó su ruido de medición (se propone como siguiente prueba: repetir intervalos de la misma corrida y medir concordancia de etiqueta).
- Las latencias incluyen la sobrecarga de Python y usan un solo hilo.
- La cota de calibración en línea (4.13) es optimista por localidad temporal.
- `kernel_ref` no distingue repeticiones; el orden de filas se verificó como temporal (autocorrelación) pero no hay identificador de corrida.

## 9. Verificación de las familias nuevas (campaña `pacca_cpu_new_families_20260918_r2`)

Comprobado sobre los `training_cpu_intervals.csv` de las 90 corridas (lectura en `pacca`, sin ejecutar cargas):

- **Nodo y CPU:** `paccaA100` (`node_id = pacca-a100`), Xeon Gold 5315Y, la misma plataforma de la campaña CPU final y de las campañas anteriores (todas las campañas CPU del directorio de resultados tienen ese perfil). No hay mezcla de plataformas.
- **Turbo:** `no_turbo = 1` esperado y observado antes de la calibración y de cada corrida, y `turbo.require_disabled: true` en el manifiesto. El turbo no explica la frecuencia no sostenida de `kmeans` y `srad`.
- **Intervalos por familia:**

| Familia | Intervalos totales | Elegibles (`training_quality_status = ok`) | compute | memory | Niveles con ≥ 1000 intervalos elegibles |
|---|---:|---:|---:|---:|---|
| `rodinia_kmeans_omp` | 150 379 | 78 413 (52.1 %) | 0 | 78 413 | F5, F6, F7, F8, REF |
| `rodinia_particlefilter_omp` | 66 406 | 65 798 (99.1 %) | 1 025 | 64 773 | los 10 niveles |
| `rodinia_srad_omp` | 135 615 | 72 800 (53.7 %) | 0 | 72 800 | F4 (1 004), F5, F6, F7, F8, REF |

- **Confirmación de "2 de 3":** los datos **no** apuntan a exactamente dos familias utilizables. `particlefilter` es la única completa en los 10 niveles. `kmeans` y `srad` son parciales y equivalentes entre sí (solo niveles bajos y nativo; casi todas las ventanas de F0 a F3 rechazadas por frecuencia no verificable: 18, 40, 15 y 23 intervalos elegibles en `kmeans`). Ninguna de las tres está en `feature_contract_source.csv` (24 familias sin ninguna Rodinia nueva) ni existe en el clúster un conjunto consolidado que las incluya. Si dos de ellas ya se incorporaron a un conjunto de entrenamiento, ese conjunto no está en las rutas revisadas, y falta saber cuáles dos son.
- **Sin ejemplos compute:** las tres son casi 100 % memory (`particlefilter`: 1.6 % compute), así que como familias de entrenamiento amplían solo la clase con más ejemplos, y como prueba externa solo pueden evaluar recall de memory (el F1 macro no está definido con una clase).
- `nw_omp` está compilado y calibrado, pero no medido.

## 11. Búsqueda de hiperparámetros anidada sobre la representación final (`nested_optuna.json`)

Pregunta: con el vector de 12 variables y el protocolo por familia, ¿ajustar hiperparámetros mejora el resultado?

Procedimiento: para cada una de las 24 familias externas, muestreador TPE de Optuna sobre el LOFO interno de las 23 familias restantes (15 pruebas para XGBoost, 8 para la logística; 360 y 192 configuraciones en total), reajuste con la mejor configuración y una sola evaluación en la familia retenida.

| Configuración | Exact. bal. | F1 macro agr. | Exactitud |
|---|---:|---:|---:|
| XGBoost fijo | **0.708** | 0.691 | 0.736 |
| XGBoost con TPE anidado | 0.682 | 0.659 | 0.694 |
| Logística fija | 0.688 | 0.763 | 0.792 |
| Logística con TPE anidado | 0.674 | 0.763 | 0.792 |

- La búsqueda **empeora** el resultado: diferencia pareada de -0.025 para XGBoost (IC95 -0.072 a +0.015 por bootstrap de familias; mejora en 10 de 24 familias).
- Causa: el criterio interno no discrimina. El mejor puntaje interno de cada pliegue va de 0.693 a 0.746 (media 0.721) mientras el resultado externo del mismo modelo va de 0.343 a 1.000. Escoger el máximo de 15 propuestas sobre un criterio con ese ruido equivale a ajustarse a los pliegues internos.
- Las configuraciones elegidas no convergen: profundidad 3 a 9, tasa de aprendizaje 0.027 a 0.317, 50 a 275 árboles.
- Coherente con la matriz de modelos (todos empatan) y con el ajuste de complejidad anidado (0.712).

**Estado de la búsqueda global anterior:** el job 7355 (7 modelos, 30 pruebas, representación anterior con `running_ratio`) cerró el 2026-09-18 con extra_trees 0.634 y xgboost 0.600 en la métrica degenerada; su repetición corregida (7390, en `pacca05`) llevaba 111 de 115 estudios al 2026-09-19. Ninguna de las dos decide el candidato, porque usan la representación anterior y la métrica descartada; su conclusión cualitativa (ningún modelo domina) coincide con la de esta batería.

## 12. Cifras del modelo final (`final_model.json`, `final_model_by_family.csv`, `final_model_by_frequency.csv`)

Una sola población (los 1 165 382 intervalos elegibles, cada uno clasificado por el pliegue que no vio su familia) y una sola configuración (XGBoost 12 variables, pesos por celda, umbral 0.85), promediando 5 semillas.

| | Sin abstención | Con umbral 0.85 |
|---|---:|---:|
| Cobertura | 1.000 | 0.726 |
| Exactitud balanceada | 0.709 | 0.723 |
| F1 macro agrupada | 0.709 | 0.772 |
| Exactitud | 0.747 | 0.811 |
| Sensibilidad compute | 0.592 | 0.617 |
| Sensibilidad memory | 0.823 | 0.902 |

Por clase sin abstención: compute P 0.619 / R 0.592 / F1 0.605 (381 990 intervalos); memory P 0.805 / R 0.823 / F1 0.814 (783 392). Matriz: 225 954 / 156 036 / 139 048 / 644 344.

Estas son las cifras que el libro reporta. Sustituyen a las de la muestra capada (cobertura 68.9 % y 69.2 %, F1 por familia 0.664), que mezclaban dos protocolos y una métrica degenerada.

## 10. Cambios aplicados al libro (solo CPU)

**Restauración.** `03_resultados.tex` recuperó de `HEAD`, adaptado a lo vigente, lo que la campaña final CPU sigue respaldando y la discusión sigue referenciando: metodología de actuación DVFS verificada, medición directa de *uncore*, validación de esfuerzo y aislamiento, validación cruzada de techos Roofline con Intel Advisor, y kernels CPU de patrón irregular con el resultado de la campaña de tres familias nuevas. No se restauró lo superado (la campaña de 126 combinaciones con turbo activo, la descripción de 9 cargas CPU) ni lo GPU, que queda para la revisión GPU.

**Reescritura de la sección del clasificador.** La subsección pasó a ser `\subsection{Clasificador de fase en CPU}` con seis apartados: qué mide la evaluación (pliegue, fuga, tope de muestreo, métrica, incertidumbre); variables de entrada y tratamiento de la correlación; selección de modelo e hiperparámetros (incluida la búsqueda TPE); resultado del modelo final; decisión selectiva y costo de inferencia; dónde está el límite; y una síntesis final. Se eliminaron los números de job de Slurm, la tabla de trazabilidad por job, la mención del *proxy* de L2 y de `bytes_moved_window`, el relato de la corrida descartada por doble balanceo y la discusión de la diferencia de 91 intervalos entre dos protocolos. Se sustituyó la métrica de F1 macro media por familia por la exactitud balanceada por celda, definida ahora en la metodología (ecuación `eq:exactitud-balanceada-celda`).

**Coherencia.** Se actualizaron los párrafos de CPU de la discusión y de las conclusiones con las cifras nuevas, y se eliminó el guion largo de toda la prosa de discusión y conclusiones. El libro compila en 144 páginas, sin referencias indefinidas ni etiquetas duplicadas.

**Figuras.** Seis figuras nuevas (`fig_cpu_*_20260919.png`) generadas por `docs/libro/scripts/generar_figuras_modelo_cpu.py`: comparación de modelos con IC, matrices de confusión, resultado por familia, calibración y curva riesgo-cobertura, curva de aprendizaje y margen al *ridge*. Se retiraron las figuras huérfanas de la versión anterior y las funciones que las generaban en `generar_figuras_resultados_cpu.py`, que ahora produce solo las cuatro figuras del conjunto de datos.

## 11. Búsqueda de hiperparámetros anidada sobre la representación final (`nested_optuna.json`)

Pregunta: con el vector de 12 variables y el protocolo por familia, ¿ajustar hiperparámetros mejora el resultado?

Procedimiento: para cada una de las 24 familias externas, muestreador TPE de Optuna sobre el LOFO interno de las 23 familias restantes (15 pruebas para XGBoost, 8 para la logística; 360 y 192 configuraciones en total), reajuste con la mejor configuración y una sola evaluación en la familia retenida.

| Configuración | Exact. bal. | F1 macro agr. | Exactitud |
|---|---:|---:|---:|
| XGBoost fijo | **0.708** | 0.691 | 0.736 |
| XGBoost con TPE anidado | 0.682 | 0.659 | 0.694 |
| Logística fija | 0.688 | 0.763 | 0.792 |
| Logística con TPE anidado | 0.674 | 0.763 | 0.792 |

- La búsqueda **empeora** el resultado: diferencia pareada de -0.025 para XGBoost (IC95 -0.072 a +0.015 por bootstrap de familias; mejora en 10 de 24 familias).
- Causa: el criterio interno no discrimina. El mejor puntaje interno de cada pliegue va de 0.693 a 0.746 (media 0.721) mientras el resultado externo del mismo modelo va de 0.343 a 1.000. Escoger el máximo de 15 propuestas sobre un criterio con ese ruido equivale a ajustarse a los pliegues internos.
- Las configuraciones elegidas no convergen: profundidad 3 a 9, tasa de aprendizaje 0.027 a 0.317, 50 a 275 árboles.
- Coherente con la matriz de modelos (todos empatan) y con el ajuste de complejidad anidado (0.712).

**Estado de la búsqueda global anterior:** el job 7355 (7 modelos, 30 pruebas, representación anterior con `running_ratio`) cerró el 2026-09-18 con extra_trees 0.634 y xgboost 0.600 en la métrica degenerada; su repetición corregida (7390, en `pacca05`) llevaba 111 de 115 estudios al 2026-09-19. Ninguna de las dos decide el candidato, porque usan la representación anterior y la métrica descartada; su conclusión cualitativa (ningún modelo domina) coincide con la de esta batería.

## 12. Cifras del modelo final (`final_model.json`, `final_model_by_family.csv`, `final_model_by_frequency.csv`)

Una sola población (los 1 165 382 intervalos elegibles, cada uno clasificado por el pliegue que no vio su familia) y una sola configuración (XGBoost 12 variables, pesos por celda, umbral 0.85), promediando 5 semillas.

| | Sin abstención | Con umbral 0.85 |
|---|---:|---:|
| Cobertura | 1.000 | 0.726 |
| Exactitud balanceada | 0.709 | 0.723 |
| F1 macro agrupada | 0.709 | 0.772 |
| Exactitud | 0.747 | 0.811 |
| Sensibilidad compute | 0.592 | 0.617 |
| Sensibilidad memory | 0.823 | 0.902 |

Por clase sin abstención: compute P 0.619 / R 0.592 / F1 0.605 (381 990 intervalos); memory P 0.805 / R 0.823 / F1 0.814 (783 392). Matriz: 225 954 / 156 036 / 139 048 / 644 344.

Estas son las cifras que el libro reporta. Sustituyen a las de la muestra capada (cobertura 68.9 % y 69.2 %, F1 por familia 0.664), que mezclaban dos protocolos y una métrica degenerada.

## 10. Cambios aplicados al libro en esta sesión (solo CPU)

`docs/libro/secciones/03_resultados.tex`, sin tocar GPU: se restauró desde `HEAD`, adaptándolo a lo vigente, lo que la campaña final CPU sigue respaldando y el capítulo de discusión sigue referenciando: metodología de actuación DVFS verificada (tres niveles, hermanos SMT, 0.995 contra 3.78); medición directa de *uncore* (`sec:resultados-uncore`); validación de esfuerzo y aislamiento (`sec:resultados-interferencia-uncore`); validación cruzada de techos Roofline con Intel Advisor (`sec:resultados-ppico`); y kernels CPU de patrón irregular (`sec:resultados-kernels-irregulares`, tabla `tab:kernels-irregulares-cpu`), ahora con el resultado de la campaña de tres familias nuevas (tabla `tab:cpu-familias-nuevas`). No se restauró lo superado: la campaña CPU de 126 combinaciones con turbo activo y su conclusión de "no son frecuencias físicas distintas", la descripción de 9 cargas CPU (sustituida por la tabla de cobertura de 43 configuraciones y 24 familias), y todo lo GPU (precisión, control de frecuencia, catálogo, warmup, clasificador anterior), que queda para la revisión GPU. Además se eliminó una etiqueta duplicada (`sec:resultados-kernels-irregulares`) que estaba puesta sobre la sección GPU. El libro compila (136 páginas), sin referencias indefinidas ni etiquetas duplicadas. El material de esta batería de calidad aún no está en el libro.

## 13. Métricas de evaluación: definición y comparación (`metrics_suite.json`, `metrics_cells.csv`)

Etapa `metrics_suite` de `cpu_quality_report.py`, modelo final, cinco semillas, evaluación sobre todos los intervalos elegibles.

| Métrica | Valor | Desv. entre semillas | IC95 por remuestreo de familias |
|---|---:|---:|---|
| Exactitud | 0.747 | 0.021 | 0.605 a 0.835 |
| Precisión / sensibilidad / F1 compute | 0.620 / 0.592 / 0.605 | 0.035 / 0.040 / 0.033 | |
| Precisión / sensibilidad / F1 memory | 0.805 / 0.823 / 0.814 | 0.016 / 0.024 / 0.016 | |
| F1 macro agrupada | 0.709 | 0.024 | |
| Exactitud balanceada clásica | 0.707 | 0.024 | 0.562 a 0.831 |
| AUC-ROC agrupada | 0.773 | 0.019 | |
| Coeficiente de Matthews (= kappa) | 0.419 | 0.047 | |
| **Exactitud balanceada por celda** | **0.709** | **0.004** | 0.633 a 0.780 |

- **Corrección de una cifra dada antes:** las cifras 0.686 (exactitud balanceada clásica), 0.745 (AUC) y 0.385 (MCC) de un cálculo rápido anterior correspondían a una sola semilla de muestreo. Los valores con cinco semillas son los de la tabla; la variabilidad entre semillas de las métricas por volumen es alta y es parte del argumento.
- **Razón empírica para preferir la métrica por celda:** desviación entre semillas 0.004 frente a 0.019 a 0.047 de las métricas por volumen; amplitud del IC95 0.15 frente a 0.23 y 0.27.
- **Sensibilidad por clase:** por volumen 0.592 (compute) y 0.823 (memory); por celdas 0.667 y 0.747. `phasic` aporta 89 136 intervalos compute con 3.7 % de acierto y explica buena parte de la diferencia.
- **Límite:** 4 de las 44 celdas tienen menos de 100 intervalos evaluados (`cpu_lulesh` compute 5, `dual_gemm` memory 9, `cpu_hpcg` compute 15, `rodinia_lavamd` memory 15). Suman 44 intervalos y pesan 9.1 % de la métrica; excluirlas la subiría de 0.709 a 0.757. Se conservan para no seleccionar datos por resultado.
- **Libro:** nuevo apartado "Métricas de evaluación" en la sección de resultados (definición de celda, figura de las 44 celdas, tabla de definiciones, ejemplo numérico con cuatro celdas reales, tabla de valores, lectura conjunta y límite). El libro pasó a 149 páginas.

## 14. Cumplimiento del plan detallado en CPU, Fases 1 y 2 (auditoría del 2026-09-19)

**Fase 1 (objetivo 1), CPU.**
- Cumplido: campaña final de 43 configuraciones (24 familias), REF más nueve niveles fijos, tres repeticiones, 1290 corridas aceptadas; frecuencia observada validada por ventana con tolerancia de 5 % y turbo desactivado; etiqueta Roofline solo con FLOPs y bytes de *uncore* (nunca proxy); catálogo versionado con checksum; matriz Pearson/Spearman y VIF documentadas con las columnas descartadas; RAPL y tiempo de cada corrida capturados (`rapl_pkg_total_delta_uj`, `telemetry_elapsed_ns_mean`).
- Parcial: representación de clases (plan §2.1.1 punto 5, 5 a 6 familias por clase): 3 familias casi puras compute (`dgemm`, `dual_gemm`, `rodinia_lavamd`) y 10 con mayoría compute; 9 casi puras memory y 14 con mayoría memory. Las tres familias nuevas (`particlefilter`, `kmeans`, `srad`) son casi 100 % memory y `nw` no está medido.
- Sin cerrar: análisis energético y de EDP de la campaña (capturado, no analizado ni reportado); decisión sobre `power_w`, que el plan §2.5 lista como variable de entrada CPU y el modelo final no usa, sin justificación registrada.

**Fase 2 (objetivo 2), CPU.**
- Cumplido: comparación de mayoritaria, árbol, regresión logística, Random Forest, Extra Trees y XGBoost; LOFO por familia; búsqueda de hiperparámetros anidada (TPE); exactitud, F1 por clase y matriz de confusión; latencia p95 y p99 en el nodo de destino; modelo serializado (`.joblib` y metadatos, en el clúster); XGBoost añadido; chequeo de fuga que aborta si entra una columna de la etiqueta.
- Variante con intensidad operacional como entrada (plan §3.3 punto 2, límite superior de desempeño): medida ahora, exactitud balanceada por celda **0.9955** frente a 0.709 sin ella. Confirma que incluirla equivale a leer la etiqueta; falta añadirla al libro como diagnóstico.
- Sin cerrar: tabla clase a frecuencia (plan §3.4 y §3.5) para CPU. El derivador existe (`fase3_daemon/policy/derive_policy_table.py`) y `common/stats.py` tiene la prueba pareada, pero no hay ninguna tabla generada en los resultados; los datos necesarios (energía y tiempo por corrida, 10 niveles) ya están en la campaña. Función de selección que combine error y latencia (§3.3 punto 5): no formalizada. Prueba externa con familias selladas: no existe para CPU.

**Incoherencias documentales detectadas.**
- `05_conclusiones.tex` afirma que el control del turbo sigue bloqueado y pide repetir la campaña; la campaña final exige turbo desactivado y registra `no_turbo = 1` antes de cada corrida, con niveles físicos distintos de 3.2 a 0.8 GHz.
- La tabla resumen del seguimiento no incluye las entradas F1-CPU-004 a F1-CPU-011 ni F2-CPU-002 y 003, y `F2-CPU-001` conserva cifras (0.664, 68.9 %) sustituidas por las de la sección 12.
- La lista de verificación del plan (§8) está sin marcar.
- El trabajo `7390` (búsqueda global de hiperparámetros en `pacca05`, más de dos días) usa la representación anterior y la métrica descartada; su resultado no decide nada.

## 15. Tabla de política CPU (clase -> frecuencia), 2026-09-19

Script: `fase2_clasificador/analysis/cpu_policy_table.py`. Datos: `docs/libro/datos/cpu_calidad_20260918/politica/`.

Método: EDP por corrida completa = (energía RAPL paquete + DRAM) x tiempo de la corrida. Todas las corridas de un kernel hacen el mismo trabajo en todos los niveles (mismos argumentos; verificado en dual_gemm N2048: 41 iteraciones en REF, F0, F4 y F8), así que el EDP es comparable. El EDP por ventana de 1 ms del derivador de Fase 3 no lo es (con ventana fija solo mide potencia) y no se usó. Clase de un kernel: la mayoritaria entre sus ventanas etiquetadas. Mediana de 3 repeticiones por kernel y nivel, prueba de Wilcoxon pareada por kernel contra REF. Se excluyen las sondas de calibración (stream_official, ert_probe). 16 kernels compute, 27 memory.

Resultado: **no actuar en ambas clases.** Ningún nivel fijo mejora el EDP agregado sobre REF.
- Compute: F0 0.3 %, F1 -2.9 %, F8 -443 % (EDP crece hasta 8x). Sin diferencia significativa.
- Memory: F1 -5.4 %, F4 -44 %; de F2 a F8 la prueba es significativa pero en sentido contrario (empeora).
- Nivel óptimo por kernel: F0 para 13 de 16 compute y 15 de 27 memory. Solo los kernels tipo STREAM (init3, first_sum, stream_add/triad, daxpy, tridiag_elim, jacobi_1d, hpcg) ganan 4 a 12 % en F1 a F4. Son casi puros memory, y el clasificador no distingue "memory de streaming" de "memory con reutilización de caché".
- **Corrección:** los tres `phasic` (F8 = 0.83 del EDP de REF) NO son evidencia a favor de DVFS. Su tiempo es casi constante entre niveles (alfa 0.005, 20.6 s en todos), porque son cargas de duración fija y no de trabajo fijo: el "ahorro" es solo menor potencia durante el mismo tiempo. Deben excluirse del análisis de EDP por corrida (y de la tabla de política).

Lectura: con RAPL de paquete completo (incluye potencia estática y otros núcleos) y trabajo fijo, la potencia dinámica ahorrada al bajar la frecuencia no compensa el alargamiento del tiempo. Es coherente con el hallazgo previo de alfa (ARC-176): pocas cargas están en la zona donde DVFS ayuda. El plan acepta "no actuar" como resultado legítimo (§3.4/§3.5).

## 16. Candidatas compute-bound nuevas (preparadas, sin medir aún)

Motivo: solo 3 familias casi puras compute. `ep` (NPB) descartada antes (cuenta números aleatorios, no FLOPs). `particlefilter`, `kmeans`, `srad` son ~100 % memory. Se prepararon 9 adaptadores RAJAPerf Base_OpenMP (MASS3DPA, DIFFUSION3DPA, CONVECTION3DPA, EDGE3D, TRAP_INT, PI_REDUCE, MAT_MAT_SHARED, LTIMES, FIR) en `scripts/pacca/rajaperf_compute_adapters/`, ya instalados en `~/hyperion-kernels/bin` de pacca (checksum idéntico al local), con `--repfact` ajustado por tiempo nativo (1 a 8 s). Todos corren y pasan su verificación. Entradas añadidas al catálogo (`cpu_rajaperf_*`, hint "intermedio"). Tamizaje: `scripts/pacca/final_campaign/cpu_compute_screen_20260919.yaml` (REF, F0, F8, 1 repetición). Pendiente: sincronizar el catálogo al clúster (git) y lanzarlo.

### 15.1 Sensibilidad al reloj (alfa) por kernel

Alfa = -pendiente de log(tiempo) contra log(frecuencia), sobre los 9 niveles fijos. Potencia media de paquete: de 105-138 W en F0 a 80-100 W en F8, es decir, 4x menos reloj ahorra solo 15-25 % de potencia (la potencia estática domina). Consecuencia: bajar reloj solo compensa con alfa <= ~0.22 (umbral de ARC-176 ya medido). Solo cinco kernels caen ahí: stream_mul 0.178, stream_triad 0.220, stream_add 0.220, first_sum 0.222, tridiag_elim 0.223; son los que ganan 4-12 % en F1-F3. Los demás memory tienen alfa 0.24-0.90; los compute 0.73-1.0. `ptrchase` (latencia pura de DRAM, alfa esperado cercano a 0) existe en el catálogo pero no está en la campaña final.
