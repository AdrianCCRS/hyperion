# Resultados corregidos de R2 — política desplegable y agregación por pliegue

**Fecha:** 2026-08-30
**Estado:** resultado exploratorio reproducible; pendiente de evaluación
confirmatoria con tamaños nuevos
**Datos:** 68 `config_id`; huellas SHA-256 registradas en los resúmenes JSON
de `resultados_selector_r2_20260830/`

## 1. Motivo de la corrección

El primer reporte eliminaba `fold` de la clave de comparación y elegía con
`idxmin` la mejor fila de modelo y de baseline dentro de cada combinación
`(régimen, estado, K)`. Eso permitía seleccionar familias y pliegues después
de observar el conjunto de prueba, y podía enfrentar filas procedentes de
pliegues distintos. Los conteos originales `2/48` y `4/48` no son resultados
válidos de una política desplegable y quedan supersedidos.

La corrección aplica cuatro reglas:

1. se congela un único modelo directo: `RandomForest + sondeo`;
2. se congela un único modelo estructurado: `ElasticNet` sin sondeo;
3. la baseline pertinente se congela por `(régimen, estado, K)` usando el
   promedio de los pliegues exploratorios, nunca por pliegue individual;
4. los pliegues de interpolación, que son disjuntos, se agregan sumando EDP;
   `extrapolation_top1` y `extrapolation_top2` permanecen separados porque
   sus conjuntos de prueba se solapan.

Esta selección sigue siendo desarrollo sobre los datos exploratorios. Su
validez externa depende de aplicar las políticas ya congeladas una sola vez
sobre los tamaños confirmatorios.

## 2. Resultado corregido

Cada alcance contiene 24 rebanadas: tres estados por ocho valores de `K`.

| Alcance | Directo supera baseline | Mejor directo/estructurado supera baseline |
|---|---:|---:|
| Interpolación agregada | 0/24 | 2/24 |
| Extrapolación `top1` | 0/24 | 0/24 |
| Extrapolación `top2` | 2/24 | 2/24 |

Las cuatro ventajas de la comparación de tres vías son:

| Alcance | Estado | K | Formulación | Mejora sobre baseline |
|---|---|---:|---|---:|
| Interpolación | `gpu_ready` | 1 | estructurada | 9,64 % |
| Interpolación | `gpu_ready` | 2 | estructurada | 4,85 % |
| Extrapolación `top2` | `cpu_ready` | 3 | estructurada | 38,02 % |
| Extrapolación `top2` | `none_ready` | 3 | estructurada | 36,16 % |

No hay ninguna victoria en `extrapolation_top1`, el extremo superior más
cercano al escenario confirmatorio. Las dos victorias de `top2` tampoco son
estables al variar `K`: en `K=5` la mejora estructurada cae a +0,71 % en
`cpu_ready` y -1,51 % en `none_ready`, ambas por debajo del piso de ruido; en
`K=10` las pérdidas son -47,03 % y -48,76 %, respectivamente.

## 3. Interpretación

La formulación estructurada mejora materialmente el fallo del target directo,
pero no lo elimina. En `extrapolation_top2`, el directo llega a pérdidas de
-364 % y -367 % en `K=10`; el estructurado las reduce a aproximadamente -47 %
y -49 %. Incorporar la composición física es útil, pero no autoriza afirmar
que el modelo haya aprendido una curva estable del horizonte.

La capa 1 de Ridge obtiene R² agrupado entre 0,941 y 0,983 para las cuatro
primitivas calientes. Es una señal predictiva real, pero no suficiente por sí
sola: un R² alto sobre costos que abarcan varios órdenes de magnitud puede
coexistir con errores de decisión importantes cerca de un cruce CPU/GPU.

El modelo estructurado seleccionado ocupa 8.179 bytes y su latencia medida es
18,08 ms en p50 y 19,15 ms en p95. Es 118 veces menor que el modelo directo
de 968.495 bytes, pero su latencia sigue siendo demasiado alta para ejecutarla
antes de cada despacho corto. R3 deberá almacenar decisiones y amortizar la
inferencia sobre el horizonte, además de medir el costo en el agente real.

## 4. Conclusión de R2

R2 no justifica adoptar todavía un modelo de dispositivo como reemplazo
general de las baselines. Sí justifica conservar la formulación estructurada
como candidata para la prueba confirmatoria, porque concentra sus ventajas en
las fronteras de cruce y reduce el error extremo del target directo.

La conclusión principal es una política híbrida: baseline por defecto y
modelo únicamente en las rebanadas precongeladas donde la ventaja exploratoria
supera el piso de ruido. La evaluación confirmatoria decidirá si esas cuatro
compuertas se conservan o si todo el eje se reduce a la política simple.

R2 no evalúa selección de frecuencia. El headroom de DVFS observado en R1,
especialmente en `gpu_ready`, permanece disponible y debe abordarse en R3/R4.
No evaluar esa capa incumpliría el núcleo del proyecto; obtener allí un
resultado negativo después de medir incertidumbre y actuación sí sería un
resultado válido.

## 5. Artefactos

- `resultados_selector_r2_20260830/direct/`: resultados del target directo,
  comparación por pliegue, agregación válida y resumen JSON.
- `resultados_selector_r2_20260830/structured/`: primitivas, resultados
  estructurados, comparación de tres vías y resumen JSON.
- Los dos `*_summary.json` registran selección, latencia, tamaño, semilla,
  rejilla de `K` y huellas SHA-256 de las entradas.
