# Protocolo congelado para la evaluación confirmatoria por tamaño

**Fecha de congelamiento:** 2026-08-30
**Estado:** congelado — modificable solo mediante enmienda fechada (§11)
**Ámbito:** §8.2 del `plan_reformulacion_selector_tamanos_20260830.md`
**Datos exploratorios permitidos:** exclusivamente
`~/hyperion-results/analysis/selector_final_20260830/` (8 160 corridas
aceptadas, 68 `config_id`)
**Datos confirmatorios reservados:** campañas `pacca_dual_cpu_big_ref_20260830`
y `pacca_dual_gpu_big_ref_20260830` (9 `config_id` nuevos: gemm, cholesky y
fft en N = 8192, 12288, 16384)

Este documento existe para impedir un ajuste retrospectivo. Todo lo que se
fija aquí quedó decidido **antes** de observar cualquier medición de las
campañas `*_big_ref_*`. Un resultado confirmatorio solo es válido si se
produce ejecutando exactamente lo que sigue.

---

## 1. Formulación del target

> **SUPERSEDIDA PARCIALMENTE por la enmienda 2026-08-30-B (§13).** La
> cantidad de interés (`y`, o su generalización a horizonte `K` de §12) no
> cambia. Lo que cambia es cómo se produce: en vez de un regresor entrenado
> directamente sobre `y`, se predicen cuatro primitivas de costo y `y` se
> deriva de ellas con la fórmula ya congelada. Esta sección describe la
> formulación directa, que sigue siendo la comparación primaria de R2 (no se
> retira, se compara contra la estructurada).

Unidad experimental: **`config_id` = operación × tamaño**. Nunca la fila
candidata, nunca la ventana de telemetría, nunca la repetición.

Target primario, por `config_id` y estado de recurso:

```text
y = log( EDP_GPU_REF / EDP_CPU_REF )
```

El EDP de cada dispositivo se lee en la región que ese dispositivo ve en el
estado correspondiente:

| estado        | región CPU | región GPU |
|---------------|-----------|-----------|
| `none_ready`  | `cold`    | `cold`    |
| `cpu_ready`   | `warm`    | `cold`    |
| `gpu_ready`   | `cold`    | `warm`    |

Interpretación: `y < 0` favorece GPU, `y > 0` favorece CPU, `|y|` pequeño es
zona de abstención (§7). La etiqueta derivada es
`device_label = "gpu" if y < 0 else "cpu"`.

Se congela la **regresión** como formulación primaria porque conserva la
magnitud del error; la clasificación binaria se reporta como contraste
secundario y no puede sustituir a la regresión si esta resulta peor.

Implementación de referencia: `classifier.selector.compact.build_compact_dataset`.

## 2. Estado de aprendizaje por estado de recurso

> **SUPERSEDIDO PARCIALMENTE por la enmienda 2026-08-30-A (§12).** Las reglas
> por estado de esta sección son correctas **únicamente para `K = 1`** (un solo
> despacho restante). La política del agente se formula sobre
> `resource_state × K`; ver §12. Las listas de "óptimo en 68/68" y "56/56 12/12"
> se conservan aquí sin modificación porque describen correctamente el caso
> `K = 1` y porque reescribirlas destruiría el registro de lo que se congeló.

Fijado con los datos exploratorios, antes de ver los confirmatorios:

- `none_ready`: CPU óptima en 68/68 → **no se entrena modelo**. Regla fija:
  CPU a REF. El conjunto confirmatorio puede refutar esta regla; si aparece
  un solo cruce a GPU, se reporta como hallazgo y no se corrige la regla
  retroactivamente.
- `cpu_ready`: CPU óptima en 68/68 → **no se entrena modelo**. Regla fija:
  permanecer en CPU (§6.3 del plan).
- `gpu_ready`: 56 GPU / 12 CPU → **única tarea de aprendizaje**. Todo lo que
  sigue sobre modelos se refiere a este estado.

Entrenar un modelo para reproducir una constante mide ruido, no capacidad
predictiva; por eso los dos primeros estados quedan excluidos por protocolo y
no por conveniencia posterior.

## 3. Lista exacta de características

### 3.1 Modelo estático (obligatorio)

Congeladas, en este orden:

1. `operation` (categórica, one-hot sobre las seis operaciones conocidas)
2. `size`
3. `log10_n`
4. `flops_per_dispatch_analytic`
5. `log10_flops_per_dispatch`
6. `logical_bytes_per_dispatch`
7. `log10_logical_bytes`
8. `arithmetic_intensity_analytic`
9. `resource_state` (categórica)

### 3.2 Modelo con sondeo (contraste de H2)

Las nueve anteriores más, del dispositivo que produjo el sondeo y de una
**única** ejecución (la primera repetición, nunca el promedio de las tres):

- `probe_time_per_dispatch_s`, `probe_energy_per_dispatch_j`,
  `probe_avg_power_w`, `probe_region_to_sampling_ratio`
- CPU: `probe_ipc`, `probe_mpki`, `probe_llc_miss_rate`,
  `probe_stall_backend_ratio`, `probe_ips`, `probe_freq_khz_observed`,
  `probe_running_ratio`
- GPU: `probe_gpu_power_mw`, `probe_gpu_util_pct`, `probe_gpu_mem_util_pct`,
  `probe_gpu_sm_clock_mhz`, `probe_gpu_temperature_c`
- el indicador `*_missing` de cada una de las anteriores
- `probe_device`

La ausencia estructural (no hay métricas CPU en un sondeo GPU y viceversa, y
`none_ready` no tiene sondeo alguno) se representa con el indicador
`*_missing`, nunca con una imputación silenciosa.

### 3.3 Prohibiciones de fuga (§7.3)

Queda **prohibido** usar como entrada: EDP de cualquier acción candidata,
acción ganadora, margen contra el segundo lugar, `is_optimal`,
`optimum_stability`, identificadores de corrida o repetición, cualquier
estadístico construido con las tres repeticiones, y cualquier información
proveniente de los tamaños reservados.

El chequeo es automático y bloqueante:
`classifier.selector.compact.assert_no_leakage`. Ninguna evaluación cuyo
conjunto de características no pase ese chequeo puede reportarse.

## 4. Familias de modelos candidatas

> **AMPLIADA por la enmienda 2026-08-30-B (§13.4).** Las cinco familias de
> abajo se conservan, ahora aplicables tanto al target directo (§1) como a
> las primitivas de costo estructuradas (§13). No se agrega XGBoost ni
> ninguna familia de mayor capacidad -- ver §13.4 para la justificación.

Deliberadamente simples: con 68 `config_id` y una sola tarea real
(`gpu_ready`), un modelo de alta capacidad ajustaría ruido.

1. Ridge
2. ElasticNet
3. regresión de Huber
4. árbol de regresión con `max_depth <= 3`
5. RandomForest regressor (`n_estimators = 200`, `max_depth <= 5`)

No se añadirá ninguna familia después de ver los datos confirmatorios. No se
usará gradient boosting ni ninguna familia ausente de esta lista.

## 5. Procedimiento de selección

1. Las particiones son **agrupadas por `config_id`** y estratificadas dentro
   de cada operación. Las tres filas de estado de un `config_id` viajan
   siempre juntas: comparten las mismas mediciones físicas.
2. Dos regímenes, ambos reportados
   (`classifier.selector.sizes.interpolation_folds` /
   `extrapolation_folds`):
   - **interpolación:** tamaños internos retenidos, con un tamaño menor y uno
     mayor de la misma operación siempre presentes en entrenamiento;
   - **extrapolación:** entrenamiento solo con los tamaños menores, prueba en
     el extremo superior.
3. Los hiperparámetros se eligen **dentro** del conjunto de entrenamiento de
   cada pliegue, mediante validación cruzada interna también agrupada por
   `config_id`. Ningún dato de prueba interviene en la selección.
4. El modelo final es uno solo: la familia con mejor `edp_sum_ratio_vs_oracle`
   promedio en **extrapolación** sobre los datos exploratorios. Esa elección
   se hace y se registra antes de tocar los datos confirmatorios.
5. Semilla fija `20260830`.

## 6. Baselines obligatorias

Las ocho de §9 del plan, implementadas en
`classifier.selector.sizes.BASELINES`, todas con parámetros ajustados
únicamente en entrenamiento:

1. `always_cpu_ref`
2. `always_gpu_ref`
3. `stay_on_ready_device`
4. `best_constant_device_train`
5. `size_threshold_train`
6. `intensity_threshold_train`
7. `operation_crossover_table_train`
8. `oracle`

Para la frecuencia: siempre REF; mejor frecuencia constante por dispositivo
estimada en entrenamiento; y no actuar si la mejora esperada no supera el
costo de conmutación.

**Regla bloqueante.** Si el modelo no supera a la mejor baseline pertinente
en la evaluación externa, se conserva la baseline y se reporta que el
Aprendizaje Automático no aportó valor adicional. Esta regla no admite
excepción por "cercanía": superar significa mejorar `edp_sum_ratio_vs_oracle`
por encima del piso de ruido de §7.

## 7. Piso de ruido y regla de abstención

Piso de ruido de medición, calibrado sobre los datos exploratorios como CV
mediano del EDP entre repeticiones de la misma acción:

- global: **3,11 %**
- por dispositivo: CPU 2,57 %, GPU 3,16 %
- **por región: `cold` 5,76 %, `warm` 1,80 %**

La descomposición por región se congela junto con el valor global porque la
región fría es aproximadamente tres veces más ruidosa que la caliente, y las
decisiones de `none_ready` descansan enteramente sobre mediciones frías.

**Regla de abstención (congelada).** El selector se abstiene de migrar de
dispositivo cuando

```text
|y_predicho| < log(1 + 0.0311)   ~= 0.0306
```

Al abstenerse aplica la política segura del estado: permanecer en el
dispositivo preparado, y CPU a REF en `none_ready`. La abstención se reporta
como cobertura y calidad, **no** se contabiliza como error del selector.

Para la capa DVFS, la actuación se habilita únicamente si se cumplen las tres
condiciones de §6.5 del plan: la ventaja supera la incertidumbre combinada,
la ganancia esperada supera el costo de actuación, y el modelo supera una
política REF en validación externa. En cualquier otro caso se usa REF.

## 8. Métricas que se reportarán

Del selector de dispositivo (§10.2), calculadas por
`classifier.selector.sizes.evaluate_devices`:

matriz de confusión (`tp_gpu`, `fp_gpu`, `fn_gpu`, `tn_cpu`);
`balanced_accuracy`; `mcc`; `precision_migrate_gpu` y `recall_migrate_gpu`;
error de la razón logarítmica de EDP; regret medio, mediano, p95 y máximo
(`regret_ratio_*`); `oracle_savings_captured_pct`; desglose por operación y
por régimen de tamaños; cobertura y calidad de la abstención.

Se reportan **las dos agregaciones de EDP** de §10.4 por separado y sin
mezclarlas: la media de razones por despacho (`regret_ratio_mean`) y la razón
de sumas (`edp_sum_ratio_vs_oracle`). En los datos exploratorios ambas ya
discrepan de forma material — una baseline con `edp_sum_ratio` de 1,003 tiene
`regret_ratio_mean` de 1,197 — de modo que reportar solo una sería
seleccionar la métrica más favorable.

## 9. Qué constituye un resultado confirmatorio

Los 9 `config_id` nuevos se evalúan **una sola vez**, con el modelo y los
umbrales ya fijados. Se reportará, en este orden y sin omitir ninguno:

1. la predicción de dispositivo por `config_id`, contra el resultado medido;
2. si `K_break_even` cae dentro de la banda extrapolada desde los tamaños
   conocidos;
3. aciertos, fallos y cambio de dominio, incluidos los fallos.

No se ajustarán hiperparámetros ni umbrales con estos datos. Después de
publicar el resultado confirmatorio se podrá reentrenar con todo el conjunto,
manteniendo el resultado original reportado por separado.

## 10. Lo que este protocolo NO afirma

- No se afirmará generalización a una séptima operación desconocida.
- No se afirmará generalización a otra plataforma.
- No se presentará el EDP por despacho como EDP de una aplicación completa.
- No se presentará una ganancia como neta antes de medir inferencia,
  actuación y transferencias.
- `K_break_even` no se presentará como entero exacto: se reporta siempre con
  su banda.

## 11. Enmiendas

Cualquier cambio a este documento posterior al 2026-08-30 debe registrarse
aquí como una entrada fechada que indique qué cambió, por qué, y si en ese
momento ya se habían observado datos confirmatorios. Un cambio no registrado
invalida el carácter confirmatorio de la evaluación.

| fecha | cambio | ¿datos confirmatorios ya observados? |
|-------|--------|--------------------------------------|
| 2026-08-30 | versión inicial congelada | no |
| 2026-08-30 | enmienda **2026-08-30-A**: política sobre `resource_state × K` (§12); §2 supersedida parcialmente | **no** — verificado: jobs 6763/6764 en estado `PENDING`, los directorios `pacca_dual_{cpu,gpu}_big_ref_20260830` no existían al redactar |
| 2026-08-30 | enmienda **2026-08-30-B**: target estructurado en tres capas (§13); §1 y §4 supersedidas parcialmente | **no** — verificado: jobs 6763/6764 en `PENDING`, sin directorios de salida al redactar |
| 2026-08-30 | enmienda **2026-08-30-C**: corrección de selección y agregación de R2 (§14) | **no** — verificado: jobs 6763/6764 en `PENDING`, sin directorios de salida al redactar |
| 2026-08-31 | enmienda **2026-08-31-A**: target relativo a REF y profundidad libre para R3-A (§15) | **no** — verificado por `squeue`: jobs 6763/6764 siguen en `PENDING` (razones "Nodes required... DOWN/DRAINED" y "Resources"), sin directorios de salida al redactar |

---

## 12. Enmienda 2026-08-30-A — política sobre `resource_state × K`

**Fecha:** 2026-08-30
**Datos confirmatorios observados al redactar:** ninguno (verificado por
`squeue` y por ausencia de los directorios de salida)
**Motivo:** la versión inicial formuló el horizonte `K` únicamente para
`none_ready` (§6.2 del plan). Para `cpu_ready` y `gpu_ready` fijó reglas
derivadas del despacho siguiente, es decir `K = 1`, y las presentó como la
política del estado. Es un error de formulación, no de medición: las
mediciones de §2 son correctas para `K = 1`.

### 12.1 Formulación corregida

La decisión de dispositivo del agente es, en los **tres** estados:

```text
decision(estado, K) = argmin_d  EDP_total(d, K | estado)
```

donde el término de arranque se paga solo cuando el dispositivo destino no
está inicializado en ese estado:

```text
d ya inicializado:   E_total = K * E_warm(d)              T_total = K * T_warm(d)
d no inicializado:   E_total = E_cold(d) + (K-1)*E_warm(d)
                     T_total = T_cold(d) + (K-1)*T_warm(d)
EDP_total(d,K)     = E_total(d,K) * T_total(d,K)
```

`K` es una **entrada conocida**, suministrada por la aplicación o por el
escenario experimental (§6.2 y §16.2 del plan). Su estimación en línea es la
Fase E2 y queda fuera del núcleo obligatorio. Todos los resultados de
dispositivo se reportarán **como función de `K`**, no para un `K` implícito.

### 12.2 Evidencia que motiva la enmienda

Calculada sobre los datos exploratorios (8 160 corridas, 68 `config_id`), con
dos implementaciones independientes que coinciden configuración a
configuración:

| desde | configuraciones que cambian de dispositivo en algún `K` |
|---|---|
| `cpu_ready` | **22/68** migran a GPU |
| `gpu_ready` | **46/68** migran a CPU |

Primeros cruces desde `cpu_ready`: Cholesky N=4096 en `K=2`, GEMM N=4096 en
`K=3`, FFT N=4096 en `K=5`. Cruces inversos desde `gpu_ready`: Stencil N=3072
en `K=2`, AXPY N=31623 y SpMV N=1 000 000 en `K=3`.

Los tres estados convergen al mismo conjunto asintótico: las **22**
configuraciones en que GPU gana en región caliente (`22 + 46 = 68`). Esta
identidad es una comprobación de consistencia interna del mapa y debe
verificarse en el conjunto confirmatorio.

En consecuencia, la regla "en `cpu_ready` permanecer siempre en CPU" (§6.3 del
plan y §2 de este protocolo) es verdadera para el despacho siguiente y falsa
como política de horizonte.

### 12.3 Alcance del resultado "política simple"

El resultado congelado en §6 —una tabla de umbrales por operación iguala al
oráculo dentro del piso de ruido, luego la regla bloqueante impide reclamar
valor para el Aprendizaje Automático— **se conserva sin cambios, acotado a
`K = 1`**. Ninguna de las ocho baselines de §6 plantea la pregunta de
horizonte, de modo que ese resultado no puede extenderse a `K > 1` sin
baselines que lo evalúen.

### 12.4 Baselines adicionales (congeladas por esta enmienda)

Se añaden a las ocho de §6, con los mismos requisitos de ajuste
exclusivamente en entrenamiento:

9. `stay_on_ready_device_k` — permanecer en el dispositivo preparado para todo
   el horizonte; es la baseline que la política de §2 representa realmente.
10. `k_break_even_table_train` — tabla empírica de `K_break_even` por operación
    y tamaño, ajustada solo con tamaños de entrenamiento, aplicada por
    interpolación al tamaño de prueba.
11. `oracle_k` — oráculo con conocimiento posterior que resuelve
    `argmin_d EDP_total(d, K | estado)` con los costos medidos.

La regla bloqueante de §6 se aplica igualmente a estas tres.

### 12.5 Reporte

Los resultados de dispositivo se reportarán en una rejilla de `K` fijada aquí
para impedir su elección posterior:

```text
K ∈ {1, 2, 3, 5, 10, 30, 100, 1000}
```

Se reportará además, por configuración, el `K` de cambio de dispositivo y su
banda de sensibilidad. `K_break_even` no se presenta como entero exacto: los
costos `cold` heredan un piso de ruido de 5,76 % frente al 1,80 % de la región
caliente (§7), y la banda observada sobre las repeticiones tiene mediana 41 % y
máximo 78 %.

### 12.6 Lo que esta enmienda NO cambia

- Los jobs 6763/6764 y sus manifiestos: los datos REF `cold`/`warm` que
  producen son exactamente los que esta formulación necesita.
- El target de §1, las características de §3, las familias de §4, el piso de
  ruido y la regla de abstención de §7.
- El estado de aprendizaje de §2 para `K = 1`.
- La decisión de no ejecutar el barrido cartesiano de frecuencias.

---

## 13. Enmienda 2026-08-30-B — target estructurado en tres capas

**Fecha:** 2026-08-30
**Datos confirmatorios observados al redactar:** ninguno (verificado por
`squeue`: jobs 6763/6764 en `PENDING`; sin directorios de salida)
**Motivo:** el resultado real de R2 sobre el target directo de §1, evaluado
en la rejilla de `K` de la enmienda 2026-08-30-A (`k_grid` congelada,
folds de interpolación y extrapolación, comparación pareada contra las
baselines) fue: el modelo supera a la mejor baseline en **2 de 48**
rebanadas `(régimen, estado, K)` -- ambas en `K = 3`, en `cpu_ready` y
`none_ready`, con mejora de 14,7 % y 13,0 % respectivamente sobre
`intensity_threshold_train` -- y pierde de forma inestable en valores de `K`
cercanos: -11,6 % en `cpu_ready` `K = 5`, -13,5 % en `none_ready` `K = 5`,
-72,4 % en `none_ready` `K = 10`. `random_forest` fue la familia con mejor
media global, pero esa media agrega comportamientos opuestos según `K`
(media 1,94-3,92 en `K` entre 1 y 10; media ~1,02 en `K` entre 30 y 1000),
señal de ajuste a un punto local y no de una curva aprendida.

El diagnóstico: `y` no es una cantidad primitiva. Es una función cerrada de
ocho costos medibles por separado -- `E` y `T`, frío y caliente, por
dispositivo -- compuestos según la fórmula de `EDP_total(d, K)` de §12.1.
Pedirle a un regresor que aprenda `y` para toda la rejilla de `K` a la vez,
con 68 `config_id`, es pedirle que redescubra esa composición sin dársela.

### 13.1 Evidencia de la reformulación (datos exploratorios, verificada)

- el costo **caliente** (`E_warm`, `T_warm`) sigue una ley de potencias en
  `log(costo) ~ log(N)` por operación, con R² entre 0,974 y 0,998 en las
  cuatro combinaciones dispositivo×magnitud, en las seis operaciones;
- el costo **frío** no correlaciona con el tamaño de la misma forma (R²
  entre 0,000 y 0,918) porque está dominado por un término de arranque
  aproximadamente constante: mediana 0,618 s con CV 0,09 para el arranque de
  GPU en tiempo, a través de las seis operaciones y trece tamaños; el
  arranque de CPU es un orden de magnitud menor y sí depende de la
  operación (CV 0,95);
- la telemetría de una única ejecución de sondeo coincide con el costo
  **frío** medido del dispositivo que la produjo (error relativo mediano
  4,88 %, dentro del piso de ruido de la región fría, 5,76 % de §7) -- no con
  el costo caliente (error > 250 %). El sondeo mide directamente una
  primitiva de costo, no una aproximación de `y`.

### 13.2 Formulación

Se reemplaza el objetivo de aprendizaje único por tres capas:

1. **Predicción**: cuatro primitivas de costo calientes (`E_warm`, `T_warm`
   × CPU, GPU), en función de `(operación, tamaño)`.
2. **Calibración**: arranque por dispositivo -- GPU como constante con su
   incertidumbre; CPU por operación si la dependencia se sostiene bajo
   validación cruzada agrupada por `config_id`.
3. **Composición**: costo frío = caliente + arranque; `EDP_total(d, K)` y
   `K_break_even` se derivan con la fórmula ya congelada en §12.1, no se
   aprenden. La decisión de dispositivo sale de esta capa, no de un cuarto
   modelo.

Cuando hay sondeo disponible, la primitiva fría predicha por la capa 1+2 se
sustituye por la medida real del dispositivo que sondeó (mejora acotada por
el 4,88 % de error ya verificado, no una mejora libre).

### 13.3 Comparación obligatoria (no reemplaza a §1, compite con ella)

El target estructurado se evalúa con el mismo procedimiento de §5, las
mismas baselines de §6, el mismo piso de ruido y regla de abstención de §7,
sobre los mismos folds que el target directo de §1. La regla bloqueante de
§6 se aplica contra la mejor de las dos formulaciones de modelo (directa o
estructurada), no solo contra las baselines: si ninguna de las dos supera a
la mejor baseline por encima del piso de ruido, se reporta que ninguna
formulación de ML aportó valor, no solo que la directa no lo hizo.

### 13.4 Familias de modelos por capa (amplía §4, no lo reemplaza)

No se introduce ninguna familia fuera de las cinco ya congeladas en §4.
Ridge es el candidato natural para la capa 1 (un ajuste lineal en
`log(costo)` contra `log(N)` es una ley de potencias); árbol de regresión
chico y RandomForest quedan disponibles como contraste no lineal en la misma
capa; ElasticNet y Huber como contraste de regularización/robustez. La capa
2 (arranque) puede no requerir ajuste de hiperparámetros si los datos
confirman que es una constante -- verificable con una media y su intervalo,
sin GridSearchCV.

**XGBoost no se agrega.** Ya se excluyó en §4 por el tamaño de muestra
(n=68); el patrón de sobreajuste local ya observado en RandomForest sobre el
target directo (gana en `K=3`, pierde hasta -72,4 % en `K` cercanos) es
razón adicional para no incorporar un modelo todavía más flexible antes de
resolver la inestabilidad con una reformulación más simple, no más compleja.

### 13.5 Extensión condicionada, no obligatoria todavía

Una cuarta capa de corrección residual -- un modelo pequeño que aprenda el
error entre la ley de potencias ideal y el costo medido, usando telemetría
de sondeo como entrada -- se registra como extensión a evaluar solo si la
composición de tres capas deja un residual sistemático y aprendible después
de aplicarse. No se implementa por anticipado.

### 13.6 Lo que esta enmienda NO cambia

- Los jobs 6763/6764 y sus manifiestos.
- La cantidad de interés `y` y su generalización a `K` de §12.
- Las características de §3, el piso de ruido y la regla de abstención de
  §7, las baselines de §6.
- El resultado de R2 sobre el target directo ya obtenido (§13 misma
  sección, párrafo de motivo): se reporta como parte del resultado, no se
  descarta.

---

## 14. Enmienda 2026-08-30-C — corrección de selección y agregación de R2

**Fecha:** 2026-08-30
**Datos confirmatorios observados al redactar:** ninguno; los jobs 6763/6764
continuaban en `PENDING` y no existían sus directorios de resultados.

### 14.1 Error corregido

El reporte inicial de §13 eliminaba `fold` de la clave antes de seleccionar
con `idxmin` la mejor baseline y el mejor modelo. Esto permitía enfrentar
filas de pliegues distintos y elegir una familia diferente después de
observar cada rebanada de prueba. Los conteos `2/48` del target directo y
`4/48` de la comparación estructurada quedan supersedidos; se conservan en
§13 solo como registro histórico del diagnóstico que motivó la enmienda B.

### 14.2 Política de evaluación corregida

1. El modelo directo se selecciona una sola vez según §5.4 y se aplica a
   todos los pliegues, estados y valores de `K`.
2. El modelo estructurado se selecciona una sola vez con el mismo criterio.
3. La baseline pertinente se congela por `(régimen, resource_state, K)` a
   partir del promedio entre pliegues exploratorios; nunca puede depender del
   pliegue individual ni del dato confirmatorio.
4. Toda comparación elemental usa exactamente el mismo
   `(fold, resource_state, K)` para las tres políticas.
5. Los tres pliegues de interpolación se agregan sumando EDP porque sus
   conjuntos de prueba son disjuntos.
6. `extrapolation_top1` y `extrapolation_top2` se reportan por separado: el
   primero está contenido en el segundo y sumarlos duplicaría configuraciones.
7. El conjunto confirmatorio recibirá estas políticas ya congeladas una sola
   vez; no se repetirá la selección después de observarlo.

### 14.3 Resultado exploratorio corregido

La comparación de tres vías supera la baseline por encima del piso de ruido
en 2/24 rebanadas de interpolación agregada, 0/24 de
`extrapolation_top1` y 2/24 de `extrapolation_top2`. El detalle, las huellas
de entrada y los artefactos por pliegue están en
`resultados_selector_r2_corregidos_20260830.md` y
`resultados_selector_r2_20260830/`.

Este resultado no autoriza todavía adoptar ML como política general de
dispositivo. Congela una política híbrida candidata —baseline por defecto y
modelo solo en las compuertas que superaron el umbral exploratorio— para que
la campaña confirmatoria decida si sobrevive fuera del conjunto de desarrollo.

## 15. Enmienda 2026-08-31-A — target relativo a REF y profundidad libre para R3-A

**Fecha:** 2026-08-31
**Datos confirmatorios observados al redactar:** ninguno; los jobs 6763/6764
continuaban en `PENDING` y no existían sus directorios de resultados.
**Ámbito:** R3-A (capa DVFS offline, `classifier/selector/dvfs.py`), sección
§6.5 del plan de reformulación. No toca R1 ni R2/estructurado.

### 15.1 Error corregido: objetivo en magnitud absoluta

La primera implementación de R3-A predecía `log(energía)` y `log(tiempo)`
absolutos por acción y calibraba el error como
`|predicho/real - 1| * 100`. Verificado sobre los 5.441 registros reales de
pacca: en configuraciones con EDP diminuto (p. ej. `axpy_N10000` en frío,
EDP≈1,66e-8 J·s) un error absoluto minúsculo del modelo se traduce en errores
porcentuales de hasta 14.000.000%, porque se divide por una magnitud casi
nula. El p95 de incertidumbre por contexto (`resource_state`, `device`)
llegaba a 121.937% incluso en el contexto más limpio, forzando abstención al
100% en todos los estados sin que eso reflejara una limitación real del
modelo.

**Corrección:** el objetivo pasa a ser el desvío logarítmico respecto a la
acción REF del mismo `config_id × resource_state`
(`log_energy_ratio = log(energía_acción) − log(energía_REF)`, análogo para
tiempo). REF se mide, nunca se predice, y su costo real ancla la
reconstrucción (`predicho = REF_medido × exp(desvío_predicho)`). Mismo split,
mismos datos reales: el p95 de incertidumbre baja a 35%–125% según contexto
(5 órdenes de magnitud de mejora). Sigue muy por encima del piso de ruido de
§7 (1,80%–5,76%): la política se abstiene en el 100% de los casos con este
solo cambio. Es un resultado honesto, no un modelo ya viable.

### 15.2 Error corregido: profundidad congelada heredada de R2

`tree` y `random_forest` heredaban de §4 el límite `max_depth <= 3 / 5`,
congelado allí para el eje de dispositivo (R2): pocas categorías, margen de
decisión enorme, la profundidad chica alcanza. R3-A tiene ~40 acciones de
frecuencia × 6 operaciones interactuando con el tamaño — un árbol de
profundidad 3 no puede ni codificar las categorías, y `fit_cost_models` de
R3-A no hace búsqueda de hiperparámetros (a diferencia de §5.3 para R2), así
que ese límite operaba sin ajuste alguno.

**Verificación antes de decidir:** sobre el mismo pliegue de calibración,
liberar `max_depth` en `random_forest` bajó la mediana del error de 36,8% a
7,4%; el p95 mejoró de 73,2% a 57,2%; el máximo se mantuvo ~320%–350%. El
error residual grande no es ruido difuso: se concentra en GPU con tamaño
chico y reloj de host bajo (`fft_N64`, `spmv_N10000`, `stencil_N64` en
`gpu:F6:*`/`gpu:F3:*`), un régimen donde el overhead de lanzamiento domina de
forma no lineal y ningún modelo de esta lista lo captura todavía.

**Decisión:** se libera `max_depth` (sin límite) para `tree` y
`random_forest` **únicamente dentro de R3-A**
(`classifier/selector/dvfs.py::_DVFS_DEPTH_OVERRIDE`). El límite de §4 para
R2 (eje de dispositivo) no cambia — sigue congelado en `max_depth <= 3 / 5`
tal como se decidió allí, porque ese límite fue correcto para ese problema
(pocas categorías, margen enorme) y liberarlo ahí ajustaría ruido.

### 15.3 Lo que esta enmienda NO cambia

- No cambia la regla de abstención de §7 ni el piso de ruido por región.
- No adopta R3-A como política operativa: con ambas correcciones aplicadas,
  el p95 de incertidumbre sigue muy por encima del piso de ruido y la
  abstención sigue siendo el resultado en la mayoría de los contextos.
- No modifica §4 para R2/estructurado — el override de profundidad es local
  al módulo `dvfs.py`, aplicado vía el parámetro `params` de
  `r2._base_estimator`, no un cambio del valor congelado allí.
- No resuelve el régimen de cola (GPU, N chico, reloj de host bajo)
  identificado en §15.2; queda como diagnóstico abierto para una iteración
  futura de R3-A (tratamiento aparte del overhead de lanzamiento, o
  exclusión explícita de ese régimen de la banda de equivalencia).
