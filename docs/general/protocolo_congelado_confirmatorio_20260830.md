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
| 2026-08-31 | enmienda **2026-08-31-B**: integración de `curve_physical`, autocorrección del hallazgo de 6.5-bis bajo compuerta honesta, frecuencias reales, diagnóstico `cpu_ready`/`none_ready` (§16) | **no** — verificado por `squeue`: jobs 6763/6764 siguen en `PENDING`, sin directorios de salida al redactar |
| 2026-08-31 | enmienda **2026-08-31-C**: contextualización del 5,0% frente a `best_constant_train` y al peso del EDP, correcciones documentales (5.440 registros, abstención 50%/25%, particiones no disjuntas) (§17) | **no** — verificado por `squeue`: jobs 6763/6764 (y 6769/6770) siguen sin directorios de salida al redactar |

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
`|predicho/real - 1| * 100`. Verificado sobre los 5.440 registros reales de
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

### 15.4 Calibración por régimen de tamaño (mismo día, misma enmienda)

**Motivo:** el error de calibración de §15.2 se agregaba por
`(resource_state, device)` únicamente. Verificado con datos reales: dentro de
`gpu_ready/gpu` (56 configs), separar por tamaño relativo a la operación
(mediana de `size` por operación en el propio `train`, sin fuga: nunca usa
`config_id` de prueba) muestra que las 26 configuraciones de tamaño grande
tienen error mediano **3,7%** (p95 27,1%) mientras las 30 de tamaño chico
tienen error mediano 9,6% (p95 66,9%). Un solo umbral por `(estado,
dispositivo)` obligaba a las configuraciones grandes a heredar la banda de
incertidumbre inflada de las chicas, y con eso el margen real medido en R1
(headroom mediano 13,46% para las 26 configs grandes de `gpu_ready`, 26/26
por encima del piso de ruido) nunca llegaba a superar el umbral de
abstención.

**Cambio:** `CostModels.uncertainty_pct_by_context` pasa a indexarse por
`(resource_state, device, size_regime)`, con `size_regime ∈ {small, large}`
calculado por operación (`dvfs._size_regimes`/`_size_regime`). No hay umbral
absoluto de tamaño válido entre operaciones (axpy solo tiene los tamaños
31623 y 100000; cholesky va de 64 a miles), así que el corte es siempre
relativo a la escala propia de cada operación.

**Resultado verificado (extrapolación, mismos pliegues, familia
`power_law`):** la abstención en `gpu_ready` baja de 100% a 50%
(`extrapolation_top1`, n=6) y 25% (`extrapolation_top2`, n=12), y el modelo
captura un ahorro real y positivo de 5,02% y 4,98% de EDP contra REF
respectivamente — el primer resultado no nulo de R3-A. Estas dos
particiones NO son réplicas independientes: `extrapolation_top1` (6
configs) está contenido en `extrapolation_top2` (12 configs), no son
disjuntas. `random_forest` no mejora de forma consistente (savings ≈0% o
levemente negativo); `power_law` es, por ahora, la única familia con señal
positiva repetida.

Este resultado sigue **sin superar la baseline no aprendida**
(`best_constant_train`) en la comparación agregada de §5.4, y en
`cpu_ready` el mismo cambio produce un ahorro levemente negativo en un
pliegue (-0,22% en `extrapolation_top2`) — se reporta sin ocultarlo. No se
adopta R3-A como política; el resultado que cambia es que, por primera vez,
el modelo actúa (no se abstiene siempre) exactamente en el estado donde R1
ya había medido el margen más grande, y cuando actúa, ahorra.

## 16. Enmienda 2026-08-31-B — integración de `curve_physical`, autocorrección del hallazgo experimental y frecuencias reales

**Fecha:** 2026-08-31
**Datos confirmatorios observados al redactar:** ninguno; los jobs 6763/6764
continuaban en `PENDING` (razones "Nodes required... DOWN/DRAINED" y
"Resources"), sin directorios de salida.
**Ámbito:** R3-A. Integra al pipeline el hallazgo experimental registrado en
la sección 6.5-bis del plan de reformulación (curva física en vez de 40
costos categóricos) y corrige un supuesto no verificado sobre frecuencias.

### 16.1 Integración de `curve_physical` como familia adicional

Se agregó `curve_physical` a `DVFS_FAMILIES` (`classifier/selector/dvfs.py`):
predice los 7 parámetros de `t(f)=t_a+t_b/f_{dev}+t_c/f_{host}` y su análogo
de energía por `config_id × resource_state`, en vez de 40 costos por acción
sin relación entre sí. Reutiliza sin cambios el resto de la arquitectura de
R3-A (calibración de incertidumbre por `(resource_state, device,
size_regime)`, compuerta de abstención, folds pareados con las demás
familias).

**Defecto encontrado durante la integración, no presente en el experimento
sin compuerta:** al someter `curve_physical` a la calibración fuera de
muestra (§5.3), un grupo de prueba atípico (`cholesky_N256`, `gpu_ready`,
reloj de host `F6`) hizo que el regresor de parámetros extrapolara un valor
absurdo, que al dividirse entre la fracción de frecuencia mínima real del
GPU (0,149, ver §16.3) produjo un log-ratio disparatado — p95 de
incertidumbre de hasta 4,69e13% en un pliegue. El experimento sin compuerta
de la sección 6.5-bis del plan nunca calibra error fuera de muestra por
separado, así que este defecto no era visible allí.

**Corrección:** se recorta el log-ratio predicho (energía y tiempo, por
separado) a `±ln(8)`, un factor 8x por eje. El rango real medido de
`log_energy_ratio`/`log_time_ratio` en las 40 acciones × 68 `config_id` del
catálogo es factor 5,67x (energía) y 7,03x (tiempo); 8x es generoso frente a
lo observado y evita que un valor de parámetro degenerado se propague sin
límite. Es una salvaguarda numérica documentada en el código, no un
resultado físico ni un hiperparámetro ajustado por desempeño.

### 16.2 Autocorrección: el hallazgo de 6.5-bis no sobrevive a la compuerta honesta

Con la salvaguarda aplicada, `curve_physical` ya no explota numéricamente,
pero **tampoco supera a `power_law`** en la comparación gated (Tabla
siguiente, extrapolación, `resource_state=all`):

| pliegue | familia | razón vs. oráculo | abstención | incertidumbre p95 |
|---|---|---|---|---|
| `extrapolation_top1` | `power_law` | **1,0121** | 83,3% | 68,1% |
| `extrapolation_top1` | `curve_physical` | 1,0137 | 100% | 189,6% |
| `extrapolation_top2` | `power_law` | **1,0130** | 66,7% | 92,2% |
| `extrapolation_top2` | `curve_physical` | 1,0136 | 100% | 3.167,2% |

Esto **corrige, no reemplaza**, el hallazgo de la sección 6.5-bis del plan.
La comparación sin compuerta (9,89% de ahorro con Ridge, contra 3,89% de la
formulación categórica) seguía siendo válida como evidencia de que la
*forma* física describe mejor los datos ($R^2$ 0,94-0,98) — eso no cambia.
Lo que no sobrevivió fue la promesa implícita de que esa forma ya estaba
lista para actuar: al exigirle la misma calibración de incertidumbre fuera
de muestra que exige a las demás familias, `curve_physical` resulta **más
inestable**, no más confiable, porque predecir 7 parámetros compartidos por
`config_id` (en vez de 40 valores independientes) hace que un solo grupo mal
predicho contamine la reconstrucción de las 32-40 acciones de ese grupo a la
vez, mientras que un regresor categórico solo se equivoca en las acciones
que predice directamente.

El procedimiento de selección de §5.4 ya elige correctamente `power_law`
como familia (verificado: `dvfs_summary.json` reporta
`"family": "power_law"` después de esta integración) — no fue necesario
intervenir la selección, la propia regla bloqueante descartó
`curve_physical` con la evidencia gated. `curve_physical` se conserva en el
código y en `DVFS_FAMILIES` como familia evaluada y descartada, con su
propia cobertura de pruebas; no se elimina, para que quede trazable por qué
no se adoptó.

### 16.3 Fracciones de frecuencia: supuesto reemplazado por telemetría real

El supuesto original (`CURVE_FREQUENCY_FLOOR = 0,35`, aplicado por igual a
CPU y GPU) quedó registrado en la sección 6.5-bis del plan como "verificado
solo indirectamente". Se verificó directamente contra `freq_khz_observed`
(CPU) y `gpu_sm_clock_mhz` (GPU) de `run_regions.csv` (16.320 filas de
telemetría real de campaña):

| nivel | fracción declarada (manifiesto) | fracción real CPU | fracción real GPU |
|---|---|---|---|
| F0 | 1,000 | 1,000 | 1,000 |
| F3 | 0,500 | 0,631 | 0,574 |
| F6 | 0,000 | **0,267** | **0,149** |
| REF | (gobernador nativo) | 0,994 | 1,000 |

El hardware nunca llega al reloj mínimo nominal declarado, y CPU y GPU
difieren entre sí en cuánto se desvían. `FREQUENCY_FRACTION` se reemplazó
por `CPU_FREQUENCY_FRACTION`/`GPU_FREQUENCY_FRACTION`, específicas por
dispositivo y derivadas de la mediana observada, no de un piso supuesto.
REF se mapea a la fracción de F0 (1,0) en ambas tablas porque su fracción
real medida (0,994 CPU, 1,000 GPU) coincide dentro del error de medición,
consistente con la verificación previa (razón de tiempo REF/F0 mediana
1,0001 en CPU).

### 16.4 Diagnóstico adicional: por qué `cpu_ready` calla y `none_ready` no mejora

Antes de esta enmienda quedaba abierto si la calibración por tamaño de la
enmienda 2026-08-31-A ayudaría también en `cpu_ready` y `none_ready`.
Verificado con el headroom de la sección \ref{sec:resultados-r1-headroom-dvfs}
del libro (`dvfs_headroom.csv`):

- **`cpu_ready`**: headroom mediano 0,25%, solo 3/68 configuraciones por
  encima del piso de ruido. No hay margen real que capturar; que el modelo
  se abstenga siempre ahí es la respuesta correcta, no una falla de
  calibración.
- **`none_ready`**: headroom mediano 4,98%, 38/68 configuraciones por
  encima del piso de ruido — sí hay margen real. Pero separar por tamaño
  (mismo mecanismo que funcionó en `gpu_ready`) no lo desbloquea: incluso el
  subconjunto de tamaño grande tiene error mediano del 18,1% (p95 68,1%),
  muy por encima del margen disponible (4,98%). La causa probable no es la
  granularidad de la calibración sino que `none_ready` descansa enteramente
  en mediciones de la región **fría** (piso de ruido 5,76%, casi 3x el de la
  región caliente, sección \ref{sec:resultados-r1-piso-ruido} del libro) —
  la misma fuente de ruido que ya explicaba la banda de sensibilidad del
  41% mediano en $K_{\text{break\_even}}$. Queda como diagnóstico abierto,
  no resuelto por esta enmienda.

### 16.5 Lo que esta enmienda NO cambia

- No cambia la conclusión de la enmienda 2026-08-31-A: `power_law` sigue
  siendo la única familia con ahorro real y positivo verificado (~5,0% en
  `gpu_ready`), y R3-A sigue sin adoptarse como política general.
- No modifica la calibración por tamaño de la enmienda 2026-08-31-A.
- No resuelve el régimen de cola (GPU, N chico, reloj de host bajo) ni el
  problema de `none_ready` recién diagnosticado en §16.4.
- No reclama que la formulación por curva física esté descartada en
  general — el ajuste de forma ($R^2$ 0,94-0,98) sigue siendo evidencia
  válida; lo que se descarta es predecir sus 7 parámetros vía Ridge/RF sobre
  descriptores estáticos como reemplazo inmediato de `power_law`. Una
  regularización más fuerte, más datos por grupo, o predecir los parámetros
  por separado en vez de conjuntamente podrían revertir esta conclusión;
  ninguna de esas variantes se probó todavía.

## 17. Enmienda 2026-08-31-C — contextualización del resultado de R3-A frente a la baseline, correcciones documentales

**Fecha:** 2026-08-31
**Datos confirmatorios observados al redactar:** ninguno; verificado por
`squeue`, jobs 6763/6764 (y 6769/6770, campañas relacionadas que aparecieron
en la cola en esta fecha) siguen sin producir directorios de salida.
**Origen:** revisión externa (Codex) sobre el estado del repositorio después
de la enmienda 2026-08-31-B, sin modificar archivos. Verificado de forma
independiente antes de aceptar cada punto.

### 17.1 Correcciones documentales (errores reales, no de interpretación)

1. El catálogo real tiene **5.440** registros
   (`wc -l candidate_summary.csv` menos encabezado), no 5.441 como decía
   §15.1 y el libro. Corregido en ambos lugares.
2. La abstención de `power_law` en `gpu_ready` reportada en §15.4 como
   "50-75%" era incorrecta. El valor real, verificado contra
   `dvfs_results.csv`, es **50% en `extrapolation_top1` (n=6) y 25% en
   `extrapolation_top2` (n=12)** — no 75%. Corregido en §15.4, en el plan
   de reformulación y en el libro.
3. `extrapolation_top1` y `extrapolation_top2` se describían como
   "particiones disjuntas" en el libro y en el plan. Son lo contrario:
   `extrapolation_top2` retiene los dos tamaños mayores por operación y
   `extrapolation_top1` solo el mayor, así que las 6 configuraciones de
   prueba de `top1` están **contenidas** dentro de las 12 de `top2`
   (`classifier.selector.sizes.extrapolation_folds`, confirmado leyendo el
   código). No son réplicas estadísticamente independientes del mismo
   hallazgo; se reportan ambas porque muestran cómo se degrada la
   extrapolación al alejarse del rango de entrenamiento, como ya hacía
   correctamente la sección de R2 (§14.2, punto 6).

### 17.2 El resultado de 5,0% contextualizado frente a `best_constant_train`

La pregunta relevante para adoptar un modelo no es si mejora sobre REF —
mejora, y de forma real — sino si mejora **lo suficiente sobre la mejor
política no aprendida**. Verificado contra `dvfs_results.csv`
(`extrapolation_top1`/`top2`, `gpu_ready`):

| política | `extrapolation_top1` | `extrapolation_top2` |
|---|---|---|
| siempre REF | 0,00% | 0,00% |
| `best_constant_train` | 4,65% | 2,48% |
| `power_law` (compuerta activa) | **5,02%** | **4,98%** |
| oráculo (techo teórico) | 14,05% | 14,90% |

La ventaja adicional de `power_law` sobre `best_constant_train` es de
**0,37 puntos porcentuales en `top1` y 2,49 en `top2`** — no cero, pero
lejos de ser una ventaja robusta. Agregando los tres estados de recurso, el
modelo resulta **0,18% peor** que `best_constant_train`
(`dvfs_summary.json`, `model_improvement_pct = -0,18`), consistente con
`adopt_model: false`.

### 17.3 Por qué un resultado localizado no mueve la conclusión agregada: peso del EDP

En `extrapolation_top2`, el EDP total a REF por estado es:

| estado | EDP a REF | % del total |
|---|---|---|
| `none_ready` | 50,11 | 51,1% |
| `cpu_ready` | 44,65 | 45,6% |
| `gpu_ready` | 3,20 | 3,3% |
| **total** | **97,96** | **100%** |

`gpu_ready` --- el único estado donde el modelo aporta algo --- pesa apenas
3,3% del EDP total del catálogo en esta partición. Un ahorro del 5% sobre
esa porción reduce el EDP global en ~0,16%, muy por debajo de cualquier piso
de ruido razonable. Esto no invalida el hallazgo local; explica por qué no
se traduce en una mejora agregada visible.

### 17.4 Tres salvedades operativas verificadas, no reflejadas en el 5,0%

1. **Cota superior offline, no ahorro neto.** La evaluación fija
   `overhead_energy_j = overhead_time_s = 0` (confirmado en
   `dvfs_summary.json`, `overhead_status: not_measured_upper_bound`). El
   costo real de conmutar frecuencia todavía no se descuenta.
2. **El modelo necesita el costo REF medido de la configuración nueva, no
   solo sus descriptores estáticos.** `predict_costs` reconstruye
   multiplicando el desvío predicho por `ref_energy_j`/`ref_time_s` de esa
   misma fila (§15.1) — una cantidad medida, no predicha, para todas las
   familias de R3-A incluida `power_law`. Operativamente exige ejecutar
   primero una acción REF de la configuración nueva; no sirve todavía como
   selector inmediato de un kernel de una sola ejecución sin ese sondeo
   previo.
3. **Las bandas de incertidumbre siguen siendo grandes.** El p95 de error
   de `power_law` en `gpu_ready` es 63,1% (`top1`) y 92,2% (`top2`) — muy
   por encima del piso de ruido. La política ahorra porque se abstiene con
   frecuencia y porque varias frecuencias forman mesetas de EDP casi
   equivalente, no porque distinga con precisión la acción óptima.

### 17.5 Conclusión revisada y política candidata

`power_law` es un hallazgo exploratorio positivo y localizado en
`gpu_ready`, no todavía una política ganadora. La política oficial de R3-A
sigue siendo la baseline (`adopt_model: false`, sin cambios respecto a
§15.4/§16). Se registra como candidata, sin adoptarla, una política híbrida:

```text
cpu_ready  -> REF (sin margen real, §16.4)
none_ready -> REF (margen real pero error de modelo demasiado alto, §16.4)
gpu_ready  -> consultar power_law
                -> ventaja predicha clara (supera piso + incertidumbre) -> actuar
                -> duda o margen pequeño -> REF (abstención)
```

Esta política queda condicionada a que R3-B (máquina de estados y actuación
real sobre hardware, §13 del plan) permita medir si el ~5,0% sobrevive
después de descontar el sondeo REF y el costo físico de cambiar frecuencia
— la pregunta que esta enmienda deja explícitamente abierta, no cerrada.

### 17.6 Lo que esta enmienda NO cambia

- No cambia ninguna cifra de fondo de §15/§16 salvo las tres correcciones
  documentales de §17.1 — los hallazgos (fix de escala, profundidad libre,
  calibración por tamaño, integración de `curve_physical` y su descarte)
  se mantienen.
- No adopta ni descarta R3-A; formaliza por qué la decisión oficial
  (`adopt_model: false`) es la correcta con la evidencia actual, y qué
  falta medir para revisarla.
- No modifica la arquitectura de `dvfs.py`; es una enmienda documental y de
  interpretación, no de código.
