# Estrategia CPU — Fase 2 (documento para el director)

**Propósito de este documento.** Complementa `Estrategia_GPU_Fase2.md`.
El eje CPU tiene un bloqueo de infraestructura real (CAP_PERFMON) que el
de GPU no tiene, así que este documento distingue con cuidado **qué está
bloqueado de verdad** de **qué se puede seguir haciendo hoy**, y por qué
ninguna de las dos cosas rompe el contrato de objetivos.

Fuente completa: `docs/general/resultados_compuertas_fase2.md`
(2026-08-21/22, antes de esta sesión) y el análisis nuevo de esta sesión
sobre esos mismos datos.

---

## 1. El bloqueo real, con precisión

`perf_event_open` para contadores de **uncore** (`uncore_imc`, ancho de
banda de memoria) requiere `CAP_PERFMON`, que se rompió en pacca
(ARC-184, regresión de un permiso ya concedido antes). Esto bloquea
**generar nuevo dataset de CPU con etiqueta de verdad** — la etiqueta
`phase_label_train` se deriva de intensidad operacional dinámica, que
depende de uncore.

**Lo que NO está bloqueado, y es la aclaración central de este
documento:** el modelo, una vez entrenado, **no depende de uncore para
inferir**. Verificado directamente en el código ya escrito
(`classifier/training/train_phase.py`):

```python
FEATURES = [
    "ipc", "mpki", "llc_miss_rate", "stall_backend_ratio",
    "ips", "running_ratio", "freq_khz_observed",
]
```

Ninguna de estas siete columnas requiere uncore. El propio diseño ya
declaraba `operational_intensity*`, `uncore_cas_count_*` y derivados como
`FORBIDDEN` — **no por el bloqueo de CAP_PERFMON, sino porque son la
fuente de la etiqueta y usarlas como feature sería fuga de datos**. El
daemon del Objetivo 3, tal como ya estaba diseñado antes de esta sesión,
nunca necesitó CAP_PERFMON para funcionar en producción. Lo que bloquea
CAP_PERFMON es **entrenar con datos nuevos**, no **desplegar**.

## 2. Lo que ya se puede hacer hoy, sin CAP_PERFMON: reanalizar lo que ya es válido

Existe una campaña completa y válida, corrida **antes** de la regresión
de CAP_PERFMON: `pacca_cpu_final_attempt03_20260820_arc174`, 424/540
corridas aceptadas, 9 kernels × 6 niveles, con uncore funcionando y
energía RAPL real. Esta sesión construyó `cpu_policy_headroom.py` y lo
corrió sobre esos datos, sin tocar ningún nodo:

| presupuesto de degradación | mejor constante | oráculo | margen del modelo |
|---|---|---|---|
| cualquiera (0.33 pts en todos) | F0 → 0% | 0.33% | **0.33 pts** |

**El margen es genuinamente pequeño en el catálogo actual de 9
kernels**, y no es un problema de resolución de grilla como en GPU
(§4). 7 de 9 kernels ya tienen su óptimo en F0 (máxima frecuencia); solo
`npb_mg` se aparta con +2.71% en F1.

## 3. Por qué CPU no responde igual que GPU al mismo remedio

En GPU, el margen escondido vivía en el salto F0→F1 porque **el reloj de
núcleo y el de memoria son dominios independientes**: bajar el núcleo no
frena la memoria. En CPU **no existe ese segundo dominio** — el propio
análisis de compuerta 0 (`resultados_compuertas_fase2.md`, antes de esta
sesión) ya lo documentó: *"bajar el reloj del núcleo también frena el
acceso a memoria (menos peticiones en vuelo, prefetch más lento)"*. El
rango de estiramiento observado al bajar 4× el reloj fue 2.21×–4.05×, no
1×–4× — ni el kernel más memory-bound se acerca a "gratis".

**Esto coincide con la literatura ya citada, no la contradice.**
`Calore2017` (Haswell CPU, ya citado) reporta explícitamente: *"los
ahorros no son grandes, pero tampoco despreciables"* — más modesto que
GPU, con la misma causa física. Su regla operativa, textual, es la
nuestra: ahorro cuando el código es memory-bound, medido por balance de
máquina vs. intensidad operacional (Roofline, ya usado en este trabajo).

## 4. Un experimento que se descartó — y por qué eso también es progreso

Se evaluó comparar la política propuesta contra el gobernador `powersave`
de Linux (dinámico bajo HWP, funcionalmente similar a `ondemand` según
`Hebbar2022`), inspirado en que ese paper logra 121–183% de mejora
justamente explotando que el gobernador por defecto detecta mal las
cargas memory-bound.

**No se implementó, porque ya se había intentado antes y falló.**
Documentado en `Registro_Cambios_Fuera_Plan_Original.md` (ARC-160,
2026-08-19): escribir `scaling_governor=powersave` en los 6 CPUs
delegados devuelve `Permission denied` en los 6 — el permiso P1
concedido por administración cubre únicamente `scaling_min_freq` /
`scaling_max_freq`, nunca el gobernador ni el EPP.

**Consecuencia útil, no solo un callejón sin salida:** el nivel `REF`
(gobernador nativo, sin pinear) en esta plataforma corre en `performance`
por defecto, y su frecuencia observada bajo carga es prácticamente
idéntica a F0 (3199.9 vs 3200.0 MHz medido). Es decir: **`REF` ya ES la
comparación contra el gobernador nativo que pide el Objetivo 4** — no
hace falta ningún permiso nuevo para satisfacerlo, porque el gobernador
nativo de este nodo, bajo carga, se comporta como "siempre máxima
frecuencia", y eso ya está en todas las tablas medidas.

## 5. El plan concreto

1. **Aplicar el mismo tamizaje por α que se usó en GPU (Anexo K.7) al
   catálogo de CPU**, buscando kernels más memory-bound que los 9
   actuales. Esto **no necesita CAP_PERFMON** — el tamizaje mide solo
   tiempo y energía RAPL a distintos `scaling_min/max_freq` pineados
   (permiso P1, ya concedido y en uso).
2. **`npb_mg` es la única señal real del catálogo actual** — es el
   equivalente CPU de `lavamd`/`dwt2d` en GPU: el candidato con margen
   genuino, a estudiar primero.
3. **Construir y evaluar el modelo (arquitectura de §1) sobre el dataset
   ya válido**, en paralelo, sin esperar nada — es exactamente el trabajo
   offline que ya empezó a fallar bajo LOKO (`resultados_compuertas_fase2.md`
   §5.bis: F1 macro 0.393 vs. 0.371 trivial) y que se puede re-intentar
   con la arquitectura corregida (predicción por carga desde un nivel de
   referencia, no clasificación de fase por ventana — mismo argumento que
   GPU §3 del documento hermano).
4. **Cuando CAP_PERFMON se repare** (acción de administrador, solicitud
   ya redactada): ampliar el dataset con los kernels nuevos que sobrevivan
   el tamizaje, con etiqueta de verdad completa.

## 6. Reconciliación honesta con la variación intra-kernel (Objetivo 2 literal)

No toda la evidencia apunta a "cero variación intra-kernel". La misma
tabla de `resultados_compuertas_fase2.md` que mostró que el óptimo casi
nunca cambia también mostró **mezcla real de clase minoritaria en
`npb_bt` (11.8% global, hasta 21.5% en F2) y `npb_lu` (19.0% global,
hasta 34.4% en F3)** — concentrada en frecuencias bajas, donde el ridge
de Roofline se desplaza y cruza el punto de operación del kernel. Es
decir: **la alternancia de fase intra-ejecución que pide el Objetivo 2
sí existe, en un subconjunto real de casos**, solo que no es el
fenómeno dominante en el catálogo actual de 9 kernels deliberadamente
"puros" (§5.1 del anteproyecto, cumplido al pie de la letra).

La arquitectura propuesta no abandona el clasificador en vivo — lo
reorienta hacia la granularidad de carga como mecanismo primario
(defendible con el precedente triple citado en el documento de GPU:
Guerreiro/Calore/Antici) y conserva la clasificación por ventana como
mecanismo secundario para los casos donde sí hay mezcla real
(`npb_bt`/`npb_lu`/`npb_mg` a frecuencias bajas) — sin inventar una
arquitectura nueva, solo priorizando la que el dato mismo respalda.

## 7. Mapeo explícito a los Objetivos Específicos

| Objetivo | Estado | Evidencia |
|---|---|---|
| 1. Caracterizar comportamiento y consumo bajo distintos estados de frecuencia (Perf+RAPL) | **Cumplido** (424/540 corridas válidas) | `pacca_cpu_final_attempt03_20260820_arc174` |
| 2. Clasificador ML de fases desde telemetría, en vivo, baja latencia | Arquitectura sin cambios; features ya libres de uncore (`train_phase.py`); alternancia real documentada en 2/9 kernels a frecuencias bajas | §1, §6 |
| 3. Daemon de espacio de usuario, inferencia + política DVFS proactiva | **No bloqueado por CAP_PERFMON** — el runtime del modelo nunca dependió de uncore; el mecanismo de actuación (`scaling_min/max_freq`) ya tiene permiso P1 | §1 |
| 4. Evaluación por EDP contra gobernador nativo | `REF` (gobernador nativo, sin pinear) ya es la comparación pedida — no requiere el permiso de `scaling_governor` que fue denegado | §4 |

---

## Riesgos abiertos, declarados sin adornar

1. El margen de 0.33 puntos es sobre el catálogo actual; no se sabe
   todavía si un tamizaje por α encontrará kernels CPU con margen real
   comparable al de GPU.
2. `powersave` como baseline dinámico queda descartado por permiso — si
   se considera valioso, requiere una solicitud nueva a administración,
   explícitamente sobre `scaling_governor`/EPP (nunca pedida hasta hoy).
3. CAP_PERFMON sigue sin fecha de resolución — el plan de §5 avanza sin
   esperarlo, pero la ampliación de dataset con etiqueta de verdad
   completa sigue condicionada a él.
4. El modelo entrenado con la arquitectura corregida (predicción por
   carga) todavía no se ha corrido — es el paso inmediato siguiente, no
   un resultado ya obtenido.

---

## Referencias citadas en este documento

Ya en `docs/libro/main.tex`: `\cite{Calore2017}`, `\cite{Hebbar2022}`,
`\cite{Guerreiro2019}`, `\cite{Antici2024}`, `\cite{Williams2009}`.
