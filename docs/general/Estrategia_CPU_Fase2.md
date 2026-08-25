# Estrategia CPU — Fase 2 (documento para el director)

**Propósito.** Complementa `Estrategia_GPU_Fase2.md`. El eje CPU tiene un
bloqueo de infraestructura real (CAP_PERFMON) que el de GPU no tiene, así
que este documento separa con cuidado **qué está bloqueado de verdad** de
**qué se puede seguir haciendo hoy**, y por qué ninguna de las dos cosas
rompe el contrato de objetivos. §8 mapea a los Objetivos Específicos; §9
da la lista de pruebas para verificar o refutar cada afirmación.

Fuente: `docs/general/resultados_compuertas_fase2.md` (2026-08-21/22) y
el análisis nuevo de esta sesión sobre esos mismos datos.

> **Nota de auditoría (2026-08-24).** Revisado contra los datos crudos; se
> corrigieron cuatro errores de la versión anterior: (a) `npb_mg` estaba
> listado entre los kernels con alternancia de fase intra-corrida cuando
> su clase minoritaria es **0.0% en los seis niveles**, (b) el conteo de
> kernels de RAJAPerf estaba inflado al doble, (c) se comparaba frecuencia
> *observada* de REF contra frecuencia *aplicada* de F0, (d) el Objetivo 4
> se declaraba satisfecho sin señalar que solo cubre uno de los dos
> gobernadores nativos disponibles. Se agregó además un riesgo nuevo
> (riesgo 6) detectado al auditar el propio script de tamizaje.

---

## 1. El bloqueo real, con precisión

`perf_event_open` sobre contadores de **uncore** (`uncore_imc`, ancho de
banda de memoria) requiere `CAP_PERFMON`, que se rompió en pacca (ARC-184,
regresión de un permiso ya concedido). Eso bloquea **generar dataset nuevo
de CPU con etiqueta de verdad**: `phase_label_train` se deriva de
intensidad operacional dinámica, que sale de uncore.

**Lo que NO está bloqueado, y es la aclaración central:** el modelo, una
vez entrenado, **no necesita uncore para inferir**. Verificado en el
código ya escrito (`classifier/training/train_phase.py`):

```python
FEATURES = [
    "ipc", "mpki", "llc_miss_rate", "stall_backend_ratio",
    "ips", "running_ratio", "freq_khz_observed",
]
```

Ninguna de esas siete columnas requiere uncore. El diseño ya declaraba
`operational_intensity*`, `uncore_cas_count_*` y derivados como
`FORBIDDEN` — **no por el bloqueo, sino porque son la fuente de la
etiqueta y usarlas como feature sería fuga de datos**. El daemon del
Objetivo 3 nunca dependió de CAP_PERFMON. Lo que CAP_PERFMON bloquea es
**entrenar con datos nuevos**, no **desplegar**.

## 2. Lo que ya se hizo hoy sin CAP_PERFMON

Existe una campaña válida corrida **antes** de la regresión:
`pacca_cpu_final_attempt03_20260820_arc174` — 424/540 corridas aceptadas,
9 kernels × 6 niveles × 10 rep, con uncore funcionando y energía RAPL
real. Esta sesión construyó `cpu_policy_headroom.py` y lo corrió sobre
esos datos, sin tocar ningún nodo. Energía RAPL (pkg+dram), contra
"siempre F0", **9 kernels**:

| presupuesto de degradación | mejor constante | oráculo | margen del modelo |
|---|---|---|---|
| ≤4% | F0 → 0% | 0.33% | **0.33 pts** |
| ≤10% | F0 → 0% | 0.33% | **0.33 pts** |
| ≤15% | F0 → 0% | 0.33% | **0.33 pts** |
| sin límite | F0 → 0% | 0.33% | **0.33 pts** |

El margen no cambia con el presupuesto porque **el óptimo casi nunca se
aparta de F0**: 7 de 9 kernels lo tienen en F0. Solo `npb_mg` se separa de
forma significativa (**+2.71% en F1**); `npb_ft` (+0.10%) y
`rajaperf_polybench_3mm_omp` (+0.14%) están dentro del ruido.

## 3. Por qué CPU no responde al mismo remedio que GPU

En GPU el margen escondido vivía en el salto F0→F1 porque **el reloj de
núcleo y el de memoria son dominios independientes**: bajar el núcleo no
frena la memoria. **En CPU no existe ese segundo dominio.** El análisis de
compuerta 0 (`resultados_compuertas_fase2.md`, previo a esta sesión) ya lo
documentó: *"bajar el reloj del núcleo también frena el acceso a memoria
(menos peticiones en vuelo, prefetch más lento)"*. El estiramiento
observado al bajar 4× el reloj fue **2.21×–4.05×**, no 1×–4×: ni el kernel
más memory-bound se acerca a "bajar el reloj sale gratis".

**Esto coincide con la literatura citada, no la contradice.**
`Calore2017` (Haswell CPU) reporta *"los ahorros no son grandes, pero
tampoco despreciables"* — más modesto que GPU, con la misma causa física.
Su regla operativa es la nuestra: ahorro cuando el código es memory-bound,
medido por balance de máquina vs. intensidad operacional (Roofline,
`\cite{Williams2009}`, ya usado en este trabajo).

**Consecuencia de diseño:** insistir en más *resolución* de grilla sobre
los mismos 9 kernels no tiene sustento — a diferencia de GPU, el problema
de CPU no es dónde se muestrea sino **qué se muestrea** (§6).

## 4. Un experimento descartado, y qué se rescató de él

Se evaluó comparar contra el gobernador `powersave` (dinámico bajo HWP,
que la documentación del kernel describe como similar a
`schedutil`/`ondemand`), inspirado en que `Hebbar2022` logra 121–183% de
mejora explotando que el gobernador por defecto detecta mal las cargas
memory-bound.

**No se implementó porque ya se intentó y falló.** ARC-160 (2026-08-19,
`Registro_Cambios_Fuera_Plan_Original.md`): escribir
`scaling_governor=powersave` en los 6 CPUs delegados devuelve
`Permission denied` en los 6 — el permiso P1 cubre únicamente
`scaling_min_freq`/`scaling_max_freq`, nunca el gobernador ni el EPP.

**Lo que sí se rescata:** en esta plataforma el driver es `intel_pstate`
con solo dos gobernadores disponibles (`performance`, `powersave`), y el
nivel `REF` corre bajo el activo, `performance`. Su frecuencia **observada
bajo carga** es indistinguible de la de F0 (**3199.88 vs 3199.95 MHz**,
ambas medias observadas sobre `npb_bt`): el gobernador nativo activo,
bajo carga, se comporta como "siempre máxima frecuencia".

**Alcance exacto para el Objetivo 4, sin sobreventa:** `REF` satisface la
comparación contra **el gobernador nativo activo** sin necesitar ningún
permiso nuevo. **No** cubre `powersave`, que es el otro gobernador nativo
disponible y el que la línea de `Hebbar2022` sugiere que sería el rival
interesante. Cerrar esa mitad exigía una solicitud de permiso sobre
`scaling_governor`/EPP que nunca se había pedido (riesgo 3) — **el
administrador la concedió el 2026-08-25** (`sudo /usr/local/bin/set_cpu_gov
<gobernador> <epp>`, restringido a los cores 0-5). El permiso ya no es el
bloqueador; **queda pendiente de verificación empírica antes de usarse en
una campaña real** (riesgo 3, actualizado), tema abierto hasta que haga
falta para la comparación de la Fase 4 — no bloquea nada de lo que corre
hoy.

## 5. El plan concreto

1. **Tamizar el catálogo por α**, buscando kernels más memory-bound que
   los 9 actuales. **No necesita CAP_PERFMON**: mide solo tiempo de pared
   y energía RAPL con `scaling_min/max_freq` pineados (permiso P1, ya en
   uso). **Encolado, ver §6.**
2. **`npb_mg` es la única señal real del catálogo actual** (+2.71% en F1)
   — el equivalente CPU de `lavamd` en GPU: el candidato con margen
   genuino, a estudiar primero.
3. **Construir y evaluar el modelo con la arquitectura corregida**
   (predicción por carga desde un nivel de referencia — detallada en
   `Estrategia_GPU_Fase2.md` §4, es la misma para ambos ejes) **sobre el
   dataset ya válido**, sin esperar nada. Es el reintento del trabajo que
   falló bajo LOKO (`resultados_compuertas_fase2.md` §5.bis: F1 macro
   0.393 del mejor modelo vs. 0.371 del predictor trivial).
4. **Cuando CAP_PERFMON se repare**: ampliar el dataset con los kernels
   que sobrevivan el tamizaje, con etiqueta de verdad completa. Protocolo
   actualizado (2026-08-25), evidencia-basado y no heredado sin más del
   núcleo: **6 niveles × 6 repeticiones** (no 10 — ver
   `docs/justifications/report/sections/repetitions_edp.tex`, análisis de
   convergencia de CV% de EDP sobre las 54 celdas de `arc174` corregido;
   n=6 conserva la misma cobertura a umbral 2% que n=3 y n=10, 40% menos
   costo). El par baseline/telemetry que mide overhead de instrumentación
   ya no se repite en cada corrida — con 540 pares medidos (media 1.95%,
   estable entre kernels) el número ya está caracterizado; el manifiesto
   nuevo declara `baseline_repetition_indices: [1]` como vigilancia contra
   desvío del instrumento, no medición completa de nuevo.

## 6. Impulso: RAJAPerf ya está en pacca, casi sin usar

Cuántos kernels usa cada trabajo citado: `Guerreiro2019` 35,
`Calore2017` **2** (una sola app, y aun así resultado real y citable),
`Hebbar2022` 43 (SPEC CPU2017 — **licencia paga, no reproducible por
nosotros**), `Antici2024` producción a escala Fugaku. **El número no
decide; decide la diversidad de régimen cubierta.**

El catálogo CPU actual (9 kernels) tiene exactamente **un** candidato
memory-bound con margen real. **RAJAPerf ya está descargado y compilado en
pacca**, y el catálogo usa **1** de sus kernels. Conteo verificado
(2026-08-24; el conteo previo estaba inflado al doble por un artefacto):

| categoría | kernels | | categoría | kernels |
|---|---|---|---|---|
| `apps` | 22 | | `algorithm` | 8 |
| `basic` | 20 | | `comm` | 6 |
| `polybench` | 13 | | `stream` | 5 |
| `lcals` | 11 | | **total** | **85** |

`raja-perf.exe` corre cualquiera con `-k NOMBRE -v Base_OpenMP`: agregar
un kernel es un wrapper, **no una compilación** — mismo binario, checksum
ya verificado.

**Job 6473 — encolado, PENDING** detrás de un job ajeno que lleva ~15 h
ocupando el nodo. Tamiza 7 candidatos de Polybench elegidos por
conocimiento algorítmico clásico (stencils y productos matriz-vector:
`JACOBI_1D`, `JACOBI_2D`, `HEAT_3D`, `FDTD_2D`, `ATAX`, `GESUMMV`, `MVT`).
Script `scripts/pacca/screen_rajaperf_cpu_alpha.sh`: bypasea el
orquestador por diseño (solo tiempo + RAPL, sin `perf`/uncore), así que no
espera a CAP_PERFMON.

## 7. Reconciliación con la variación intra-kernel (Objetivo 2 literal)

No toda la evidencia apunta a "cero variación intra-kernel". La tabla de
clase minoritaria de `resultados_compuertas_fase2.md` muestra **mezcla
real** en:

| kernel | REF | F0 | F1 | F2 | F3 | F4 | global |
|---|---:|---:|---:|---:|---:|---:|---:|
| `npb_lu` | 0.0 | 0.0 | 0.0 | 8.5 | 34.4 | 30.9 | **19.0%** |
| `npb_bt` | 0.6 | 0.8 | 19.4 | 21.5 | 14.0 | 10.2 | **11.8%** |
| `rajaperf_polybench_3mm_omp` | 5.7 | 5.6 | 4.8 | 3.7 | 2.5 | 1.7 | 3.2% |

Los otros seis están por debajo de 1%, y **`npb_mg`, `npb_cg` y
`dgemm_n2048` son 0.0% en los seis niveles** — una sola clase de principio
a fin.

Dos matices que importan y que la versión anterior de este documento no
distinguía:

- En `npb_bt`/`npb_lu` la mezcla aparece **solo a frecuencias bajas**: es
  el ridge de Roofline desplazándose y cruzando el punto de operación
  (ARC-175), no fases que se alternen por naturaleza del algoritmo.
- **`rajaperf_polybench_3mm_omp` es el único con mezcla a REF/F0**
  (5.7%/5.6%), es decir a frecuencia máxima, donde el ridge no se movió.
  Es el caso más interesante para un clasificador en vivo, y **es
  precisamente de la suite que §6 propone ampliar**.

La arquitectura no abandona el clasificador en vivo: lo reorienta hacia
la granularidad de carga como **mecanismo primario** (con el precedente
triple Guerreiro/Calore/Antici) y conserva la clasificación por ventana
como **mecanismo secundario** para los casos con mezcla real —
`npb_lu`, `npb_bt` y `rajaperf_polybench_3mm_omp`. No es una arquitectura
nueva: es priorizar la que el dato respalda, sin descartar la otra.

## 8. Mapeo a los Objetivos Específicos

| Objetivo | Estado | Evidencia |
|---|---|---|
| 1. Caracterizar comportamiento y consumo bajo distintos estados de frecuencia (Perf+RAPL) | **Cumplido** (424/540 válidas), ampliándose vía RAJAPerf | `..._arc174`; §2, §6 |
| 2. Clasificador ML en vivo, baja latencia | Features ya libres de uncore; granularidad primaria = carga, secundaria = ventana en los 3 kernels con mezcla real | §1, §7 |
| 3. Daemon de espacio de usuario con política DVFS | **No bloqueado por CAP_PERFMON**: el runtime nunca dependió de uncore, y la actuación (`scaling_min/max_freq`) ya tiene permiso P1 | §1 |
| 4. Evaluación por EDP contra gobernador nativo | **Parcial**: `REF` cubre el gobernador nativo *activo* (`performance`) sin permisos nuevos; permiso para `powersave` concedido (2026-08-25), pendiente de verificación empírica antes de usarse | §4; riesgo 3 |

## 9. Lista de pruebas — cómo verificar o refutar esta propuesta

| # | Qué verifica | Cómo | Pasa si | Si falla |
|---|---|---|---|---|
| **C1** | ¿El tamizaje encuentra margen en CPU? (§5.1, §6) | Job 6473: ajustar α por candidato sobre tiempo medido | ≥1 candidato con α < 0.226 **y** r² > 0.95 | El catálogo CPU no tiene margen accesible: reportar como resultado negativo cuantificado, no insistir con más resolución |
| **C2** | **El pineo de frecuencia se sostuvo bajo carga** (riesgo 6) | Muestrear `scaling_cur_freq` *durante* la corrida, no antes | Media observada dentro de 5% del objetivo en los 6 CPUs | Los tiempos del tamizaje son inválidos: repetir con verificación bajo carga (lección CAL-07/ARC-164) |
| **C3** | El tamizaje no está contaminado por I/O (lección `myocyte`, Anexo L.1) | Verificar tamaño de los archivos que RAJAPerf escribe por corrida | Salida despreciable frente al tiempo de cómputo | Un α bajo puede ser costo fijo de I/O, no memory-boundness: descontarlo antes de aceptar el candidato |
| **C4** | Robustez de la conclusión de 0.33 pts (§2) | Reejecutar `cpu_policy_headroom.py` optimizando **EDP** en vez de energía | La conclusión no cambia cualitativamente | Si bajo EDP aparece margen, la métrica —no la física— era la limitante, igual que pasó en GPU (Anexo M) |
| **C5** | `REF` ≡ gobernador nativo activo (§4) | Leer `scaling_governor` en cada corrida, no asumirlo | `performance` en los 6 CPUs delegados | La comparación del Objetivo 4 no es contra lo que se afirma: corregir antes de reportar |
| **C6** | Anti-fuga de etiqueta (§1) | Test unitario: inyectar a propósito una columna `FORBIDDEN` | El test **falla** ruidosamente | El guardarraíl no protege: todo resultado del modelo queda invalidado hasta arreglarlo |
| **C7** | ¿La arquitectura corregida mejora sobre el trivial? (§5.3) | Reentrenar por carga, LOKO, contra el predictor trivial (0.371 F1 macro) y contra la mejor constante elegida **solo en folds de entrenamiento** | Supera al trivial de forma significativa, no por décimas | La granularidad por carga tampoco alcanza en CPU: el aporte del eje CPU es la caracterización (Objetivo 1), y hay que decirlo |
| **C8** | Mecanismo secundario por ventana (§7) | Entrenar solo sobre `npb_lu`/`npb_bt`/`3mm_omp`, que sí tienen mezcla | Mejora sobre el trivial *dentro* de ese subconjunto | La mezcla existe pero no es aprendible con estas features: documentarlo como límite medido |
| **C9** | Latencia de inferencia (Objetivo 2) | p50/p99 de una inferencia (ya medido antes: 103 µs árbol, 33 ms ensembles) | p99 « período de decisión del daemon | Descartar los ensembles y quedarse con árbol/regresión, como ya sugieren los datos previos |
| **C10** | Consistencia documento↔dato | Reejecutar `cpu_policy_headroom.py` y comparar contra §2 | Coinciden | Actualizar: los números deben salir del script, nunca copiarse a mano |

---

## Riesgos abiertos, sin adornar

1. **El margen de 0.33 pts es sobre el catálogo actual**; el job 6473
   sigue PENDING, sin resultado. No se sabe si el tamizaje encontrará
   kernels CPU con margen comparable al de GPU.
2. **Los 7 candidatos se eligieron por conocimiento algorítmico**, no por
   OI medida — misma aproximación declarada que se usó en GPU, con el
   mismo riesgo de que alguno no sea memory-bound en la práctica.
3. **`powersave` — el permiso ya se concedió (2026-08-25), pero no está
   verificado.** El administrador entregó `set_cpu_gov <gobernador> <epp>`,
   restringido a los cores 0-5 y sin parámetro de rango de CPUs. Su propio
   correo argumenta que exclusividad + *pinning* de la carga (ya
   implementados en el runner, `--exclusive` + `sched_setaffinity` vía
   `--pin-workload-cpus`) bastan para neutralizar la interferencia de los
   hermanos SMT (16-21). **Nuestra propia Prueba A de ARC-162 contradice
   esa conclusión para `scaling_min_freq`/`scaling_max_freq`**: con la
   carga ya confinada a 0-5 por `taskset` (la misma receta), el reloj
   físico no escaló (razón F0/F4 = 0.995) hasta que el candado de
   frecuencia se aplicó TAMBIÉN a los hermanos 16-21 (razón = 3.78,
   ARC-163). Si el mismo mecanismo de coordinación de P-state aplica a
   gobernador/EPP, `set_cpu_gov` sobre 0-5 podría no bajar el reloj físico
   compartido — y el script entregado no permite aplicarlo también a
   16-21. **Pendiente una prueba tipo Prueba A/B antes de usarlo en
   cualquier campaña real** (`ert_probe` bajo `powersave power`, medir
   `scaling_cur_freq`/potencia real). Abierto a propósito hasta que haga
   falta para la comparación de la Fase 4 contra ambos gobernadores
   nativos — no bloquea ninguna corrida ni prueba en curso hoy.
4. **CAP_PERFMON sigue sin fecha.** El plan de §5 avanza sin esperarlo,
   pero **la ampliación del dataset con etiqueta de verdad sí depende de
   él**: el bypass ARC-191 exime solo datasets 100% GPU, no kernels
   `device: cpu` reales.
5. **El modelo con la arquitectura corregida todavía no se ha corrido** —
   es el paso siguiente, no un resultado obtenido.
6. **Riesgo nuevo, detectado al auditar el propio script de tamizaje
   (`screen_rajaperf_cpu_alpha.sh`):** pinea `min=max` y espera 1 s fijo,
   pero **no verifica la frecuencia real bajo carga**. ARC-160/164
   documentó que bajo `intel_pstate`+HWP con EPP=`performance` el
   decaimiento hacia un techo más bajo tarda **segundos**, y que
   `scaling_cur_freq` leído en reposo no refleja el pineo. Si el
   transitorio contamina el arranque de cada corrida, los tiempos —y por
   lo tanto α— quedan sesgados. **La prueba C2 existe para detectarlo**;
   hasta que pase, los resultados del job 6473 deben leerse como
   provisionales.

---

## Referencias

Ya en `docs/libro/main.tex`: `\cite{Calore2017}`, `\cite{Hebbar2022}`,
`\cite{Guerreiro2019}`, `\cite{Antici2024}`, `\cite{Williams2009}`.
