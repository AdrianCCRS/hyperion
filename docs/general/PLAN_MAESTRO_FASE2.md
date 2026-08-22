# Plan maestro de Fase 2 — Hyperion

**Versión:** 2026-08-22
**Estado del nodo:** paccaA100 ocupado por terceros; `pacca01` (partición
`normal`) ocioso y disponible para análisis.
**Documentos que este reemplaza como referencia única:**
`estado_y_decisiones_fase2.md`, `resultados_compuertas_fase2.md`,
`opciones_modelo_fase2.md`, `diseno_campana_final_fase2.md`,
`estado_fase2_compuertas_20260822.md`. Los cinco siguen siendo válidos como
registro histórico; este es el que se ejecuta.

> **Regla que gobierna todo este plan.** Ninguna campaña de más de una hora
> se lanza sin un pre-vuelo que falsee primero sus supuestos, y ningún
> supuesto verificable leyendo código o reanalizando datos existentes se
> verifica gastando nodo. Esta regla nació de perder cobertura completa en
> F0–F2 para dos kernels y de descubrir después, gratis, dos defectos que
> habrían invalidado 4 h de medición.

---

# PARTE 0 — Resumen ejecutivo

**La idea original no funcionó, y ya se sabe por qué.** No es la
plataforma: es el catálogo. El régimen físico donde bajar la frecuencia
ahorra energía **existe** en paccaA100 —`stream_official` está en
α = 0.154, por debajo del umbral de viabilidad 0.226— pero **ninguno de los
nueve kernels del dataset lo toca**: su α mínimo es 0.242 y ninguno supera
el 74 % del ancho de banda alcanzable.

**Tres problemas estructurales, en orden de gravedad:**

1. **El catálogo no cubre el régimen memory-bound real.** Ningún kernel
   satura memoria. Cinco no llegan al 30 % del ancho de banda de STREAM.
2. **El catálogo no es una muestra.** Seis de los nueve kernels de CPU son
   NPB clase B: misma suite, misma clase, mismos flags, misma estructura.
   LOKO sobre pliegues no independientes no mide generalización.
3. **El catálogo no tiene variación de fase intra-ejecución.** Clase
   minoritaria del 4.0 % de media, cuatro kernels en 0.0 %.

**Lo que se hace al respecto:** reestructurar el dataset en tres ejes
(régimen, diversidad, fase), validando cada supuesto con pruebas cortas
antes de comprometer horas. Las pruebas están en la Parte IV con su costo,
dónde corren, y qué conclusión sale de cada resultado posible.

---

# PARTE I — La idea original y por qué se cayó

## I.1 Lo que se planteó

Un clasificador supervisado recibe telemetría por ventana de ejecución,
predice el régimen (compute-bound / memory-bound), y una política usa esa
predicción para elegir la frecuencia que minimiza el EDP.

## I.2 Lo que se midió

| Evidencia | Resultado |
|---|---|
| Óptimo de EDP por kernel | Frecuencia **máxima** en 9 de 9 |
| Ahorro bajo cualquier presupuesto | ~0.7 % de media |
| Óptimo distinto entre fases | 8 de 9 kernels con ≥99 % en un solo nivel |
| Clasificador binario bajo LOKO | F1 **0.393** contra trivial **0.371** |
| Random Forest binario | 0.358 — **peor** que el trivial |
| Regresor sobre `b` continuo bajo LOKO | R² **negativo en los 9 pliegues** |
| Clase minoritaria intra-kernel | **4.0 %** de media; cuatro kernels en 0.0 % |

## I.3 Por qué se cayó — la cadena causal

```
El catálogo solo contiene cargas limitadas por núcleo
        ↓
alpha >= 0.242 en las 9000 celdas medidas  (umbral: 0.226)
        ↓
El óptimo de EDP es la frecuencia máxima, siempre
        ↓
La etiqueta no varía dentro de una ejecución (4.0 % minoritaria)
        ↓
El modelo no tiene nada que aprender que el trivial no sepa
        ↓
LOKO: R² negativo en los nueve pliegues
```

Cada eslabón está medido. **El primero es el único que hay que romper**;
los demás se caen solos si ése cae.

---

# PARTE II — Inventario completo de aristas

Todo lo encontrado, con su estado. **Cerrado** = medido y resuelto.
**Abierto** = sin resolver. **Retractado** = se afirmó y resultó falso.

## II.1 Física de la plataforma

| # | Arista | Estado | Evidencia |
|---|---|---|---|
| A1 | Umbral de viabilidad de EDP en CPU: **α ≤ 0.226** | Cerrado | Derivado de `P_rel·(1+r·α)² < 1` con el modelo de potencia real |
| A2 | Umbral en GPU: **α ≤ 0.639** | Cerrado | Mismo derivado; rango dinámico GPU 2.54× |
| A3 | **Piso de potencia alto en CPU**: 116.5 W a 3200 MHz, 83.4 W a 800 MHz → rango dinámico de solo **1.40×** | Cerrado | La restricción estructural del eje CPU. Es la razón de que el umbral sea tan bajo |
| A4 | El ridge del Roofline **depende de la frecuencia**: 8.733 → 2.992 FLOP/byte de 3200 a 800 MHz (×0.343) | Cerrado | Propiedad del nodo, idéntica entre kernels. 2 de 9 kernels cambian de etiqueta |
| A5 | **El uncore NO está acoplado** al reloj de núcleo | Cerrado (retractado lo contrario) | STREAM conserva **78.4 %** de su ancho de banda al 25 % del reloj (pendiente 0.29) |
| A6 | **α > 1 en cuatro kernels** (dgemm 1.159, ft 1.327, lavamd 1.092, 3mm 1.056) con r² 0.96–0.999 | **ABIERTO** | Imposible bajo el modelo de Amdahl. Descartado el uncore, la causa se desconoce |

## II.2 El catálogo — la raíz del problema

| # | Arista | Estado | Evidencia |
|---|---|---|---|
| B1 | **`stream_official`: α = 0.1538** (r² 0.982) — **bajo el umbral** | Cerrado | Ajuste directo sobre duraciones reales |
| B2 | `ert_probe`: α = 0.2277 — justo en el umbral | Cerrado | Ídem |
| B3 | Los 9 kernels del dataset: **α mínimo 0.242** sobre 9000 celdas | Cerrado | C2, job 6424 |
| B4 | **Ninguno satura memoria.** Máximo 74 % del ancho de banda de STREAM (npb_mg); cinco por debajo del 30 % | Cerrado | Job 6427 |
| B5 | **6 de 9 kernels de CPU son NPB clase B** — misma suite, misma clase, mismos flags | **ABIERTO** | Los pliegues de LOKO no son independientes |
| B6 | **Un solo programa multifásico** (`phasic.c`, instanciado ×3 períodos) | **ABIERTO** | Pseudo-replicación: n = 1, no n = 3 |
| B7 | Todos los kernels son "puros" — un régimen dominante por programa | **ABIERTO** | Las aplicaciones reales son mixtas |
| B8 | Un solo tamaño por kernel; NPB entero en clase B | **ABIERTO** | El eje de tamaño mueve el régimen y no se está usando |

### Anchos de banda medidos (job 6427), referencia STREAM F0 = 76.80 GB/s

| kernel | BW F0 (GB/s) | % de STREAM | pendiente BW vs f |
|---|---|---|---|
| npb_mg | 57.18 | 74 % | 0.681 |
| npb_cg | 48.35 | 63 % | 0.889 |
| npb_sp | 43.73 | 57 % | 1.024 |
| npb_ft | 21.76 | 28 % | 0.960 |
| dgemm_n2048 | 21.01 | 27 % | 1.011 |
| npb_lu | 5.84 | 8 % | 1.009 |
| npb_bt | 3.88 | 5 % | 0.984 |
| 3mm_omp | ~0.18 | <1 % | 0.968 (ruido) |
| lavamd_omp | ~0.07 | <1 % | 0.571 (ruido) |

Pendiente ≈ 1.0 significa "no saturaba memoria, solo emite peticiones más
despacio". Pendiente ≈ 0 significaría saturación real. STREAM: 0.29.

## II.3 Rejilla de frecuencia

| # | Arista | Estado |
|---|---|---|
| C1 | Niveles medidos: 3200, 2600, 2000, 1400, 800 MHz (+ REF) | Cerrado |
| C2 | **Hueco 2600–3200 MHz sin una sola medición** | **ABIERTO** |
| C3 | Bajo objetivo de energía con holgura `s`: `f* = f_ref/(1+s/α)`. Con α≈0.9 y s=5 % → **~3032 MHz**; con α=0.5 → **~2909 MHz**. Para los 9 kernels cae en **2831–3051 MHz** | **ABIERTO** — todo dentro del hueco |
| C4 | La campaña de rejilla fina en esa zona (job 6391) se **canceló por un razonamiento equivocado** | Debe volver |

## II.4 Validez de la medición

| # | Arista | Estado | Nota |
|---|---|---|---|
| D1 | **Conteo de bytes de uncore CORRECTO** | Cerrado (retractada la duda) | 13.0 ventanas por intervalo, exacto e idéntico en los 6 niveles sobre 540 corridas. STREAM 76.80 GB/s contra 59.50 declarados = 1.291 = factor write-allocate |
| D2 | `bytes_moved_uncore_real / delta_t_ns` **fila por fila está MAL** | Cerrado | Hay que agrupar por intervalo. Este error produjo una falsa alarma de "inflación ×17" |
| D3 | La coordenada de avance por **instrucciones retiradas es invariante a la frecuencia** (0.34 % peor caso) | Cerrado | Habilita la alineación de fases entre niveles |
| D4 | `windows.csv` tiene `repetition = 1` siempre; el índice real está en el nombre del directorio | Cerrado | Agrupar por esa columna fusiona las 10 repeticiones |
| D5 | Turbo debe estar desactivado en toda corrida de frecuencia fija | Cerrado | `with_cpu_turbo_disabled.sh`, obligatorio |
| D6 | `--exclusive` obligatorio para RAPL de paquete y uncore (E11) | Cerrado | |
| D7 | `catalog_path` roto en los 20 manifiestos tras la reorganización | Cerrado | Cualquier relanzamiento habría fallado al arrancar |

## II.5 Los kernels de fase (instrumentos)

| # | Arista | Estado |
|---|---|---|
| E1 | `phasic.c` compilaba a **`mulsd`/`addsd` escalar sobre xmm** (~1/16 del pico) | Cerrado — ahora `vfmadd132pd` sobre `zmm` |
| E2 | Causa raíz de E1: eran los **únicos kernels sin script de build** | Cerrado — `scripts/pacca/build_phase_kernels.sh`, que verifica el ancho con objdump |
| E3 | `-mprefer-vector-width=512` **no es redundante** con `-march=native` | Cerrado (ARC-125 ya lo sabía) |
| E4 | Las marcas `PHASE` eran offsets **sin ancla de reloj** | Cerrado — `T0_MONOTONIC_NS`, mismo reloj que el colector |
| E5 | `gpu_phasic.cu` **sin compilar** (requiere nvcc en paccaA100) | **ABIERTO** |
| E6 | `ptrchase` es **monofásico** (memoria pura), no multifásico | Por diseño — es sonda de α, no sujeto |

## II.6 Eje GPU

| # | Arista | Estado |
|---|---|---|
| F1 | Solo **4 de 8 kernels** ejercitan la GPU de verdad (dgemm 98.5 %, gaussian 99.2 %, lavamd 100 %, heartwall 93.9 % activos) | **ABIERTO** |
| F2 | `rodinia_lud` y `rodinia_heartwall` **salen con código 0 en una máquina sin GPU** | **ABIERTO** — su `success_check` no valida CUDA |
| F3 | El **piso de utilización crea sesgo dependiente de frecuencia**: `lud` aceptado 0/3 en REF y F0 pero 3/3 en F1–F4, porque util≥5 % crece al bajar el reloj | **ABIERTO** — bloquea las campañas GPU |
| F4 | Tasa real de filas GPU utilizables: **42.6 %** (no 7.4 %, que fue error de denominador) | Cerrado |
| F5 | `rodinia_lavamd` GPU: **α = 0.201**, 100 % activo, hasta 247 W | Cerrado — el único resultado GPU valioso |
| F6 | El "óptimo de −30.5 % EDP en F4" de `lud` es **ahorro de potencia en reposo**, no adaptación DVFS | Cerrado |
| F7 | El shim de blocking-sync no sincronizaba: datos GPU **anteriores a 2026-08-19 corrieron en modo spin** | Cerrado |
| F8 | `test.avi` de heartwall estaba **truncado** y `success_check` no lo detectaba | Cerrado |
| F9 | El umbral de GPU (α ≤ 0.639) es **mucho más laxo** que el de CPU por el rango dinámico 2.54× | Cerrado — el eje GPU es intrínsecamente más prometedor |

## II.7 El modelo

| # | Arista | Estado |
|---|---|---|
| G1 | `b = σ(−k·log₁₀(OI/I_ridge(f)))`. En OI = ridge da 0.5, así que umbralizar reproduce la etiqueta de Fase 1 | Cerrado — **acuerdo 1.000000** sobre 9.95M ventanas |
| G2 | `b` **sí varía dentro de una ejecución** donde la etiqueta binaria era plana (npb_cg abarca 0.113 con minoritaria 0.0016 %) | Cerrado |
| G3 | **15.3 %** de la varianza de `b` es intra-corrida | Cerrado |
| G4 | **No se ha separado señal de ruido** en ese 15.3 % | **ABIERTO** — falta autocorrelación |
| G5 | α **sí varía entre tramos**: npb_ft sd 0.302 (0.40→1.33), npb_mg sd 0.194 | Cerrado |
| G6 | Pero **0 de 9000 celdas** bajan del umbral → la política derivada es constante | Cerrado |
| G7 | **C3 falla**: R² negativo en los 9 pliegues, gana 5/9, mejora de MAE 3.9 % | Cerrado |
| G8 | Fuga de etiqueta: `operational_intensity*`, `i_ridge_used`, `flops_measured_window`, `bytes_moved_*`, `uncore_cas_count_*`, `phase_label_*` **prohibidos como entrada** | Cerrado |
| G9 | Pseudo-replicación: 10M ventanas pero **n efectivo = 9** | **ABIERTO** — agravado por B5 |
| G10 | **D2 sin decidir**: si la segunda salida es α o `f_opt` | **ABIERTO** |

---

# PARTE III — Diagnóstico corregido

## III.1 Lo que NO es el problema

- **No es la plataforma.** El uncore no está acoplado (A5) y existe el
  régimen viable (B1). Bajar la frecuencia SÍ paga para cargas
  suficientemente memory-bound en este nodo.
- **No es la instrumentación.** El conteo de bytes está validado contra
  física conocida (D1), la etiqueta continua reproduce exactamente la
  binaria (G1), y la coordenada de avance es invariante a la frecuencia
  (D3).
- **No es la rejilla de frecuencia**, al menos no para el objetivo de EDP.
  Sí lo es para el objetivo de energía con holgura (C2/C3).
- **No es el diseño del target.** `b` recupera estructura que el binario
  destruía (G2).

## III.2 Lo que SÍ es el problema

**El catálogo, en tres ejes simultáneos:**

| Eje | Defecto | Consecuencia |
|---|---|---|
| **Régimen** | Ningún kernel satura memoria (B4); α mínimo 0.242 contra umbral 0.226 (B3) | El óptimo es siempre la frecuencia máxima. No hay decisión que aprender |
| **Diversidad** | 6 de 9 son NPB clase B (B5); un solo tamaño por kernel (B8); todos "puros" (B7) | LOKO no mide generalización porque los pliegues no son independientes |
| **Fase** | Clase minoritaria 4.0 %, cuatro kernels en 0.0 %; un solo programa multifásico (B6) | No hay variación intra-ejecución que clasificar |

**Los tres hay que romperlos.** Romper solo el de régimen daría un modelo
que acierta pero no se puede evaluar. Romper solo el de diversidad daría
una evaluación honesta de un modelo sin nada que predecir.

## III.3 El margen de maniobra que sí existe

1. **El régimen viable existe** — STREAM lo prueba (B1). Hay que poblarlo.
2. **El eje GPU tiene umbral 2.8× más laxo** (F9) y ya hay un caso
   confirmado dentro (lavamd, α = 0.201, F5).
3. **El objetivo de energía con holgura** tiene su óptimo en una zona sin
   medir (C3), y ese es el objetivo general de la tesis, no el EDP puro.
4. **El eje de tamaño no se ha usado** (B8) y es la palanca más barata:
   mueve el régimen sin inventar kernels.

---

# PARTE IV — Plan maestro de pruebas

Cada prueba lleva: qué falsea, dónde corre, cuánto cuesta, y **qué
conclusión sale de cada resultado posible**. Ordenadas por dependencia.

> **Convención de nodos.** `pacca01` (partición `normal`, 104 CPU, 257 GB,
> ocioso) para todo lo que sea aritmética sobre datos ya medidos.
> `paccaA100` (partición `GPU`, `--exclusive`) solo para medir.
> **La latencia de inferencia es la única magnitud de análisis que debe
> medirse en paccaA100**, porque es una afirmación sobre el hardware de
> despliegue.

---

## OLA 0 — Gratis, sobre datos existentes. Empezar aquí.

### T0.1 — Autocorrelación de `b`: ¿señal o ruido? *(cierra G4)*

- **Falsea:** que el 15.3 % de varianza intra-corrida sea estructura de
  fase y no ruido de muestreo.
- **Cómo:** autocorrelación a rezago 1..k de `b` dentro de cada corrida, a
  lo largo del índice de ventana. Comparar contra la misma serie permutada
  al azar (control nulo).
- **Dónde/costo:** `pacca01`, ~10 min.
- **Conclusiones:**
  - ACF(1) alta y decaimiento lento → **es señal**. La primera salida del
    modelo tiene base y B6/B7 son menos urgentes.
  - ACF(1) ≈ 0 → **es ruido**. `b` no aporta nada intra-ejecución con este
    catálogo y hay que apoyarse enteramente en cargas nuevas.

### T0.2 — Explicar α > 1 *(cierra A6)*

- **Falsea:** las tres hipótesis restantes tras descartar el uncore:
  (a) contención de SMT o de núcleos no delegados;
  (b) efecto térmico/de licencia AVX que reduce la frecuencia efectiva a
  alta frecuencia y no a baja; (c) el ajuste sin intercepto amplificando un
  sesgo sistemático en `T(f_ref)`.
- **Cómo:** contrastar `freq_khz_observed` real contra la solicitada por
  nivel y kernel; recomputar α usando frecuencia observada en vez de
  nominal; revisar temperatura de paquete por nivel.
- **Dónde/costo:** `pacca01`, ~15 min.
- **Conclusiones:**
  - Si α cae bajo 1 al usar frecuencia observada → era un artefacto de
    usar la frecuencia nominal. **Todos los α del proyecto hay que
    recalcularlos**, y el umbral se compara contra los nuevos.
  - Si persiste → hay un mecanismo físico sin identificar y debe quedar
    documentado como limitación explícita.

### T0.3 — Reevaluar C3 con pliegues honestos *(ataca B5/G9)*

- **Falsea:** que el fallo de C3 sea del modelo y no de la evaluación.
- **Cómo:** rehacer LOKO agrupando por **suite** y no por kernel — los seis
  NPB como un solo pliegue. Da 4 pliegues reales (NPB, DGEMM, Rodinia,
  RAJAPerf) en vez de 9 falsos. Reportar también el techo: R² entrenando y
  probando dentro del mismo kernel.
- **Dónde/costo:** `pacca01`, ~30 min.
- **Conclusiones:**
  - Si el techo intra-kernel es alto y LOKO-por-suite sigue en R² negativo
    → el problema es **generalización entre regímenes**, y solo se arregla
    poblando regímenes nuevos.
  - Si el techo intra-kernel también es bajo → **los rasgos no contienen
    la información**, y hay que revisar el vector de entrada antes que el
    catálogo.

### T0.4 — Mapa de régimen del catálogo actual *(informa B4/B8)*

- **Falsea:** dónde cae cada kernel en el plano (α, % de ancho de banda
  saturado), y cuánto habría que moverlo para cruzar el umbral.
- **Cómo:** tabla y gráfico de α contra fracción de BW de STREAM, con la
  línea de umbral 0.226. Extrapolar qué fracción de saturación
  correspondería a α = 0.226.
- **Dónde/costo:** `pacca01`, ~10 min.
- **Conclusión:** da el **objetivo cuantitativo** para los kernels nuevos:
  "hay que llegar a ≥ X % de saturación de memoria". Sin esto, agregar
  cargas es a ciegas.

---

## OLA 1 — Pre-vuelos cortos en paccaA100. Cada uno < 30 min.

### T1.1 — Pre-vuelo de fases *(job 6420, YA EN COLA)*

- **Manifiesto:** `campaign_pacca_phase_preflight.yaml`
- **Costo:** 27 corridas, ~20 min.
- **Falsea cinco supuestos:**

  | | Supuesto | Conclusión si falla |
  |---|---|---|
  | S1 | `ptrchase` alcanza α ≤ 0.226 | Si `ptrchase` —latencia pura— no baja del umbral cuando STREAM sí, el modelo de α está mal especificado, no el catálogo |
  | S2 | Los kernels nuevos pasan validación de frecuencia en F4 | Huecos donde está la física; hay que ajustar `grace`/`tail_grace` antes de la campaña grande |
  | S3 | La fase se resuelve en ventanas de 1 ms | `phasic_p010` no vale y la campaña grande no debe gastar corridas en él |
  | S4 | La etiqueta de verdad cruza con `windows.csv` | El ancla `T0_MONOTONIC_NS` no funciona y `phasic` pierde su razón de ser |
  | S5 | La fase de cómputo es comparable con las cargas reales | Si su potencia está muy por debajo de `3mm_omp`, su EDP no transfiere |

- **Prior actual:** fuerte a favor de S1. STREAM (limitado por ancho de
  banda) da 0.154; `ptrchase` está limitado por **latencia**, que es un
  régimen aún menos sensible al reloj.

### T1.2 — Pre-vuelo de tamaños NPB *(ataca B8, la palanca más barata)*

- **Falsea:** que subir la clase de NPB mueva el régimen lo suficiente para
  cruzar el umbral.
- **Diseño:** `npb_cg` y `npb_mg` —los dos más cerca de saturar (63 % y
  74 %)— en clase **C**, contra su clase B actual. Solo F0 y F4, 3 reps.
  **2 kernels × 2 clases × 2 niveles × 3 reps = 24 corridas.**
- **Requiere antes:** compilar `cg.C.x` y `mg.C.x` con
  `scripts/felix/build_npb.sh` (ya pone `-march=native`), y verificar que
  la huella de memoria cabe.
- **Costo:** ~25 min.
- **Conclusiones:**
  - α baja al subir de clase → **el eje de tamaño funciona**. Se extiende a
    todas las clases viables y el catálogo se multiplica por un eje
    físicamente significativo, sin inventar kernels ni perder
    comparabilidad con la literatura. Es el mejor resultado posible.
  - α no se mueve → el tamaño no es la palanca; hay que ir a cargas
    intrínsecamente distintas (T1.3).

### T1.3 — Pre-vuelo de cargas nuevas *(ataca B5/B7)*

- **Falsea:** que una aplicación multifásica real aporte lo que los
  sintéticos no pueden (credibilidad externa) y lo que NPB no aporta
  (independencia de pliegue).
- **Candidata:** **HPCG** — estándar, citable, OpenMP, y con fases
  documentadas en la literatura (SpMV, Gauss-Seidel simétrico, multigrid,
  productos punto). No depende de que nosotros afirmemos su etiqueta.
- **Diseño:** HPCG, F0 y F4, 3 reps = 6 corridas. ~20 min incluyendo build.
- **Conclusiones:**
  - α < 0.226 y fases detectables → **es el sujeto que faltaba**, y
    `phasic` queda claramente como instrumento y no como evidencia.
  - α alto → HPCG tampoco cubre el régimen; la evidencia de régimen tendrá
    que descansar en los sintéticos, y hay que decirlo explícitamente como
    limitación.

### T1.4 — Pre-vuelo de rejilla fina *(cierra C2/C3)*

- **Falsea:** que exista una frecuencia entre 2600 y 3200 MHz que mejore la
  energía con holgura de tiempo acotada.
- **Diseño:** los 3 kernels con α más bajo (npb_sp 0.499, npb_mg 0.664,
  npb_cg 0.782) en **3000 y 2800 MHz**, más F0 como ancla. 3 reps.
  **3 kernels × 3 niveles × 3 reps = 27 corridas**, ~25 min.
- **Conclusiones:**
  - Hay un mínimo de energía interior con degradación < 5 % → **el objetivo
    de energía con holgura sí tiene decisión que tomar**, y el modelo
    recupera su razón de ser aunque el EDP puro no la dé. Este es el
    segundo mejor resultado posible del plan.
  - No lo hay → el objetivo con holgura tampoco produce decisión en CPU, y
    el peso del trabajo se traslada al eje GPU (F9).

---

## OLA 2 — Campañas completas. Solo lo que la Ola 1 haya validado.

### T2.1 — Campaña de fases *(job 6412, EN `hold`)*

Su diseño depende de T1.1. Ajustar antes de liberar:
- Si S3 falla, quitar `phasic_p010`.
- Si S2 falla, subir `grace_seconds`/`tail_grace_seconds`.
- Si S1 falla, replantear: el problema no sería el catálogo.

### T2.2 — Campaña final de CPU, reestructurada

**No se diseña hasta tener T0.1–T0.4 y T1.1–T1.4.** La estructura prevista,
en los tres ejes del diagnóstico:

| Eje | Contenido previsto | Depende de |
|---|---|---|
| Régimen | `ptrchase`, STREAM como kernel de dataset (no solo calibración), clases NPB altas | T1.1, T1.2 |
| Diversidad | HPCG u otra suite; ≥2 tamaños por kernel; reducir el peso de NPB por debajo del 50 % | T1.2, T1.3 |
| Fase | `phasic` (instrumento) + al menos una aplicación multifásica real (sujeto) | T1.1, T1.3 |
| Frecuencia | Añadir 3000 y 2800 MHz a la rejilla | T1.4 |

### T2.3 — Eje GPU

**Bloqueado por F3**, que es un defecto de criterio, no de medición: el
piso de utilización del 5 % acepta más corridas a baja frecuencia que a
alta, sesgando el dataset por construcción. Antes de cualquier campaña GPU:

1. Reemplazar el piso de utilización por un criterio **invariante a la
   frecuencia** (energía sobre el reposo), como campo opcional del
   manifiesto para no alterar lo ya encolado.
2. Arreglar `success_check` de `rodinia_lud` y `rodinia_heartwall`, que
   salen con código 0 sin GPU (F2).
3. Compilar `gpu_phasic.cu` (E5) — requiere nvcc en paccaA100.
4. Verificar si `myocyte`/`dwt2d`/`backprop` se recuperan con tamaños
   mayores.

**El eje GPU merece prioridad alta** por F9: su umbral es 2.8× más laxo y
ya hay un caso confirmado dentro (F5).

---

# PARTE V — Reestructuración del dataset

## V.1 Criterios que debe cumplir el catálogo nuevo

| Criterio | Métrica | Objetivo |
|---|---|---|
| Cobertura de régimen | Kernels con α ≤ 0.226 | **≥ 3**, hoy 0 |
| Cobertura de saturación | Kernels sobre el 80 % del BW de STREAM | **≥ 3**, hoy 0 |
| Independencia de pliegues | Fracción del catálogo de una sola suite | **≤ 40 %**, hoy 67 % |
| Eje de tamaño | Kernels con ≥ 2 tamaños medidos | **≥ 4**, hoy 0 |
| Variación de fase | Kernels con clase minoritaria ≥ 20 % | **≥ 2**, hoy 0 |
| Multifásicos reales | Programas multifásicos no sintéticos | **≥ 1**, hoy 0 |

Estos números son el criterio de aceptación del dataset. **Si la campaña
final no los cumple, no sirve, sin importar cuántas corridas tenga.**

## V.2 Por qué los tamaños y las variantes importan tanto

El régimen de un kernel no es una propiedad del algoritmo: es del
algoritmo **contra la jerarquía de memoria de este nodo**. La LLC del
Gold 5315Y es de 12 MiB por socket. Un kernel cuyo conjunto de trabajo
quepa en LLC es compute-bound por construcción; el mismo kernel con el
conjunto 40× mayor puede saturar DRAM.

Todos los NPB del catálogo están en **clase B**. Existen S, W, A, B, C, D.
Mover la clase es la forma canónica y citable de barrer ese eje, no cuesta
escribir código, y produce puntos comparables con toda la literatura de
NPB. **Es la palanca de mejor relación resultado/esfuerzo del plan**, y por
eso T1.2 va temprano.

Lo mismo aplica a `dgemm_n2048` (que a n4096 pasaría a otro régimen) y a
`3mm_omp` (`--sizefact`).

## V.3 Sintéticos: instrumentos, no sujetos

`ptrchase` y `phasic` establecen **si el mecanismo es físicamente posible**
y aportan etiqueta de verdad que ninguna aplicación real da. No pueden ser
la base de evidencia sobre cargas reales. El libro debe distinguirlos
explícitamente, o la defensa se cae con la pregunta obvia.

---

# PARTE VI — Diseño del modelo (D2)

## VI.1 La arquitectura

Regresión de doble salida, Optuna para hiperparámetros — la arquitectura
que pidió el director. Lo que cambia es qué predice la segunda salida.

- **Entrada** (por ventana): `ipc`, `mpki`, `llc_miss_rate`,
  `stall_backend_ratio`, `ips`, `running_ratio`, `freq_khz_observed`.
- **Prohibido** (G8): todo aquello de lo que se deriva el target.
- **Salida 1 — `b` ∈ [0,1]:** score continuo de acotamiento. 0 = compute,
  1 = memory. **Confirmada** por G1 y G2.
- **Salida 2 — abierta:** α o `f_opt`.

## VI.2 La decisión pendiente sobre la salida 2

| | α | `f_opt` |
|---|---|---|
| A favor | Da la **magnitud** de la ganancia (necesaria para el costo de conmutación); sirve para cualquier objetivo sin reentrenar; continua y suave | Es lo que pidió el director literalmente; interpretación directa |
| En contra | Un paso más de derivación | El argmin es discontinuo; hay que reentrenar por objetivo; **hoy es constante** (siempre la frecuencia máxima) |

**Qué la decide:** T1.1 y T1.4. Si `ptrchase` baja del umbral y/o la
rejilla fina revela un mínimo interior, `f_opt` deja de ser constante y la
elección pasa a ser un compromiso genuino que vale discutir con el
director. Si ninguna de las dos, α es la única con algo que predecir.

## VI.3 Protocolo de evaluación

- **LOKO por suite, no por kernel** (T0.3), mientras 6 de 9 sean NPB.
- Reportar siempre el **peor pliegue**, no solo la media.
- Baseline trivial obligatorio en cada reporte.
- La **latencia de inferencia** se mide en paccaA100, nunca en `pacca01`.

---

# PARTE VII — Registro de riesgos

| Riesgo | Impacto | Mitigación | Estado |
|---|---|---|---|
| α > 1 sin explicar (A6) | Si es artefacto de frecuencia nominal, **todos los α están mal** | T0.2 | Abierto, prioridad alta |
| El 15.3 % de `b` es ruido (G4) | La salida 1 pierde su justificación | T0.1 | Abierto |
| Los rasgos no contienen la señal | Ningún catálogo lo arregla | T0.3 (techo intra-kernel) | Abierto |
| Sesgo del piso GPU (F3) | Todo dataset GPU nace sesgado | Criterio invariante a frecuencia | Abierto, bloquea GPU |
| Pseudo-replicación de `phasic` (B6) | 3 entradas de 1 fuente no son n=3 | Tratarlas como un solo kernel en LOKO | Abierto |
| `gpu_phasic` sin compilar (E5) | No hay instrumento de fase en GPU | Requiere nvcc en paccaA100 | Abierto |
| El nodo sigue ocupado | Toda la Ola 1 se retrasa | La Ola 0 no depende de él | Vigente |

---

# PARTE VIII — Notas de ejecución

## VIII.1 Estado de la cola

| Job | Qué | Partición | Estado |
|---|---|---|---|
| 6420 | Pre-vuelo de fases (T1.1) | GPU | PENDING, tras el array de terceros |
| 6412 | Campaña de fases (T2.1) | GPU | **HELD** — liberar solo tras leer 6420 |
| 6424 | Compuertas C1/C2/C3 | normal | COMPLETADO |
| 6426 | Acoplamiento de uncore | normal | COMPLETADO (conclusión retractada) |
| 6427 | Auditoría de bytes | normal | COMPLETADO |

## VIII.2 Recetas

**Análisis en el nodo libre:**
```bash
sbatch -p normal -c 16 -t 01:00:00 -J hyp_x \
  -o ~/hyperion-results/analysis/x_%j.out \
  -e ~/hyperion-results/analysis/x_%j.err \
  --wrap "export PYTHONPATH=/home/latorresn/hyperion; \
          source ~/hyperion-venv/bin/activate; \
          python3 /home/latorresn/hyperion/classifier/analysis/SCRIPT.py ..."
```

**Sincronización:** siempre `git push` local → `git pull` en pacca. Nunca
`cat | ssh`, nunca comparación manual de checksums.

**Construir los kernels de fase:** `scripts/pacca/build_phase_kernels.sh`,
que verifica con objdump el ancho de vector realmente emitido y actualiza
los checksums que van al catálogo.

**Reordenar la cola:** `scontrol hold` / `scontrol release`, nunca
`scancel`. La cuenta es compartida: **jamás cancelar trabajos ajenos.**

## VIII.3 Trampas conocidas

- `bytes_moved_uncore_real / delta_t_ns` fila por fila **está mal** (D2).
  Agrupar por intervalo de uncore.
- La columna `repetition` vale 1 siempre (D4). Usar el nombre del
  directorio.
- Los directorios `__baseline` y `rep00` no son corridas de telemetría.
- `set -u` rompe `module load` en este clúster.
- El venv correcto es `~/hyperion-venv`, no el anaconda del sistema.
- `-march=native` solo no da AVX-512: hace falta
  `-mprefer-vector-width=512` (E3).

## VIII.4 Orden de ejecución recomendado

```
AHORA (sin nodo):      T0.1  T0.2  T0.3  T0.4
CUANDO SE LIBERE:      T1.1 (ya en cola)  →  T1.2  T1.4  T1.3
LUEGO:                 ajustar y liberar T2.1
                       diseñar T2.2 con los criterios de V.1
EN PARALELO:           desbloquear el eje GPU (T2.3), que tiene
                       el umbral más laxo y un caso ya confirmado
```

**T0.2 es la más urgente de las cuatro gratuitas:** si α > 1 resulta ser un
artefacto de usar la frecuencia nominal, todos los α del proyecto —incluido
el 0.242 que define el diagnóstico— hay que recalcularlos, y varias
conclusiones de este documento cambian.

---

# ANEXO A — Resultados de la Ola 0 (job 6429, 2026-08-22)

## A.1 T0.1 — La estructura de fase es SEÑAL, no ruido. *(cierra G4)*

Autocorrelación a rezago 1 de `b` dentro de cada corrida, contra la misma
serie permutada al azar (control nulo que conserva la distribución marginal
y destruye solo el orden temporal):

| kernel | ACF(1) | ACF(1) nulo | persistencia (ventanas a 1/e) |
|---|---|---|---|
| npb_cg | 0.9967 | −0.0028 | **200** |
| npb_ft | 0.9662 | 0.0008 | 17 |
| npb_bt | 0.9626 | −0.0002 | 14 |
| npb_lu | 0.9582 | −0.0008 | 13 |
| 3mm_omp | 0.9485 | −0.0005 | 50 |
| lavamd_omp | 0.9321 | −0.0021 | 9 |
| npb_sp | 0.9150 | 0.0007 | 6 |
| dgemm_n2048 | 0.8996 | −0.0037 | 5 |
| npb_mg | 0.8995 | −0.0064 | 5 |

**Inequívoco.** ACF entre 0.90 y 0.997 contra un nulo de ~0.000 en los
nueve. El 15.3 % de varianza intra-corrida de C1 **es estructura real**.

Y da algo que no se tenía: **la escala temporal de las fases es de 5 a 200
ventanas, o sea 5–200 ms.** La transición de P-state medida en este nodo es
de ~10–11 ms, así que `npb_cg` (200 ventanas) es holgadamente conmutable y
`dgemm`/`npb_mg` (5 ventanas) están en el límite. Ese contraste es
exactamente lo que `phasic_p010/p100/p1000` fue construido para barrer.

## A.2 T0.2 — α > 1 explicado: es el FILTRO DE CALIDAD *(cierra A6)*

| kernel | α filtrado | α crudo | retención F0 | retención F4 |
|---|---|---|---|---|
| rodinia_lavamd_omp | 1.2042 | **1.0272** | 0.869 | 0.982 |
| dgemm_n2048 | 1.1506 | **0.9308** | 0.809 | 0.950 |
| 3mm_omp | 1.0516 | **1.0067** | 0.957 | 0.990 |
| npb_ft | 0.8843 | **0.7705** | 0.872 | 0.962 |
| **npb_mg** | 0.6379 | **0.3848** | **0.602** | 0.820 |
| npb_lu | 0.8917 | 0.8401 | 0.942 | 0.984 |
| npb_cg | 0.7807 | 0.7599 | 0.975 | 0.994 |
| npb_sp | 0.5070 | 0.4906 | 0.968 | 0.987 |
| npb_bt | 0.9070 | 0.9016 | 0.994 | 0.998 |

**En los nueve kernels, la retención en F0 es menor que en F4.** El filtro
de validación de frecuencia rechaza más ventanas a alta frecuencia que a
baja, así que `T(F0)` queda sistemáticamente subestimado, el cociente
`T(F4)/T(F0)` se infla, y con él α. `npb_mg` pierde el 40 % de sus ventanas
en F0.

La hipótesis (b) queda descartada: la frecuencia observada es exactamente
la nominal en los seis niveles.

### Consecuencia grave: **todos los α del proyecto están sesgados al alza**

`npb_mg` pasa de 0.638 a **0.385**. En C2 su distribución por tramo tenía
p05 = 0.400 y mínimo 0.242; corregida hacia abajo en la misma proporción,
**una fracción real de sus tramos caería por debajo del umbral 0.226**.

### T0.2b — NUEVA TAREA, URGENTE, sin nodo

Ni el α filtrado ni el crudo son correctos: el filtrado tiene el sesgo de
retención, y el crudo incluye tiempo corrido a la frecuencia equivocada. La
duración correcta es la que **el propio kernel reporta por stdout** — NPB
imprime `Time in seconds`, y el catálogo ya tiene
`runtime_seconds_stdout_pattern` para extraerlo. Es inmune a cualquier
filtrado de ventanas.

**Recalcular todos los α con esa duración antes de que nada dependa de
ellos**, incluidos el 0.242 que sostiene el diagnóstico y la tabla de C2.

## A.3 T0.3 — Los rasgos SÍ contienen la información *(el resultado clave)*

**Techo intra-kernel** (entrenar y probar dentro del mismo kernel; no es
generalización, es el techo de lo aprendible con estos rasgos):

| kernel | R² intra-kernel |
|---|---|
| npb_cg | **0.977** |
| npb_mg | 0.883 |
| dgemm_n2048 | 0.877 |
| npb_lu | 0.833 |
| npb_sp | 0.812 |
| npb_bt | 0.657 |
| npb_ft | 0.657 |
| 3mm_omp | 0.333 |
| lavamd_omp | 0.158 |

**LOKO por suite** (cuatro pliegues reales en vez de nueve falsos):

| suite excluida | kernels | MAE modelo | MAE trivial | R² |
|---|---|---|---|---|
| NPB | 6 | **0.3821** | 0.4223 | −7.6 |
| RAJAPerf | 1 | **0.2007** | 0.2762 | −11.3 |
| Rodinia | 1 | **0.3191** | 0.4329 | −251.3 |
| DGEMM | 1 | 0.4119 | 0.2222 | −85.1 |

**Lectura, y es la conclusión más importante de toda la Ola 0.** El techo
es alto (0.66–0.98) en siete de nueve kernels: **el vector de rasgos
contiene la información necesaria para predecir `b`.** El fallo de C3 no es
del modelo ni de los rasgos, es de **generalización entre regímenes**.

Eso es precisamente un problema de catálogo, y por tanto **arreglable**. Si
el techo hubiera sido bajo, ningún catálogo lo habría arreglado y habría
que rehacer el vector de entrada.

Los dos techos bajos (3mm 0.333, lavamd 0.158) son los dos kernels con
tráfico DRAM prácticamente nulo (0.2 % y 0.1 % del de STREAM): su `b` está
dominado por ruido de medición del OI. Consistente.

## A.4 T0.4 — El objetivo cuantitativo para los kernels nuevos

α (con duraciones crudas) contra fracción del ancho de banda de STREAM:

| kernel | α | BW F0 (GB/s) | % de STREAM |
|---|---|---|---|
| *stream_official* | *0.154* | *76.80* | *100* |
| npb_mg | 0.385 | 57.18 | 74.5 |
| npb_sp | 0.491 | 43.73 | 56.9 |
| npb_cg | 0.760 | 48.35 | 63.0 |
| npb_ft | 0.771 | 21.76 | 28.3 |
| npb_lu | 0.840 | 5.84 | 7.6 |
| npb_bt | 0.902 | 3.88 | 5.1 |
| dgemm_n2048 | 0.931 | 21.01 | 27.4 |
| 3mm_omp | 1.007 | 0.19 | 0.2 |
| lavamd_omp | 1.027 | 0.07 | 0.1 |

Interpolando entre `npb_mg` (74.5 %, α 0.385) y STREAM (100 %, α 0.154), el
umbral α = 0.226 cae alrededor del **90–92 % de saturación**.

**Ése es el objetivo cuantitativo del catálogo nuevo: llegar a ~90 % del
ancho de banda de STREAM.** `npb_mg` está a 15 puntos, y la clase C
multiplica su conjunto de trabajo por 8 — es la apuesta de T1.2.

La relación no es monótona (`npb_cg` a 63 % tiene α mayor que `npb_sp` a
57 %), así que esto es una guía de diseño, no una ley. La saturación de
ancho de banda no es lo mismo que el acotamiento por latencia, y `ptrchase`
ataca precisamente el segundo.

## A.5 Cómo queda el diagnóstico tras la Ola 0

| Antes de la Ola 0 | Después |
|---|---|
| ¿El 15.3 % es señal o ruido? | **Señal**, ACF 0.90–0.997 contra nulo 0.000 |
| ¿Por qué α > 1? | **Sesgo de retención del filtro**; todos los α están inflados |
| ¿Fallan los rasgos o el catálogo? | **El catálogo.** Techo intra-kernel 0.66–0.98 |
| ¿Cuánto hay que mejorar el catálogo? | **~90 % de saturación de memoria** |

Los cuatro riesgos abiertos de la Parte VII que dependían de la Ola 0 quedan
cerrados. Se abre uno nuevo, T0.2b, y es urgente.
