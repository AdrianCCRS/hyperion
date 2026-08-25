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
| 6420 | Pre-vuelo de fases (T1.1) | GPU | **RECHAZADO** — 0/27 aceptadas, causa: CAP_PERFMON (ARC-184) |
| 6412 | Campaña de fases (T2.1) | GPU | **HELD** |
| 6430 | Pre-vuelo de tamaño (T1.2) | GPU | **CANCELADO** a mano — mismo problema detectado a los 2 min |
| 6431 | Pre-vuelo de rejilla fina (T1.4) | GPU | **HELD** |
| 6424 | Compuertas C1/C2/C3 | normal | COMPLETADO |
| 6426 | Acoplamiento de uncore | normal | COMPLETADO (conclusión retractada, ver Anexo B) |
| 6427 | Auditoría de bytes | normal | COMPLETADO |
| 6443 | Sonda de potencia de reposo GPU (v1) | GPU | FALLÓ (campo `sm` inválido en `--query-supported-clocks`) |
| 6447 | Sonda de potencia de reposo GPU (v2) | GPU | COMPLETADO, ver Anexo C |
| — | T0.2b/c/d, auditoría uncore, `gpu_phasic` | GPU/normal/directo | Todos completados fuera de `sbatch` de campaña (ver más abajo) |

### Hallazgo crítico: hoy no se puede correr NINGUNA campaña, ni CPU ni GPU

`MAN-07` exige que **toda** campaña declare `stream_official` y `ert_probe`
en `calibration:` (son las únicas fuentes de ancho de banda/FLOPs para el
ridge de CPU) — son kernels `device: cpu` por defecto. Eso hace
`has_cpu_kernel = True` **siempre**, sin importar si el catálogo de la
campaña es puramente GPU. Con `has_cpu_kernel = True`, `E12` exige
`uncore.enabled = True` (si no, rechaza); y con `uncore.enabled = True`,
`E13` (ARC-184) bloquea porque `perf` no puede leer `uncore_imc`.

Es un candado doble por diseño, no un error: intentar sortearlo poniendo
`uncore.enabled: false` simplemente cambia cuál de los dos preflights
bloquea (E12 en vez de E13). **No hay combinación de campos del manifiesto
que evite el bloqueo mientras el admin no reponga `CAP_PERFMON`.** Todo lo
hecho durante esta sesión posterior al descubrimiento (compilar
`gpu_phasic`, medir potencia de reposo, T0.2b/c/d) se hizo **fuera** del
harness de campaña — invocando binarios y `nvidia-smi` directamente, o
reanalizando datos ya escritos — precisamente porque el harness está
cerrado.

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

---

# ANEXO B — T0.2b/T0.2c: alpha recalculado (jobs directos + 6433)

## B.1 alpha por CORRIDA, tres estimaciones

Duración tomada de lo que el propio kernel imprime por stdout
(`Time in seconds`), inmune a cualquier filtrado de ventanas. r² 0.976–0.9997.

| kernel | α filtrado (original) | α crudo | **α stdout** |
|---|---|---|---|
| npb_mg | 0.6379 | 0.3848 | **0.3834** |
| npb_sp | 0.5070 | 0.4906 | **0.4895** |
| npb_cg | 0.7807 | 0.7599 | **0.7572** |
| npb_ft | 0.8843 | 0.7705 | **0.7766** |
| npb_lu | 0.8917 | 0.8401 | **0.8410** |
| npb_bt | 0.9070 | 0.9016 | **0.9019** |
| dgemm_n2048 | **1.1506** | 0.9308 | **0.9339** |

**Las ventanas crudas y el stdout coinciden a menos de 0.008 en α.** Eso
valida las duraciones crudas y confirma que la estimación sesgada era la
filtrada. `3mm_omp` y `lavamd_omp` no declaran patrón de runtime en el
catálogo y no son evaluables por esta vía.

**Ya no hay ningún α > 1 a nivel de corrida:** el máximo es 0.934.

### Por qué el sesgo golpea más a los kernels cortos

`npb_mg` dura **0.847 s** en F0. Con `grace_seconds = 0.15` y
`tail_grace_seconds = 0.15`, las ventanas descartadas por transitorio son
0.30 s, el **35 % de la corrida**. Eso explica casi exactamente su retención
del 60.2 % en F0, y por qué es el kernel donde más cambia α (0.638 → 0.383).
La misma ventana de gracia sobre una corrida de 27 s (`npb_bt`) es el 1 %,
y su α apenas se mueve (0.907 → 0.902).

**Consecuencia de diseño para las campañas nuevas:** la ventana de gracia
debe ser una fracción del tiempo de corrida, no un valor absoluto, o los
kernels cortos quedan sistemáticamente sesgados.

## B.2 alpha por TRAMO con duraciones sin filtrar

| kernel | α medio | sd | p05 | α mín | % celdas ≤ 0.226 |
|---|---|---|---|---|---|
| npb_sp | 0.5071 | 0.0106 | 0.4925 | 0.4858 | 0.0 |
| npb_mg | 0.6664 | 0.1978 | 0.3951 | 0.2593 | 0.0 |
| npb_cg | 0.7824 | 0.0946 | 0.7611 | **0.2011** | 0.1 |
| npb_lu | 0.8917 | 0.0264 | 0.8493 | 0.8269 | 0.0 |
| npb_bt | 0.9070 | 0.0049 | 0.8986 | 0.8906 | 0.0 |
| npb_ft | 0.9597 | 0.2990 | 0.3839 | 0.2477 | 0.0 |
| 3mm_omp | 1.0522 | 0.0344 | 0.9845 | 0.9467 | 0.0 |
| lavamd_omp | 1.1075 | 0.0816 | 1.0862 | 0.8319 | 0.0 |
| dgemm_n2048 | 1.1530 | 0.0658 | 1.0600 | 1.0199 | 0.0 |

**1 celda de 9000 (0.01 %)** por debajo del umbral. Mínimo global 0.2011.

## B.3 CORRECCIÓN a la conclusión de T0.2

En el Anexo A.2 se concluyó que el filtro de calidad explicaba α > 1. **Eso
es cierto solo a nivel de CORRIDA.** Quitando el filtro, los α por TRAMO
apenas se movieron (dgemm 1.159 → 1.153; lavamd 1.092 → 1.108), y tres
kernels siguen por encima de 1.

Así que hay **dos efectos distintos**, no uno:

1. **Nivel de corrida:** sesgo de retención del filtro. Resuelto y
   cuantificado. El α por corrida es fiable y está validado por tres vías
   independientes.
2. **Nivel de tramo:** persiste un α > 1 que el filtro no explica. La causa
   más probable es **desalineación de los tramos**: la coordenada de avance
   por instrucciones es invariante al 0.34 % *sobre la corrida completa*,
   pero dentro de un centil una deriva pequeña hace que la celda *k* en F4
   no cubra exactamente el mismo trabajo que en F0. Además la duración de
   una celda es ~1 % de la corrida, así que el ruido relativo es mucho mayor.

### Qué se puede afirmar y qué no

- **Fiable:** el α por corrida. Mínimo del dataset **0.3834** (`npb_mg`).
- **Con reservas:** el α por tramo. Su magnitud y su dispersión pueden
  contener artefacto de binado. La afirmación de C2 de que "α varía entre
  tramos" sigue apoyada por T0.1 (la estructura de fase es real, ACF
  0.90–0.997), pero **la magnitud de esa variación no es de fiar** hasta
  validar el binado contra la etiqueta de verdad de `phasic`.
- Nótese que el α por corrida es la combinación **ponderada por duración**
  de los α por tramo, no su media simple. Que `npb_mg` tenga α por corrida
  0.383 y media por tramo 0.666 significa que **sus tramos LARGOS son los de
  α bajo** — un dato físicamente interesante y favorable, porque son los que
  más pesan en el consumo.

## B.4 El diagnóstico central NO cambia

| | Antes | Después de corregir |
|---|---|---|
| α mínimo por corrida | 0.242 (era por tramo) | **0.3834** |
| α mínimo por tramo | 0.242 | 0.2011 |
| Celdas bajo umbral | 0 de 9000 | **1 de 9000** |
| Kernels bajo umbral | 0 de 9 | **0 de 9** |

**Ningún kernel del dataset alcanza el régimen viable.** La corrección mueve
los números pero no el diagnóstico: el catálogo sigue siendo el problema, y
el plan de las Olas 1 y 2 sigue en pie sin cambios.

## B.5 Tarea nueva que sale de aquí

**T0.2d — validar el binado de tramos contra la etiqueta de verdad de
`phasic`.** Cuando 6420 entregue datos, comparar los tramos que produce la
coordenada de avance contra las marcas `PHASE` reales. Es la única forma de
saber cuánto del α > 1 por tramo es artefacto. Sin nodo adicional: usa los
datos del pre-vuelo que ya está en cola.

**Y un cambio de diseño para las campañas nuevas:** `grace_seconds` y
`tail_grace_seconds` deben expresarse como fracción del tiempo de corrida.

---

# ANEXO C — T2.3: criterio de actividad GPU invariante a la frecuencia (ARC-185)

## C.1 Motivación medida

`gpu_util_pct` de NVML es una fracción de TIEMPO (cuánto del intervalo de
muestreo hubo algún kernel corriendo). Con reloj más lento el mismo trabajo
tarda más, así que un kernel genuinamente ocioso puede cruzar cualquier
piso fijo en los niveles bajos sin haber hecho más trabajo real (F3 del
inventario: `rodinia_lud`, ocioso por diseño de esa prueba, pasaba de
0.0 % en REF/F0 a 3.5 % en F4).

## C.2 Potencia de reposo medida en vivo (job 6447, paccaA100, 20 s/nivel)

| nivel | MHz | potencia media | mín | máx |
|---|---|---|---|---|
| F0 | 1410 | 53.82 W | 47.38 W | 54.45 W |
| F1 | 1110 | 40.11 W | 39.52 W | 56.39 W |
| F2 | 810 | 36.23 W | 35.95 W | 36.71 W |
| F3 | 510 | 34.50 W | 34.33 W | 34.91 W |
| F4 | 210 | 33.80 W | 33.64 W | 34.03 W |

La potencia de reposo **varía 1.59×** entre F4 y F0 (33.8 W → 53.8 W), así
que un piso único de potencia sería tan defectuoso como el piso de
utilización único. Confirma que la línea de reposo debe medirse **por
nivel**, como ya proponía el plan.

El máximo de F1 (56.39 W) es un valor atípico aislado (rango 39.5–56.4 W
frente a una media de 40.1 W) — probablemente un pico transitorio durante
la transición de reloj, capturado por estar al inicio de la ventana de
muestreo. No afecta la mediana ni el criterio, que usa la media.

## C.3 Implementación

`orchestrator/validation.py::validate_windows()` y
`classifier/features/load.py::filter_gpu_trainable()` aceptan ahora
`idle_power_mw_by_level` + `active_power_margin_mw` (opcionales). Cuando se
proveen, el criterio pasa a ser:

```
gpu_power_mw - idle_power_mw(nivel) >= margen
```

en vez del piso de utilización fijo. **Sin esos dos parámetros el
comportamiento es idéntico al anterior** — ninguna campaña ya en cola
cambia. Un nivel sin línea de reposo medida falla cerrado (no se asume 0).

Enrutado end-to-end: `manifest.gpu.idle_power_mw_by_level` /
`manifest.gpu.active_power_margin_mw` → `campaign.py` → `validate_windows()`.
9 tests nuevos (4 en `test_validation.py`, 3 en `test_load.py`), todos
verdes junto con los 92 preexistentes.

## C.4 Margen recomendado -- RETRACTADO, ver Anexo G

**El margen único de 20 000 mW propuesto aquí resultó ser un error grave**,
encontrado al día siguiente (ARC-189): rechaza F4 completo incluso para
kernels con trabajo GPU genuino y sostenido, porque el exceso de potencia
de trabajo real también se reduce con el reloj -- a F4 puede ser tan bajo
como ~1.3 W, muy por debajo de cualquier margen calibrado contra F0. La
tabla original de la Ola de idle-power (C.2) medía solo la línea de
REPOSO; nunca se contrastó contra el exceso de un kernel activo de
verdad, y por eso el margen quedó mal calibrado en la dirección
peligrosa (rechaza datos buenos, no solo malos). El margen correcto,
según el mecanismo y el arreglo, va en el Anexo G.

## C.5 Pendiente para activar el criterio en campañas reales

**El ejemplo original de esta sección tenía `active_power_margin_mw: 20000`
como escalar único — RETRACTADO por ARC-189 (Anexo G): rechaza F4 completo
incluso con trabajo GPU real.** El campo debe ser un mapa por nivel:

```yaml
gpu:
  enabled: true
  idle_power_mw_by_level:
    F0: 53820.7
    F1: 40105.9
    F2: 36225.1
    F3: 34500.9
    F4: 33804.0
  active_power_margin_mw:      # provisional, ver Anexo G.4 -- necesita re-medición
    F0: 6000
    F1: 3000                   # F1 tiene un pico de reposo sin explicar (Anexo G.4), no confiar aún
    F2: 1800
    F3: 1200
    F4: 800
```

Estos valores son de **20 s por nivel**, suficientes para decidir el
criterio pero no para fijarlos en el catálogo final: antes de una campaña
GPU completa, remedir con una ventana más larga (~60 s) y con más
repeticiones por nivel, siguiendo el mismo criterio de "no lanzar sin
pre-vuelo" del resto del plan.

## C.6 Lo que queda abierto en el eje GPU (F1, F2, F3 restantes)

- **F1/F2** (`rodinia_lud`/`rodinia_heartwall` con `success_check:
  {type: exit_code}`, no validan CUDA): **no modificado directamente**
  (tocar la fuente de terceros aumenta huella sin necesidad clara), pero
  su riesgo real quedó bastante más acotado por dos cosas que sí se
  hicieron esta sesión:
  1. Los manifiestos fijan `--nodelist=paccaA100 --gres=gpu:1`, así que
     Slurm ya impide ejecutar sin GPU real.
  2. **Con el criterio de potencia de C.3 activo, una corrida cuya GPU no
     hizo trabajo real (fallo silencioso de CUDA) se queda en la potencia
     de reposo durante toda la corrida → cero ventanas "usables" →
     rechazada por VAL-09/I10**, exactamente el mismo mecanismo que ya
     protege contra un kernel genuinamente ocioso. Antes de C.3 esto NO
     era cierto de forma confiable (el piso de utilización tenía el sesgo
     de F3), así que activar `idle_power_mw_by_level` cierra la mayor
     parte de F1/F2 como efecto colateral, sin tocar Rodinia. Queda como
     mitigación, no como prueba positiva de que el kernel usó CUDA — para
     eso haría falta instrumentar la fuente.
- **E5 (compilación de `gpu_phasic.cu`): RESUELTO.** `nvcc` no está en
  ningún module ni en el `PATH` por defecto, pero sí existe en
  `/home/latorresn/latorresn/cuda-12.3` (CUDA 12.3, confirmado contra
  `compute_cap` real de la A100 = 8.0). Compilado con
  `scripts/pacca/build_gpu_phase_kernels.sh` (ARC-186), catalogado como
  `gpu_phasic_p010/p100/p1000` (mismo patrón que `phasic` de CPU).

  **Hallazgo colateral importante, genérico para cualquier kernel CUDA
  futuro**: `nvcc` **no produce binarios reproducibles byte a byte**.
  Confirmado en vivo: dos builds consecutivas de la MISMA fuente, mismos
  flags, mismo nodo, dieron `sha256` distintos (divergen desde el byte
  897688). Es su pipeline de compilación en varias etapas
  (`cudafe`/`ptxas`) el que incrusta nombres de archivo temporal
  aleatorios en metadatos de depuración — el código ejecutable en sí es
  idéntico: tras `strip --strip-all` las dos builds dan el **mismo**
  `sha256`. El script de build despoja el binario **antes** de calcular
  el checksum que va al catálogo, precisamente para que una reconstrucción
  legítima futura no quede bloqueada por `C02` (verificación de checksum
  del binario). GCC sí es reproducible (verificado con el mismo método
  sobre `phasic`/`ptrchase`): el problema es específico de `nvcc`.

  **Limitación estructural documentada, no resuelta**: `CAT-10` exige
  `operational_intensity_flops_per_byte` estático por kernel para todo
  `device: gpu`, a diferencia de CPU donde el OI se mide dinámicamente por
  ventana vía `uncore_imc`. Como `gpu_phasic` alterna dos regímenes
  extremos dentro de una sola corrida, su `phase_label_train` automático
  será **constante** durante toda la corrida — el pipeline estándar de
  etiquetado no puede capturar la alternancia en GPU con la telemetría
  actual. Lo que sí sirve del kernel (y es lo que T0.2d ya explotó): las
  marcas de verdad reales (`T0_MONOTONIC_NS` + `PHASE`) cruzadas
  *offline* contra `gpu_power_mw`/`gpu_sm_clock_mhz` por ventana.

---

# ANEXO D — T0.2d: validación del binado contra la etiqueta de verdad (ARC-187)

## D.1 Cómo se hizo posible sin uncore ni preflight nuevo

Los datos usados son del job 6420 (pre-vuelo de fases), **rechazado**
por VAL-09/I10 a causa del problema de CAP_PERFMON (ARC-184) — pero el
rechazo solo invalida las columnas derivadas de `uncore` (OI,
`phase_label_uncore_real`). `delta_instructions`, `t_start_ns`/`t_end_ns`
(telemetría de `perf_event_open`, independiente de uncore) y las marcas
`PHASE`/`T0_MONOTONIC_NS` por stdout (ARC-177) son reales y completas
(`running_ratio = 1.0`). El primer intento del script filtraba por
`quality_status == "ok"`, que descartaba las 20 266 ventanas de cada
corrida por error — `intensity_undefined` solo significa "sin OI", no
"sin telemetría". Corregido para excluir solo los estados que sí
invalidan esas columnas.

## D.2 Resultado

| kernel (período de fase) | pureza media del bin | pureza p05 | acuerdo F0 vs F4 |
|---|---|---|---|
| `phasic_p010` (10 ms) | 0.514 | 0.507 | 0.70 |
| `phasic_p1000` (1 s) | **0.988** | 0.917 | **0.87** |

**`phasic_p010` no es un caso de desalineación** — es un problema de
escala. Con 100 bins sobre ~20 s de corrida, cada bin cubre ~200 ms, es
decir ~20 ciclos completos de una fase de 10 ms. La baja pureza (0.51,
esencialmente aleatoria) es la consecuencia matemática esperada de
promediar 20 alternancias dentro de un mismo bin, no evidencia de que el
binado esté mal alineado. Confirma, eso sí, algo útil para S3: 100 bins es
resolución insuficiente para fases de 10 ms — habría que usar más bins o
directamente descartar `phasic_p010` de la campaña de fases (S3 ya lo
sugería).

**`phasic_p1000` es la prueba limpia** (el período de fase, 1 s, es 5× más
largo que un bin, ~200 ms): la pureza intra-nivel es alta (0.988 de
media, 0.917 en el peor bin) — el binado SÍ aísla fases reales dentro de
un mismo nivel. Pero el **acuerdo entre F0 y F4 es solo 0.87**: en ~13 %
de los bins, la fase dominante en F0 no coincide con la de F4, pese a que
cada bin es internamente casi puro en ambos niveles por separado.

## D.3 Interpretación — y por qué no generaliza sin más a los 9 kernels reales

13 % de desacuerdo es real, pero antes de aplicarlo al resto del proyecto
hay que entender POR QUÉ pasa, y la explicación más plausible es
específica de `phasic`, no un defecto general de la coordenada de avance:

`phasic` alterna entre una fase de FMA denso en instrucciones (cientos de
miles de retiros por milisegundo) y una fase de persecución de punteros
muy pobre en instrucciones (un salto de puntero por iteración, dominado
por latencia de memoria, no por reloj). Si esa densidad de instrucciones
por segundo de pared responde de forma DISTINTA a la frecuencia en cada
fase —lo cual es casi seguro: la fase de cómputo es sensible al reloj por
definición, la de memoria no— entonces la fracción de instrucciones que
corresponde a "haber completado el bin k" deja de ser la misma posición
temporal real entre F0 y F4. La invariancia del 0.34 % (ARC-175) se validó
sobre aplicaciones con densidad de instrucciones razonablemente homogénea
a lo largo de la corrida; `phasic` la viola a propósito, porque para eso
se construyó (alternar dos regímenes extremos).

**Esto es tranquilizador para C2**, no alarmante: los 9 kernels reales del
dataset (NPB, DGEMM, RAJAPerf, Rodinia) no alternan deliberadamente entre
regímenes de densidad de instrucciones tan opuestos dentro de una sola
corrida — son individualmente más homogéneos. El 13 % de desacuerdo medido
aquí es probablemente una cota superior, no una cifra transferible a
`npb_ft`/`npb_mg`/etc. tal cual.

## D.4 Lo que sigue abierto

- El α > 1 residual por tramo (Anexo B.3) sigue sin una causa CONFIRMADA
  para los kernels reales — este anexo da una hipótesis de mecanismo
  (densidad de instrucciones dependiente de fase y de frecuencia) y una
  cota de cuánto podría pesar en el peor caso (`phasic`), pero no mide el
  efecto directamente sobre `npb_ft`/`dgemm`/etc.
- Si se usa `phasic` como instrumento de validación de latencia de
  detección más adelante, hacerlo con `phasic_p100`/`p1000`, nunca
  `p010` — confirmado que 100 bins no lo resuelven.

---

# ANEXO E — corrección al Anexo B: sí queda un α > 1 por corrida (ARC-187)

El Anexo B afirmó "ya no hay ningún α > 1 a nivel de corrida" — esa
afirmación se basó en 7 de 9 kernels, porque `3mm_omp` y `lavamd_omp` no
tenían patrón de runtime declarado y quedaron fuera de T0.2b. Con el hueco
cerrado (ver arriba, mismo ARC-187), la tabla completa es:

| kernel | α (duración propia del kernel) | r² | fuente |
|---|---|---|---|
| npb_mg | 0.3834 | 0.9761 | patrón propio |
| npb_sp | 0.4895 | 0.9942 | patrón propio |
| npb_cg | 0.7572 | 0.9969 | patrón propio |
| npb_ft | 0.7766 | 0.9997 | patrón propio |
| npb_lu | 0.8410 | 0.9997 | patrón propio |
| npb_bt | 0.9019 | 0.9996 | patrón propio |
| dgemm_n2048 | 0.9339 | 0.9996 | patrón propio |
| 3mm_omp | 0.9827 | 0.9998 | respaldo (arnés) |
| **lavamd_omp** | **1.0289** | **0.9999** | **patrón propio** |

**`rodinia_lavamd_omp` da α = 1.029 usando su propio "Total time" impreso
por el binario** (no el respaldo del arnés), con un ajuste casi perfecto
(r² = 0.9999). No es un artefacto de medición del filtro de calidad (esa
causa ya se descartó en el Anexo B para el resto) ni del respaldo
(`3mm_omp`, que sí usa el respaldo, da 0.983 — por debajo de 1).

**Sigue sin explicación confirmada.** A diferencia de los kernels
multifásicos (`phasic`, Anexo D), `lavamd_omp` es un algoritmo único y
coherente durante toda la corrida, así que la hipótesis de "densidad de
instrucciones que difiere por fase" no aplica aquí. Candidatas sin
verificar: (a) una porción de "Total time" no delegada a los cores fijados
en frecuencia (arranque de hilos OpenMP, E/S) que no escala con el reloj y
así infla el cociente; (b) algún efecto de latencia de memoria que se
acumula más que proporcionalmente al bajar el reloj para el patrón de
acceso específico de LavaMD. Ninguna de las dos está medida — quedan como
pendiente explícita, no como explicación.

**Corroboración que descarta una de las dos candidatas.** T0.2 (Anexo A.2)
ya había medido, con un método INDEPENDIENTE (duración cruda sumada desde
`windows.csv`, medida por el arnés, no impresa por el kernel), α = 1.0272
para este mismo kernel — prácticamente idéntico al 1.0289 de aquí. Si la
causa fuera "una porción de tiempo no delegada a los cores fijados que el
propio print del kernel cuenta pero el arnés no", las dos mediciones
deberían discrepar; no lo hacen. Descarta razonablemente la candidata (a)
y deja (b) — algún efecto de latencia de memoria— como la más plausible,
aunque sigue sin medirse directamente.

**No cambia el diagnóstico central**: `lavamd_omp` sigue muy por encima
del umbral 0.226, así que no afecta la conclusión de que ningún kernel del
dataset alcanza el régimen viable. Cambia sí la afirmación puntual de que
"ya no queda ningún α > 1" — no es cierta, y queda corregida aquí.

---

# ANEXO F — Matriz experimental propuesta, CPU y GPU, con tamaños (ARC-188)

Todo lo de este anexo está **diseñado y listo** (manifiestos, catálogo,
checksums), pero **nada puede ejecutarse hasta que el admin resuelva
CAP_PERFMON** (§VIII.1). Se distingue lo ya medido (dataset base) de lo
añadido y de lo condicional a un pre-vuelo.

## F.1 Matriz CPU

### Núcleo ya medido (dataset `arc174`, 540 corridas, referencia)

| kernel | suite | tamaño | niveles | reps |
|---|---|---|---|---|
| npb_bt, npb_mg, npb_cg, npb_sp, npb_ft, npb_lu | NPB-OMP | clase **B** | REF,F0,F1,F2,F3,F4 | 10 |
| dgemm_n2048 | DGEMM-OpenBLAS | N=2048 | REF,F0,F1,F2,F3,F4 | 10 |
| rodinia_lavamd_omp | Rodinia-OpenMP | boxes1d=24 | REF,F0,F1,F2,F3,F4 | 10 |
| rajaperf_polybench_3mm_omp | RAJAPerf-OpenMP | `--sizefact 1 --repfact 10` | REF,F0,F1,F2,F3,F4 | 10 |

6/9 kernels = NPB (67% de una sola suite — el defecto de diversidad, B5).

### Añadido 1 — eje de tamaño (T1.2, catálogo listo, pre-vuelo diseñado)

| kernel | tamaño nuevo | binario | estado |
|---|---|---|---|
| npb_cg → **npb_cg_c** | clase B → **C** | `cg.C.x`, construido y verificado en pacca | listo |
| npb_mg → **npb_mg_c** | clase B → **C** | `mg.C.x`, construido y verificado en pacca | listo |
| npb_sp_c, npb_ft_c, npb_bt_c, npb_lu_c | clase **C** | binarios existen, catalogados, **sin pre-vuelo propio** | catálogo listo, falta diseñar T1.2b si `cg_c`/`mg_c` funcionan |

Pre-vuelo diseñado: `npb_cg`+`npb_cg_c`+`npb_mg`+`npb_mg_c` × {F0,F4} × 3 rep
= 36 corridas (`campaign_pacca_size_preflight.yaml`). Objetivo cuantitativo
(Anexo A.4): llegar a ≥90% del ancho de banda de STREAM; `npb_mg` está en
74.5% a clase B.

### Añadido 2 — rejilla fina de frecuencia (T1.4, listo)

Los 3 kernels de α más bajo (npb_sp, npb_mg, npb_cg) × {F0, **3000 MHz**,
**2800 MHz**} × 3 rep = 27 corridas (`campaign_pacca_fine_grid_preflight.yaml`).
Cierra el hueco 2600–3200 MHz, donde cae el óptimo bajo el objetivo de
energía-con-holgura (2831–3051 MHz para los 9 kernels).

### Añadido 3 — eje de fase (T1.1, pre-vuelo YA CORRIÓ pero rechazado por uncore)

| kernel | qué es | tamaño |
|---|---|---|
| ptrchase | sonda monofásica de α (memoria pura) | buffer 512 MiB |
| phasic_p010/p100/p1000 | instrumento multifásico sintético | buffer 512 MiB, período 10 ms/100 ms/1 s |

Campaña completa diseñada (`campaign_pacca_phase_probe.yaml`, job 6412 en
`hold`): 4 kernels × 8 niveles (REF,F0,F1,F2,F3,1200,1000,F4) × 10 rep =
320 corridas. **`phasic_p010` es sospechoso de aportar poco** (T0.2d: 100
bins no resuelven su período de 10 ms) — candidato a excluir de la
campaña final si el pre-vuelo lo confirma.

### Pendiente sin catalogar — carga real multifásica (T1.3)

HPCG u otra suite real con fases documentadas en la literatura, para no
depender solo de sintéticos como evidencia de régimen. **Sin binario, sin
catálogo, sin pre-vuelo** — sigue siendo una propuesta, no un plan
ejecutable todavía.

### Matriz CPU total propuesta (si todos los pre-vuelos confirman)

| eje | kernels | niveles | reps | corridas |
|---|---|---|---|---|
| Núcleo (ya medido) | 9 | 6 | 10 | 540 |
| Tamaño | 2–6 (`*_c`) | 2 (F0,F4) | 10 (a confirmar) | 40–120 |
| Rejilla fina | 3 | 2 (3000,2800) | 10 | 60 |
| Fase | 3–4 | 8 | 10 | 240–320 |
| **Total** | | | | **~900–1050 corridas** |

## F.2 Matriz GPU

### Kernels reales, por actividad medida (fresca, job de auditoría 2026-08-23)

| kernel | tamaño actual | % activo | inclusión propuesta |
|---|---|---|---|
| rodinia_gaussian | 4096×4096 | 90.7% | **sí**, sin cambios |
| gpu_dgemm_n4096 | N=4096 | 86.4% | **sí**, sin cambios |
| rodinia_heartwall | 1000 cuadros | 83.5% | **sí** (con la mitigación de C.3 para F1/F2) |
| rodinia_lavamd | boxes1d=70 | 53.0%, muy variable | **sí, con reservas** — el único con α medido (0.201), bajo el umbral GPU (0.639) |
| rodinia_dwt2d | 16384×16384 | 21.2% | **no**, salvo que se explique por qué un tamaño ya grande sigue débil (no es un problema de tamaño obvio) |
| rodinia_backprop | 1048560 | 19.1% | condicional — probar tamaño mayor primero |
| rodinia_myocyte | 100000 | 10.8% | condicional — probar tamaño mayor primero |
| rodinia_lud | 2048×2048 | 1.2% | **no**, sin repetir tamaño mayor primero (ver F3, sesgo confirmado) |

### Eje de fase GPU (nuevo esta sesión, catálogo listo)

| kernel | tamaño | estado |
|---|---|---|
| gpu_phasic_p010/p100/p1000 | buffer 2048 MiB, período 10 ms/100 ms/1 s | compilado, catalogado, **sin pre-vuelo propio todavía** |

Limitación documentada (Anexo C.6): `phase_label_train` automático será
constante para este kernel — solo sirve vía cruce offline de las marcas
`PHASE` contra `gpu_power_mw`/`gpu_sm_clock_mhz`, no para entrenar
directamente.

### Rejilla de frecuencia GPU (medida en vivo, job 6447)

| nivel | MHz reales | potencia de reposo |
|---|---|---|
| F0 | 1410 | 53.82 W |
| F1 | 1110 | 40.11 W |
| F2 | 810 | 36.23 W |
| F3 | 510 | 34.50 W |
| F4 | 210 | 33.80 W |

Eje CPU durante el barrido GPU: fijo en REF/F4 (2 niveles), como en las
campañas GPU existentes — el interés está en el reloj de GPU, no de CPU.

### Matriz GPU total propuesta

| eje | kernels | niveles GPU | niveles CPU | reps | corridas |
|---|---|---|---|---|---|
| Núcleo activo | 4 (gaussian, dgemm_n4096, heartwall, lavamd) | 6 (REF,F0-F4) | 2 | 3 | 144 |
| Fase | 3 (`gpu_phasic_*`) | 6 | 2 | 3 | 108 |
| Condicionales (tamaño mayor) | 2 (backprop, myocyte) | 6 | 2 | 3 | 72 (si se habilitan) |
| **Total** | | | | | **~250–320 corridas** |

Mucho más barato que CPU porque el eje activo real son 4 kernels, no 9 —
consecuencia directa de que solo la mitad del catálogo GPU hace trabajo
real medible.

---

# ANEXO G — ARC-189: el margen de potencia debía ser por nivel, no único

## G.1 Cómo se encontró

Al investigar por qué `rodinia_dwt2d` mostraba solo 21.2% de actividad
pese a tener potencia elevada (ver G.3), se contrastó el criterio de
potencia (C.3) contra tres kernels **ya confirmados como activos por
otras vías** (`rodinia_heartwall`, `rodinia_gaussian`, `gpu_dgemm_n4096`)
en F0 y F4. El resultado fue el mismo en los tres: con el margen único de
20 000 mW recomendado el día anterior, **`frac_excess>=20W` daba
exactamente 0.000 en F4 para los tres**, pese a que `gpu_util_pct` los
mostraba entre 35.6% y 94.5% activos ahí mismo.

## G.2 El mecanismo

El exceso de potencia de trabajo GPU real escala con el reloj casi tanto
como la propia potencia de reposo. Medido sobre ventanas con
`gpu_util_pct >= 50` (actividad inequívoca):

| kernel | exceso mínimo F0 | exceso mínimo F4 | razón |
|---|---|---|---|
| rodinia_heartwall | 9 477 mW | 1 287 mW | 7.4× |
| rodinia_gaussian | 11 353 mW | 3 059 mW | 3.7× |
| gpu_dgemm_n4096 | 10 357 mW | 3 753 mW | 2.8× |

Un margen de 20 000 mW, calibrado el día anterior **solo contra el ruido
de una sonda en reposo** (job 6447, sin ninguna carga real corriendo),
nunca se contrastó contra el exceso de un kernel activo de verdad. En F4
ningún kernel activo lo habría superado — el criterio construido
explícitamente para no rechazar trabajo real por el sesgo de frecuencia
(F3) tenía **el mismo sesgo, en la misma dirección, por una causa
distinta**.

## G.3 El arreglo

`gpu_active_power_margin_mw` deja de ser un escalar y pasa a ser un mapa
por nivel, igual que `gpu_idle_power_mw_by_level`, en
`validate_windows()` y `filter_gpu_trainable()`. Un `float` suelto se
sigue aceptando por compatibilidad, con la advertencia explícita en el
docstring de que es casi con certeza un error. 4 tests nuevos (3 en
`test_validation.py`, 1 en `test_load.py`), incluido uno que reproduce el
bug exacto (margen único rechaza F4 con trabajo real; margen por nivel lo
acepta). 570 tests totales, todos verdes.

## G.4 Margen provisional — necesita re-medición, no está cerrado

Con el mínimo real por nivel (tres kernels, arriba) y el ruido de reposo
de la sonda (job 6447), un margen conservador y seguro:

| nivel | margen propuesto | mínimo real observado | ruido de reposo (rango) |
|---|---|---|---|
| F0 | 6 000 mW | 9 477–12 739 | hasta ±6 440 (ver abajo) |
| F1 | 3 000 mW | 4 175–7 159 | **hasta +16 284, atípico** |
| F2 | 1 800 mW | 2 528–3 324 | ±500 |
| F3 | 1 200 mW | 1 888–2 528 | ±400 |
| F4 | 800 mW | 1 287–3 753 | ±400 |

**No se toma como definitivo.** F1 mostró un pico de reposo de +16 284 mW
sobre su propia media en la sonda de 20 s (job 6447) — un solo evento, de
causa no confirmada (candidata: transitorio de asentamiento justo tras
`nvidia-smi -lgc`, sin período de espera antes de muestrear). Si ese pico
es recurrente y no un artefacto de la sonda corta, un margen de 3 000 mW
en F1 no lo cubre con seguridad. **Antes de usar esta tabla en una
campaña real**: remedir la línea de reposo con una sonda más larga (≥60 s
por nivel), excluyendo explícitamente los primeros 1–2 s tras fijar el
reloj (mismo principio que `frequency_settle` ya aplica en CPU), y
verificar si el pico de F1 persiste.

## G.5 `rodinia_dwt2d` — el caso que originó todo esto, sigue sin resolver

El comentario del catálogo que decía "potencia nunca supera 46.6 W" **no
se pudo reproducir**: en F0 la potencia real va de 59.6 a 83.5 W (media
65.1 W), muy por encima del reposo (53.8 W); 97% de las ventanas superan
60 W. Corregido en el catálogo (`orchestrator/schemas/kernels/catalog.yaml`).

Lo que explica el 21.2% de `gpu_util_pct`: `dwt2d` tiene potencia elevada
con utilización NVML baja — la firma de un kernel memory-bound (OI = 2.17,
ya declarado) que ocupa poco la SM pero sí mueve DRAM real.
`gpu_util_pct` mide fracción de tiempo con *algún* kernel corriendo, no
intensidad de trabajo — no es el metro correcto para este patrón.

**Pero tampoco es un caso limpio para el criterio de potencia.** El
exceso sobre reposo se desploma con la frecuencia mucho más que en
`dgemm`/`gaussian`/`heartwall`: en F4 cae a ~1 W, indistinguible del
ruido de la propia sonda. Con potencia y utilización en desacuerdo, y ese
desacuerdo variando con el nivel, **la inclusión de `dwt2d` en la matriz
final queda abierta** — no se resuelve solo corrigiendo el comentario.

## G.6 Qué falta

1. Re-medir la línea de reposo GPU con sonda larga (≥60 s), excluyendo el
   transitorio de asentamiento, y confirmar o descartar el pico de F1.
2. Recalibrar el margen por nivel contra esa medición corregida.
3. Decidir `dwt2d` con un criterio adicional (p.ej. energía total sobre
   reposo integrada en toda la corrida, no ventana a ventana) en vez de
   los dos que ya mostraron discrepancia.
4. Repetir la auditoría de actividad (F.2) para `backprop`/`myocyte` con
   el criterio de potencia YA corregido antes de decidir si necesitan
   tamaño mayor — el 19.1%/10.8% de la tabla anterior se midió con
   `gpu_util_pct`, que para kernels memory-bound livianos puede estar
   subestimando lo mismo que en `dwt2d`.

---

# ANEXO H — ARC-191: bypass MAN-07→E12→E13 para campañas 100% GPU, validado con job 6457

## H.1 Por qué

CAP_PERFMON se rompió en pacca (ARC-184): `uncore_imc` ilegible, lo que
bloqueaba **toda** campaña — incluidas las 100% GPU, que no necesitan
uncore en absoluto. El usuario autorizó explícitamente un bypass del
núcleo de validación ("Estoy consciente que es un fix al core de
telemetría y validación pero es necesario. Asegúrate de ser cauteloso y
no romper nada en el proceso") para no quedar detenido un día completo
mientras se gestiona el permiso a nivel de administrador.

## H.2 El mecanismo del bloqueo

MAN-07 exige incondicionalmente que todo manifiesto declare kernels de
calibración de CPU (`stream_official`/`ert_probe`) en `calibration:` —
eso no se toca. El bug real estaba en `run_campaign_preflight()` (E12):
calculaba `has_cpu_kernel` combinando `calibration:` **y** `kernels:`,
así que la sola presencia de `stream_official` en calibración forzaba
`uncore.enabled=True` aunque el dataset (`kernels:`) fuera 100% GPU. E13
entonces fallaba por CAP_PERFMON roto, para campañas que nunca iban a
leer uncore.

**Confirmado por lectura de código antes del fix**: los kernels de
calibración pasan por `run_single()` y parsean su stdout con regex
directamente (`_measure_bw_and_flops_peak()`), nunca por
`postprocess_run()`/`validate_windows()` — no dependen de uncore de
ninguna forma.

## H.3 El arreglo

En `run_campaign_preflight()`, E12 pasa a mirar solo `dataset_entries`
(de `kernels:`, lo que realmente se corre y valida), no la unión con
`calibration:`. `entries` (calibración+kernels) se preserva sin cambios
para los demás checks (binario/checksum/success_check/memoria). 2 tests
nuevos que reproducen el escenario real exacto: calibración=CPU,
kernels=100% GPU, `uncore.enabled=false` → E12 y E13 pasan; y un segundo
test que confirma que E12 **sigue bloqueando** si aparece un kernel de
CPU real en `kernels:`.

## H.4 Validación con job real (6457)

`campaign_pacca_gpu_uncore_disabled_preflight.yaml`: 4 kernels GPU, 3
niveles GPU (REF/F0/F4), 2 niveles CPU (REF/F4), 3 rep = 72 corridas,
márgenes de potencia provisionales (solo F0/F4, sin REF todavía). Job
6457: **COMPLETADO**, 48/72 aceptadas, **24/24 rechazadas exactamente en
`gpuREF`** — por diseño: `gpuREF` no tenía margen declarado en ese
manifiesto y el criterio falla cerrado (`return False`) cuando no hay
margen para el nivel, en vez de aceptar por defecto. Ni una corrida
inesperada del lado F0/F4. El mecanismo se comportó exactamente como se
diseñó, de punta a punta, con datos reales.

## H.5 Qué NO cubre este bypass

Sigue bloqueado cualquier dataset con al menos un kernel de CPU real en
`kernels:` — el eje de CPU permanece completamente detenido hasta que se
repare CAP_PERFMON a nivel de administrador (reporte en
`docs/retoma/pacca/Reporte_Perf_Sin_CAP_PERFMON_Efectivo.md`, reabierto,
no enviado todavía).

---

# ANEXO I — ARC-194: sonda de reposo v2 y márgenes finales de los 6 niveles

## I.1 Resultado de la sonda larga (job 6461, 60 s/nivel, asentamiento excluido)

| nivel | MHz | reposo medio | p95 | máx |
|---|---|---|---|---|
| REF | 210 (autorregulado) | 33 654.0 | 33 780 | 33 860 |
| F0 | 1410 | 53 621.8 | 54 090 | 55 620 |
| F1 | 1110 | 39 856.8 | 40 020 | **40 570** |
| F2 | 810 | 36 084.9 | 36 250 | 36 610 |
| F3 | 510 | 34 441.0 | 34 620 | 34 820 |
| F4 | 210 | 33 677.6 | 33 810 | 33 870 |

**El pico de F1 desapareció.** Con el transitorio de asentamiento excluido
y 300 muestras (antes 100), el máximo de F1 es 40 570 — nada parecido al
56 390 espurio de la v1. Confirma que era un transitorio de la sonda
corta, no una inestabilidad real del nivel.

**`REF` en reposo iguala a `F4`** (33 654 vs 33 678): sin reloj fijado, la
GPU se autorregula al mínimo cuando no hay carga, sin importar el
gobernador.

## I.2 Exceso real en REF — mucho mayor que en cualquier nivel fijo

| kernel | exceso mínimo en REF | exceso mínimo en F0 |
|---|---|---|
| rodinia_heartwall | 41 875 | 9 477 |
| rodinia_gaussian | 28 060 | 11 353 |
| gpu_dgemm_n4096 | 27 180 | 10 357 |

Bajo carga, `REF` deja subir el reloj libremente (boost), así que su
exceso real es mucho mayor que el de F0 pese a que su reposo es igual al
de F4. Es el nivel más fácil de discriminar de los seis.

## I.3 Márgenes finales por nivel

Derivados como: por encima de 2–4× el techo de ruido de reposo (p95/máx
menos la media) Y por debajo de la mitad del exceso mínimo real observado.

| nivel | idle (mW) | margen (mW) | techo de ruido | exceso real mínimo | factor de seguridad |
|---|---|---|---|---|---|
| REF | 33 654.0 | 800 | ~206 | 27 180 | 34× bajo la señal |
| F0 | 53 621.8 | 4 000 | ~2 000 | 9 477 | 2× sobre ruido, 2.4× bajo señal |
| F1 | 39 856.8 | 2 000 | ~713 | 4 175 | 2.8× sobre ruido, 2× bajo señal |
| F2 | 36 084.9 | 1 200 | ~525 | 2 528 | 2.3× sobre ruido, 2.1× bajo señal |
| F3 | 34 441.0 | 900 | ~379 | 1 888 | 2.4× sobre ruido, 2.1× bajo señal |
| F4 | 33 677.6 | 800 | ~193 | 1 287 | 4.1× sobre ruido, 1.6× bajo señal |

**Ya no son provisionales** — a diferencia de la tabla de G.4, esta se
derivó de una sonda de reposo limpia (60 s, asentamiento excluido) y de
exceso real medido en tres kernels de referencia en los 6 niveles. F4
queda con el margen de seguridad más ajustado (1.6×); si algún kernel
nuevo muestra un exceso real menor a 1287 mW en F4, revisar antes de
confiar en el criterio ahí.

---

# ANEXO J — Núcleo activo GPU, 6 niveles completos, job 6462 (2026-08-23)

## J.1 Manifiesto

`campaign_pacca_gpu_nucleo_activo.yaml`: mismos 4 kernels de H.4/I.2
(`rodinia_gaussian`, `gpu_dgemm_n4096`, `rodinia_heartwall`,
`rodinia_lavamd`), pero con los **6 niveles GPU completos**
(REF+F0-F4), 2 niveles CPU (REF/F4), 3 rep = 144 corridas, con la tabla
de márgenes **final** de I.3 (ya no la provisional de H.4, que solo
cubría F0/F4). `uncore.enabled: false` bajo el mismo mecanismo ARC-191
de H.2-H.3 — este manifiesto no prueba un mecanismo nuevo, extiende la
cobertura de niveles del ya probado en job 6457.

## J.2 Resultado

Job 6462: `COMPLETED`, exit 0:0, 1:01:15. `campaign_metadata.json`:
**144/144 aceptadas, 0 rechazadas, 0 saltadas**,
`frequency_restored_verified: true`. Por nivel GPU: 24/24 en cada uno de
los 6 — a diferencia de 6457, `gpuREF` **ya no se rechaza en bloque**,
porque este manifiesto sí declara su margen (800 mW, I.3).

## J.3 Spot-check de datos reales (no solo el agregado)

`verdict.json` de 4 corridas (una por kernel, niveles REF/F1/F2/F3):
las 4 con `accepted: true, message: "ok"`. `windows.csv` — potencia GPU
real (`gpu_power_mw`) contra piso+margen del nivel:

| corrida | rango medido (mW) | piso+margen (mW) | lectura |
|---|---|---|---|
| gaussian, REF/gpuREF | 57 073–117 926 | 34 454 | exceso amplio, no al filo |
| heartwall, REF/gpuF1 | 42 126–65 975 | 41 857 | aceptado, pero el mínimo de ventana queda muy cerca del umbral |
| dgemm_n4096, REF/gpuF2 | 37 074–167 181 | 37 285 | mínimo puntual por debajo del umbral; el run se acepta por criterio agregado, no por ventana individual |
| lavamd, REF/gpuF3 | 36 374–72 309 | 35 341 | holgado |

`quality_status` de la corrida `gaussian/gpuREF` (6380 ventanas):
`intensity_undefined` (2560) y `pmu_degraded` (2397) dominan — esperado
con `uncore.enabled=false` (H.2), no una regresión nueva.

## J.4 Lectura honesta

Este job cierra la fila "Núcleo activo" de la matriz GPU de F.2 de
punta a punta, con datos reales verificados por muestreo — no solo el
0 rechazadas del agregado. La única señal a vigilar hacia adelante es
F1/F2: sus mínimos de ventana individuales rondan el umbral (H.4/I.3 ya
advertía que F1 y F2 tienen el factor de seguridad más ajustado después
de F4); el criterio de aceptación es agregado por corrida, así que no
invalida lo aceptado aquí, pero si un kernel nuevo tiene un perfil de
potencia más plano que estos cuatro, conviene revisar caso a caso antes
de confiar ciegamente.

**Lo que este job NO resuelve** (ver lista abierta en G.6 y el cierre de
sesión anterior): eje de CPU sigue bloqueado por CAP_PERFMON; `dwt2d`
sigue fuera de la matriz sin decisión; `lavamd_omp` α=1.029 sin causa
confirmada; binning de `phasic` con 13% de discrepancia F0 vs F4 sin
validar contra los 9 kernels reales de CPU; NPB clase C catalogados
pero nunca corridos; limitación arquitectónica de `gpu_phasic` (CAT-10,
etiqueta constante) sin resolver.

---

# ANEXO K — Hueco del oráculo en GPU: el catálogo actual no tiene margen (2026-08-23)

Análisis offline sobre las 144 corridas del job 6462, cero costo de nodo.
Scripts: `classifier/analysis/gpu_oracle_headroom.py` y
`gpu_alpha_calibration_kernels.py`.

## K.0 Prerrequisito verificado

`gpu_energy_valid=1` en el **100%** de las filas `gpu_telemetry`, con
`gpu_energy_delta_mj` poblado, comprobado en F0, F2 y F4. La medición de
energía GPU es fiable; el análisis se apoya en piso firme.

## K.1 La métrica correcta es energía TOTAL, no solo GPU

En estas corridas la GPU aporta solo ~47% de la energía total
(gaussian CPU=REF: 396.9 J GPU vs 453.1 J CPU+DRAM). Bajar el reloj de
GPU alarga la corrida y la CPU delegada sigue consumiendo durante esa
extensión. Evaluar solo el lado GPU sobreestimaría el ahorro de forma
grosera: en `gaussian` de F0 a F1 la energía **de GPU baja** 400→339 J,
pero la **total sube** 858→913 J, porque la CPU suma +117 J. Todo lo que
sigue usa energía total.

## K.2 El hueco del oráculo es prácticamente nulo

| kernel | CPU | mejor nivel | ahorro E% | costo t% |
|---|---|---|---|---|
| rodinia_gaussian | REF | REF | 0.91 | −0.82 |
| gpu_dgemm_n4096 | REF | REF | 2.14 | −2.15 |
| rodinia_heartwall | REF | REF | 0.41 | −0.56 |
| rodinia_lavamd | REF | F1 | 1.34 | +10.02 |
| rodinia_gaussian | F4 | REF | 1.15 | −0.82 |
| gpu_dgemm_n4096 | F4 | F0 | 0.00 | 0.00 |
| rodinia_heartwall | F4 | F0 | 0.00 | 0.00 |
| **rodinia_lavamd** | **F4** | **F1** | **7.67** | **+4.87** |

**Los "ahorros" de REF sobre F0 no son ahorros de DVFS.** REF bajo carga
hace boost hasta ~1410 MHz, o sea el mismo punto de operación que F0; el
costo de tiempo NEGATIVO (REF resulta más rápido *y* más barato) delata
que es ruido de medición entre dos puntos equivalentes, no una decisión
de frecuencia. Descontando eso, **3 de 4 kernels no tienen absolutamente
nada que ganar**: el reloj máximo ya es el óptimo.

El único margen real es `rodinia_lavamd`: 7.67% de ahorro en F1 con 4.87%
de costo en tiempo (CPU=F4).

## K.3 Por qué: alpha lo predice 4/4

| kernel | CPU | alpha | r2 | ¿margen? |
|---|---|---|---|---|
| rodinia_heartwall | REF | 0.771 | 0.9905 | no |
| gpu_dgemm_n4096 | REF | 0.619 | 0.9950 | no |
| rodinia_gaussian | REF | 0.607 | 0.9953 | no |
| **rodinia_lavamd** | REF | **0.179** | 0.9648 | **sí** |
| **rodinia_lavamd** | F4 | **0.062** | 0.9726 | **sí** |

Con el umbral heredado del eje CPU (α < 0.226 ⇒ el DVFS paga), **alpha
acierta el resultado del oráculo en los 4 kernels**. El instrumento de la
Fase 1 transfiere íntegro al eje GPU — resultado metodológico reutilizable
en el documento.

## K.4 El mecanismo: solo se escala el reloj de SM

`nvidia-smi -lgc` fija el reloj **gráfico/SM**; el reloj de **memoria
queda intacto**. De ahí se sigue la predicción: un kernel limitado por
ancho de banda de DRAM debe ser casi insensible. Verificado con los
kernels de calibración, que ya corrieron en los 6 niveles (cero costo):

| kernel | α | T(F4)/T(F0) con recorte de reloj 6.7× |
|---|---|---|
| `gpu_stream_bw` (ancho de banda puro) | **0.071** | **1.485** |
| `rodinia_heartwall` (cómputo) | 0.771 | 5.52 |

Mismo recorte de reloj, 3.7× de diferencia en penalización. **El margen de
DVFS en GPU vive en los kernels limitados por memoria.**

(`gpu_ert_probe_fp64` quedó descartado del tamizaje: su tiempo NO es
monótono al bajar el reloj — F4 sale más rápido que F3 — señal de que
autoajusta su tamaño de problema. Su α no es interpretable y el script lo
marca automáticamente.)

## K.5 Hallazgo colateral: CPU al mínimo CUESTA energía

En los 4 kernels, `CPU=REF` domina a `CPU=F4` en energía total:

| kernel | E_tot CPU=REF | E_tot CPU=F4 | penalización |
|---|---|---|---|
| rodinia_gaussian | 850.1 | 894.3 | +5.2% |
| gpu_dgemm_n4096 | 907.7 | 1111.6 | +22.5% |
| rodinia_heartwall | 685.0 | 754.8 | +10.2% |
| rodinia_lavamd | 731.7 | 2307.6 | **+215%** |

Frenar la CPU alarga la corrida más de lo que ahorra en potencia:
race-to-idle gana. `lavamd` es el caso extremo (5.7 s → 16.8 s), lo que
además revela que su carga tiene un componente de CPU dominante.

**Esto no invalida la decisión de CPU-al-mínimo como control experimental**
(aislar el efecto de la GPU es un propósito legítimo y distinto), pero
sí significa que, **como política**, CPU-al-mínimo es la opción equivocada
en los 4 kernels medidos.

## K.6 Conclusión y consecuencia

> **RETRACTADO por el Anexo L (job 6463).** La conclusión de abajo — "es
> el catálogo, no la plataforma" — resultó **falsa** al tamizar los
> candidatos memory-bound: los tres fallaron igual, y la causa es un piso
> de potencia estática de la plataforma (~117 W) que cierra la ventana
> del DVFS para casi cualquier kernel. Es la plataforma. Se conserva el
> texto original para dejar la traza del error. El α=0.071 de
> `gpu_stream_bw` citado como prueba tampoco alcanza: el umbral real
> exigido por el piso estático es α ≈ 0.03, no 0.226.

Segunda confirmación independiente, ahora en el eje GPU, de lo ya
concluido en CPU: **es el catálogo, no la plataforma.** La plataforma sí
tiene margen de DVFS explotable (`gpu_stream_bw`, α=0.071, lo demuestra);
los kernels elegidos son los que no lo tienen.

**No construir el modelo de ML sobre estos 4 kernels.** El techo teórico
que podría capturar cualquier modelo es ~1.9% de energía promedio, con
CV de tiempo entre repeticiones de hasta 4.8% — un modelo que "gane" por
ese margen no es defendible en una sustentación.

Consecuencia operativa: alpha pasa de ser una métrica descriptiva a ser
un **instrumento de tamizaje barato** para reconstruir el catálogo GPU
alrededor de kernels limitados por memoria. Ver K.7.

## K.7 Candidatos a tamizar, en orden de prioridad

1. **`rodinia_dwt2d`** — el que se excluyó por dudoso (Anexo G.5) es ahora
   el más prometedor. Su firma (OI=2.17, potencia alta con `gpu_util_pct`
   baja) es exactamente la de un kernel limitado por ancho de banda. La
   ambigüedad del criterio de actividad que lo dejó fuera es un *síntoma*
   de la propiedad que ahora se busca, no un defecto.
2. **`gpu_stream_bw` / `babelstream_cuda`** — α=0.071 ya medido, binario
   ya construido. Promoverlo de solo-calibración a kernel de dataset.
3. **`backprop` y `myocyte`** — se excluyeron por actividad baja medida con
   `gpu_util_pct`, el criterio que G.6 ítem 4 ya marcó como inadecuado
   justamente para kernels memory-bound livianos. Reauditar con potencia.

Tamizaje propuesto: 5 niveles fijos × 1–2 rep, **solo tiempo**, sin
matriz completa. Conservar únicamente los que den α < 0.226, y recién
entonces correr campaña completa sobre los sobrevivientes + `lavamd`.

## K.8 Qué queda decidido sobre la arquitectura del modelo

Con estos datos, el nivel óptimo es **constante dentro de cada kernel**
y varía **entre** kernels — el escenario (a) de la discusión previa: un
selector por kernel, no un clasificador de fase por ventana. **CAT-10
deja de ser bloqueante.** No está probado que no exista variación
intra-corrida (no se midió), pero con 3 de 4 kernels en "usar el máximo"
no hay evidencia que justifique la inversión en el esquema (b).

---

# ANEXO L — El piso de potencia estática cierra la ventana del DVFS en GPU (job 6463)

Tamizaje de α sobre los candidatos memory-bound de K.7. Job 6463,
COMPLETADO, exit 0, 23:39, **54/54 aceptadas, 0 rechazadas**. (Predije que
`dwt2d` sería rechazado en F4 por su exceso de potencia de ~1 W; **me
equivoqué**, los márgenes del Anexo I lo manejaron sin problema.)

## L.1 El ajuste de α resultó INVALIDO — y el r2 lo delató

| kernel | α | intercepto | 1−α | r2 |
|---|---|---|---|---|
| rodinia_myocyte | 0.161 | 1.241 | 0.839 | **0.535** |
| rodinia_backprop | 0.157 | 1.197 | 0.843 | **0.628** |
| rodinia_dwt2d | 0.306 | 1.456 | 0.694 | **0.530** |

Los α salen bajos (aparentarían "el DVFS paga"), pero **el modelo no
ajusta**: r2 de 0.53–0.63 y el intercepto incompatible con 1−α, que el
modelo exige. **Estos α no se pueden reportar.** La causa está en los
tiempos crudos: `dwt2d` da F3=17.601 s y F4=**17.538 s** — más rápido con
el reloj recortado 2.4× — y `myocyte` satura en F3→F4 (21.688→22.012).
Ese punto de F4, con el mayor apalancamiento en la regresión (x=6.71),
aplana la recta y produce un α bajo espurio.

**Descartadas dos explicaciones, ambas por verificación directa:**

1. *¿Falla el actuado de frecuencia en F4?* No. El reloj de SM medido
   bajo carga da min = max = objetivo exacto en los 5 niveles y los 3
   kernels (1410/1110/810/510/**210**). Actuación impecable.
2. *¿Rodinia hace menos trabajo en F4 saliendo con código 0?* (el modo de
   fallo del incidente `test.avi`.) No. El stdout de `dwt2d` es idéntico
   en F0/F3/F4: mismo `inputsize 805306368` cargado completo, las mismas
   3 etapas de DWT, stderr vacío.

## L.2 La causa real: la potencia total apenas se mueve

`P_cpu` es **constante en ~82–87 W** en todos los niveles y todos los
kernels. La CPU delegada (6 núcleos, gobernador `performance`) no baja su
consumo porque la GPU vaya más lenta — solo espera más tiempo, cobrando
igual.

| kernel | P(F0) | P(F4) | caída de P | T(F4)/T(F0) |
|---|---|---|---|---|
| gpu_dgemm_n4096 | 246.7 W | 144.5 W | 41.4% | 4.57 |
| rodinia_heartwall | 177.5 W | 121.2 W | 31.7% | 5.52 |
| rodinia_gaussian | 158.4 W | 120.2 W | 24.1% | 4.55 |
| rodinia_myocyte | 140.4 W | 117.0 W | 16.6% | 2.16 |
| rodinia_dwt2d | 135.6 W | 116.5 W | 14.1% | 3.18 |
| rodinia_lavamd | 127.2 W | 113.7 W | 10.6% | 2.08 |
| rodinia_backprop | 89.5 W | 101.1 W | **−12.9%** | 2.14 |

Todos convergen a un piso de **~113–121 W** por más que el reloj baje
6.7×. `backprop` es patológico: su potencia **sube** al bajar el reloj,
así que no puede pagar en ningún escenario.

## L.3 El criterio exacto, sin modelo de por medio

Como E = P·T, bajar el reloj paga si y solo si

    T(f)/T(F0)  <  P(F0)/P(f)

No depende de Amdahl ni de α — solo de energía y tiempo medidos, así que
sobrevive a que el ajuste sea inválido. Aplicado a los **7 kernels × 5
niveles** medidos hasta hoy (35 combinaciones), **explica el 100% de los
casos**, y solo uno lo satisface:

| caso | T/T(F0) | P(F0)/P | ¿paga? |
|---|---|---|---|
| **rodinia_lavamd @ F1** | **1.100** | **1.115** | **SI, −1.34%** |
| rodinia_myocyte @ F1 | 1.280 | 1.125 | no, +13.77% |
| rodinia_dwt2d @ F1 | 1.544 | 1.107 | no, +39.47% |
| rodinia_backprop @ F1 | 1.287 | 0.954 | no, +34.94% |
| … las 31 restantes | — | — | no |

El único ganador lo es por un margen de **1.5 puntos**. No es una política,
es una casualidad al filo del ruido.

## L.4 Corrección: es la plataforma, no el catálogo

K.6 concluyó "es el catálogo, no la plataforma" y **eso era falso**. La
prueba estaba diseñada para refutarlo y lo refutó: los tres kernels más
memory-bound del catálogo (OI de 0.06, 0.45 y 2.17 — más bajos que
cualquiera de los 4 originales) fallaron exactamente igual.

El umbral real que impone el piso estático no es α < 0.226 (heredado del
eje CPU) sino, de la tabla de L.2, **una holgura de tiempo de solo
1.12–1.20× frente a un recorte de reloj de 6.7×** — o sea **α ≈ 0.03**.
Ningún kernel real del catálogo se acerca; ni siquiera `gpu_stream_bw`
(α=0.071, T(F4)/T(F0)=1.485) lo lograría.

## L.5 Dónde SI está la energía

En `dwt2d` a F0, `E_cpu` = 466.7 J de 747.2 J totales: **el 62% de la
energía de una corrida GPU la gasta una CPU que está esperando.** Ese es
el lever grande, no el reloj de la GPU.

Cuidado: **no** se resuelve bajando la frecuencia de la CPU — K.5 ya midió
que CPU al mínimo empeora la energía total en los 4 kernels (hasta +215%),
porque alarga la corrida más de lo que ahorra. La vía plausible es reducir
el número de núcleos delegados o liberar los núcleos mientras la GPU
trabaja, no frenarlos. **Sin medir todavía.**

## L.6 Consecuencia para el modelo de ML

**El eje GPU no sostiene un modelo de DVFS.** De 35 combinaciones medidas
hay exactamente una decisión no trivial, con 1.34% de ahorro y 1.5 puntos
de margen. Un clasificador entrenado sobre eso no puede superar de forma
defendible a la constante "usar siempre el reloj máximo".

Esto vale como **resultado negativo riguroso**, que es un entregable
legítimo: criterio analítico explícito, validado sobre 7 kernels y 35
combinaciones, con el mecanismo físico identificado y cuantificado (piso
estático de ~117 W). No es "no funcionó"; es "aquí está la condición que
debe cumplirse, y esta plataforma no la cumple".

---

# ANEXO M — CORRECCIÓN: la métrica era mía, no de la física (2026-08-23)

El Anexo L concluyó que el DVFS de GPU "no puede pagar" en esta
plataforma. **Esa conclusión dependía de una decisión metodológica que
tomé yo, no de una restricción física**, y al contrastarla con la
literatura resultó ser más estricta que el estándar del campo.

## M.1 El error

L midió **energía total (GPU + paquete CPU + DRAM)**. Lo justifiqué en
K.1 diciendo que evaluar solo la GPU "sobreestimaría el ahorro de forma
grosera". Pero la literatura de DVFS de GPU mide **energía de GPU vía
NVML**, no del nodo entero — porque es lo que una política de DVFS de
GPU realmente controla. El piso de ~84 W de la CPU delegada es una
propiedad del nodo de medición, no del mecanismo bajo estudio.

Se verificó además que ese piso **no** es un artefacto del arnés: el shim
de blocking-sync (ARC-70) compiló y se aplicó en los jobs 6462 y 6463
(cero advertencias en los logs), así que la CPU no está girando en vacío.
Los 84 W son el piso real del paquete.

## M.2 El resultado con la métrica de la literatura

Energía solo-GPU (NVML), CPU=REF, contra "siempre F0":

| kernel | mejor | ahorro E_gpu | costo t | ganancia EDP |
|---|---|---|---|---|
| rodinia_lavamd | F1 | **25.11%** | +10.02% | **17.60%** |
| rodinia_heartwall | F1 | **18.24%** | +33.12% | 1.03% |
| rodinia_gaussian | F1 | **15.35%** | +25.62% | 1.69% |
| rodinia_myocyte | F1 | **7.66%** | +28.02% | 0.00% |
| rodinia_dwt2d | F0 | 0.00% | — | — |
| gpu_dgemm_n4096 | REF | 2.11% | −2.15% | artefacto REF/F0 |
| rodinia_backprop | REF | 24.30% | −0.54% | artefacto (E_gpu de 8–49 J, no fiable) |

**Cuatro kernels con ahorro real de 7.7% a 25.1%**, y el nivel óptimo
varía entre kernels (F1 / F0 / REF) — o sea, hay señal aprendible.

Estos valores caen dentro del rango publicado: 8.7–23.1% en entrenamiento
de DNN, y 20.2–26.7% con escalado consciente de la aplicación en V100 y
A100. El montaje no está roto; la métrica estaba sobre-restringida.

## M.3 Anexo L no se retracta — se reencuadra

El hallazgo de L (piso estático de ~117 W, criterio
`T(f)/T(F0) < P(F0)/P(f)`) **sigue siendo correcto y medido**. Lo que
cambia es su estatus: no es "el DVFS de GPU no sirve", sino **"en este
nodo la CPU absorbe el ahorro de la GPU"** — una limitación real del
alcance del resultado, que la literatura habitualmente no reporta.

Reportar **ambas métricas** es más fuerte que cualquiera por separado:
alineado con el campo en la métrica primaria (solo GPU), y con una
advertencia cuantificada que el campo suele omitir.

## M.4 Dos mandos que quedaron sin usar

1. **Granularidad de la grilla.** Toda la acción está entre F0 (1410 MHz)
   y F1 (1110 MHz): ahí se capturan 15–25% de ahorro, pero con 10–33% de
   degradación. Mi grilla salta 300 MHz de una vez. Con presupuesto
   iso-latencia de 10%, casi todo el ahorro desaparece; con 15%,
   `lavamd@F1` reaparece con 25.11%. **El óptimo está dentro del salto que
   no muestreé.** La A100 ofrece muchos escalones intermedios.
2. **Reloj de memoria.** Varios trabajos escalan núcleo **y** memoria
   (`--query-supported-clocks=mem,gr`). Aquí solo se usó `-lgc` (núcleo).
   Para un kernel limitado por ancho de banda, el mando de memoria es
   probablemente el que importa — y explicaría por qué `dwt2d` y
   `stream_bw` no respondieron al de núcleo.

## M.5 Consecuencia

Se revierte la conclusión de L.6 ("el eje GPU no sostiene un modelo").
**Sí lo sostiene**, con la métrica del campo. Lo pendiente antes de
entrenar: (a) grilla fina entre 1410 y 1110 MHz, (b) probar el mando de
memoria, (c) fijar el presupuesto de degradación como parámetro explícito
de la política, no como supuesto implícito.

## M.6 Lección de método

Dos conclusiones fuertes seguidas ("es el catálogo, no la plataforma" en
K.6; "el DVFS no puede pagar" en L.6) resultaron ser artefactos de
decisiones propias no contrastadas contra la práctica establecida. Ambas
se emitieron con datos correctos y análisis correcto sobre una premisa
elegida sin verificar. **Contrastar la premisa con la literatura antes de
declarar un resultado negativo**, no después.

Referencias consultadas: MDPI Computation 8(2):37 (2020) — predicción de
energía/rendimiento con escalado de frecuencia de núcleo y memoria;
ICPP 2019, "Predictable GPUs Frequency Scaling for Energy and
Performance"; arXiv:1610.01784 — survey y estudio de medición de DVFS en
GPU; ACM (2023) — escalado consciente de la aplicación, 26.7% V100 /
20.2% A100.

---

# ANEXO N — Matriz experimental actualizada, CPU y GPU (2026-08-24)

**Supersede al Anexo F.** F se escribió antes de K/L/M (métrica de
energía corregida) y antes de la campaña 6462. La estructura de F
(núcleo/tamaño/rejilla/fase) ya no aplica igual; esta versión refleja lo
que realmente corrió, lo que está en cola, y lo que es condicional a un
tamizaje pendiente — sin inventar resultados que todavía no llegaron.

Formato: **CONFIRMADO** (dato real en disco) / **EN COLA** (manifiesto
válido, corriendo o esperando nodo) / **CONDICIONAL** (depende de que un
tamizaje pendiente pase un umbral).

## N.1 Matriz GPU

### Núcleo activo — CONFIRMADO (job 6462)

| kernel | tamaño | niveles GPU | CPU | reps | corridas |
|---|---|---|---|---|---|
| rodinia_gaussian | 4096×4096 | 6 (REF,F0-F4) | REF,F4 | 3 | 36 |
| gpu_dgemm_n4096 | N=4096 | 6 | REF,F4 | 3 | 36 |
| rodinia_heartwall | 1000 cuadros | 6 | REF,F4 | 3 | 36 |
| rodinia_lavamd | boxes1d=70 | 6 | REF,F4 | 3 | 36 |
| **Total** | | | | | **144** |

Resultado (Anexo M, energía de GPU vía NVML): ahorro real de 7.7–25.1%
en `lavamd`/`heartwall`/`gaussian`/`myocyte` a F1, `lavamd`@F1 gana 17.6%
de EDP. `dgemm_n4096` y `dwt2d` sin ahorro en esta grilla.

**Decisión que toma esta matriz, respecto a F**: el eje CPU=F4 se
**elimina** hacia adelante. K.5 midió que CPU al mínimo empeora la
energía total 5.2–215% en los 4 kernels — confirmado, no una hipótesis.
Todo lo que sigue usa **CPU=REF únicamente**, la mitad de corridas por
kernel sin perder información real.

### Grilla fina — EN COLA (job 6471)

| kernel | niveles GPU | CPU | reps | corridas |
|---|---|---|---|---|
| gaussian, dgemm_n4096, heartwall, lavamd, myocyte, backprop, dwt2d (7) | 10 (REF,F0,4 intermedios 1170-1350MHz,F1-F4) | REF | 3 | 210 |

Responde si el margen de 15-25% cabe dentro de un presupuesto de
degradación ≤4-10% (hoy solo cabe sin límite o con EDP). Márgenes de
potencia de los 4 niveles nuevos **interpolados**, riesgo declarado en
`campaign_pacca_gpu_fine_grid_dataset.yaml`.

### Diversidad de carga (dwt2d) — EN COLA (job 6472)

| kernel | tamaños | niveles GPU | CPU | reps | corridas |
|---|---|---|---|---|---|
| rodinia_dwt2d_s192/s2048/s4096/s8192 + rodinia_dwt2d (16384) | 5 | 6 | REF | 3 | 90 |

### Candidatos memory-bound tamizados — CONFIRMADO, resultado negativo (job 6463)

`rodinia_myocyte`, `rodinia_backprop`, `rodinia_dwt2d` (16384): el ajuste
de α resultó **inválido** (r²=0.53-0.63, Anexo L.1) por el piso de
potencia estática — no se usa como filtro por sí solo. Con la métrica
GPU-only corregida (Anexo M), `myocyte` sí muestra 7.66% de ahorro real
en F1; los otros dos no. **Ya están en la grilla fina (arriba)**, no se
proponen corridas nuevas.

### RAJAPerf CUDA — CONDICIONAL, sin build todavía

~55 kernels adicionales disponibles en RAJAPerf (Polybench+apps), pero
la variante CUDA nunca se compiló en pacca (solo existe la OpenMP). Antes
de cualquier corrida GPU: build nuevo (`ENABLE_CUDA=On`), verificar
reproducibilidad, despojar y fijar checksum — mismo procedimiento que
`gpu_phasic`. **No hay pre-vuelo diseñado todavía.**

### Matriz GPU total, si todo confirma

| eje | estado | corridas |
|---|---|---|
| Núcleo activo | confirmado | 144 |
| Grilla fina | en cola | 210 |
| Diversidad dwt2d | en cola | 90 |
| RAJAPerf CUDA | condicional, sin build | — |
| **Total medido/en cola** | | **444** |

## N.2 Matriz CPU

### Núcleo — CONFIRMADO (dataset `arc174`, previo a esta sesión)

| kernel | suite | tamaño | niveles | reps |
|---|---|---|---|---|
| npb_bt, npb_mg, npb_cg, npb_sp, npb_ft, npb_lu | NPB-OMP | clase B | REF,F0-F4 | 10 |
| dgemm_n2048 | DGEMM-OpenBLAS | N=2048 | REF,F0-F4 | 10 |
| rodinia_lavamd_omp | Rodinia-OpenMP | boxes1d=24 | REF,F0-F4 | 10 |
| rajaperf_polybench_3mm_omp | RAJAPerf-OpenMP | `--repfact 10` | REF,F0-F4 | 10 |
| **Total** | | | | **540 (424 aceptadas)** |

Resultado (`cpu_policy_headroom.py`, esta sesión): margen del modelo
sobre la mejor constante = **0.33 puntos**. `npb_mg` es la única señal
real (+2.71% en F1). Causa física: núcleo y memoria comparten dominio de
reloj en CPU (a diferencia de GPU) — no es un problema de grilla.

### Tamizaje RAJAPerf/Polybench — EN COLA (job 6473)

| candidato | fuente | por qué |
|---|---|---|
| Polybench_JACOBI_1D, JACOBI_2D | stencil 1D/2D | bajo reuso de dato, clásico memory-bound |
| Polybench_HEAT_3D, FDTD_2D | stencil 3D/2D con tiempo | ídem |
| Polybench_ATAX, GESUMMV, MVT | matriz-vector | O(n²) trabajo sobre O(n²) datos, AI baja |

5 niveles (F0-F4, mismos MHz que el dataset base) × tiempo+RAPL, sin
uncore. Criterio de paso: **α < 0.226** (mismo umbral derivado del modelo
de potencia real de CPU, Anexo A.1 de este documento). Bypasea el
orquestador — `scripts/pacca/screen_rajaperf_cpu_alpha.sh`, reusa el
binario `raja-perf.exe` ya compilado y con checksum verificado, cero
compilación nueva.

**Sin resultado todavía** — job 6473 pendiente en cola SLURM.

### Si el tamizaje pasa — CONDICIONAL, matriz de seguimiento

Para cada candidato con α < 0.226: catalogar como kernel `device: cpu`
(exec_path=`raja-perf.exe`, `-k <NOMBRE> -v Base_OpenMP`), correr
6 niveles × **6 rep** (2026-08-25: bajado de 10 a 6, no heredado sin más
del núcleo — ver `docs/justifications/report/sections/repetitions_edp.tex`,
análisis de convergencia de CV% de EDP sobre las 54 celdas de `arc174`
corregido: n=6 conserva la misma cobertura a umbral 2% que n=3 y n=10
(48/54 celdas) por 40% menos costo. Los dos kernels que sí muestran
varianza de EDP real y no resuelta ni a n=10 —`npb_ft`, `rodinia_lavamd`—
ya son parte del catálogo confirmado, no de este tamizaje; cualquier
sobreviviente cuyo propio tamizaje ya sugiera comportamiento inestable
debe tratarse como candidato a repeticiones adicionales dirigidas, no
forzar n=10 para todo el catálogo ampliado). Costo por kernel
sobreviviente: 36 corridas (72 con el par baseline/telemetry).

**Bloqueo real para esta parte**: aunque el tamizaje puede correr hoy sin
uncore, la campaña de dataset completa (con `phase_label_train` real)
para cualquier candidato que pase **sí exige uncore** (E12/E13, CAP_PERFMON)
porque son kernels `device: cpu` reales, no bypasea como GPU (ARC-191 solo
exime datasets 100% GPU). Esta etapa queda condicionada al permiso, aunque
el tamizaje que decide si vale la pena no lo esté.

### Matriz CPU total, si todo confirma

| eje | estado | corridas |
|---|---|---|
| Núcleo | confirmado | 540 (424 aceptadas) |
| Tamizaje RAJAPerf | en cola, sin uncore | ~35×7=245 mediciones de tiempo/energía (no "corridas" en el sentido de dataset) |
| Ampliación con sobrevivientes | condicional a tamizaje Y a CAP_PERFMON | 36×N sobrevivientes (n=6, ver justificación EDP) |

## N.3 Lo que cambió respecto al Anexo F, resumido

1. **CPU=F4 sale de la matriz GPU** — confirmado que empeora, no solo
   sospechado.
2. **La rejilla fina de CPU (T1.4 en F) se reemplaza por el tamizaje de
   catálogo (N.2)** — el hallazgo de esta sesión es que el problema de
   CPU no es resolución de grilla (a diferencia de GPU), es diversidad de
   catálogo. Insistir en más resolución sobre los mismos 9 kernels no
   tiene sustento nuevo.
3. **El eje de fase sintética (`phasic`/`gpu_phasic`, T1.1/T1.3 de F)
   sigue en pie sin cambios** — no lo tocó nada de esta sesión.
4. **RAJAPerf pasa de "1 kernel usado" a "banco de ~55 disponibles,
   parcialmente tamizado"** — el hallazgo nuevo más grande de esta
   sesión en términos de matriz.
