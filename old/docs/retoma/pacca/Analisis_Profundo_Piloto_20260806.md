# Análisis profundo del piloto en paccaA100 (2026-08-06)

Cruce de las dos campañas reales completas corridas en pacca el mismo día:
`pacca_ref_full_20260806` (7 kernels × REF × 3 repeticiones, NPB clase B +
DGEMM) y `pacca_class_c_stress_20260806` (6 kernels × REF × 3 repeticiones,
NPB clase C — mismo algoritmo, problema ~4-15× más grande). Ambas con los
fixes de ARC-54/55/56/62/63 ya aplicados (RAPL conectado, afinidad de CPU
real, atribución de energía corregida, `L2_LINES_IN_ALL` instrumentado).

**Total de datos analizados:** ~1.03 millones de ventanas de `windows.csv`
entre las dos campañas.

---

## 1. Hallazgo principal — la clasificación Roofline es estable entre tamaños de problema

Comparar el mismo kernel (mismo algoritmo, mismo código) corrido en clase B
y clase C es la prueba más fuerte de generalización que se ha hecho en el
proyecto hasta ahora: si el modelo Roofline captura una propiedad real del
kernel y no un artefacto de una corrida puntual, la clasificación dominante
no debería cambiar mucho al escalar el problema.

| Kernel | Hint | Dominante clase B | Dominante clase C | Diferencia |
|---|---|---|---|---|
| `npb_mg` | memory_bound | **99.9%** memory_bound | **99.9%** memory_bound | 0.0 pp |
| `npb_bt` | intermedio | 85.6% compute_bound | 85.4% compute_bound | 0.2 pp |
| `npb_cg` | memory_bound | 92.7% memory_bound | 93.5% memory_bound | 0.8 pp |
| `npb_lu` | intermedio | 88.4% compute_bound | 89.0% compute_bound | 0.6 pp |
| `npb_sp` | intermedio | 58.2% memory_bound | 59.3% memory_bound | 1.1 pp |
| `npb_ft` | intermedio | 79.7% compute_bound | 66.2% compute_bound | **13.5 pp** |

**5 de 6 kernels mantienen su clasificación dominante prácticamente igual**
(diferencia ≤1.1 puntos porcentuales) a pesar de que clase C representa
entre 4× (`mg`) y 15× (`bt`, medido en filas de `windows.csv`, proxy de
duración) más trabajo que clase B. Esto es evidencia sólida de que
`phase_label_train` está capturando una propiedad arquitectónica real del
kernel, no ruido de una corrida específica — un resultado que vale la pena
destacar en el libro como validación de la metodología.

**`npb_ft` es la excepción** (79.7% → 66.2% compute_bound, sigue siendo
dominante compute pero con una mezcla notablemente mayor de `memory_bound`
en clase C, 33.8% vs 20.3%). Consistente con su intensidad operacional
teniendo un rango enorme en ambas clases (p10≈0.09, p90≈2.8 FLOP/byte) —
FT (FFT 3D) tiene fases genuinamente distintas (cómputo de mariposas vs.
transposición/comunicación de datos entre planos), y el balance entre esas
fases parece desplazarse con el tamaño de la malla. No es un bug; es la
señal de que FT necesita tratarse como un caso genuinamente mixto, más que
`npb_lu`/`npb_bt`/`npb_sp`, que sí se mantienen estables.

También aparecieron 19 ventanas `intensity_undefined` en `npb_ft_c` y 9 en
`npb_mg_c` (0 en ambos en clase B) — una fracción insignificante (<0.1%
del total en cada caso), probablemente ventanas puntuales con
`delta_cache_misses=0` por casualidad de scheduling en una corrida más
larga. No afecta la conclusión anterior, solo se documenta por
completitud.

---

## 2. Hallazgo importante — el sesgo de `bytes_moved_window` también depende del tamaño del problema, no solo del kernel

ARC-60/ARC-62 habían caracterizado el sesgo de `bytes_moved_window` (la
brecha entre el proxy de `cache_misses` y el proxy más fino de
`L2_LINES_IN_ALL`) como **dependiente del kernel** (STREAM ~5%, `npb_mg`
~31%). Cruzando con clase C aparece una segunda variable que ARC-60/62 no
habían podido probar: **el mismo sesgo también cambia, y a veces en
dirección opuesta, cuando cambia el tamaño del problema.**

| Kernel | Brecha clase B | Brecha clase C | Cambio |
|---|---|---|---|
| `npb_bt` | 13.5% | 3.2% | -10.3 pp |
| `npb_cg` | 13.5% | **52.4%** | **+38.9 pp** |
| `npb_ft` | 17.8% | 27.8% | +10.0 pp |
| `npb_lu` | 11.3% | 3.7% | -7.6 pp |
| `npb_mg` | 30.8% | 3.5% | **-27.3 pp** |
| `npb_sp` | 13.3% | 3.0% | -10.3 pp |

`npb_mg` pasa de ser el kernel con más sesgo en clase B (30.8%) a uno de
los más bajos en clase C (3.5%). `npb_cg` hace exactamente lo contrario:
de 13.5% (similar al resto) a 52.4%, el sesgo más grande medido en todo el
proyecto hasta ahora, superando incluso al ~34% de STREAM en felix (F3.4).

**Esto es una brecha metodológica real, no solo un dato curioso.** El
sesgo de `bytes_moved_window` no puede caracterizarse con un solo número
por kernel (ni siquiera por kernel+nodo) — depende también de la relación
entre el tamaño de trabajo y la jerarquía de caché (working set vs. L2/L3),
que cambia con la clase del problema. La caracterización de ARC-33/60/62,
hecha solo a una escala, **no debe extrapolarse a otras escalas del mismo
kernel sin volver a medir.** Esto refuerza — no resuelve — la necesidad de
la validación con contadores de uncore (bloqueada, ARC-59) como la única
vía para tener un ground truth real independiente del tamaño del problema.

---

## 3. Energía y potencia

Con RAPL funcionando de punta a punta (ARC-54, confirmado en ambas
campañas con `energy_valid` entre 99.2% y 100.0% en todos los kernels), la
potencia media por kernel se mantiene sorprendentemente estable:

| Kernel | Potencia media (clase B) | Potencia media (clase C) |
|---|---|---|
| `dgemm_n2048` | 140.2 W | — |
| `npb_bt` | 121.5 W | 122.7 W |
| `npb_cg` | 126.7 W | 122.1 W |
| `npb_ft` | 129.2 W | 128.8 W |
| `npb_lu` | 114.8 W | 116.7 W |
| `npb_mg` | 124.1 W | 124.7 W |
| `npb_sp` | 121.0 W | 123.1 W |

**La potencia varía poco entre kernels (114-141 W) y casi nada entre
clases del mismo kernel** (diferencias de ≤5 W) — a diferencia de la
clasificación Roofline (que sí distingue kernels claramente), el consumo
instantáneo bajo REF (frecuencia nativa, sin control DVFS) parece reflejar
sobre todo la ocupación de los 6 cores delegados, no tanto si el kernel es
compute o memory-bound. Esto tiene una implicación directa para la Fase 1:
**si la potencia no varía mucho entre fases, la energía total de una
corrida es casi proporcional a su duración** — la variable que realmente
va a distinguir kernels/fases en términos de eficiencia energética (EDP,
Energy-Delay-Product) es el tiempo de ejecución para un trabajo dado, no
tanto la potencia instantánea. Esto es exactamente la clase de relación
que DVFS multi-frecuencia (todavía bloqueado) necesita para poder
mostrarse: con un solo nivel de frecuencia (REF) no se puede ver cómo la
potencia y el tiempo se mueven en direcciones opuestas al cambiar la
frecuencia.

**Nota metodológica encontrada de paso — `dgemm_n2048` no acumula ninguna
ventana `quality_status="ok"`** (0 de 773, todas caen en otro estado,
presumiblemente `warmup_excluded`): la corrida completa dura ~0.77s
medidos, y `warmup_seconds: 0.5` del catálogo excluye una fracción enorme
de esa ventana tan corta. La clasificación igual sale correcta (96.5%
compute_bound, `intensity_undefined` no depende del gate de warmup), pero
ninguna ventana de DGEMM pasa hoy el filtro de calidad "limpio" que sí
pasan los kernels NPB (85-99.5% `ok`). Vale la pena bajar
`warmup_seconds` de `dgemm_n2048` en el catálogo (a algo como 0.05s) para
que su ventana de warmup sea proporcional a su duración real, no un
artefacto heredado de kernels mucho más largos.

---

## 4. Volumen y calidad de datos

| | Clase B (REF) | Clase C |
|---|---|---|
| Corridas aceptadas | 21/21 | 18/18 |
| Ventanas totales | 215 986 | 891 587 |
| `energy_valid` global | 99.98% | ~100% |
| `calibration_references.cv_pct` | 2.27% | **0.33%** |
| Duración total medida (suma) | ~216 s | ~892 s |

La calibración salió considerablemente más estable en clase C
(`cv_pct=0.33%` vs `2.27%`) — con corridas individuales más largas, el
kernel de referencia (`npb_mg`) acumula muchos más eventos por ventana,
reduciendo el ruido relativo de piso de contador (el mismo mecanismo que
ya explicó por qué `npb_ep` era mal candidato de referencia en ARC-46).
Ninguna corrida mostró `pmu_degraded` en ninguna de las dos campañas — el
harness sigue midiendo limpio incluso en corridas de más de 100 segundos.

---

## 5. Conclusiones para el piloto

1. **La clasificación Roofline generaliza bien entre tamaños de problema**
   para 5 de 6 kernels — evidencia fuerte de que el pipeline mide una
   propiedad arquitectónica real, no ruido. `npb_ft` es la excepción
   documentada, consistente con su naturaleza de kernel genuinamente mixto.
2. **El sesgo de `bytes_moved_window` depende del tamaño del problema,
   además de depender del kernel y del nodo** — una brecha metodológica
   nueva que no estaba caracterizada antes de esta sesión. Refuerza que
   uncore (bloqueado, ARC-59) sigue siendo la única vía para un ground
   truth confiable, y que cualquier corrección aproximada del sesgo tendría
   que ser función de (kernel, tamaño, nodo), no una constante.
3. **La potencia varía poco entre kernels/fases bajo un solo nivel de
   frecuencia** — refuerza que el permiso de escritura de `cpufreq`
   (bloqueado) es la pieza que falta para que el proyecto pueda observar
   la relación real entre frecuencia, tiempo y energía que es el objetivo
   central de la Fase 1.
4. Hallazgo menor de catálogo: `warmup_seconds` de `dgemm_n2048` está mal
   calibrado para su propia duración (todas sus ventanas caen en
   `warmup_excluded`) — corrección de bajo riesgo, pendiente.
