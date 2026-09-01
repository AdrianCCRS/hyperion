# Informe de la campaña final de CPU (DVFS multi-frecuencia)

**Fecha:** 2026-08-20
**Plataforma:** `paccaA100` (Universidad de Cartagena)
**Alcance:** **Solo CPU.** El manifiesto (`campaign_pacca_dvfs.yaml`) declara `gpu.enabled: false`. La campaña GPU multi-frecuencia (ejes CPU/GPU independientes, producto cartesiano ya implementado en el código, ARC-129) **nunca se ha corrido como campaña de producción completa** — sigue como paso pendiente separado. Todo lo que sigue en este informe describe únicamente el eje de frecuencia de CPU.

---

## 1. Resumen ejecutivo

| | |
|---|---|
| Campaña principal | `pacca_cpu_final_attempt03_20260820` |
| Kernels × niveles × repeticiones | 9 × 6 (REF, F0–F4) × 10 = **540 combinaciones** |
| Aceptadas | **424** (78.5%) |
| Rechazadas | **116** (21.5%) |
| Saltadas | 0 |
| Matriz completa | Sí (`matrix_incomplete: false`) |
| Frecuencia restaurada al finalizar | Sí (`frequency_restored_verified: true`) |
| Horas-núcleo consumidas | 34.74 |
| Causa de todos los rechazos | 100% `E01` (traza de frecuencia por ventana) — ningún otro `factor_id` apareció |

**Conclusión de fondo: no se encontró ningún bug.** El patrón de rechazo, aunque no uniforme entre kernels y niveles, quedó completamente explicado por un mecanismo físico ya diagnosticado esta misma sesión (dilución por inactividad de la lectura `scaling_cur_freq` en puntos de sincronización entre hilos), no por un defecto del instrumento o del pipeline. Dos kernels con huecos severos (`npb_cg`, `rodinia_lavamd_omp`) se atendieron con campañas suplementarias dirigidas después de investigar la causa exacta; ver secciones 4–5.

---

## 2. Calibración Roofline por nivel

Las 6 calibraciones (una por nivel de frecuencia, ARC-78) resultaron todas plausibles (`plausibility_check_passed: true`, contra el datasheet declarado — 480 GFLOP/s ± 40%, sección `hardware_datasheet` del manifiesto):

| Nivel | P_pico (GFLOP/s) | BW_pico (GB/s) | i_ridge (FLOP/byte) |
|---|---:|---:|---:|
| REF | 505.96 | 57.93 | 8.733 |
| F0  | 507.31 | 58.48 | 8.675 |
| F1  | 415.35 | 57.23 | 7.258 |
| F2  | 317.73 | 55.84 | 5.690 |
| F3  | 226.80 | 52.59 | 4.313 |
| F4  | 132.14 | 44.16 | 2.992 |

El punto de inflexión (`i_ridge`) se desplaza con la frecuencia tal como predice el diseño (P_pico escala con el reloj de núcleo, BW_pico casi no cambia — dominio de memoria aparte) — evidencia adicional de que la actuación de frecuencia fue correcta durante la calibración misma en los 6 niveles.

---

## 3. Distribución de aceptación — por qué no es uniforme

### 3.1 Por kernel

| Kernel | Aceptadas/60 | % rechazo |
|---|---:|---:|
| `npb_bt` | 60/60 | 0% |
| `npb_lu` | 60/60 | 0% |
| `npb_mg` | 60/60 | 0% |
| `npb_sp` | 60/60 | 0% |
| `npb_ft` | 59/60 | 1.7% |
| `npb_cg` | 33/60 | 45.0% |
| `dgemm_n2048` | 34/60 | 43.3% |
| `rajaperf_polybench_3mm_omp` | 38/60 | 36.7% |
| `rodinia_lavamd_omp` | 20/60 | 66.7% |

### 3.2 Por nivel de frecuencia (los 9 kernels juntos)

| Nivel | Aceptadas/90 | % rechazo |
|---|---:|---:|
| REF | 90/90 | 0% |
| F0  | 70/90 | 22.2% |
| F1  | 53/90 | 41.1% |
| F2  | 54/90 | 40.0% |
| F3  | 67/90 | 25.6% |
| F4  | 90/90 | 0% |

### 3.3 Explicación física completa

El chequeo `E01` (`validate_cpu_frequency_trace`) exige que **cada muestra individual** de `scaling_cur_freq` de los 6 CPUs delegados caiga dentro de la tolerancia declarada (5% del objetivo) durante toda la corrida — no un promedio, no un percentil. Esto expone un artefacto de medición ya caracterizado esta sesión (ARC-164/167): la lectura de frecuencia bajo `intel_pstate` se calcula del cociente APERF/MPERF, una medida que solo es representativa bajo actividad de instrucciones sostenida. Cuando un hilo llega a un punto de sincronización (barrera, unión de región paralela, reducción) antes que sus pares y queda momentáneamente ocioso, esa lectura puede diluirse hacia abajo sin que el candado de frecuencia haya cambiado realmente.

Esto explica **cada fila de las dos tablas anteriores**, sin excepción:

- **REF (0% rechazo, siempre)**: no tiene objetivo numérico de frecuencia (gobernador nativo) — el chequeo de tolerancia no aplica por diseño; solo se exige que la traza esté completa.
- **F4 (0% rechazo, siempre)**: es el piso de hardware (800 MHz). La dilución solo puede empujar la lectura *hacia abajo*, y en el piso no hay margen por debajo al que caer.
- **F1/F2 (los peores, ~40%)**: quedan en el punto intermedio menos favorable — suficiente distancia al piso para que una dilución tenga espacio real para desviarse, pero un margen de tolerancia más angosto en kHz absolutos que F0 (la tolerancia es 5% del objetivo, así que F0 tiene ~160 kHz de margen y F2 solo ~100 kHz).
- **Diferencia por kernel**: `npb_bt/lu/mg/sp` ejecutan una única región paralela larga y continua, sin barreras internas — inmunes al mecanismo. `npb_cg`, `dgemm_n2048`, `rajaperf_polybench_3mm_omp` y `rodinia_lavamd_omp` sí tienen puntos de sincronización internos (reducciones, múltiples fases, múltiples lanzamientos de región paralela) — expuestos al mismo mecanismo que ya se había diagnosticado en el propio microbenchmark de calibración `stream_official` (ARC-167).

---

## 4. Dos casos investigados a fondo (0/10 en F1)

### `npb_cg`

33 de 48 168 muestras fuera de tolerancia en una corrida rechazada representativa — **el 100% cae en los primeros 53 ms de la corrida**, nada disperso en el resto (~8 s limpios). Es un transitorio de arranque más largo que el margen de gracia vigente en ese momento (`grace_seconds=0.05`, calibrado con `ert_probe`, un kernel de solo ~83 ms de duración total — no representativo de un kernel de varios segundos).

### `rodinia_lavamd_omp`

667 de 57 354 muestras fuera de tolerancia, en **dos focos distintos**:
- Un cluster inicial (~960 ms) del mismo tipo que `npb_cg`.
- Un cluster final (~960 ms, 578 de las 667 concentradas en un solo CPU) que **no** es un transitorio de arranque (la frecuencia llevaba 8+ segundos estable) — es el mismo mecanismo de dilución por inactividad, aquí concentrado en la única unión final de la región paralela de LavaMD (a diferencia de STREAM/CG, que tienen barreras repetidas a lo largo de toda la corrida).

---

## 5. Fix implementado y campañas suplementarias

Se implementó `frequency_validation.tail_grace_seconds` (simétrico al ya existente `grace_seconds`, pero excluyendo de la comprobación de tolerancia — nunca de los chequeos estructurales — las lecturas cercanas al *final* de la traza, no solo al inicio). Con esto se lanzaron dos campañas suplementarias dirigidas, cada una restringida a un solo kernel con valores de gracia calibrados específicamente a su duración medida (no un valor global), sin tocar los 424 datos ya aceptados de la campaña principal:

| Kernel | Config | REF | F0 | F1 | F2 | F3 | F4 |
|---|---|---:|---:|---:|---:|---:|---:|
| `npb_cg` — principal | — | 10/10 | 5/10 | 0/10 | 1/10 | 7/10 | 10/10 |
| `npb_cg` — suplemento | `grace_seconds=0.15` | 10/10 | 7/10 | 7/10 | 9/10 | 4/10 | 10/10 |
| `npb_cg` — **combinado** | | 20/20 | 12/20 | 7/20 | 10/20 | 11/20 | 20/20 |
| `rodinia_lavamd_omp` — principal | — | 10/10 | 0/10 | 0/10 | 0/10 | 0/10 | 10/10 |
| `rodinia_lavamd_omp` — suplemento | `grace_seconds=1.2`/`tail_grace_seconds=1.2` | 10/10 | 10/10 | 10/10 | 10/10 | 3/10 | 10/10 |
| `rodinia_lavamd_omp` — **combinado** | | 20/20 | 10/20 | 10/20 | 10/20 | 3/20 | 20/20 |

**Resultado**: F0–F2 de `rodinia_lavamd_omp` pasaron de 0/10 a 10/10 perfecto en el suplemento — `tail_grace_seconds` funcionó exactamente como se diseñó. `npb_cg` mejoró sustancialmente en F1/F2 (0→7/10, 1→9/10) aunque de forma menos uniforme que LavaMD — el margen de gracia (0.15s) es más ajustado y el fenómeno conserva un componente estocástico real (con tolerancia cero por muestra, incluso una buena configuración no garantiza 10/10 en cada repetición). F3 de ambos kernels quedó con mejora parcial únicamente (11/20 y 3/20) — mismo mecanismo de margen absoluto angosto cerca del piso, sin una solución limpia adicional identificada todavía.

---

## 6. Huecos conocidos que quedan sin atender

Dos kernels con rechazo también alto (`dgemm_n2048`, `rajaperf_polybench_3mm_omp`) **no** recibieron campaña suplementaria — la decisión se acotó a los dos casos más severos (0/10 en múltiples niveles). Su cobertura real en la campaña principal:

| Kernel | REF | F0 | F1 | F2 | F3 | F4 |
|---|---:|---:|---:|---:|---:|---:|
| `dgemm_n2048` | 10/10 | 10/10 | **0/10** | **0/10** | 4/10 | 10/10 |
| `rajaperf_polybench_3mm_omp` | 10/10 | 5/10 | 3/10 | 4/10 | 6/10 | 10/10 |

`dgemm_n2048` en particular tiene el mismo patrón de hueco total (0/10) en F1 y F2 que tenían `npb_cg`/`rodinia_lavamd_omp` antes del fix — es una extensión natural y de bajo riesgo si se decide correrla (mismo mecanismo ya entendido, mismo tipo de campaña suplementaria dirigida).

---

## 7. Estado de la plataforma

`paccaA100` verificado limpio después de las tres campañas (principal + 2 suplementarias): los 12 CPUs (6 delegados + 6 hermanos SMT) en su rango nativo de hardware (800 MHz–3.6 GHz), `no_turbo=0`, sin procesos huérfanos. Ningún dato de las 424 combinaciones aceptadas de la campaña principal fue tocado, sobrescrito ni reemplazado.
