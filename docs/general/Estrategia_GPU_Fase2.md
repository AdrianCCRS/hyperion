# Estrategia GPU — Fase 2 (documento para el director)

**Propósito de este documento.** No es un cambio de objetivos del
anteproyecto — es la corrección de cómo los estamos cumpliendo, basada en
evidencia empírica propia contrastada contra la literatura ya citada en
el marco teórico. Cada sección cierra con el mapeo explícito a los
Objetivos Específicos 1–4.

Fuente completa de la evidencia: `docs/general/PLAN_MAESTRO_FASE2.md`,
Anexos K, L, M (2026-08-23). Este documento es la síntesis ejecutiva de
esos tres anexos, más el rediseño de modelo que se deriva de ellos.

---

## 1. Qué se hizo (Objetivo Específico 1)

Se recolectó telemetría de bajo nivel (NVML para GPU, RAPL+Perf para el
lado CPU delegado) sobre 7 kernels GPU en 6 niveles de frecuencia, con 3
repeticiones cada uno (job 6462, 144 corridas), más un tamizaje dirigido
de 3 kernels adicionales (job 6463, 54 corridas). Los binarios GPU se
corrigieron de un problema de reproducibilidad de `nvcc` no detectado
antes (ARC-193) y se validaron checksums antes de cada campaña. Esto
cumple el Objetivo 1 al pie de la letra: caracterización de carga
computacional y consumo energético bajo distintos estados de frecuencia,
con NVML como interfaz de potencia.

## 2. El error que se corrigió, y por qué importa contarlo

La primera medición de energía sumó **GPU + paquete CPU + DRAM** (energía
total del nodo). Con esa métrica, el DVFS de GPU parecía no poder pagar
en absoluto: un recorte de reloj de 6.7× solo compraba 10–41% de caída de
potencia, y la conclusión inicial fue "es la plataforma, no el catálogo"
(Anexo L).

**Esa métrica era más estricta que el estándar del campo.** La literatura
de DVFS de GPU mide energía de **GPU vía NVML únicamente**
[Fan2020, Guerreiro2019, Mei2016] — es lo que una política de GPU
realmente controla; el piso de potencia del host es una propiedad del
nodo de medición, no del mecanismo bajo estudio. Con la métrica correcta,
el resultado se revierte (Anexo M):

| kernel | mejor nivel | ahorro E_gpu | costo de tiempo | ganancia EDP |
|---|---|---|---|---|
| `rodinia_lavamd` | F1 (1110 MHz) | **25.11%** | +10.02% | **17.60%** |
| `rodinia_heartwall` | F1 | **18.24%** | +33.12% | 1.03% |
| `rodinia_gaussian` | F1 | **15.35%** | +25.62% | 1.69% |
| `rodinia_myocyte` | F1 | **7.66%** | +28.02% | — |

Rango 8.7–25.1%, dentro de lo publicado (8.7–23.1% en entrenamiento DNN;
20.2–26.7% con escalado consciente de la aplicación en V100/A100
[AppAware2023]). El hallazgo del Anexo L (piso estático de potencia del
host, criterio `T(f)/T(F0) < P(F0)/P(f)`) **no se retracta** — se
reencuadra: describe correctamente por qué la energía *total del nodo* no
mejora mucho, un hallazgo real y citable en sí mismo, pero no es la
métrica que decide si la política de GPU vale la pena.

## 3. El diseño del modelo: por qué NO clasifica fase intra-ejecución, y por qué eso no rompe el Objetivo 2

**Lo que se creía necesario originalmente:** un clasificador de fase por
ventana, análogo al de CPU, que decida en cada instante si el kernel está
en régimen compute-bound o memory-bound.

**Lo que impedía eso (CAT-10):** en GPU, la intensidad operacional se
declara **estática por kernel** en el catálogo (a diferencia de CPU, que
la mide dinámicamente por ventana vía uncore), así que la etiqueta de
fase automática sale **constante** durante toda la corrida de un kernel.

**Lo que el propio dato mostró (Anexo K.8, `gpu_policy_headroom.py`):**
incluso si se pudiera medir la fase dinámicamente, **no habría nada que
clasificar dentro de una corrida** — el nivel óptimo es constante por
kernel en el rango de niveles medido; lo que varía es **entre** kernels.
CAT-10 dejó de ser un bloqueador porque el fenómeno que impedía medir
(alternancia intra-ejecución) no está presente en los kernels de este
catálogo, con esta grilla de frecuencias.

**Esto no es una desviación del Objetivo 2 — es exactamente la
arquitectura que ya usa la literatura publicada del campo:**

- `Guerreiro2019` (ya citado en el libro): clasifica la **aplicación**,
  no la fase; entrena con benchmarks sintéticos, observa contadores **en
  una sola frecuencia de referencia**, predice el comportamiento en el
  resto. 16% de ahorro promedio, **0.74% de desviación promedio respecto
  al óptimo real**.
- `Calore2017` (ya citado): probaron ajuste función-por-función y lo
  abandonaron explícitamente ("*clock tuning on a function-by-function
  basis is not convenient*"), pivotando a una frecuencia constante para
  todo el programa.
- `Antici2024` (ya citado, MCBound, producción real en Fugaku):
  clasifican el **job completo**, F1-macro ≥ 0.89.

Tres grupos, tres papers, misma decisión de granularidad. El Objetivo 2
pide un modelo clásico de ML (Árboles de Decisión / Bosques Aleatorios)
que clasifique **"las fases de ejecución… basándose en la telemetría"**,
en tiempo de ejecución y con baja latencia. Nada en ese texto exige que
la fase cambie *dentro* de una corrida — exige que la inferencia sea
**en vivo** y de **baja latencia**. El clasificador propuesto sigue
siendo eso: observa telemetría real durante la ejecución y decide una
frecuencia; lo que cambia es la granularidad de la unidad que clasifica
(carga/kernel, no ventana de 1 ms), que es la misma granularidad que
usan los sistemas ya publicados y evaluados con éxito.

## 4. Arquitectura propuesta

- **Entrada:** telemetría observada en **un solo nivel de referencia**
  (F0) — `gpu_util_pct`, `gpu_mem_util_pct`, potencia sobre reposo,
  `gpu_util_pct / gpu_mem_util_pct` (proxy de memory-boundness) — más la
  frecuencia candidata como feature.
- **Salida:** dos regresiones por (carga, nivel): `E(f)/E(F0)` y
  `T(f)/T(F0)`.
- **Política:** elegir el nivel de mínima energía sujeto a un
  presupuesto de degradación de tiempo — o, más fiel al Objetivo 4,
  **minimizar EDP directamente**, sin presupuesto arbitrario (ver §6).
- **Fuga de etiqueta:** prohibido todo lo que venga de la corrida del
  nivel candidato (su tiempo, su energía, su potencia); solo features de
  F0 + la frecuencia candidata.
- **Evaluación:** leave-one-kernel-out contra la **mejor constante única**
  (no contra "siempre F0") — es el rival real, y es el número que decide
  si vale la pena entrenar algo en absoluto.

## 5. El margen real, medido, no supuesto

`gpu_policy_headroom.py` compara tres políticas (siempre F0, mejor
constante única, oráculo por kernel) para cuantificar cuánto puede ganar
un modelo por encima de una regla trivial:

| presupuesto de degradación | mejor constante | oráculo | margen del modelo |
|---|---|---|---|
| ≤4% (literatura, `Mei2016`) | F0 → 0% | 0.58% | 0.58 pts |
| ≤15% | F0 → 0% | 4.76% | 4.76 pts |
| sin límite | F1 → 7.71% | 11.41% | 3.70 pts |

Con la grilla de 6 niveles, el margen defendible es angosto porque el
salto F0→F1 es de 300 MHz (10–33% de costo de tiempo) — casi nada cae
dentro de un presupuesto estricto. **La grilla, no la física, es la
causa**: la EDP de `lavamd@F1` ya gana 17.6% pese a superar el 4% de
degradación, porque el Objetivo 4 pide evaluar por EDP, no por un
recorte de tiempo arbitrario tomado de otra plataforma.

## 6. En marcha ahora mismo (sin esperar más nodo)

- **Job 6471** (grilla fina, 210 corridas): 4 escalones nuevos entre F0
  (1410 MHz) y F1 (1110 MHz) — exactamente donde vive el margen medido.
  Márgenes de potencia de esos 4 niveles **interpolados, no medidos**
  (riesgo declarado: si están mal, el síntoma es rechazo masivo I10, no
  un error silencioso).
- **Job 6472** (barrido de tamaño de `dwt2d`, 90 corridas): 5 tamaños del
  mismo kernel (192 a 16384 px), mismo binario/checksum, márgenes
  **medidos** (Anexo I) — sube la diversidad de carga sin el riesgo de
  compilar nada nuevo.
- **Screening por α ya validado como instrumento** (Anexo K.4): la
  hipótesis física es que `nvidia-smi -lgc` solo escala el reloj de SM,
  no el de memoria, así que el margen vive en kernels memory-bound.
  Verificado con calibración gratuita: `gpu_stream_bw` (ancho de banda
  puro) da α=0.071 frente a α=0.6–0.8 de los kernels de cómputo.

## 7. Riesgos abiertos, declarados sin adornar

1. Márgenes interpolados del job 6471 — a confirmar cuando termine.
2. Nunca se probó el **reloj de memoria** como segundo mando de DVFS —
   varios trabajos citados lo escalan junto al de núcleo
   [Fan2020, Guerreiro2019]; podría ser el mando relevante para
   `dwt2d`/`stream_bw`.
3. Con 7–11 kernels, el LOKO entrena sobre 6–10 — sigue siendo un piloto,
   no un resultado estadísticamente robusto.
4. La OI de los 3 tamaños intermedios de `dwt2d` está **interpolada**, no
   medida con `ncu` — no se usa como feature del modelo (evita CAT-10 por
   diseño), pero está documentado en el catálogo para no ocultarlo.

## 8. Mapeo explícito a los Objetivos Específicos

| Objetivo | Estado | Evidencia |
|---|---|---|
| 1. Caracterizar comportamiento y consumo bajo distintos estados de frecuencia (NVML) | **Cumplido**, ampliándose | Jobs 6462/6463/6471/6472 |
| 2. Clasificador ML de fases desde telemetría, en vivo, baja latencia | **Reinterpretado a granularidad de carga/kernel**, con precedente publicado triple (Guerreiro/Calore/Antici) | Anexo K.8, §3 de este documento |
| 3. Daemon de espacio de usuario, inferencia + política DVFS proactiva | Sin cambios de diseño; pendiente de implementación sobre el modelo de §4 | — |
| 4. Evaluación por EDP contra gobernador nativo | **EDP ya es la métrica primaria del análisis** (§5); GPU no tiene gobernador nativo en el sentido de CPU — el "nativo" es `native_governor`/REF, ya incluido en todas las tablas | Anexo K.2, `gpu_policy_headroom.py` |

---

## Referencias citadas en este documento

Ya en `docs/libro/main.tex`: `\cite{Guerreiro2019}`, `\cite{Calore2017}`,
`\cite{Antici2024}`. Pendientes de agregar (verificadas, ver
`docs/libro/referencias_pendientes_dvfs_gpu.md`): `Fan2020`, `Mei2016`,
`AppAware2023`.
