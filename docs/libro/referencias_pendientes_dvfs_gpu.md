# Referencias pendientes — DVFS de GPU y coordinación CPU–GPU

Staging para `docs/libro/main.tex`, que usa `\begin{thebibliography}{99}`
con entradas `\bibitem` en estilo IEEE. **No se pegaron directamente en
`main.tex`** porque en `thebibliography` toda entrada listada aparece en
la bibliografía final aunque no se cite, y ensuciaría el documento hasta
que efectivamente se usen.

Cada entrada indica **qué afirmación del trabajo respalda**, para poder
citarla donde corresponde sin releer el paper entero.

> **ESTADO DE VERIFICACIÓN.** Los autores de las entradas marcadas
> `[AUTORES VERIFICADOS]` se confirmaron contra la página del editor o
> arXiv. Las marcadas `[⚠ AUTORES SIN VERIFICAR]` tienen título, venue y
> DOI/URL correctos, pero **la lista de autores NO se confirmó** — hay
> que completarla antes de citarlas. No se inventó ningún nombre.

---

## 1. Métrica de energía: el campo está dividido

Respalda la decisión de **reportar energía de GPU y energía de sistema
por separado** (Anexos L y M): no existe un estándar único, así que la
elección debe declararse explícitamente en vez de asumirse.

`[AUTORES VERIFICADOS]`
```latex
\bibitem{Mei2016} X. Mei, Q. Wang, and X. Chu, ``A Survey and Measurement
Study of GPU DVFS on Energy Conservation,'' arXiv preprint
arXiv:1610.01784, Oct. 2016. doi: 10.48550/arXiv.1610.01784.
```

Punto clave para citar: el survey documenta que distintos trabajos miden
potencia de forma distinta (solo GPU vs. sistema completo) y que hay
trabajo previo enfocado en energía **a nivel de sistema** que concluye
que el DVFS de GPU afecta la energía del sistema **menos** que el DVFS de
CPU. **Esto corrobora el hallazgo del Anexo L como resultado conocido, no
como un artefacto de nuestro montaje.**

---

## 2. Predicción ML de la configuración óptima núcleo+memoria

Respalda el diseño del modelo: **dos mandos de frecuencia (núcleo y
memoria), no uno**, y el planteo de predecir la mejor configuración en
vez de barrerla.

`[AUTORES VERIFICADOS]`
```latex
\bibitem{Fan2020} K. Fan, B. Cosenza, and B. Juurlink, ``Accurate Energy
and Performance Prediction for Frequency-Scaled GPU Kernels,''
\textit{Computation}, vol. 8, no. 2, p. 37, 2020.
doi: 10.3390/computation8020037.
```

`[⚠ AUTORES SIN VERIFICAR — probablemente el mismo grupo que Fan2020]`
```latex
\bibitem{Fan2019} K. Fan, B. Cosenza, and B. Juurlink, ``Predictable GPUs
Frequency Scaling for Energy and Performance,'' in \textit{Proc. 48th
Int. Conf. Parallel Processing (ICPP)}, 2019.
doi: 10.1145/3337821.3337833.
```

---

## 3. Coordinación CPU–GPU: la respuesta del campo a "la CPU se come el ahorro"

Respalda directamente la pregunta abierta del proyecto. La postura de la
literatura **no** es excluir la CPU de la medición, sino **co-optimizar**
ambos dominios; regular uno solo "no desbloquea todo el potencial".

`[⚠ AUTORES SIN VERIFICAR]`
```latex
\bibitem{CoCap2016} ``Co-Cap: Energy-efficient Cooperative CPU-GPU
Frequency Capping for Mobile Games,'' in \textit{Proc. ACM Symp. Applied
Computing (SAC)}, 2016. doi: 10.1145/2851613.2851671.
```

`[⚠ AUTORES SIN VERIFICAR]`
```latex
\bibitem{Synergistic2018} ``Synergistic CPU-GPU Frequency Capping for
Energy-Efficient Mobile Games,'' \textit{ACM Trans. Embedded Computing
Systems}, 2018. doi: 10.1145/3145337.
```

`[⚠ AUTORES SIN VERIFICAR]`
```latex
\bibitem{CoDVFS2025} ``CoDVFS: Improving the Energy Efficiency of AI
Servers Through Coordinated DVFS,'' Springer, 2025.
doi: 10.1007/978-981-95-8405-5\_23.
```

Punto clave: `CoDVFS` usa **optimización bayesiana** para converger a la
configuración conjunta CPU+GPU sin barrer la grilla completa — relevante
si el espacio de búsqueda crece al agregar el mando de memoria.

`[⚠ AUTORES SIN VERIFICAR]`
```latex
\bibitem{Coordinated2013} ``Coordinated Energy Management in Heterogeneous
Processors,'' in \textit{Proc. Int. Conf. High Performance Computing,
Networking, Storage and Analysis (SC)}, 2013. [Online]. Available:
\url{https://casl.gatech.edu/wp-content/uploads/2013/08/AMD_SC2013_FINAL_pub.pdf}
```

---

## 4. Magnitudes de referencia para comparar nuestros resultados

Sirven para situar el 7.7–25.1% medido en el Anexo M.

| fuente | ahorro reportado | costo de rendimiento |
|---|---|---|
| Mei et al. (survey) | **19.28%** | **≤ 4%** |
| escalado consciente de aplicación (ACM 2023) | 26.7% (V100) / 20.2% (A100) | no confirmado |
| impacto en entrenamiento DNN | 8.7–23.1% | no confirmado |

**El objetivo defendible a batir es ~19% de ahorro con ≤4% de
degradación.** Nuestro mejor caso actual (`lavamd` @ F1: 25.1% de ahorro
de energía de GPU con +10.0% de tiempo) ahorra más pero **excede el
presupuesto de degradación** de la referencia — argumento a favor de la
grilla fina entre 1410 y 1110 MHz.

`[⚠ AUTORES SIN VERIFICAR]`
```latex
\bibitem{AppAware2023} ``Improving GPU Energy Efficiency through an
Application-Aware Approach,'' ACM, 2023.
doi: 10.1145/3627703.3629584.
```

---

## 5. Tesis doctoral de referencia (metodología completa)

`[⚠ AUTORES: J. Guerreiro — VERIFICAR resto de metadatos]` — ojo: el
libro **ya cita** `\bibitem{Guerreiro2019}` (Parallel Computing). Esta es
la tesis, un documento distinto y más extenso del mismo autor.

```latex
\bibitem{Guerreiro2020} J. Guerreiro, ``DVFS Modeling for Energy-Efficient
GPU Computing,'' Ph.D. dissertation, INESC-ID / Instituto Superior
Técnico, Univ. Lisboa, 2020. [Online]. Available:
\url{https://hpcas.inesc-id.pt/~handle/papers/PhD_JoaoGuerreiro_2020.pdf}
```

---

## 6. Tercer mando: número de unidades de cómputo

El survey de la sección 1 reporta un trabajo que construyó 448
configuraciones barriendo **8 conteos de unidades de cómputo (4, 8, …,
32) × 8 frecuencias de núcleo (300–1100 MHz) × 8 frecuencias de memoria
(475–1375 MHz)**.

Relevancia directa: respalda que el **conteo de recursos** es un mando
legítimo junto a la frecuencia — lo que en nuestro caso apunta al número
de núcleos de CPU delegados, dado que se midió que la *frecuencia* de CPU
mueve su potencia solo 0.7–3.8% (Anexo M / L.5).

La cita primaria de ese trabajo está dentro del survey (`Mei2016`,
referencia [24] de ese documento); **hay que rastrearla ahí antes de
citarla directamente.**

---

## 7. CPU: la literatura YA CITADA responde "qué hicieron ellos" (2026-08-23)

El hallazgo de `docs/general/resultados_compuertas_fase2.md` (clasificador
de fase NO supera la línea base trivial bajo LOKO: F1=0.393 vs 0.371;
α varía 0.642 ENTRE kernels pero solo 0.004–0.056 DENTRO de cada uno) es
el mismo patrón que GPU. Los tres papers de CPU que el libro **ya cita**
resuelven exactamente ese problema, de dos formas distintas:

### 7.1 Estrategia A: clasificar la APLICACIÓN, no la fase

`\cite{Guerreiro2019}` (GPU, pero el diseño transfiere íntegro): entrena
offline con benchmarks sintéticos, clasifica cualquier aplicación nueva
usando eventos de PMU recolectados **a una sola frecuencia de
referencia**, y predice el comportamiento en el resto de frecuencias.
16% de ahorro promedio, 36% pico, **0.74% de desviación promedio respecto
al óptimo real**. Con presupuesto de degradación ≤10%: hasta 26%.

`\cite{Calore2017}` (Haswell CPU + K80 GPU, ambos): probaron ajuste
**función por función** y lo abandonaron explícitamente por no ser
conveniente ("*clock tuning on a function-by-function basis is not
convenient*"), pivotando a **una frecuencia constante para todo el
programa** — ~7% de ahorro en GPU sin costo de rendimiento. Para CPU, la
regla que dan es literalmente la nuestra: *"fair energy savings are
possible by tuning the processor clock to lower values in all cases in
which the code is memory-bound"* — clasificación por balance de máquina
vs. intensidad operacional, el mismo Roofline que ya usamos
(`\cite{Williams2009}`).

Es la MISMA arquitectura que ya rediseñé para GPU tras el Anexo M:
features observados en un solo nivel de referencia, predicción por
kernel/aplicación, sin necesitar variación de fase intra-corrida. **No es
una arquitectura nueva para CPU — es la misma que GPU, y GPU ya tomó la
decisión correcta antes de saberlo.**

### 7.2 Estrategia B: control reactivo contra un gobernador débil

`\cite{Hebbar2022}` (Intel Core i7, SPEC CPU2017) es la respuesta
alternativa, y explica algo que no habíamos considerado: comparan contra
el gobernador **ondemand** real de Linux, no contra "siempre máxima
frecuencia". Hallazgo clave: *"el gobernador ondemand tiende a mantener
el procesador en la frecuencia más alta incluso cuando ~90% de los ciclos
activos están detenidos"* — el gobernador por defecto es malo detectando
cargas memory-bound. Su técnica muestrea una razón de PMU barata
(`stall_backend_ratio` o equivalente) **cada 100 ms** y la mapea
LINEALMENTE (continuo, no clasificación binaria) a una frecuencia,
durante toda la ejecución. Resultado: 121–183% de mejora en eficiencia
energética — **contra ondemand**, no contra máxima frecuencia.

Advertencia que dejan documentada: una variante que muestreó una métrica
más ruidosa (CPI directo) cambió de frecuencia demasiado seguido y generó
oscilaciones anchas (1.5→4.0 GHz) que empeoraron el consumo — la
inestabilidad del controlador es un riesgo real, no solo teórico.

### 7.3 Tercera confirmación independiente: clasificar el JOB, no la fase

`\cite{Antici2024}` (MCBound, SC24, producción real en Fugaku): F1-macro
≥ 0.89 clasificando trabajos completos como memory/compute-bound, **a
nivel de job**, no de fase. Tres papers, tres grupos, misma decisión de
granularidad.

### 7.4 Consecuencia para el acoplamiento CPU cuando vuelva CAP_PERFMON

1. El rediseño de GPU (predicción por kernel desde un nivel de
   referencia, sin clasificación de fase) **se acopla directo a CPU sin
   inventar nada nuevo** — es la arquitectura que la literatura ya usa
   para ambos dispositivos.
2. **Cambiar el baseline de comparación** de "siempre performance" a un
   gobernador real (`ondemand`/`schedutil` en Linux) es una vía barata y
   citable para revertir el resultado "desastroso": si el gobernador de
   pacca también ignora `stall_backend_ratio` alto, ahí puede vivir un
   ahorro grande que "siempre F0" nunca iba a mostrar.
3. Explorar un controlador reactivo (Hebbar-style) es una segunda línea
   de trabajo, independiente de si el dataset tiene o no variación
   intra-kernel — ataca el problema desde el lado del baseline, no desde
   el lado del dataset.
