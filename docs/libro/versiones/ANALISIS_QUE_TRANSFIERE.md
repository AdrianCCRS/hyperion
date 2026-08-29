# Qué sobrevive al pivote y qué no — análisis para la reescritura del libro

Fecha: 2026-08-28. Acompaña a `main_v1_clasificador_fase_intrakernel_20260828.tex`,
copia íntegra del documento tal como quedó bajo la formulación anterior.

El propósito de este archivo es evitar que la reescritura tire trabajo válido por
asociación: el pivote cambió **el objeto que el modelo predice**, no el
instrumento que lo mide ni la física que lo explica. La mayor parte de la Fase 1
no solo sobrevive — pasa de ser "caracterización preliminar" a ser **la evidencia
que obliga al nuevo diseño**, que es un papel más fuerte del que tenía.

---

## 1. El punto de partida: qué cambió exactamente

| | Formulación anterior | Formulación vigente |
|---|---|---|
| Unidad de decisión | Ventana de ~1 ms dentro de un kernel; luego, la carga completa | **Una llamada a una operación** = una fase de una aplicación HPC multifásica |
| Etiqueta objetivo | `compute-bound` / `memory-bound` (proxy) | **Configuración óptima** `(dispositivo, frecuencia)` por `argmin` EDP (directa) |
| Rol del dispositivo | Dos modelos separados, CPU y GPU | Un solo modelo; el dispositivo es parte de la decisión |
| Momento de la decisión | Reactivo, durante la ejecución | **Antes** de despachar cada operación |
| Catálogo | 9 cargas CPU + 12 GPU, heterogéneas entre sí | 6 operaciones × rejilla de tamaños × 2 dispositivos = 68 `config_id` pareados |

El cambio de etiqueta es el más importante y el menos obvio: `compute/memory-bound`
nunca fue el objetivo, era un **proxy** de "qué frecuencia conviene". El pivote
elimina el intermediario y predice directamente la cantidad que interesa. Varios
resultados de la Fase 1 son, leídos hoy, la demostración empírica de que ese proxy
era el eslabón débil.

---

## 2. Transfiere íntegro y sin reinterpretación

Esto puede reusarse tal cual, cambiando a lo sumo referencias cruzadas.

### 2.1 Todo el aparato de verificación del instrumento

Es, con diferencia, el activo más sólido del trabajo y **no depende en absoluto de
qué predice el modelo**. Incluye la disciplina de no dar por funcional ningún
mecanismo sin comprobarlo contra el hardware real:

- Atribución por `perf_event_open` con PID + herencia, con el hijo detenido antes
  de su primera instrucción.
- Presupuesto de contadores simultáneos verificado empíricamente (no deducido del
  modelo de CPU), y el episodio de `nmi_watchdog` reservando un PMC.
- Codificación cruda de eventos gateada por `cpu family`/`model` exactos.
- Medición directa de FLOPs por hardware ponderada por ancho vectorial, tras
  eliminar por completo el prorrateo.
- Bytes reales de DRAM por `uncore_imc`, y el episodio del permiso "concedido"
  que no coincidía con el estado verificable del sistema.
- Verificación de la actuación DVFS **bajo carga**, que refutó la relectura de
  `scaling_min/max_freq` como prueba suficiente mientras el turbo global seguía
  activo.
- Calibración propia de los techos Roofline y su validación cruzada con una
  herramienta independiente.
- Overhead del propio instrumento medido (media 1.95 % sobre 540 pares), no
  supuesto despreciable.

Todo esto se conserva. De hecho la campaña dual lo reutilizó sin cambios y volvió
a pagar dividendos: la corrupción de frecuencia propagada entre jobs y el bug de
redondeo a pasos de 100 MHz se detectaron con esta misma disciplina.

### 2.2 Metodología de repeticiones

El estudio de convergencia de CV% (n=3..10) con el re-análisis combinatorio
C(10,3), y su conclusión de que 3 repeticiones son un mínimo operativo defendible
para clasificación pero **no** están probadas para evaluación de EDP. Transfiere
sin cambios y sigue siendo la justificación vigente de la política de
repeticiones, incluido el caso FFT.

### 2.3 Aspectos éticos, plataforma, y el criterio de inclusión en catálogo

El criterio de tres condiciones (verificación interna de corrección, reporte
explícito del volumen de trabajo, volumen expresado en FLOPs) sigue aplicando
íntegro al catálogo dual.

---

## 3. Transfiere, pero cambia de papel — y gana fuerza

Aquí está lo más valioso del análisis: resultados que bajo la formulación anterior
eran "malas noticias" o "limitaciones", y que bajo la nueva son **la justificación
del diseño**.

### 3.1 La homogeneidad de régimen — de fracaso a fundamento

**Hallazgo:** fracción media de clase minoritaria del 4.0 % por carga; 4 de 9
cargas con exactamente 0.0 % en los seis niveles. Clasificador por ventana bajo
partición que excluye una carga completa: $F_1$ macro 0.393 frente a 0.371 del
predictor trivial, con los ensambles de árboles por debajo del trivial.

**Papel anterior:** una limitación incómoda del muestreo, que "se aborda ampliando
el catálogo".

**Papel nuevo:** es el **resultado experimental que refuta la premisa de la
formulación anterior**. No mide una insuficiencia de los modelos ni de las
características: mide la ausencia del fenómeno. Y esa ausencia se buscó con
insistencia — LULESH, HPCG, GAP, CHOLMOD, C8, características temporales — sin
encontrarla. Un resultado negativo obtenido con esa persistencia es publicable y
es exactamente lo que legitima cambiar de unidad de decisión.

**Cómo escribirlo:** no como limitación en un rincón, sino como sección propia de
Resultados con su lugar en la cadena argumental hacia la nueva metodología.

### 3.2 El rango dinámico de potencia en CPU — de mala noticia a motivo del eje dispositivo

**Hallazgo:** bajar el reloj 4× (3200→800 MHz) reduce la potencia solo un 28 %
(107–143 W → 80–89 W), mientras el tiempo se estira entre 2.21× y 4.05×. Óptimo de
EDP fuera de la frecuencia máxima en **0 de 9 cargas** en CPU; ahorro medio del
0.7 %.

**Papel anterior:** el DVFS de CPU casi no sirve en esta plataforma — una
conclusión negativa que dejaba al trabajo sin margen que demostrar.

**Papel nuevo:** es precisamente **por qué la decisión no puede ser solo de
frecuencia**. Si en un dispositivo el piso de potencia estática es tan alto que
alargar la ejecución nunca compensa, entonces el grado de libertad que sí tiene
recorrido es *en qué dispositivo ejecutar*. El resultado deja de ser un callejón
sin salida y pasa a ser el argumento cuantitativo del eje CPU/GPU.

Contrasta además con GPU, donde sí hay margen: +7.4 % de energía media con ≤10 %
de degradación, hasta +27.5 %; y +9.5 % con ≤20 %, hasta +37.4 %. La **asimetría
entre dispositivos** es un hallazgo propio y es justamente lo que un selector
puede explotar.

### 3.3 El desplazamiento del punto de inflexión Roofline con la frecuencia

**Hallazgo:** el ridge se mueve de 8.733 a 2.992 FLOP/byte al bajar de 3200 a
800 MHz (×0.34).

**Papel nuevo:** explica por qué una etiqueta `compute/memory-bound` **no es una
propiedad de la carga sino del par (carga, frecuencia)**. La mezcla observada en
`npb_bt` y `npb_lu` no era alternancia algorítmica: era el ridge cruzando el punto
de operación. Es el argumento físico más limpio contra usar esa etiqueta como
objetivo de aprendizaje, y a favor de predecir el EDP directamente. Debe quedar
cerca de la justificación de la nueva formulación, no perdido entre resultados.

### 3.4 α (sensibilidad a la frecuencia) entre cargas frente a dentro de cargas

**Hallazgo:** α recorre 0.384–1.026 **entre** cargas, con $R^2$ de 0.976–0.9998,
pero es prácticamente constante **dentro** de cada una.

**Papel nuevo:** sostiene directamente que la unidad predecible es la operación,
no el instante. Es la contraparte cuantitativa y positiva de §3.1: no solo "no hay
variación intra-carga que aprender", sino "sí hay variación inter-carga, grande y
bien ajustada, que un modelo puede aprovechar".

### 3.5 La frontera de medición de energía

**Hallazgo:** dónde se decide medir la energía cambia la conclusión.

**Papel nuevo:** es el antecedente directo del **contrato cold/warm**. Ese contrato
no salió de la nada: es la misma pregunta metodológica, ahora resuelta con
marcadores absolutos en `CLOCK_MONOTONIC` que delimitan qué parte del proceso
cuenta como costo de despacho. Conviene presentarlos como continuidad, no como dos
cosas distintas.

### 3.6 La asimetría de instrumentación CPU/GPU

**Hallazgo:** en GPU no puede producirse intensidad operacional por ventana a
ningún costo; la telemetría del acelerador tiene su propia cadencia (~100 ms
frente a ~0.26 ms de la ventana de CPU), y sus filas son registros propios, no
ventanas de CPU enriquecidas.

**Papel nuevo:** justifica que el dataset de nivel 2 agregue por `config_id` en
lugar de intentar una etiqueta por ventana homogénea entre dispositivos. También
explica por qué el selector observa un agregado por operación y no una serie
temporal.

### 3.7 Cargas que no ejercitan el acelerador

**Hallazgo:** `rodinia_lud` con α=0.030 y $R^2$ negativo, potencia plana en
59–62 W: la GPU está en reposo. La "mejora del 30.5 % en EDP" era ahorro de
potencia en reposo, no adaptación sobre trabajo efectivo.

**Papel nuevo:** es el precedente conceptual del **criterio de actividad por
potencia** que hoy usa la campaña dual (líneas de reposo medidas por nivel, job
6714, con margen por nivel). Y es un ejemplo excelente de autocorrección honesta:
una lectura preliminar favorable fue retirada al entender la física. Conservar tal
cual.

---

## 4. Queda como historia, no como método vigente

- La formulación de Fase 2 anterior (predecir $E(f)/E(f_{ref})$ y
  $T(f)/T(f_{ref})$ con la carga completa como unidad) queda superada por el
  selector. Su justificación sigue siendo válida como paso intermedio del
  razonamiento, y merece una mención en la Discusión como etapa del recorrido, no
  como diseño vigente.
- El clasificador por ventana como "mecanismo secundario" para `npb_lu`,
  `npb_bt` y `3mm_omp`: cerrado en negativo. No reabrir. La memoria del proyecto
  (`intra-kernel-phase-hunt-negative`) lo registra explícitamente.
- Las cifras concretas de campañas de Fase 1 sobre el catálogo viejo (126/126,
  546/546, etc.) siguen siendo válidas **como lo que fueron**, pero no describen
  el dataset del selector. No mezclarlas.

---

## 5. Qué exige el marco teórico que hoy no tiene

Tres huecos reales, detectados al contrastar el marco conceptual actual con lo que
la nueva metodología necesita explicar:

1. **Selección de dispositivo en sistemas heterogéneos.** El marco cubre DVFS,
   Roofline, PMU, RAPL y NVML, pero no dice nada sobre *dónde ejecutar*. Es el
   grado de libertad central de la nueva formulación y hoy no está fundamentado.
   **Atención:** no hay en la bibliografía verificada del documento (43 entradas)
   ninguna referencia que cubra colocación de trabajo CPU/GPU. Hay que buscarla y
   verificarla; la regla de no inventar citas es dura.

2. **Costo de despacho y su amortización.** Transferencias H2D/D2H, creación de
   contexto CUDA, y el hecho de que ese costo se paga una vez por proceso y no por
   operación. Es lo que sostiene el contrato cold/warm y el umbral de
   amortización. Medido en este trabajo: init de CUDA de ~6.5 s a reloj alto
   frente a ~20.6 s a reloj mínimo, contra un bucle medido de ~2.14 s.

3. **El EDP como objetivo directo frente a etiquetas intermedias.** El marco
   presenta el EDP como métrica de evaluación (Fase 4) pero no como *objetivo de
   aprendizaje*. El pivote lo convierte en el target, y esa decisión merece
   fundamento explícito — apoyada, además, en §3.3, que muestra por qué el proxy
   era frágil.

El resto del marco conceptual (potencia CMOS, DVFS, Roofline, `perf_event`, RAPL,
NVML, fundamentos de ML, ventanas y muestreo) se mantiene: describe el
instrumento y la física, ninguno de los dos cambió.

---

## 6. Riesgo a vigilar en la reescritura

El documento anterior ya había hecho **un** pivote de granularidad (de ventana a
carga completa, §`sec:fase2-reformulacion`) con su justificación escrita. La
reescritura hace un segundo. Conviene que el texto no dé la impresión de una
metodología que se mueve cada vez que un resultado incomoda. La forma honesta de
presentarlo es como una **cadena de evidencia que converge**: cada cambio de
unidad estuvo forzado por una medición concreta, y el destino final —
la operación como unidad de decisión — es también la lectura más literal de los
objetivos específicos, que hablan de "las fases de ejecución de las aplicaciones"
y nunca de ventanas temporales.
