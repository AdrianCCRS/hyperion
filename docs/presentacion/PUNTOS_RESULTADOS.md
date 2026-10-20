# Resultados: puntos obligatorios de la sustentación

Lista de control para las diapositivas de Resultados y Conclusiones. Cada punto sale del libro
(`docs/libro/secciones/03_resultados.tex`, `04_discusion.tex`, `05_conclusiones.tex`); la etiqueta
entre corchetes es la sección de origen. Cifras exactas como están en el libro: no redondear
distinto ni reformular el alcance.

Leyenda: **[O]** obligatorio en el cuerpo · **[A]** al anexo (por si preguntan) · **⚠** matiz
que no puede faltar cuando se muestra la cifra.

---

## R1 · Datos (OE1, caracterización)

- **[O]** CPU: **1470 corridas** aceptadas (49 configuraciones × 10 estados: nativo + F0–F8 × 3
  repeticiones) → **1 309 755 intervalos elegibles** (91.9 %); 40.2 % compute / 59.8 % memory.
  [sec:resultados-cpu]
- **[O]** GPU: **483 corridas**, 19 kernels, **16 familias**; 171 compute / 312 memory.
  [sec:resultados-gpu-dataset-final]
- **[O]** El *ridge* baja con el reloj: **8.69 → 2.92 FLOP/byte** (nativo → F8), porque baja el techo
  de cómputo y el ancho de banda se mantiene. Una misma carga puede cambiar de clase con la
  frecuencia. [sec:resultados-cpu-dvfs]
- **[A]** Validación de la medición: Intel Advisor concuerda con los techos (ridge nativo 7.1–9.5);
  la frecuencia solo responde al restringir también los hermanos SMT (razón 0.995 → 3.78); GPU:
  ridge 14.97 (FP32) y 6.90 (FP64); precisión declarada ≠ ejecutada (hotspot retirado).

## R2 · Clasificador CPU (OE2)

- **[O]** XGBoost, 6 variables de Perf, LOFO sobre 30 familias: **0.728** de exactitud balanceada
  por celda (IC95 0.669–0.790). [sec:resultados-cpu-final]
- **[O]** El protocolo importa: mezclando intervalos al azar da **0.874**; dejando fuera la familia,
  0.728. Esa es la cifra honesta para programas nuevos. [sec:resultados-cpu-final, Discusión]
- **[O]** El techo es **información, no modelo**: los modelos con capacidad suficiente quedan en el
  mismo IC, Optuna no mejora, 25-vecinos sin ajuste llega a 0.684, hay perfiles gemelos entre
  clases opuestas, y con la intensidad operacional como entrada (prohibida) da 0.9989.
  [sec:resultados-cpu-limite]
- **[O]** Abstención τ = 0.85: decide el **72.2 %** de los intervalos, con 0.738 sobre lo decidido.
  ⚠ La confianza está mal calibrada fuera de familia (ECE 0.126; npb_ft falla con confianza alta).
  [sec:resultados-cpu-selectiva]
- **[A]** Matrices de confusión, resultado por familia (14 familias > 0.9; npb_ft 0.314), selección
  de XGBoost por latencia, invarianza de la frecuencia (0.42 % de cambios en F0→F1).

## R3 · Clasificador GPU (OE2)

- **[O]** Random forest de **3 señales invariantes** (utilización GPU, utilización de memoria y su
  desviación): **0.819** LOFO. [sec:resultados-gpu-invarianza]
- **[O]** Se eligió **por invarianza, no por exactitud**: tres modelos indistinguibles (RF 0.807,
  ET 0.806, RL 0.792); al fijar el reloj, la regresión logística cambia el **26.7 %** de sus
  predicciones; el RF sin reloj ni potencia mantiene (0.819). [sec:resultados-gpu-invarianza]
- **[O]** τ = 0.90: **0.900** sobre lo decidido, cobertura media por celda **65.7 %**; cero errores en
  memory; precisión de la clase accionable 0.937. [sec:resultados-gpu-invarianza]
- **[A]** GEMM y LUD cambian de régimen con el reloj (irresolubles sin fuga de etiqueta); ECE
  0.0868; importancia por permutación.

## R4 · Costo de la inferencia (OE2/OE3, pregunta de investigación)

- **[O]** Inferencia dentro del agente (ONNX Runtime): **16.4 µs** CPU, **97 µs** GPU (mediana). Es
  despreciable frente a los ~3 s que toma detectar una fase. [sec:resultados-agente-latencias]
- **[O]** ⚠ Pero el **proceso de CPU completo** sí cuesta: **6–14 W** de sobrecarga sin actuar (lectura
  de 9 contadores en 6 núcleos cada 1 ms, despertares, inferencia; causa no aislada). El de GPU no
  tiene costo medible. [sec:resultados-fase4-costo]
- **[A]** Cambio de reloj GPU 0.35–0.37 s; escritura de límites CPU 2.85 ms en paralelo;
  exportación ONNX sin cambios de etiqueta (Δp ≤ 3×10⁻⁷); restauración ante SIGTERM/SIGINT; piso de
  coordinación respetado en 99.2 % de las muestras.

## R5 · Política de frecuencia (OE3)

- **[O]** CPU: **no actuar** en ninguna clase. F1 en memory empeora 5.4 %; el oráculo por kernel
  ganaría solo 2.2 %. [sec:resultados-cpu-politica]
- **[O]** La causa es el **piso de potencia**: ~85 W, el **84 %** de la potencia a 3.2 GHz, no baja con
  el reloj de los núcleos delegados. En GPU el piso es el **55.8 %** → sí queda margen.
  [sec:resultados-cpu-politica, sec:resultados-gpu-politica]
- **[O]** GPU: memory → **F1 (1260 MHz)**, compute → no actuar. **+8.9 %** de EDP (IC95 3.0–13.3,
  p = 0.020; 7 de 8 familias). ⚠ Es estimación **interna**; la evidencia de generalización es
  **6.5 %** LOFO con IC −4.0 a 13.1 (incluye el cero). [sec:resultados-gpu-politica]
- **[O]** Mecanismo común: la pendiente β de log T vs log f (compute ≈ 0.84, memory ≈ 0.19). La clase
  Roofline marca la oportunidad, no la garantiza: **rodinia_gaussian** (memory, β 0.94) pierde
  27.3 %. [sec:resultados-gpu-politica, Discusión]
- **⚠** Alcance CPU: solo frecuencia de los **6 núcleos delegados**, turbo off, sin *uncore*. No es una
  conclusión sobre la CPU en general.

## R6 · Evaluación frente a la base (OE4)

Métrica: EDP CPU+GPU = (E_pkg0 + E_pkg1 + E_GPU) · T, razón frente a la base (< 1 mejora).
Cifra principal en escenarios por bloques: media geométrica por bloque.

- **[O]** Matriz inicial: el agente de **GPU no cambia nada** (1.00, sombra 1.00–1.01); el de **CPU
  empeora**: 1.15 (A) y 1.06 (B) en CPU, 1.04 / 1.06 en conjunto (**p = 0.0079**). La **sombra** ya
  explica la mayor parte → el costo es observar, no actuar. [sec:resultados-fase4-edp]
- **[O]** **E-A**: EDP CPU+GPU **0.962 (−3.8 %, IC95 2.3–5.4 %)**, energía GPU −7.2 %, duración sin cambio
  (0.992, IC incluye 1); **5 de 5 bloques**. [sec:resultados-fase4-E]
- **[O]** **CloverLeaf** (externa, continua): **0.961 (−3.9 %, IC95 3.2–4.6 %)**, energía GPU −6.4 %,
  duración 1.001; 5 de 5 bloques. ⚠ Una sola decisión memory sostenida: se comportó como F1 fijo y
  no puntúa clasificación. [sec:resultados-fase4-cloverleaf]
- **⚠** p = 0.0625 es el **mínimo posible** con 5 bloques: dirección consistente, no confirmación
  estadística; IC calculados *post hoc*.
- **[O]** Por qué el ahorro es "modesto": la CPU es el **64 %** de la energía medida y el agente de GPU no
  la toca. [sec:resultados-fase4-E, Discusión]
- **[O]** Powersave: todas las conclusiones se mantienen bajo ambos gobernadores (E-A 0.974,
  CloverLeaf 0.968). Responde a "frente a los gobernadores nativos". [sec:resultados-fase4-powersave]

## R7 · ¿Aporta clasificar por fase? (OE4, el resultado clave)

- **[O]** En **E-A, F1 fijo iguala al agente** (agente/F1 fijo 0.998, IC 0.990–1.006): la fase DGEMM ya
  corría bajo 1260 MHz por el límite de potencia de la A100. Hay que decirlo, no esconderlo.
  [sec:resultados-fase4-E]
- **[O]** **E-C** (heartwall, compute sensible al reloj, entre stencil y SpMV): agente/F1 fijo **0.972
  (−2.8 %, IC95 1.5–4.2 %)**, **6 de 6 bloques, p = 0.031**; el agente frente a la base 0.976, mientras
  F1 fijo **no mejora** la base (1.004). Heartwall: 29.9 s base, 30.3 s agente, 33.6 s F1 fijo. Se
  reproduce bajo powersave (−2.2 %). [sec:resultados-fase4-EC]
- **⚠** E-C es una **prueba de mecanismo**: escenario construido con el kernel más favorable y una
  familia vista. No es validación externa.

## R8 · Lo que no funcionó (OE4)

- **[O]** **CPU**: el agente empeora el EDP (no se recomienda desplegarlo en esta plataforma).
- **[O]** **E-B** (familias inéditas): paridad (sombra 1.028, activo 1.016; descriptivo); el agente se
  **abstuvo** en BabelStream en 9 de 11 repeticiones; F1 fijo ahorró más (0.892 vs 0.966 en energía
  GPU, exploratorio). El límite es la **cobertura** ante familias nuevas.
- **[O]** **LAMMPS**: paridad (1.003–1.009); el agente no aplicó F1 (sin decisiones o solo compute).
- **[A]** Modelo GPU entrenado por corrida y aplicado por ventana de 3 s (triad: 64 de 109
  abstenciones); bimodalidad de myocyte; primera decisión 13–16 s tras el inicio de fase.

## R9 · Contraste con el estado del arte (Discusión)

- **[O]** Ahorro menor que la literatura en GPU: Ali et al. 29.6 % (GA100), DRLCap 22 %, Simsek 7.82 %
  de **energía de GPU**, frente a **7.2 / 6.4 %** aquí. Razones: (1) aquí la métrica es EDP CPU+GPU,
  (2) EDP penaliza tiempo → F1, el nivel más cercano al nativo, (3) el controlador de la A100 ya
  ahorra al tocar el límite de potencia, (4) una regla por clase, sin caracterizar cada aplicación.
  Medido solo en GPU, el orden es el de Simsek et al.

## R10 · Respuesta a la pregunta de investigación (Conclusiones)

> ¿En qué medida el agente … reduce el consumo … sin que el costo de la inferencia degrade el
> rendimiento global … frente a los gobernadores nativos?

- **[O]** **GPU: sí**, con fases de memoria largas: −3.8 / −3.9 % de EDP CPU+GPU sin alargar la ejecución,
  bajo ambos gobernadores; la inferencia (97 µs, una vez por episodio) no tiene costo medible.
- **[O]** **CPU: no**, en esta plataforma: el piso de potencia no deja margen y la sobrecarga total del
  proceso (que incluye y acota la inferencia) aumenta el EDP.
- **[O]** Recomendación: **desplegar solo el proceso de GPU**.
- **[O]** La clasificación por fase aporta cuando hay fases de cómputo sensibles al reloj (E-C); donde
  no las hay, una regla fija rinde igual.

## Trabajo futuro (una diapositiva o tarjeta, no más)

- Ampliar familias externas GPU memory-bound (bajar abstenciones) y evaluar el modelo congelado una vez.
- Repetir base / agente / F1 fijo en aplicaciones externas con fases de cómputo sensibles al reloj.
- Reentrenar GPU por ventana; CUPTI para fronteras de fase; DCGM (`dram_active`).
- CPU: reducir primero el costo de observación.

---

## Reglas de redacción para estas diapositivas

1. Una idea por diapositiva; la cifra grande y el matiz ⚠ en una línea corta.
2. Nunca presentar el 8.9 % sin el 6.5 % fuera de muestra, ni E-A sin decir que F1 fijo lo iguala.
3. "EDP CPU+GPU" siempre con ese nombre (no "EDP" a secas en Fase 4).
4. p = 0.0625 → "5 de 5 bloques"; no decir "significativo".
5. Lo [A] va al anexo; no se elimina información, se mueve.
