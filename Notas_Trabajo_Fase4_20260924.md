# Mapa de resultados de la Fase 4

**Propósito.** Esta nota organiza el estado experimental para discutirlo y decidir los siguientes pasos. No reemplaza el capítulo de resultados ni declara conclusiones nuevas para el libro.

**Corte de información.** 24 de septiembre de 2026. D, la medición previa de F y el confirmatorio E-A terminaron. El brazo F1 fijo del confirmatorio no produjo celdas válidas por el criterio de reloj declarado antes de medir.

## 1. Cómo leer los nombres

Hay dos niveles de nombres que no deben mezclarse.

| Nivel | Nombres | Significado |
|---|---|---|
| Escenario de Fase 4 | A, C, D, E y F | Experimento completo que responde una pregunta concreta. |
| Aplicación dentro de un escenario | A y B, por ejemplo E-A y E-B | Conjunto de fases concreto. A suele contener familias vistas durante el entrenamiento y B familias inéditas. |

Por tanto, **E-A no es el escenario A original**. Es la aplicación de familias conocidas del escenario E.

Los brazos se leen así.

| Brazo | Qué mide |
|---|---|
| REF o `base` | Referencia principal. Gobernador nativo, sin daemon. |
| `sombra` | El daemon clasifica pero no escribe frecuencias. Aísla su costo de observación e inferencia. |
| `activo` | El daemon clasifica y ejecuta su política. |
| `activo_gpu` | Solo el daemon de GPU actúa. Es el brazo pertinente para evaluar la política GPU sin contaminarla con el costo del daemon de CPU. |
| `activo_gpu_cpuobs` | El daemon de GPU actúa y el de CPU solo observa. Cuantifica el costo adicional de tener la CPU bajo observación. |
| `base_noturbo` | Control secundario sin turbo. No reemplaza a REF. |

La métrica principal es el EDP del nodo, `(E_CPU + E_GPU) × T`. La energía y el EDP de GPU sola explican el mecanismo cuando la política solo controla la GPU.

## 2. La pregunta experimental se divide en cuatro

1. ¿El daemon identifica el régimen de una fase durante la ejecución?
2. ¿Mantener el daemon introduce un costo relevante aun si no cambia frecuencias?
3. ¿Cambiar la frecuencia de CPU mejora el EDP en paccaA100?
4. ¿Cambiar la frecuencia de GPU mejora el EDP y bajo qué condiciones?

La Fase 4 no necesita que la respuesta sea positiva en todos los dispositivos. El aporte consiste en medir el alcance real de la política y sus límites.

## 3. Resultados ya disponibles

| Escenario | Pregunta | Diseño que importa | Resultado observado | Lectura de trabajo |
|---|---|---|---|---|
| A, matriz inicial | Funcionamiento general del agente en CPU, GPU y ejecución conjunta | Dos aplicaciones por alcance, familias vistas e inéditas, REF, sombra y activo | GPU quedó cerca de la paridad con REF. CPU activo tuvo EDP de 1.7 a 2.0 y el alcance conjunto de 1.3 a 1.6 frente a REF. | El resultado negativo de CPU no prueba todavía que actuar sea la causa, porque sombra ya tenía casi todo el costo. |
| Diagnóstico de A | De dónde viene el costo del daemon de CPU | Variación de periodo, fijación del consumidor y número de hilos de ONNX Runtime | El periodo de muestreo no explica el costo. La configuración inicial creó competencia de CPU mediante hilos de inferencia. | Se separó un problema de implementación del efecto físico de la política. |
| C, daemon de CPU corregido | Si la política de CPU mejora después de eliminar el costo grande del daemon | CPU y conjunto, daemon con un hilo de inferencia | Sombra quedó entre 1.036 y 1.094 de REF. Activo quedó entre 1.05 y 1.16 de REF y no mejoró frente a sombra. | La política CPU no aporta ganancia de EDP en esta plataforma, incluso con un daemon de costo mucho menor. |
| E-A, familias GPU vistas | Si el agente GPU gana en la región donde la Fase 2 predijo ahorro | Tres fases largas `memory_bound`, una de cómputo, sin daemon CPU en el brazo principal | `activo_gpu` redujo la energía GPU a 0.929 de REF, mantuvo el tiempo en 1.000 y redujo el EDP del nodo a **0.975** de REF. Las tres réplicas activas quedaron por debajo de las tres de REF. | **Este es el resultado positivo del daemon.** La política GPU entrega 7.1 % menos energía de GPU y 2.5 % menos EDP del nodo cuando detecta fases largas de memoria de familias conocidas. |
| E-B, familias GPU inéditas | Si la misma mejora se generaliza a memoria no vista | Dos fases declaradas de memoria y una de cómputo | Mediana de `activo_gpu` en 1.014 de EDP del nodo y 1.003 de energía GPU frente a REF. BabelStream solo recibió F1 en una réplica. Myocyte no abrió fases por utilización de GPU insuficiente. | No hay mejora agregada. El límite observado es la abstención o falta de detección, no una pérdida atribuible a aplicar F1. La única réplica donde actuó en BabelStream redujo 12 % la energía GPU y 5 % el EDP del nodo, pero es evidencia descriptiva de una sola réplica. |
| D, LAMMPS | Si el agente es inocuo en una aplicación HPC de terceros con actividad GPU limitada | Cuatro benchmarks, REF, sombra, `activo_gpu` y `activo_gpu_cpuobs`, tres repeticiones | Razón mediana de EDP de `activo_gpu` frente a REF entre 0.998 y 1.008 según benchmark. Con observación CPU, entre 1.030 y 1.043. En `lj` y `eam` hubo una decisión GPU `compute_bound` por celda, sin bajar frecuencia; en `chain` y `rhodo` no hubo decisión. | Validez externa acotada al costo y a la abstención de actuar. No aporta evidencia de ahorro GPU porque la política no aplicó F1. |
| Confirmatorio E-A, job 7684 | Reproducir el ahorro y comparar la decisión dinámica con F1 fijo | Cinco bloques con REF, sombra, `activo_gpu` y `fijo_gpu_f1` | Las cinco corridas válidas del agente quedaron por debajo de las cinco de REF. Razón de medianas de EDP del nodo 0.9716; energía GPU 0.9282 y duración 0.9990. Sombra quedó en 0.9935. Las cinco celdas F1 fijas quedaron inválidas por reloj observado fuera de 1230–1290 MHz bajo carga. | Se reprodujo la mejora frente a REF con una nueva campaña. No se obtuvo la comparación confirmatoria frente a F1 fijo. |

En el confirmatorio E-A, el modelo GPU emitió 20 decisiones sobre 20 fases, sin abstenciones ni errores respecto de las clases declaradas. En cada una de las cinco ejecuciones identificó tres fases de memoria y una de cómputo; solicitó F1 en memoria y liberó el reloj en cómputo. En la última fase de memoria el registro indica `written=false` porque F1 ya estaba aplicado. Estas son cinco repeticiones de las mismas cuatro familias y no miden generalización a familias nuevas. En D, `lj` y `eam` produjeron una decisión `compute_bound` por celda sin reducción de reloj; `chain` y `rhodo` no produjeron decisiones. Como D carece de verdad de fase para cada intervalo, esas decisiones no se puntúan como aciertos o errores.

## 4. El resultado que puede mostrarse con claridad

La afirmación defendible hoy es la siguiente.

> En paccaA100, el daemon de GPU redujo el EDP del nodo en 2.5 % frente a REF cuando la aplicación estuvo dominada por fases `memory_bound` largas de familias presentes en el entrenamiento. La duración no cambió y la energía de GPU se redujo 7.1 %.

La evidencia que sostiene esa afirmación tiene cuatro partes.

| Evidencia | Dato |
|---|---|
| Acción correcta | En E-A el daemon reconoció las fases de memoria y aplicó F1 a 1260 MHz. En la última fase no escribió porque la GPU ya mantenía ese reloj desde la fase anterior. |
| Costo del daemon GPU | Sombra quedó en 0.995 de EDP de REF. No hay costo medible que explique la mejora del brazo activo. |
| Efecto físico controlado | El brazo activo bajó la energía GPU a 0.929 de REF con tiempo relativo de 1.000. |
| Traslado a nodo completo | CPU aportó cerca de 64 % de la energía de E-A y no fue controlada. Por ello, el ahorro de GPU se redujo a 0.975 en EDP del nodo. |

No conviene formular este resultado como una mejora general para cualquier aplicación GPU. Sus condiciones observadas son fases de memoria largas, actividad de GPU suficiente para abrir una fase y una clasificación que no se abstenga.

## 5. Qué no debe confundirse con el resultado positivo

### CPU

La aplicación CPU-A de los escenarios A y C dedicó 71 % de su tiempo a NPB CG, una fase `memory_bound`. Por ello, C sí prueba una mezcla mayoritariamente de memoria con el daemon corregido. Aun así, una sola aplicación compuesta no permite afirmar que no existe ninguna carga CPU donde DVFS pueda ayudar.

La conclusión se sostiene para paccaA100 y la política evaluada al combinar tres niveles de evidencia. El barrido de Fase 2 cubrió 28 kernels `memory_bound` a través de nueve niveles de frecuencia. En F1, la razón mediana de EDP fue 1.010 frente a REF y no hubo mejora agregada. El ajuste físico de potencia mostró que cerca de 81 % de la potencia de paquete no escala con el reloj. Finalmente, dos pruebas selladas sobre familias nuevas intentaron aislar subgrupos con alta espera de memoria y tráfico de DRAM, y ambas refutaron la regla propuesta. Por tanto, la formulación precisa es que no se encontró una región generalizable donde actuar sobre la frecuencia de CPU mejore el EDP bajo este hardware y estas señales. El escenario C confirma esa conclusión en el lazo del daemon, pero no la establece por sí solo.

### Aplicaciones GPU cortas o mixtas de la matriz inicial

En la matriz A, las fases de memoria eran demasiado cortas o el agente se abstuvo. El resultado cercano a paridad no contradice E-A. Muestra que el ahorro requiere tiempo suficiente después de la primera decisión.

### E-B

E-B no es evidencia de que F1 dañe las familias inéditas. Es evidencia de que el clasificador actual no siempre activa en ellas. Debe permanecer junto a E-A al presentar los resultados, porque delimita la generalización de la ganancia.

### Daemon de CPU en observación

En E-A, añadir el daemon de CPU en sombra llevó el EDP del nodo a 1.003. Esto no invalida la ganancia del agente GPU. Define la configuración que conviene evaluar y reportar como positiva, un agente de GPU sin daemon CPU cuando la política CPU es `no_actuar`.

## 6. Escenarios pendientes y su función

| Estado | Escenario | Qué falta resolver | Resultado esperado antes de medir |
|---|---|---|---|
| Terminado, job 7683 | Medición previa de F | REF frente a solicitud de F1 en DGEMM CUTLASS, Conv2D y LavaMD, cinco repeticiones por nivel | Conv2D redujo el EDP de GPU sola a 0.894 de REF, pero el EDP del nodo quedó en 1.017 por el aumento de tiempo. LavaMD quedó en 0.960 de EDP GPU y 1.024 de EDP nodo. En DGEMM el reloj observado bajo solicitud de F1 fue 1140–1155 MHz, igual que en REF, por lo que ese par no valida una comparación a 1260 MHz sostenidos. |
| Terminado, job 7684 | Confirmatorio E-A | Cinco réplicas por REF, sombra, agente GPU y F1 fijo solicitado, con orden bloqueado y aleatorizado | El agente redujo el EDP del nodo 2.84 % en la mediana frente a REF, con separación completa entre los cinco valores de cada brazo. Las cinco celdas F1 fijas se invalidaron: aproximadamente 122–123 muestras cargadas por celda quedaron fuera de 1230–1290 MHz. |
| Propuesto, sin ejecutar | F | Determinar si conmutar por fase supera a mantener F1 fijo durante toda una aplicación balanceada | Requiere un diseño nuevo con un brazo fijo cuyo reloj solicitado pueda sostenerse bajo carga. La comparación fallida de E-A no autoriza a afirmar esta ventaja. |

La propuesta F no reemplaza E-A. Responde una pregunta adicional. E-A demuestra que el daemon puede ahorrar frente a REF. F determinaría si la decisión por fase agrega valor frente a dejar la GPU siempre en F1.

**Nota estadística del confirmatorio.** El protocolo registró una prueba exacta bilateral de permutaciones y citó 0.00794 como mínimo para cinco contra cinco sin restricciones. La ejecución se aleatorizó en cinco bloques, con una observación de cada brazo por bloque. Para respetar esa asignación, una prueba de permutaciones pareada solo puede intercambiar las etiquetas dentro de cada bloque y su menor valor bilateral con cinco bloques es 0.0625. Las cinco diferencias agente menos REF fueron negativas. La prueba sin restricciones arroja 0.00794, pero permite asignaciones que el diseño no podía producir; por ello no debe presentarse como la prueba exacta del diseño bloqueado. Esta discrepancia metodológica se debe declarar al interpretar el confirmatorio. La magnitud y la reproducción del efecto siguen siendo observaciones válidas.

## 7. Secuencia para explicarlo en una reunión

1. Se probó el agente completo y se separó el costo de observar del efecto de actuar mediante el brazo sombra.
2. En CPU, incluso tras corregir el costo del daemon, la física de potencia y tiempo no deja una región útil. El resultado es negativo y delimitado.
3. En GPU, la matriz general resultó neutra porque las fases fueron cortas o hubo abstenciones.
4. Se declaró antes de medir un escenario de aplicabilidad con memoria larga y familias vistas.
5. Allí el agente GPU obtuvo la mejora reproducible de 2.5 % en EDP del nodo y 7.1 % en energía GPU.
6. La variante con familias inéditas definió el límite de generalización. El agente solo ahorra cuando toma una decisión confiable.
7. D muestra paridad del agente GPU sin actuación útil en LAMMPS. El confirmatorio E-A reprodujo la mejora frente a REF, mientras que sus cinco celdas F1 fijas inválidas dejaron abierta la ventaja frente a ese brazo.

## 8. Mensajes que se pueden reutilizar después en el libro

- El daemon no se evalúa solo por exactitud de clasificación. Sombra permite separar el costo de mantenerlo activo del efecto de la frecuencia que escribe.
- La política de CPU fue descartada para esta plataforma por medición experimental, no por una falla de clasificación.
- La política GPU tiene una región de aplicabilidad medida. En E-A, el agente preservó el tiempo y redujo la energía GPU, lo que se tradujo en una mejora menor pero consistente del EDP del nodo.
- El ahorro del nodo está limitado por la fracción de energía que corresponde a CPU y que el agente GPU no controla.
- Las abstenciones en familias inéditas son un límite de generalización visible y cuantificado. No deben ocultarse ni presentarse como daño de F1 cuando no hubo actuación.

## 9. Archivos de respaldo

| Información | Fuente local |
|---|---|
| Matriz inicial A | `docs/libro/datos/fase4_20260924/fase4_matrix_main.csv` |
| Escenario E | `docs/libro/datos/fase4_20260924/fase4_E.csv` y `docs/libro/datos/fase4_20260924/tabla_E.tex` |
| Escenario D | `~/hyperion-results/final/fase4_D/results.csv` en pacca, job 7682 |
| Confirmatorio E-A | `~/hyperion-results/final/fase4_EA_confirmatorio/results.csv` y `cells/*/fixed_gpu_lock.txt` en pacca, job 7684 |
| Tablas y figuras de Fase 4 | `docs/libro/datos/fase4_20260924/` |
| Diseño de A, C, D y E | `Plan_Fase4_Evaluacion.md` y `Plan_Fase4_Escenario_E.md` |
| Medición previa de F | `scripts/pacca/hyp_fase4_F_f1compute.sbatch` |
| Confirmatorio E-A | `Protocolo_Fase4_EA_Confirmatorio.md` y `scripts/pacca/hyp_fase4_EA_confirmatorio.sbatch` |
