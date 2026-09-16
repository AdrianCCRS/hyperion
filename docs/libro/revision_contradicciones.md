# Revisión de contradicciones internas del libro

Fecha: 2026-09-07.

Alcance: fuente vigente `main.tex` y sus seis capítulos incluidos desde `secciones/`. Se excluyó `main_fase02.tex`, que no forma parte del documento compilado. Esta revisión contrasta afirmaciones internas; no certifica la ejecución real de experimentos, no audita las referencias externas y no comprueba la correspondencia del PDF con las fuentes. No se modificó el manuscrito. Las líneas corresponden al estado revisado.

El problema predominante es la convivencia de estados distintos del proyecto sin un corte temporal uniforme: resultados históricos, protocolos nuevos y componentes implementados se presentan como si describieran un mismo estado final.

## Contradicciones de mayor impacto

### 1. Fases pendientes frente a componentes y evaluaciones descritos como realizados

- `secciones/05_conclusiones.tex:3`: los objetivos de entrenamiento, implementación del agente y evaluación EDP «no se presentan como ejecutados».
- `secciones/03_resultados.tex:3`: las Fases 2–4 quedan fuera del alcance de los resultados.
- `secciones/02_metodologia.tex:583`: «El agente de control se implementó»; la línea 607 declara implementados y verificados el bucle GPU y otros componentes.
- `secciones/02_metodologia.tex:621`: «El reporte de comparación se generó» y «La comparación se reportó».

**Problema:** las conclusiones niegan un avance que la metodología sí atribuye al trabajo. La limitación de la línea 625 —sin datos del escenario del agente— acota la evaluación, pero no reconcilia el resto de las afirmaciones.

**Corrección:** fijar el corte del informe y distinguir explícitamente diseño propuesto, componente implementado, prueba simulada y validación experimental. Si se conservan los avances parciales de Fases 3–4, deben aparecer con ese alcance en resultados, discusión y conclusiones; no basta con cambiar todos los verbos al futuro.

### 2. Turbo CPU resuelto en resultados y pendiente en conclusiones

- `secciones/03_resultados.tex:27`: «Resuelto el bloqueo de turbo» inicia la investigación de interferencia SMT.
- `secciones/03_resultados.tex:37`: la actuación CPU queda lista para una campaña con niveles verificados.
- `secciones/05_conclusiones.tex:9`: el control CPU todavía requiere autoridad administrativa sobre `no_turbo` o `max_perf_pct`.
- `secciones/05_conclusiones.tex:13`: la primera tarea futura es resolver el control del turbo.

**Problema:** se dan dos estados finales incompatibles del mismo bloqueo. Es legítimo conservar la invalidez de las campañas antiguas; no lo es mantener su causa como impedimento actual después de narrar su resolución.

**Corrección:** fechar los hitos y actualizar las limitaciones al último estado acreditado. Separar «actuador corregido» de «campaña final ejecutada y aceptada».

### 3. Política EDP derivada de un barrido que los resultados no consideran válido

- `secciones/02_metodologia.tex:599`: la tabla de política se habría derivado del barrido de Fase 1, con medianas por kernel y contrastes estadísticos, y serializado como cuatro entradas.
- `secciones/03_resultados.tex:18`: la campaña CPU no permite atribuir efectos de frecuencia sobre tiempo, energía o EDP.
- `secciones/03_resultados.tex:59`: la campaña GPU multi-frecuencia de producción sigue pendiente.

**Problema:** no queda identificado un conjunto experimental válido que sustente la derivación descrita. La política GPU «no actuar» por falta de medición de latencia no equivale a demostrar estadísticamente que ninguna frecuencia mejora el EDP.

**Corrección:** identificar campaña, datos y artefacto de política si existen. De lo contrario, describir el procedimiento implementado o propuesto, sin afirmar que produjo una política respaldada por el barrido experimental.

### 4. Unidad de entrenamiento incompatible con la entrada del clasificador y el lazo CPU

- `secciones/02_metodologia.tex:507`: una fila CPU corresponde a un intervalo del controlador de memoria, con deltas agregados; la clasificación tiene esa granularidad.
- `secciones/02_metodologia.tex:511`: una fila GPU corresponde a una corrida completa y sus agregados posteriores al calentamiento.
- `secciones/02_metodologia.tex:575`: el modelo recibe telemetría de una «ventana de ejecución», sin distinguir dispositivos.
- `secciones/02_metodologia.tex:583`: el lazo CPU usa ventanas de aproximadamente 1 ms y las considera coherentes con la granularidad de Fase 1.

**Problema:** el texto confunde la cadencia de adquisición con la unidad del modelo. El propio libro asigna a CPU intervalos un orden de magnitud mayores. Para GPU tampoco explica cómo producir durante una fase las características definidas sobre una corrida completa.

**Corrección:** definir por dispositivo la ventana de características, el momento en que están disponibles, la cadencia de inferencia y la cadencia de actuación. Un sondeo a 1 ms puede existir, pero no debe presentarse como una observación etiquetada independiente a esa resolución.

### 5. Precisión mixta excluida por el método y admitida en resultados

- `secciones/01_marco_referencia.tex:101`: una carga mixta exige definir previamente el tratamiento de la mezcla.
- `secciones/02_metodologia.tex:377`: el método de un techo por precisión solo se declara válido para cargas empíricamente homogéneas.
- `secciones/03_resultados.tex:46`: backprop y myocyte contienen 82.3 % y 71.3 % de operaciones FP64; se conservan y se cambia su precisión declarada porque la clasificación no varía.

**Problema:** corregir la precisión declarada no elimina las fracciones restantes de 17.7 % y 28.7 %. La justificación de estabilidad de clase introduce una excepción que el método no formaliza.

**Corrección:** documentar una regla explícita y verificable para admitir mezclas cuya clase sea robusta bajo los tratamientos considerados, o excluirlas conforme a la regla actual. No describirlas como homogéneas.

### 6. FLOPs exclusivamente medidos frente a DGEMM GPU calculado analíticamente

- `secciones/02_metodologia.tex:18`: el conteo reportado por una carga nunca es fuente del trabajo aritmético, solo contraste de una medición directa.
- `secciones/02_metodologia.tex:360`: la caracterización GPU se describe mediante conteos de Nsight Compute.
- `secciones/03_resultados.tex:48`: DGEMM GPU usa conteo analítico, porque los contadores convencionales no capturan su ruta Tensor Core.
- `secciones/02_metodologia.tex:315`: DGEMM GPU figura también como carga de conjunto de datos, no exclusivamente como referencia de calibración.

**Problema:** existe una excepción efectiva a la procedencia exigida del numerador. El contraste de dos cálculos no debe denominarse medición directa por los contadores descritos.

**Corrección:** declarar por separado la fuente de FLOPs y bytes de esta carga, su papel en entrenamiento o calibración y el techo compatible con su ruta de ejecución. Si solo es informativa, retirarla del bloque de entrenamiento.

## Contradicciones de protocolo y cifras

### 7. Dos composiciones del catálogo «final»

- `secciones/02_metodologia.tex:306`: seis familias NPB; las líneas 307–309 añaden DGEMM, LavaMD y 3MM. Son nueve cargas CPU de conjunto de datos, más dos de calibración.
- `secciones/03_resultados.tex:7`: el catálogo final tiene nueve cargas CPU en total: siete de conjunto de datos y dos de calibración.
- `secciones/02_metodologia.tex:326`: se menciona una ampliación para la campaña definitiva.

**Problema:** puede tratarse de dos versiones legítimas, pero la tabla y el resultado final no identifican esas versiones ni la campaña a la que pertenece cada una.

**Corrección:** separar catálogo histórico validado, ampliación candidata y catálogo definitivo aceptado, con número de familias y campaña correspondiente. No reemplazar simplemente siete por nueve en los resultados históricos.

### 8. Campaña «completa»: 66 corridas frente a una matriz de 126

- `secciones/02_metodologia.tex:547`: siete kernels, seis niveles y tres repeticiones, presentados como campaña real completa, pero se reportan 66 corridas.
- `secciones/02_metodologia.tex:410`: la matriz es el producto cartesiano de cargas, frecuencias y repeticiones.

**Problema:** 7 × 6 × 3 = 126, no 66. Contar aceptadas y rechazadas no explica por sí mismo las 60 combinaciones restantes.

**Corrección:** identificar si las 66 corridas pertenecen a una ejecución parcial, a una matriz efectiva distinta o a un subconjunto analizado, y proporcionar su relación con las 126 combinaciones previstas.

### 9. Referencia a una tabla de frecuencias que describe otra rejilla

- `secciones/03_resultados.tex:14`: REF y F0–F4 remiten a `tab:frecuencias`.
- `secciones/03_resultados.tex:16`: esos cinco niveles son 3600, 2900, 2200, 1500 y 800 MHz.
- `secciones/02_metodologia.tex:422`: la tabla referenciada contiene REF y once niveles fijos, empezando por 3200 MHz.
- `secciones/02_metodologia.tex:428`: sí distingue la rejilla histórica de la fina.

**Problema:** la evolución está explicada, pero la referencia concreta conduce a una tabla incompatible con el experimento reportado.

**Corrección:** crear una tabla histórica o remitir a la explicación de la rejilla histórica; conservar una tabla separada para el barrido definitivo.

### 10. Compilación por defecto frente a banderas modificadas

- `secciones/02_metodologia.tex:285`: todas las cargas de datos se compilan con opciones por defecto de cada suite.
- `secciones/03_resultados.tex:96`: se corrigió la compilación de los seis NPB para solicitar el conjunto de instrucciones nativo.

**Problema:** las opciones de producción ya no son las predeterminadas, según los propios resultados.

**Corrección:** documentar las opciones efectivamente utilizadas y su versión. Distinguir modificación de banderas de compilación de modificación del código fuente.

### 11. Generalización por familia frente a partición por corrida

- `secciones/02_metodologia.tex:328`: la familia algorítmica gobierna la validación; separar tamaños no demuestra generalización a patrones nuevos.
- `secciones/02_metodologia.tex:577`: permite partición por corrida o carga.
- `secciones/05_conclusiones.tex:13`: vuelve a dejar pendiente esa elección por corrida o carga.

**Problema:** separar corridas puede colocar la misma familia en entrenamiento y prueba. Esa evaluación responde a una pregunta más limitada que la generalización por familia declarada antes.

**Corrección:** establecer familias como grupos para la evaluación de patrones no observados. Si se desea evaluar además nuevas ejecuciones de familias conocidas, presentar ese experimento y su alcance por separado.

### 12. Dominio de frecuencia por núcleo lógico frente al control de hermanos SMT

- `secciones/02_metodologia.tex:43`: la tabla de plataforma declara «Dominio de frecuencia: Por núcleo lógico».
- `secciones/01_marco_referencia.tex:748`: explica que los hermanos SMT comparten núcleo físico y no tienen frecuencia necesariamente independiente.
- `secciones/03_resultados.tex:29`: la corrección experimental exige limitar también a los hermanos no usados por la carga.

**Problema:** la tabla confunde la exposición de límites por CPU lógica con el dominio físico de actuación demostrado en resultados.

**Corrección:** distinguir interfaz por CPU lógica, acoplamiento físico entre hermanos y conjunto de CPU que debe restringirse. Ajustar también la afirmación de `secciones/02_metodologia.tex:631` de que toda actuación se limita a los núcleos asignados, aclarando la asignación exclusiva y la cobertura de los hermanos.

## Ambigüedades adicionales que requieren reconciliación

1. **Inventario de la auditoría GPU.** `03_resultados.tex:42` anuncia siete kernels Rodinia; la línea 44 identifica tres puros; la 46 describe tres mixtos; la 48 llama «cuarto caso» a DGEMM de cuBLAS. El reparto tres más cuatro no cubre los siete Rodinia anunciados y mezcla suites. Falta identificar el veredicto de heartwall, presente en `02_metodologia.tex:313`, y separar DGEMM de la auditoría Rodinia.

2. **Invariancia GPU afirmada frente a convergencia exigida.** `02_metodologia.tex:371` y `:388` tratan la intensidad como invariable con el instante o el número de repeticiones; `:373` exige comprobar su convergencia al aumentar el trabajo. La distinción entre intensidad representativa estabilizada y mediciones transitorias podría reconciliarlas, pero debe formularse como condición verificada, no como garantía universal del algoritmo.

3. **Disponibilidad de los gobernadores de comparación.** El marco distingue los modos de `intel_pstate` y los gobernadores genéricos (`01_marco_referencia.tex:692`), mientras la metodología fija `ondemand` y `schedutil` (`02_metodologia.tex:613`) sobre una plataforma descrita con `intel_pstate`. La verificación relatada en `:625` usa archivos simulados. Falta documentar el modo real y la disponibilidad en el nodo o declarar escenarios condicionados a ella. Esto es una laguna de evidencia interna, no una demostración de incompatibilidad del nodo.

4. **Promesa de discusión no cumplida.** `02_metodologia.tex:607` afirma que la falta de integración CPU y las alternativas de detección GPU se retoman en discusión. `04_discusion.tex` se concentra en hallazgos de Fase 1 y no desarrolla esos puntos.

## Cambios históricos que no se consideran contradicciones

- Diez eventos PMU en la prueba inicial y nueve en producción: la reserva NMI y la retirada de L2 se explican.
- Diagnóstico GPU a 765 MHz y validación posterior a 600/1200 MHz: el libro conserva explícitamente el primero como no concluyente y no atribuye causalidad al cambio de controlador.
- Campaña aceptada por integridad de adquisición pero inválida para comparar estados DVFS: el texto distingue ambos criterios de aceptación.
- Ventanas CPU de aproximadamente 1 ms y bytes de aproximadamente 10 ms: la agregación y difusión de la etiqueta están explicadas. La contradicción aparece al definir el modelo y el lazo, no en esa coexistencia de señales.

## Orden sugerido de corrección

Primero, fijar fecha de corte y estado real de las cuatro fases, del turbo y de las campañas que respaldan la política. Segundo, unificar los contratos de granularidad, procedencia de etiquetas, precisión y partición. Tercero, reconciliar catálogos, matrices, tablas y compilación. Finalmente, actualizar resumen, discusión y conclusiones contra ese estado único.
