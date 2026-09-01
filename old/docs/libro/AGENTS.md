# AGENTS.md — docs/libro/

Contexto de orientación para cualquier agente de IA (Codex, Claude Code u otro) que trabaje en el documento de tesis (`main.tex`) de este directorio. Léelo completo antes de editar. Complementa al `AGENTS.md` de la raíz del repositorio (que describe el código del orquestador, no el libro).

## 1. Regla dura, no negociable

**Nunca inventar referencias bibliográficas.** El autor ha sido explícito y repetido en esto: no le sirve una cita "plausible" generada por un modelo de lenguaje; debe ser un trabajo real y verificable.

Antes de añadir cualquier `\bibitem` nuevo a `main.tex`:
1. Verificar que el paper/documento existe realmente (buscarlo, confirmar autores, venue, año).
2. Verificar que el DOI dado resuelve al documento correcto (no basta con que "suene" a un DOI válido).
3. Si no se puede verificar con confianza, no se cita — se dice explícitamente en el texto que no se encontró precedente, en vez de inventar uno.

Este documento ya pasó por rondas de auditoría externa de sus citas (ver §4). Cualquier cita nueva debe pasar el mismo estándar.

## 2. Fuente de contenido autorizada

- `docs/general/plan_trabajo_grado.md` es la fuente original de la propuesta (objetivos, marco conceptual base, las 29 referencias `[1]`–`[29]` con DOI real). Los objetivos específicos y el objetivo general en `main.tex` son **verbatim** de ese plan — no se modifican sin instrucción explícita del autor.
- El trabajo técnico real de Fase 1 (harness C++, orquestador Python, hallazgos de campañas en pacca) documentado en `docs/retoma/`, `docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md` y el propio código en `orchestrator/` es la base para el contenido *adicional* del Marco Conceptual y toda la Metodología — no proviene del plan original.

## 3. Estado actual de `main.tex`

Redactado completo con resultados reales de la Fase 1, actualizado hasta los hallazgos del 2026-08-14. La campaña CPU limpia de reemplazo terminó con 126/126 combinaciones aceptadas e incorpora la medición directa de bytes de `uncore`, el ridge Roofline corregido y el presupuesto PMU reducido a nueve eventos. Sin embargo, la verificación posterior de `scaling_cur_freq` durante la carga demostró que los límites escritos por núcleo no retenían el reloj efectivo mientras el turbo global permanecía activo; esa campaña conserva valor para clasificación en el régimen realmente observado, pero no constituye un barrido DVFS válido ni evidencia del efecto de seis frecuencias distintas. El lector de frecuencia por ventana ya está implementado y verificado; una nueva campaña DVFS de CPU requiere primero resolver el control de turbo. La auditoría de precisión aritmética del catálogo GPU también está incorporada. `nvidia-smi -lgc` ya era el mecanismo operativo de bloqueo de reloj GPU y fue verificado directamente bajo carga: con el estado actual de la plataforma se sostuvieron objetivos de 600 y 1200 MHz durante 305 y 174 lecturas activas, respectivamente, con restauración posterior. La versión NVIDIA 610.57.04/CUDA 13.3 se registra como contexto de esa prueba, no como la causa ni el inicio de la funcionalidad. El diagnóstico anterior de 765 MHz se conserva como observación histórica no concluyente, no como prueba de que el actuador estuviera averiado. La campaña GPU multi-frecuencia completa sigue pendiente. Capítulos de Resultados, Discusión y Conclusiones tienen contenido real, no marcadores. Fases 2-4 (clasificador, agente, EDP) siguen fuera de alcance y se incorporarán al cerrarse cada una — no inventar cifras de esas fases.

El documento se optimizó de volumen el mismo día (72→63 páginas): el Marco Conceptual se condensó a la mitad, eliminando duplicación con Metodología — la medición de FLOPs por PMU (encoding, tabla de ponderación FMA, validación empírica) vive ahora únicamente en la sección 2.3 (Metodología), no repetida en el marco conceptual. Si vas a tocar cualquiera de las dos secciones, revisa la otra para no reintroducir la duplicación. Objetivo de extensión del autor: 80-90 páginas para el documento completo (las 4 fases), fuera de referencias y anexos — hay margen amplio todavía para Fases 2-4.

Decisiones ya tomadas por el autor, no reabrir sin que él lo pida:
- **Director:** Gilberto Javier Díaz Toro (no el placeholder original de la plantilla).
- **Plataforma experimental en Metodología:** solo paccaA100/Unicartagena. No mencionar felix/SC3 en ningún punto del documento (felix fue banco de pruebas descartado, sin RAPL — ver memoria de proyecto).
- **Marco Legal:** sección omitida a propósito (`\section{Marco Legal}` con comentario, sin contenido). El autor la definirá él mismo más adelante.
- **Estado del Arte:** portado del plan tal cual; su expansión queda pendiente y explícitamente diferida — no tocado en la optimización de volumen de 2026-08-11 más allá de lo estrictamente necesario.

## 4. Disciplina epistémica establecida (no aflojarla)

Varias secciones del Marco Conceptual y la Metodología fueron revisadas específicamente para separar con precisión "qué se mide" de "qué se estima/asume". Ejemplos ya resueltos que sirven de patrón para cualquier edición futura:

- La notación con sombrero ($\widehat{\text{FLOPs}}_i$) para un estimador por prorrateo ya no existe en el documento — el prorrateo se eliminó por completo del instrumento en ARC-100, y `main.tex` solo describe medición directa (`FLOPs_i^{HW}`, ecuación `eq:flops-medidos`, en Metodología §2.3). Si encuentras la notación con sombrero en algún borrador o memoria antigua, es historia superada, no el estado vigente.
- Afirmaciones sobre PMU/hardware evitan generalizaciones absolutas ("los contadores de FLOPs sobrecuentan" es incorrecto como propiedad general; depende del evento y la microarquitectura — hay que matizar).
- Cuando se menciona un método más riguroso que no se implementó (p. ej. PEBIL/BBV para FLOPs exactos), se declara honestamente por qué no se usó, no se presenta como si fuera equivalente a lo hecho.

Si vas a tocar estas secciones, mantén ese nivel de precisión — no lo simplifiques de vuelta a afirmaciones más fuertes de lo que el instrumento real soporta.

## 5. Verificación técnica contra el código real

Cualquier afirmación sobre "qué mide/hace el instrumento" en la Metodología debe poder verificarse contra el código real, no inventarse por plausibilidad. Puntos de verdad concretos:
- Contadores realmente adquiridos: `telemetry/src/perf_reader.cpp` (actualmente 9: instructions, cycles, cache-references, cache-misses, stalled-cycles-backend y los 4 sub-eventos de `FP_ARITH_INST_RETIRED`). `L2_LINES_IN_ALL` se retiró deliberadamente en ARC-132 porque `nmi_watchdog=1` reserva un PMC y forzaba multiplexación al solicitar diez; no volver a contarlo como señal activa ni reintroducirlo sin repetir la validación en hardware.
- Catálogo de kernels: `orchestrator/schemas/kernels/catalog.yaml` (9 entradas de CPU: 7 dataset + 2 calibración; 12 entradas de GPU: 8 dataset + 4 calibración). Tras ARC-110, `rodinia_hotspot` se retiró por mezcla de precisión no resuelta y se reemplazó por `rodinia_gaussian`; `rodinia_backprop`/`rodinia_myocyte` corrigieron su `gpu_precision` declarada de fp32 a fp64. Entre las cuatro calibraciones GPU hay tres microbenchmarks propios para los techos Roofline y una DGEMM de cuBLAS informativa.
- Lógica de post-procesamiento / `phase_label_train`: `orchestrator/postprocess.py`. FLOPs por ventana se **miden** directamente por hardware (`flops_measured_window`), no se prorratean — el mecanismo de prorrateo se eliminó por completo en ARC-100; no reintroducirlo sin que el usuario lo pida explícitamente.
- Bytes de CPU y granularidad de la etiqueta: `phase_label_train` usa exclusivamente `uncore_imc` real. Los conteos de `perf stat -I` ya son deltas por intervalo (piso práctico de ~10 ms), no acumulados; el post-procesamiento suma los FLOPs de las ventanas CPU cubiertas y difunde la intensidad/etiqueta del intervalo. `cache_misses × line_size` permanece solo como columna auxiliar y nunca como respaldo de clasificación.
- Frecuencia CPU observada: las campañas anteriores a ARC-135 tenían una lectura única posterior a la carga difundida a todas las ventanas. El instrumento actual lee `scaling_cur_freq` en C++ sobre el mismo tick que la PMU. No describir el dato antiguo como muestreo por ventana ni presentar una relectura exitosa de `scaling_min_freq`/`scaling_max_freq` como prueba de frecuencia efectiva bajo carga.
- Decisiones de diseño fuera del plan original, con su justificación y número ARC: `docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md`.

Si el texto del libro y el código no coinciden, el código manda — corrige el texto, no al revés.

## 6. Formato

`main.tex` usa `\bibitem`/`thebibliography` (no BibTeX/.bib), citación numérica vía `natbib`. Las 29 referencias del plan más las añadidas durante este trabajo están todas dentro de un único bloque `\begin{thebibliography}{99}...\end{thebibliography}` al final del documento — añadir ahí, no crear un archivo `.bib` aparte.
