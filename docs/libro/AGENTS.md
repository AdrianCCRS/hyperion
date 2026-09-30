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

- `old/docs/general/plan_trabajo_grado.md` es la fuente original de la propuesta (objetivos, marco conceptual base y las referencias iniciales con DOI real). La pregunta de investigación, el objetivo general y los objetivos específicos de `secciones/00_frontmatter.tex` son **verbatim** de ese plan y no se modifican sin instrucción explícita del autor.
- El alcance vigente del proyecto lo fijan `Plan_Detallado_Realineacion_Hyperion.md` y `Seguimiento_Cambios_Plan_Director.md` (raíz del repositorio). Las decisiones de diseño anteriores fuera del plan original, con su número ARC, están en `old/docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md`.

## 3. Estado actual del libro

El libro está completo con las cuatro fases (instrumento y datos, clasificadores y política, agente, evaluación) y en revisión final (unas 78 páginas). `main.tex` solo incluye los capítulos de `secciones/` mediante `\input`; se edita cada capítulo en su archivo. Las cifras salen de `datos/` y las figuras de `scripts/` (ver `datos/README.md` y `README.md`, que es el material complementario citado en el libro como \matcomp).

La serie de la Fase 4 con el gobernador `powersave` (job 7789, terminado 2026-09-29) ya está integrada: matriz inicial, E-A, E-B, D y CloverLeaf tienen su párrafo de contraste `performance`/`powersave` en `secciones/03_resultados.tex`, con las cifras derivadas de `datos/fase4_20260930/` vía `scripts/generar_tablas_fase4_powersave_20260930.py`.

La última revisión crítica del texto está en `recordatorios/revision_panel_libro_final_20260929.md`, con el estado de cada hallazgo.

Decisiones ya tomadas por el autor, no reabrir sin que él lo pida:
- **Director:** Gilberto Javier Díaz Toro (no el placeholder original de la plantilla).
- **Plataforma experimental:** solo paccaA100/Unicartagena. No mencionar felix/SC3 en ningún punto del documento (felix fue banco de pruebas descartado, sin RAPL).
- **Marco Legal:** omitido a propósito; el autor lo definirá.
- **Estilo:** sin el guion largo (`---`) en la prosa; se usan comas, paréntesis o dos puntos. El guion medio (`--`) de rangos y de "Energía--Retardo" sí se usa.
- **Contenido:** solo la metodología y el resultado finales; no se narran bugs ni depuración del instrumento.

## 4. Disciplina epistémica establecida (no aflojarla)

Varias secciones del Marco Conceptual y la Metodología fueron revisadas específicamente para separar con precisión "qué se mide" de "qué se estima/asume". Ejemplos ya resueltos que sirven de patrón para cualquier edición futura:

- La notación con sombrero ($\widehat{\text{FLOPs}}_i$) para un estimador por prorrateo ya no existe en el documento — el prorrateo se eliminó por completo del instrumento en ARC-100, y `main.tex` solo describe medición directa (`FLOPs_i^{HW}`, ecuación `eq:flops-medidos`, en Metodología §2.3). Si encuentras la notación con sombrero en algún borrador o memoria antigua, es historia superada, no el estado vigente.
- Afirmaciones sobre PMU/hardware evitan generalizaciones absolutas ("los contadores de FLOPs sobrecuentan" es incorrecto como propiedad general; depende del evento y la microarquitectura — hay que matizar).
- Cuando se menciona un método más riguroso que no se implementó (p. ej. PEBIL/BBV para FLOPs exactos), se declara honestamente por qué no se usó, no se presenta como si fuera equivalente a lo hecho.

Si vas a tocar estas secciones, mantén ese nivel de precisión — no lo simplifiques de vuelta a afirmaciones más fuertes de lo que el instrumento real soporta.

## 5. Verificación técnica contra el código real

Cualquier afirmación sobre "qué mide/hace el instrumento" en la Metodología debe poder verificarse contra el código real, no inventarse por plausibilidad. Puntos de verdad concretos:
- Contadores realmente adquiridos: `common/telemetry/src/perf_reader.cpp` (9: instructions, cycles, cache-references, cache-misses, `CYCLE_ACTIVITY.STALLS_MEM_ANY` como ciclos detenidos por memoria, y los 4 sub-eventos de `FP_ARITH_INST_RETIRED`). En paccaA100 el evento genérico `stalled-cycles-backend` no está mapeado; no describir la variable como ciclos detenidos del *backend*. `L2_LINES_IN_ALL` se retiró deliberadamente en ARC-132 porque `nmi_watchdog=1` reserva un PMC y forzaba multiplexación al solicitar diez; no volver a contarlo como señal activa ni reintroducirlo sin repetir la validación en hardware.
- Catálogo de kernels: `fase1_telemetria/catalog/catalog.yaml` (el libro reporta 30 familias en CPU y 16 en GPU; verificar conteos contra el catálogo y `datos/`). Tras ARC-110, `rodinia_hotspot` se retiró por mezcla de precisión no resuelta y se reemplazó por `rodinia_gaussian`; `rodinia_backprop`/`rodinia_myocyte` corrigieron su `gpu_precision` declarada de fp32 a fp64. Entre las cuatro calibraciones GPU hay tres microbenchmarks propios para los techos Roofline y una DGEMM de cuBLAS informativa.
- Lógica de post-procesamiento / `phase_label_train`: `fase1_telemetria/postprocess.py`. FLOPs por ventana se **miden** directamente por hardware (`flops_measured_window`), no se prorratean — el mecanismo de prorrateo se eliminó por completo en ARC-100; no reintroducirlo sin que el usuario lo pida explícitamente.
- Bytes de CPU y granularidad de la etiqueta: `phase_label_train` usa exclusivamente `uncore_imc` real. Los conteos de `perf stat -I` ya son deltas por intervalo (piso práctico de ~10 ms), no acumulados; el post-procesamiento suma los FLOPs de las ventanas CPU cubiertas y difunde la intensidad/etiqueta del intervalo. `cache_misses × line_size` permanece solo como columna auxiliar y nunca como respaldo de clasificación.
- Frecuencia CPU observada: las campañas anteriores a ARC-135 tenían una lectura única posterior a la carga difundida a todas las ventanas. El instrumento actual lee `scaling_cur_freq` en C++ sobre el mismo tick que la PMU. No describir el dato antiguo como muestreo por ventana ni presentar una relectura exitosa de `scaling_min_freq`/`scaling_max_freq` como prueba de frecuencia efectiva bajo carga.
- Agente: `fase3_daemon/` (procesos C++ de CPU y GPU; la detección de episodios de GPU está en `gpu_loop_cpp/include/gpu_activity_tracker.hpp`). Evaluación: `fase4_evaluacion/` (índice de campañas en su `README.md`).

Si el texto del libro y el código no coinciden, el código manda — corrige el texto, no al revés.

## 6. Formato

La bibliografía usa BibTeX: las entradas están en `main.bib` y el estilo es `IEEEtran` con `natbib` en modo numérico, así que la numeración sigue el orden de primera cita sin intervención manual. Añadir referencias nuevas solo en `main.bib`, respetando la regla del §1. Compilar con `latexmk -pdf main.tex` (el PDF no se versiona).
