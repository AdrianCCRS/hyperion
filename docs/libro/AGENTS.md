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

Escrito y revisado hasta el final del capítulo de **Metodología** (inclusive). Capítulos de Resultados, Discusión y Conclusiones son marcadores (`% [PENDIENTE]`) — la Fase 1 es la única ejecutada; no hay cifras finales que reportar todavía.

Decisiones ya tomadas por el autor, no reabrir sin que él lo pida:
- **Director:** Gilberto Javier Díaz Toro (no el placeholder original de la plantilla).
- **Plataforma experimental en Metodología:** solo paccaA100/Unicartagena. No mencionar felix/SC3 en ningún punto del documento (felix fue banco de pruebas descartado, sin RAPL — ver memoria de proyecto).
- **Marco Legal:** sección omitida a propósito (`\section{Marco Legal}` con comentario, sin contenido). El autor la definirá él mismo más adelante.
- **Estado del Arte:** portado del plan tal cual; su expansión queda pendiente y explícitamente diferida.

## 4. Disciplina epistémica establecida (no aflojarla)

Varias secciones del Marco Conceptual y la Metodología fueron revisadas específicamente para separar con precisión "qué se mide" de "qué se estima/asume". Ejemplos ya resueltos que sirven de patrón para cualquier edición futura:

- El estimador de FLOPs por ventana (`§Estimación de la intensidad operacional observada`) usa notación con sombrero — $\widehat{\text{FLOPs}}_i$ para el valor **estimado**, $\text{FLOPs}_i$ sin sombrero para el valor real no observado, $\text{FLOPs}_i^{\text{HW}}$ para el valor medido por hardware en la campaña de calibración futura (`§Validación del estimador de FLOPs por ventana`, aún no ejecutada — está en tiempo condicional/futuro a propósito, no reportar como si ya se hubiera hecho).
- Afirmaciones sobre PMU/hardware evitan generalizaciones absolutas ("los contadores de FLOPs sobrecuentan" es incorrecto como propiedad general; depende del evento y la microarquitectura — hay que matizar).
- Cuando se menciona un método más riguroso que no se implementó (p. ej. PEBIL/BBV para FLOPs exactos), se declara honestamente por qué no se usó, no se presenta como si fuera equivalente a lo hecho.

Si vas a tocar estas secciones, mantén ese nivel de precisión — no lo simplifiques de vuelta a afirmaciones más fuertes de lo que el instrumento real soporta.

## 5. Verificación técnica contra el código real

Cualquier afirmación sobre "qué mide/hace el instrumento" en la Metodología debe poder verificarse contra el código real, no inventarse por plausibilidad. Puntos de verdad concretos:
- Contadores realmente adquiridos: `telemetry/src/perf_reader.cpp` (actualmente 10: instructions, cycles, cache-references, cache-misses, stalled-cycles-backend, L2_LINES_IN_ALL, y los 4 sub-eventos de `FP_ARITH_INST_RETIRED` — ARC-97, presupuesto exacto sin holgura en pacca).
- Catálogo de kernels: `orchestrator/schemas/kernels/catalog.yaml` (9 entradas de CPU: 7 dataset + 2 calibración — no "8", error ya corregido en el texto del libro).
- Lógica de post-procesamiento / `phase_label_train`: `orchestrator/postprocess.py`. FLOPs por ventana se **miden** directamente por hardware (`flops_measured_window`), no se prorratean — el mecanismo de prorrateo se eliminó por completo en ARC-100; no reintroducirlo sin que el usuario lo pida explícitamente.
- Decisiones de diseño fuera del plan original, con su justificación y número ARC: `docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md`.

Si el texto del libro y el código no coinciden, el código manda — corrige el texto, no al revés.

## 6. Formato

`main.tex` usa `\bibitem`/`thebibliography` (no BibTeX/.bib), citación numérica vía `natbib`. Las 29 referencias del plan más las añadidas durante este trabajo están todas dentro de un único bloque `\begin{thebibliography}{99}...\end{thebibliography}` al final del documento — añadir ahí, no crear un archivo `.bib` aparte.
