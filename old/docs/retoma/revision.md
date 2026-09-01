# Revisión de pares simulada — Fase 1 (main.tex)

Generada con el skill `academic-paper-reviewer` (modo `full`, adaptado a un Trabajo de Grado de pregrado sin venue objetivo — ver nota de adaptación en la Decisión Editorial). Panel de 5 asientos ciegos entre sí: Ajuste/EIC, Metodología, Dominio, Perspectiva, Abogado del Diablo. Fecha: 2026-08-18/19.

---

## Decisión Editorial (Fase 2 — síntesis)

*Nota de adaptación (transparencia obligatoria per Iron Rule #1 del skill): esta síntesis usa la estructura sustantiva de `editorial_decision_template.md` pero omite el andamiaje de infraestructura que esta revisión ad-hoc no tiene instanciado (artefactos de proveniencia con SHA-256, refs de transporte `R<n>` como sistema de trazabilidad automatizada, IDs de sub-claim). El contenido —consenso, desacuerdo, decisión, hoja de ruta— es completo y real, construido exclusivamente a partir de los 5 reportes de Fase 1 incluidos abajo, sin fabricar nada no dicho por ellos.*

### Información del documento
- **Título**: Diseño e Implementación de un Agente en Espacio de Usuario para la Gestión Dinámica de Frecuencia (DVFS) en Sistemas Heterogéneos mediante Modelos Ligeros de Machine Learning
- **Tipo**: Trabajo de Grado (pregrado, UIS) — revisión de la Fase 1
- **Fecha de decisión**: 2026-08-18

### Panel (5 asientos, ciegos entre sí)

| Revisor | Recomendación | Confianza |
|---|---|---|
| Ajuste/EIC (comité de sustentación) | Minor Revision | 4 |
| R1 — Metodología | Minor Revision | 5 |
| R2 — Dominio | Minor Revision | 4 |
| R3 — Perspectiva | Minor Revision | 3 |
| Abogado del Diablo | N/A (solo hallazgos) | 1 CRITICAL, 5 MAJOR, 2 MINOR |

### Decisión

**Minor Revision**

Consenso unánime (4/4) de los asientos evaluadores en el nivel de decisión. El hallazgo CRITICAL del Abogado del Diablo se adjudica explícitamente abajo — no se ignora, pero tampoco eleva la decisión a Major Revision.

### Adjudicación del hallazgo CRITICAL (obligatoria, Iron Rule #4)

**C1 (Abogado del Diablo)**: el título, el objetivo general y la pregunta de investigación comprometen un agente DVFS evaluado frente a gobernadores nativos en energía/rendimiento/EDP; tras todo el trabajo reportado, cero puntos de dato válidos existen sobre esa relación (CPU invalidado por turbo, GPU no ejecutado).

**Adjudicación: sustancia validada, severidad reducida de CRITICAL a MAJOR (primera prioridad de la hoja de ruta).**

**Razonamiento**: la tensión que señala el Abogado del Diablo es real y tres asientos la tocaron independientemente por ángulos distintos —EIC (Evidence Sufficiency: `PARTLY_MEETS`), Metodología (S5, honestidad epistémica) y Perspectiva (W5, objetivos sin calificar de alcance)— sin que ninguno la considerara bloqueante. La razón por la que no es CRITICAL en el sentido de "invalida el documento" es que **el propio documento la declara explícita y repetidamente** (Resumen, nota de estado, apertura de Metodología, apertura de Resultados, Conclusiones) como un hallazgo negativo, no como algo ocultado que un lector debe descubrir. Un CRITICAL en este framework significa "invalida la reclamación central o impide la aceptación sin corrección" — pero la reclamación central de *este documento específico* (que reporta explícitamente ser Fase 1 de 4, no la tesis completa) nunca fue "demostramos el efecto DVFS", sino "construimos y verificamos el instrumento que lo medirá". El Abogado del Diablo tiene razón en que el **título general del trabajo de grado** sí promete eso, y ahí es donde el hallazgo aterriza con fuerza real: no como defecto científico de lo reportado, sino como un vacío de calificación de alcance en los puntos de entrada del documento (título, objetivo general, RQ) que no llevan la misma disciplina de "esto es Fase 1 solamente" que sí tienen el Resumen y las Conclusiones. Por eso se degrada a MAJOR y se convierte en el ítem #1 de revisiones obligatorias, no se descarta.

### Análisis de consenso

#### Puntos de acuerdo

**[CONSENSO-3]** (EIC, Metodología, Perspectiva coinciden; Dominio silencioso por estar fuera de su remit — no en desacuerdo):
La disciplina epistémica del documento —distinguir con rigor verbal y estructural lo ya ejecutado (Fase 1, pasado) de lo planeado (Fases 2-4, futuro), sin fugas de lenguaje tentativo hacia capítulos que se presentan como cerrados— es genuina y verificada independientemente por tres revisores mediante búsquedas dirigidas distintas. El propio Abogado del Diablo lo reconoce en su sección de Observaciones no-defecto.

**[CONSENSO-2, corroboración independiente]**: EIC (W2) y Abogado del Diablo (M5) llegan, por ángulos distintos —escaneabilidad para un jurado vs. sustitución de argumentación científica por bitácora de ingeniería— a la misma conclusión estructural: la narrativa de "bug → causa raíz → corrección → reverificación", repetida ~8 veces con estructura casi idéntica, necesita síntesis tabular. Que dos revisores sin comunicación entre sí lleguen al mismo punto por razones distintas es una señal de corroboración real, no de un artefacto de un solo lector.

#### Puntos de desacuerdo

**Desacuerdo 1: severidad de la brecha entre lo prometido (título/objetivo) y lo entregado (solo instrumento, Fase 1)**
- **Vista de EIC/Metodología/Perspectiva**: brecha esperable y ya divulgada honestamente; no bloquea, solo pide mejor calificación de alcance en los puntos de entrada.
- **Vista del Abogado del Diablo**: la brecha es tan grande (cero datos sobre la variable dependiente central) que el documento sustituye el compromiso científico prometido por el título con un compromiso de ingeniería distinto.
- **Tipo de desacuerdo**: severidad, no existencia — todos coinciden en que la brecha existe.
- **Resolución del editor**: ver adjudicación de C1 arriba. Se valida la sustancia, se limita el alcance de la corrección a los puntos de entrada del documento (no se exige nueva adquisición de datos).

**Desacuerdo 2: si construir un instrumento propio en vez de adoptar LIKWID/PAPI necesita justificación explícita**
- **Vista del Abogado del Diablo (m2)**: sí, varios bugs reportados son problemas que herramientas maduras ya resuelven.
- **Vista de los demás**: ningún revisor estándar lo cuestionó — implícitamente aceptan que un instrumento propio con requisitos específicos (PID+inherit, ventanas de muestreo, integración con el orquestador de campaña) es una decisión de diseño razonable no cuestionada.
- **Resolución del editor**: hallazgo MINOR válido pero no consensuado por el resto del panel — se incluye como sugerencia, no como obligación.

### Razonamiento de la decisión

Los cinco asientos, evaluando ángulos sin superposición por diseño (rigor experimental, cobertura de literatura, ajuste/coherencia global, aplicabilidad práctica, y el ataque adversarial), coinciden en que el aporte de Fase 1 —instrumento de telemetría con verificación cruzada exhaustiva y protocolo Roofline reproducible— es sólido y honesto en su registro, con debilidades todas reparables mediante texto adicional o reorganización, ninguna mediante nueva adquisición de datos. El único hallazgo con potencial de bloquear la decisión (C1, DA) se adjudicó explícitamente y se resolvió a MAJOR por la razón dada arriba: el documento ya divulga lo que el Abogado del Diablo señala, solo falta que esa divulgación llegue a los puntos de entrada (título/objetivo/RQ) con la misma disciplina que ya tiene en Resumen y Conclusiones. Ningún hallazgo restante justifica Major Revision (ninguno exige repetir experimentos o rediseñar el protocolo); de ahí Minor Revision.

### Revisiones obligatorias (Must Fix)

**REV-1 — Calificar el alcance en título/objetivo general/pregunta de investigación** *(adjudicación de C1-DA, EIC W2/S4, Perspectiva W5)*
- **Problema**: título, objetivo general y RQ comprometen el agente DVFS completo sin calificación de fase; el lector solo descubre el alcance real en Resumen/Conclusiones.
- **Requisito**: añadir una frase de calificación de alcance en el objetivo general y/o la RQ (p. ej. "...validado en la plataforma de referencia descrita, como primera fase de un proyecto de cuatro fases") y una nota equivalente cerca del título en portada/Resumen si el formato de la Escuela lo permite.
- **Criterio de aceptación**: un lector que solo lea título+objetivo general+RQ debe poder inferir, sin llegar a Conclusiones, que el documento reporta un instrumento y protocolo, no el agente completo evaluado.

**REV-2 — Tabla-resumen de defectos de instrumento encontrados/corregidos** *(EIC W2 + DA M5, corroborado)*
- **Problema**: ~8 episodios narrados en prosa densa sin vista tabular consolidada.
- **Requisito**: tabla (cuerpo o apéndice) con columnas Defecto / Causa raíz / Evidencia de confirmación / Corrección, remitiendo al párrafo narrativo para el detalle.
- **Criterio de aceptación**: un jurado puede ubicar el estado de cada defecto en <30 segundos sin leer la prosa completa.

**REV-3 — Análisis individual de Antici2024/MCBound en el Estado del Arte** *(Dominio W1)*
- **Problema**: el antecedente más cercano por título a la tercera línea temática (caracterización/clasificación online de cargas) solo aparece en clústeres de cita, nunca analizado individualmente.
- **Requisito**: párrafo que contraste explícitamente alcance de MCBound (CPU-only vs. CPU-GPU, online vs. offline, mecanismo de actuación) contra el aporte reclamado.
- **Criterio de aceptación**: el argumento de brecha en la línea de cierre del Estado del Arte se sostiene también frente a este antecedente específico.

**REV-4 — Cuantificar o matizar la causa del error de 7.48% en validación de FLOPs** *(Metodología W1)*
- **Problema**: la explicación causal (fase de verificación no cronometrada) se presenta como establecida sin medición que la aísle cuantitativamente.
- **Requisito**: o bien una medición adicional que aísle esos FLOPs, o declarar explícitamente que la explicación es plausible pero no verificada cuantitativamente.
- **Criterio de aceptación**: la afirmación causal en el texto tiene el mismo respaldo cuantitativo que el resto de cifras de precisión del documento, o se recalifica su certeza.

**REV-5 — Dimensionar el costo de no-portabilidad** *(Perspectiva W1)*
- **Problema**: se declara que el instrumento está gateado a una microarquitectura, pero no qué fracción del protocolo es reutilizable vs. qué exige repetirse.
- **Requisito**: párrafo en Limitaciones/Trabajo Futuro que distinga la parte reutilizable (lógica del protocolo) de la parte no portable (encodings PMU, ridge point).
- **Criterio de aceptación**: un lector puede estimar, aunque sea cualitativamente, el esfuerzo de portar el instrumento a otro nodo.

**REV-6 — Modelo de privilegios mínimos para la Fase 3** *(Perspectiva W2)*
- **Problema**: el agente de Fase 3 actuará autónomamente con privilegios elevados sin que se anticipe un modelo de contención.
- **Requisito**: mención explícita en Trabajo Futuro de que la Fase 3 deberá abordar privilegio mínimo / límites de tasa / mecanismo de reversión ante comportamiento anómalo.
- **Criterio de aceptación**: Trabajo Futuro reconoce este riesgo, no solo el cierre técnico de Fase 1.

**REV-7 — Estrategia de estabilización cerca del *ridge point* para inferencia en línea** *(Perspectiva W3, corroborado por DA M1 desde el ángulo de sesgo del catálogo)*
- **Problema**: excluir ventanas ambiguas del catálogo de entrenamiento es válido para curar datos, pero no es una opción para un agente de control en producción.
- **Requisito**: reconocer en el diseño previsto de Fase 3 que se necesitará histéresis/banda muerta/probabilidad de clase, no una etiqueta binaria heredada del criterio de curación.
- **Criterio de aceptación**: la sección "Fase 3" menciona explícitamente esta estrategia como pendiente de diseño.

**REV-8 — Explicación alternativa no descartada para la corrección de `-lgc`** *(DA M4)*
- **Problema**: el salto de fallo a éxito en el bloqueo de reloj GPU coincide con un cambio de versión de driver NVIDIA no aislado como variable.
- **Requisito**: reconocer explícitamente el cambio de versión de driver como explicación alternativa no descartada (el texto ya evita inferir causalidad — falta nombrar la alternativa).
- **Criterio de aceptación**: el párrafo correspondiente menciona el cambio de driver como posible factor de confusión, no solo la ausencia de inferencia causal.

**REV-9 — Reencuadrar la novedad del "aporte metodológico central"** *(DA M3)*
- **Problema**: el principio de verificación empírica exhaustiva se presenta como aporte metodológico propio pese a citarse Georges et al. (2007) como su fuente para otro fin en el mismo documento.
- **Requisito**: reformular la frase de Conclusiones para presentar el aporte como *aplicación disciplinada y verificada* de un principio establecido a una plataforma nueva con hallazgos específicos, no como novedad metodológica per se.
- **Criterio de aceptación**: no hay tensión entre cómo se cita Georges et al. en Metodología y cómo se reclama el aporte en Conclusiones.

**REV-10 — Matizar la precisión aparente del punto de inflexión CPU** *(DA M2)*
- **Problema**: cifras a dos decimales (0.29%, 7.48%) conviven sin discusión con un rango de ridge point de 7.0–9.3 FLOP/byte (~33%) que decide clasificaciones.
- **Requisito**: una frase que reconozca explícitamente que la frontera de clasificación tiene una incertidumbre mucho mayor que la precisión de validación de FLOPs, para que no se lean como equivalentes.
- **Criterio de aceptación**: el lector no infiere que la clasificación compute/memory-bound tiene la misma certeza que la medición de FLOPs.

### Revisiones sugeridas (Should Fix)

- **S1**: completar o remitir formalmente la sección Marco Legal (EIC W1).
- **S2**: enlazar Trabajo Futuro con los Objetivos Específicos 2-4 pendientes, no solo con el cierre técnico de Fase 1 (EIC W3).
- **S3**: declarar el criterio de tolerancia esperado *antes* de la validación cruzada con Advisor (Metodología W2).
- **S4**: reportar el CV% de estabilidad observado para la carga de referencia (Metodología W3).
- **S5**: reportar los parámetros numéricos del criterio de calentamiento (umbral CV%, factor de margen, fracción de meseta) (Metodología W4).
- **S6**: desglosar el 4.6% no-`ok` de la campaña insignia por categoría de calidad (Metodología W5).
- **S7**: acotar o sustentar la afirmación de vacío regional/nacional (Dominio W2).
- **S8**: nota al pie explicando qué son los identificadores ARC-XXX la primera vez que aparecen (Perspectiva W4).
- **S9**: justificar brevemente por qué se construyó instrumento propio en vez de adoptar LIKWID/PAPI (DA m2).
- **S10**: aclarar la distribución de autoría entre los dos autores (DA m1) — verificar primero si la Escuela lo exige.

### Hoja de ruta (orden de trazabilidad, no de prioridad de trabajo)

- [ ] REV-1 (must_fix) — calificar alcance en título/objetivo/RQ
- [ ] REV-2 (must_fix) — tabla de defectos de instrumento
- [ ] REV-3 (must_fix) — análisis individual de MCBound
- [ ] REV-4 (must_fix) — cuantificar o matizar causa del 7.48%
- [ ] REV-5 (must_fix) — dimensionar costo de no-portabilidad
- [ ] REV-6 (must_fix) — privilegios mínimos Fase 3
- [ ] REV-7 (must_fix) — estabilización cerca del ridge point
- [ ] REV-8 (must_fix) — alternativa de versión de driver
- [ ] REV-9 (must_fix) — reencuadrar novedad metodológica
- [ ] REV-10 (must_fix) — matizar precisión del ridge point CPU
- [ ] S1..S10 (should_fix/consider) — ver lista arriba

*El autor decide qué de esto se aborda, en qué orden, y qué se declara fuera de alcance (`will_address`/`wont_address`/`not_on_point`) — esta hoja de ruta no impone orden de trabajo, solo trazabilidad hacia el hallazgo que la originó.*

### Cierre

Se invita a una versión revisada del documento que aborde los puntos señalados por el panel, en particular la calificación de alcance en los puntos de entrada (REV-1) y la síntesis tabular de la narrativa de depuración (REV-2), que son los dos únicos hallazgos corroborados por más de un asiento independientemente. El resto de revisiones obligatorias son puntuales y no requieren nueva adquisición de datos ni re-ejecución de campañas — coherente con que ninguno de los cinco asientos cuestionó la validez de lo efectivamente reportado en la Fase 1, solo su presentación y su relación con lo que el título del trabajo completo promete.

---

## Apéndice: reportes completos de los 5 revisores

### Reporte 1 — Ajuste/EIC (comité de sustentación)

# Peer Review Report

## Manuscript Information
- **Title**: Diseño e Implementación de un Agente en Espacio de Usuario para la Gestión Dinámica de Frecuencia (DVFS) en Sistemas Heterogéneos mediante Modelos Ligeros de Machine Learning
- **Manuscript ID**: N/A (trabajo de grado, UIS, Ingeniería de Sistemas — sin ID de manuscrito)
- **Review Date**: 2026-08-18
- **Review Round**: Ronda 1

---

## Reviewer Information

### Reviewer Role
Journal-Fit Reviewer (rol adaptado a comité de sustentación de pregrado, per protocolo Fase 1 del panel)

### Reviewer Identity
Director de comité evaluador de trabajos de grado en ingeniería de sistemas/computación, con experiencia en proyectos de arquitectura/HPC que combinan instrumento propio + evaluación empírica.

### Review Focus
Evalúo si el alcance declarado de Fase 1 constituye un aporte metodológico independiente correcto y suficiente para un trabajo de grado de pregrado; si la escritura distingue con claridad lo implementado/verificado de lo planeado; y si el documento sería defendible ante un jurado hoy mismo, sin los resultados de la campaña final. No entro en el detalle técnico de la instrumentación PMU/RAPL/NVML en sí (eso corresponde a otro asiento del panel); mi lente es de conjunto: coherencia título→objetivos→resultados→conclusiones, honestidad de registro, y madurez del aporte como pieza autocontenida.

---

## Overall Assessment

### Recommendation
- [x] **Minor Revision** — Listo con ajustes menores para sustentación (no requiere nueva revisión completa del panel)

### Confidence Score
4 — Alta confianza. El documento es legible de principio a fin, el alcance está declarado explícitamente en múltiples puntos, y mi rol (coherencia global, honestidad de registro, encaje disciplinar) no depende de validar cada detalle de la instrumentación de bajo nivel.

### Summary Assessment
El trabajo reporta la Fase 1 (de cuatro) de un proyecto de agente DVFS en espacio de usuario: construcción y validación empírica de un instrumento de telemetría CPU/GPU y un protocolo de etiquetado Roofline con calibración propia. Metodológicamente es sólido — cada mecanismo del instrumento (FLOPs por hardware, bytes de *uncore*, actuación de frecuencia CPU/GPU, techos Roofline) se somete a verificación cruzada independiente, y varios defectos reales del propio instrumento se documentan con causa raíz y corrección. La escritura distingue con disciplina lo verificado de lo pendiente: la campaña CPU de 126/126 corridas se declara explícitamente insuficiente como evidencia DVFS por un hallazgo de turbo no controlado, y la campaña GPU multi-frecuencia se declara no ejecutada — nunca se presenta un resultado no obtenido como obtenido. Revisé exhaustivamente el Marco de Referencia y la sección de Fase 1 buscando lenguaje futuro/condicional residual tras el pedido explícito de reescritura en lenguaje definitivo, y no encontré ninguno; las Fases 2-4, correctamente, sí usan futuro. Las debilidades detectadas son de forma y de cierre administrativo (sección Marco Legal vacía, narrativa de depuración poco tabulada, hoja de ruta de "Trabajo Futuro" desconectada de los objetivos específicos 2-4), no de validez científica. Es defendible hoy con ajustes menores.

---

## Strengths

### S1: Ninguna afirmación de resultado no obtenido — verificado exhaustivamente tras el pedido de reescritura
El autor pidió específicamente eliminar todo lenguaje futuro/condicional ("se espera", "se planea", "próximamente") del capítulo de Metodología en su parte de Fase 1 y del Marco de Referencia. Revisé el documento completo con búsqueda dirigida sobre esas secciones y no encontré ninguna instancia residual; el hallazgo de turbo en CPU se declara con la fórmula más fuerte posible de honestidad negativa.
**Evidence Anchor**: `text: "la campaña no se presenta como un barrido DVFS válido ni como evidencia del efecto de la frecuencia sobre tiempo, energía o EDP" (§3.2, Validación de la actuación DVFS en CPU)`

### S2: Aporte metodológico genuino e independiente de las Fases 2-4
El procedimiento reproducible de etiquetado *compute-bound*/*memory-bound* mediante calibración empírica del ridge point, con trazabilidad de calidad por ventana, es un aporte que se sostiene por sí mismo aunque el clasificador y el agente nunca se completen.
**Evidence Anchor**: `text: "cuyo aporte metodológico central es un procedimiento reproducible para etiquetar ventanas de ejecución como compute-bound o memory-bound mediante una calibración empírica del modelo Roofline sobre la plataforma de medición, con trazabilidad explícita de la calidad de cada observación" (Resumen)`

### S3: Validación cruzada independiente que expuso un defecto real de ~8x
La verificación de los techos Roofline contra Intel Advisor (mecanismo de medición genuinamente distinto) encontró que el microbenchmark propio de densidad aritmética compilaba a SSE2 en vez de AVX-512, produciendo un techo de cómputo subestimado casi 8 veces.
**Evidence Anchor**: `text: "el techo de cómputo, en cambio, no validó: el microbenchmark propio de densidad aritmética medía apenas 71.8–71.9 GFLOP/s, una brecha de casi 8× frente a la referencia de Advisor" (§3.6, Validación cruzada de los techos Roofline)`

### S4: Declaración de alcance repetida y consistente en los puntos de entrada del documento
El alcance ("solo Fase 1 está ejecutada") se declara en el Resumen, en la nota de estado del documento, en la introducción del capítulo de Metodología y en la apertura del capítulo de Resultados.
**Evidence Anchor**: `text: "Este capítulo describe en detalle el diseño de la primera fase, por ser la que se encuentra ejecutada al momento de redacción del presente documento, y presenta el diseño previsto para las fases restantes" (§2.1, Enfoque metodológico)`

---

## Weaknesses

### W1: Sección Marco Legal vacía, sin resolución declarada para el momento de sustentación
**Problem**: La sección 1.2 (Marco Legal) contiene únicamente un comentario LaTeX con la lista de elementos candidatos y la nota de que los autores no la redactarán "para no atribuir normativa que no haya sido verificada".
**Evidence Anchor**: `absence: §1.2 Marco Legal — expected contenido sustantivo o remisión explícita a un anexo/exención institucional; checked el cuerpo de la sección (líneas 518-531, solo comentario LaTeX)`
**Why it matters**: Un jurado de sustentación puede señalar esto como un vacío formal el mismo día de la defensa si no se resuelve o al menos se justifica explícitamente ante el comité antes de esa fecha.
**Suggestion**: Redactar al menos los elementos de bajo riesgo (licencia de publicación del código/dataset, condiciones de uso de la infraestructura institucional) que ya son verificables por los propios autores.
**Severity**: Minor | **Confidence**: 4

### W2: Narrativa de depuración densa y no tabulada en Resultados y Discusión
**Problem**: Varios hallazgos de instrumento se narran en párrafos únicos de 15-20 líneas que mezclan síntoma, hipótesis, prueba de aislamiento y corrección. Ausencia de una tabla-resumen de defectos encontrados/corregidos.
**Evidence Anchor**: `absence: §3 (Resultados) y §4 (Discusión) — expected tabla-resumen de defectos de instrumento; checked estructura de tablas del documento`
**Why it matters**: No afecta la validez de los hallazgos, pero reduce la capacidad del jurado de auditar rápidamente el volumen de trabajo de depuración real.
**Suggestion**: Añadir una tabla-resumen (aunque sea en apéndice) que liste cada defecto con causa raíz y corrección, remitiendo al párrafo narrativo correspondiente.
**Severity**: Minor | **Confidence**: 3

### W3: "Trabajo Futuro" no se conecta explícitamente con los Objetivos Específicos 2-4
**Problem**: La sección de Trabajo Futuro lista tres tareas de corto plazo, todas orientadas a cerrar las campañas pendientes de Fase 1, sin mencionar explícitamente que los Objetivos Específicos 2-4 siguen íntegramente pendientes.
**Evidence Anchor**: `text: "En el corto plazo, tres tareas siguen directamente habilitadas..." (§5.3, Trabajo Futuro)`
**Why it matters**: Un jurado que revise objetivos contra conclusiones esperará ver que el trabajo futuro cubre explícitamente el camino hacia los objetivos 2-4.
**Suggestion**: Añadir una o dos frases en Trabajo Futuro que enlacen explícitamente las tareas de corto plazo con los objetivos específicos aún pendientes.
**Severity**: Minor | **Confidence**: 4

---

## Detailed Comments

### Título & Resumen
El título describe el proyecto completo (agente + DVFS + ML), no solo lo entregado en este documento. El propio Resumen lo aclara en su último párrafo. Es una resolución aceptable dado que el título corresponde al trabajo de grado completo, no a este entregable parcial.

### Introducción
La pregunta de investigación y los objetivos están bien formulados. La Introducción sí contiene una instancia de lenguaje prospectivo, pero se refiere al resultado esperado del proyecto completo, no a un resultado de Fase 1 presentado como ya obtenido.

### Metodología (foco: Fase 1)
Es la sección más fuerte del documento. Cada decisión metodológica está justificada con su razón física o estadística. Confirmé que todo el texto de Fase 1 está en pasado/presente definitivo — sin excepciones encontradas.

### Resultados / Discusión
El hallazgo de turbo en CPU (§3.2) es el ejemplo más importante de disciplina de reporte: una campaña de 126/126 corridas aceptadas se declara explícitamente inválida como evidencia DVFS.

### Conclusiones
Responden honestamente al estado de cada objetivo específico. Una frase que reconecte explícitamente "esta fase no responde aún la pregunta de investigación general" cerraría el círculo de forma más nítida.

### Limitaciones
Sección fuerte: cinco limitaciones concretas, cada una con su consecuencia metodológica explícita.

---

## Questions for Authors

1. La sección Marco Legal permanece vacía. ¿Existe ya una conversación con la Escuela sobre si esta sección puede quedar remitida a un anexo?
2. ¿Cuál es el criterio de "listo" que el comité debería usar para aceptar la Fase 1 como cerrada: la validación del instrumento (ya lograda) o la obtención de estados DVFS físicamente distintos (aún pendiente)?
3. ¿Planean los autores incluir, en la versión final de las Conclusiones, una frase explícita que aclare que la pregunta de investigación general permanece abierta hasta el cierre de la Fase 4?

## Minor Issues

### Layout
- El capítulo de Agradecimientos permanece vacío (`% [PENDIENTE: redactar agradecimientos]`).
- Varios párrafos en Resultados/Discusión superan las 15-20 líneas sin subdivisión (ver W2).

---

## Criterion-Bound Judgements

**Nota sobre encaje de venue**: no existe una revista/venue objetivo confirmada — es un trabajo de grado de pregrado. Se trata la dimensión de "ajuste editorial" como `criteria_binding_unavailable`.

| Dimension | Judgement | Decision bearing? |
|---|---|---|
| Originality | MEETS | Yes |
| Methodological Rigor | EXCEEDS | Yes |
| Evidence Sufficiency | PARTLY_MEETS | Yes |
| Argument Coherence | MEETS | No |
| Writing Quality | PARTLY_MEETS | No |
| Literature Integration | MEETS | Yes |
| Significance & Impact | PARTLY_MEETS | Yes |

La recomendación de Minor Revision se sostiene porque ningún criterio decision-bearing está en DOES_NOT_MEET: Evidence Sufficiency y Significance & Impact están en PARTLY_MEETS por diseño explícito y honesto del propio documento (fase declarada como parcial), reparables solo completando Fases 2-4 — no son defectos de este entregable.

---

### Reporte 2 — Metodología (Peer Reviewer 1)

# Peer Review Report

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Metodología)

### Reviewer Identity
Ingeniero de sistemas con especialización en medición de rendimiento (PMU, `perf_event_open`, modelo Roofline), con experiencia diseñando protocolos de benchmarking reproducibles en HPC.

### Review Focus
Rigor del diseño experimental de la Fase 1 (matriz, control de sesgos, calibración), validez de las verificaciones empíricas citadas en el texto, y trazabilidad de cada afirmación cuantitativa hacia una corrida real citada en el propio documento.

---

## Overall Assessment

### Recommendation
- [x] **Minor Revision**

### Confidence Score
5 — dominio central de mi experticia.

### Summary Assessment
El documento reporta el instrumento y protocolo experimental de la Fase 1 con calidad metodológica notablemente alta para un trabajo de pregrado: el instrumento no da por válida ninguna capacidad de hardware sin verificación directa, y reporta con inusual honestidad hallazgos que invalidan parcialmente su propia campaña insignia (turbo global no controlado). La validación cruzada de FLOPs y de los techos Roofline contra Advisor son controles de validez ejemplares. Las principales debilidades son de reporte cuantitativo más que de diseño: una explicación causal no cuantificada para el error de 7.48%, un criterio de tolerancia no declarado para aceptar la discrepancia de ancho de banda frente a Advisor, y parámetros numéricos del criterio de calentamiento no reportados. Ninguna socava la validez central de lo reportado; recomiendo revisión menor.

---

## Strengths

### S1: Calibración empírica de los techos Roofline en lugar de valores nominales de fábrica
**Evidence Anchor**: `text: §2.2.1 (líneas 388-398)`

### S2: Medición directa de FLOPs con encoding validado y confirmado a escala de campaña (~1.29M ventanas, 100% con FLOPs medidos)
**Evidence Anchor**: `text: §2.4.6 (línea 817)`

### S3: Validación cruzada de la calibración Roofline con Intel Advisor — expuso un defecto real de 8×
**Evidence Anchor**: `text: §3.6 (líneas 939-943)`

### S4: Taxonomía de calidad no destructiva — ninguna observación se descarta silenciosamente
**Evidence Anchor**: `table: Tabla 2.4 — línea 747`

### S5: Honestidad epistémica sobre la invalidez de la campaña insignia (turbo)
**Evidence Anchor**: `text: §3.2 (línea 882)`

### S6: Reproducibilidad respaldada por artefactos versionados, no solo declarada
**Evidence Anchor**: `text: §2.4.7 (líneas 823-825)`

### S7: Detección de interferencia entre mecanismos del instrumento (uncore vs. presupuesto de PMU), diagnosticada en dos capas independientes
**Evidence Anchor**: `text: §3.4.1 (líneas 928-934)`

---

## Weaknesses

### W1: Explicación del error de 7.48% en la validación de FLOPs no cuantificada independientemente
**Evidence Anchor**: `text: §2.4.6 (línea 815)` | **Severity**: Major | **Confidence**: 5

### W2: Criterio de tolerancia no declarado para aceptar la discrepancia de ancho de banda frente a Advisor
**Evidence Anchor**: `text: §3.6 (línea 941)` | **Severity**: Minor | **Confidence**: 4

### W3: Control de estabilidad por CV descrito pero nunca reportado con valores concretos
**Evidence Anchor**: `absence: Capítulo 3 — expected valor de CV% de estabilidad` | **Severity**: Minor | **Confidence**: 4

### W4: Parámetros numéricos del criterio de calentamiento no reportados en el cuerpo del texto
**Evidence Anchor**: `text: §2.4.5 (línea 756)` | **Severity**: Minor | **Confidence**: 4

### W5: Desglose de la taxonomía de calidad no reportado para la campaña insignia (solo agregado 95.4% ok)
**Evidence Anchor**: `text: §3.2 (línea 878)` | **Severity**: Minor | **Confidence**: 4

---

## Detailed Comments

### Diseño de la matriz experimental y control de sesgos
Cinco medidas bien fundamentadas conceptualmente. El punto débil no es el diseño sino su ejecución en la primera instancia (turbo global), reconocido sin ambigüedad como resultado, no como defecto de diseño ocultado.

### Muestra y catálogo de cargas
Criterio de inclusión apropiado y aplicado consistentemente. Ameritaría aclaración si la ampliación del catálogo con dos cargas adicionales se decidió antes o después de observar las etiquetas resultantes del catálogo original (ver Questions for Authors #1).

### Medición de FLOPs y su validación empírica
Diseño de verificación metodológicamente sólido; único punto abierto es la falta de cuantificación independiente de la causa del error de 7.48%.

### Validación cruzada de Roofline con Advisor
La corrección selectiva de la bandera de ancho vectorial (aplicada solo donde se encontró la regresión, verificada caso por caso) evita la falacia de sobregeneralización.

### Reproducibilidad
Mecanismos concretos y verificables. No se reporta si el repositorio incluye ya los datasets crudos (ver Questions for Authors #4).

---

## Questions for Authors

1. ¿La ampliación del catálogo con las dos cargas adicionales se decidió antes o después de observar las etiquetas resultantes del catálogo original de siete cargas?
2. Para el error de 7.48% en FLOPs: ¿se dispone de una medición que aísle específicamente los FLOPs de la fase de verificación numérica del kernel?
3. ¿Qué criterio de tolerancia se tenía en mente, antes de ejecutar la validación cruzada con Advisor, para distinguir una discrepancia "esperable" de una atribuible a un defecto real?
4. ¿El repositorio público incluye ya los datos crudos y procesados, no solo el código?

---

## Criterion-Bound Judgements

| Dimensión | Juicio | ¿Determinante? |
|---|---|---|
| Rigor del diseño experimental | MEETS | Sí — sostiene el Minor Revision |
| Validez de verificaciones empíricas citadas | PARTLY_MEETS | Sí — motiva revisiones puntuales |
| Reproducibilidad | MEETS | No |
| Transparencia en reporte de calidad de datos | PARTLY_MEETS | Sí |
| Distinción Fase 1 (cerrada) vs. Fases 2-4 (prospectivas) | EXCEEDS | No — fortaleza |

Ninguno de los hallazgos (W1-W5) es Critical; todos son reparables mediante adiciones de texto o reporte de datos que, según el propio documento, ya existen en el repositorio.

---

### Reporte 3 — Dominio (Peer Reviewer 2)

# Peer Review Report

## Reviewer Information

### Reviewer Role
Peer Reviewer 2 (Domain)

### Reviewer Identity
Investigador senior en sistemas de gestión de energía (DVFS, governors, arquitecturas heterogéneas CPU-GPU), familiarizado con el estado del arte de clasificadores ligeros para control en tiempo real.

### Review Focus
Cobertura y actualidad del marco conceptual/estado del arte, correcta ubicación del aporte frente a trabajo previo, y solidez de la fundamentación de EDP y Roofline.

---

## Overall Assessment

### Recommendation
**Minor Revision**

### Confidence Score
4

### Summary Assessment
El documento presenta un marco conceptual sólido y un Estado del Arte organizado en cuatro líneas temáticas claramente diferenciadas, con síntesis crítica genuina en lugar de mera enumeración de referencias. El modelo Roofline y el EDP están correctamente fundamentados. El capítulo de Discusión dialoga de forma sustantiva con la literatura revisada. El posicionamiento final del aporte evita el sobreclaim. La debilidad más notable es que MCBound (Antici2024) — el trabajo más cercano por título al problema de caracterización online que aborda la tercera línea del Estado del Arte — nunca se analiza individualmente. La recomendación de revisión menor refleja que el resto del andamiaje conceptual es riguroso y que la corrección requerida es acotada y específica.

---

## Strengths

### S1: Estado del Arte organizado en líneas temáticas con síntesis crítica, no enumeración
**Evidence Anchor**: `text: §Estado del Arte`

### S2: Aplicación profunda del modelo Roofline, no solo su cita nominal
**Evidence Anchor**: `text: §2.2`

### S3: Reconocimiento explícito de los límites de aplicabilidad de Roofline (GPU, precisión)
**Evidence Anchor**: `text: §2.2 "un único I_ridge genérico no es representativo de ninguna de las dos"`

### S4: Diálogo sustantivo con la literatura en el capítulo de Discusión
**Evidence Anchor**: `text: §Discusión`

### S5: Posicionamiento del aporte sin sobreclaim
**Evidence Anchor**: `text: §Estado del Arte "el aporte de este trabajo no radica en afirmar que la clasificación de fases o el control online sean completamente inéditos"`

---

## Weaknesses

### W1: MCBound (Antici2024) — el antecedente más cercano por título — nunca se analiza individualmente
**Problem**: Aparece solo en clústeres de cita en la Introducción, ausente del párrafo de la tercera línea del Estado del Arte donde sí se discuten individualmente Shekofteh2019 y Littman2025.
**Evidence Anchor**: `absence: §Estado del Arte (líneas 533-549) — expected análisis individual de Antici2024/MCBound`
**Severity**: Major | **Confidence**: 4

### W2: La afirmación de brecha regional/nacional no cuenta con evidencia de una búsqueda sistemática
**Evidence Anchor**: `absence: línea 291`
**Severity**: Minor | **Confidence**: 3

---

## Detailed Comments

### Literature Review / Theoretical Framework
Cobertura adecuada; única ausencia individual notable es Antici2024/MCBound.

### Academic Argument Quality
Terminología precisa y consistente; lógica argumentativa sólida en el núcleo técnico.

### Contribution to the Field
Aporte incremental bien delimitado, posicionamiento claro, riesgo de sobreclaim bajo.

---

## Questions for Authors

1. ¿Se consideró explícitamente el alcance de Antici2024 (MCBound) frente al de este trabajo?
2. ¿Se realizó alguna búsqueda sistemática que sustente la afirmación de ausencia "a nivel nacional y regional"?

---

## Criterion-Bound Judgements

| Dimensión | Juicio | ¿Determinante? |
|---|---|---|
| Literature Integration | PARTLY_MEETS | Sí — motiva Minor Revision |
| Argument Coherence (dominio) | MEETS | No |
| Originality / Positioning | MEETS | Sí |
| Significance & Impact | NOT_ASSESSED | No |
| Methodological Rigor | NOT_ASSESSED | No |

La recomendación de Minor Revision se explica por W1 (Major, reparable con una adición acotada al Estado del Arte); W2 es de impacto menor.

---

### Reporte 4 — Perspectiva (Peer Reviewer 3)

# Peer Review Report

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Perspective)

### Reviewer Identity
Ingeniero de software con experiencia en sistemas de producción/observabilidad, ajeno al mundo académico de HPC.

### Review Focus
Trazabilidad y honestidad epistémica del texto, aplicabilidad práctica más allá del nodo específico usado, y supuestos no cuestionados.

---

## Overall Assessment

### Recommendation
**Minor Revision**

### Confidence Score
3

### Summary Assessment
El documento reporta la Fase 1 de un proyecto de cuatro fases orientado a un agente de DVFS guiado por ML. Su aporte central es un instrumento de medición rigurosamente auto-verificado, con una narrativa de depuración que documenta explícitamente varios defectos reales. El rasgo más notable, desde mi perspectiva, es una disciplina epistémica poco común: distingue con cuidado "lo escrito en sysfs" de "lo efectivamente medido en el reloj", y separa consistentemente lo ya ejecutado de lo planeado. No encontré fugas del tiempo futuro/condicional hacia los capítulos de resultados o conclusiones. Las debilidades identificadas no son de rigor experimental sino de aplicabilidad más allá del nodo único validado, de una voz de interesado (operadores de clúster/seguridad) ausente, y de trazabilidad hacia referencias internas no explicadas. Recomiendo revisión menor.

---

## Strengths

### S1: Narrativa de depuración estilo "postmortem" con verificación directa contra hardware real
**Evidence Anchor**: `text: "aceptar una escritura en los límites por núcleo demuestra que el kernel almacenó la solicitud, no que el hardware la respetó bajo carga"`

### S2: Conservación explícita de observaciones degradadas en vez de descarte silencioso
**Evidence Anchor**: `text: "ninguna observación se descarta silenciosamente"`

### S3: Separación consistente y verificable entre lo ejecutado y lo planeado
**Evidence Anchor**: `text: "Las Fases 2–4 ... están fuera del alcance de estos resultados y se reportarán al cerrarse el desarrollo completo del proyecto"`

### S4: Gateo explícito de los mecanismos específicos de plataforma
**Evidence Anchor**: `text: "ese mismo gateo es lo que impide, por diseño, que el instrumento aplique sin verificación una codificación cruda a un nodo distinto de este"`

---

## Weaknesses

### W1: El costo operativo de la no-portabilidad se declara pero no se dimensiona
**Evidence Anchor**: `text: sec. Limitaciones` | **Severity**: Major | **Confidence**: 4

### W2: Ausencia de la voz del operador/administrador de clúster como interesado (privilegios elevados sobre infraestructura compartida)
**Evidence Anchor**: `text: sec. Aspectos éticos` | **Severity**: Major | **Confidence**: 4

### W3: La estrategia de manejo de ambigüedad cerca del *ridge point* no se traslada a la inferencia en línea
**Evidence Anchor**: `text: "se retiró del catálogo"` | **Severity**: Major | **Confidence**: 3

### W4: Referencias a identificadores internos de *issue tracker* (ARC-XXX) sin explicación para el lector externo
**Evidence Anchor**: `text: "una auditoría posterior (ARC-110) encontró que..."` | **Severity**: Minor | **Confidence**: 4

### W5: Los Objetivos no llevan la misma calificación de alcance que las Limitaciones
**Evidence Anchor**: `text: sec. Objetivos` | **Severity**: Minor | **Confidence**: 3

---

## Detailed Comments

### Resultados
Reporta explícitamente que la campaña de 126/126 combinaciones "no se presenta como un barrido DVFS válido" pese a estar completa — equivalente a "todos los tests pasaron pero el despliegue no demuestra lo que creíamos".

### Discusión
Generaliza correctamente sus hallazgos como principio metodológico, pero solo en el marco de validez experimental, no de operación futura del agente en producción (ver W2, W3).

---

## Questions for Authors

1. ¿Qué fracción del protocolo esperan que sea directamente reutilizable en un segundo nodo con otra microarquitectura?
2. ¿Qué mecanismo previsto en la Fase 3 manejará una ventana de telemetría en línea que caiga en la zona ambigua del ridge point?
3. Para la Fase 3, ¿qué modelo de privilegio mínimo o mecanismo de contención se contempla, más allá de lo cubierto en Aspectos Éticos para el contexto experimental?

---

## Criterion-Bound Judgements

| Dimensión | Juicio | ¿Determinante? |
|---|---|---|
| Argument Coherence | MEETS | No |
| Writing Quality | MEETS | No |
| Significance & Impact | PARTLY_MEETS | Sí — W1/W2 son Major |

Ninguno de mis hallazgos cuestiona la validez de lo reportado en Fase 1; son adiciones de discusión que no requieren nuevos experimentos.

---

### Reporte 5 — Abogado del Diablo

## Devil's Advocate Review

### Reconocimiento breve de fortalezas genuinas
El documento es inusualmente transparente sobre sus propios resultados negativos —admite explícitamente que la campaña CPU insignia no demuestra seis estados DVFS efectivos por interferencia de turbo, y que la campaña GPU multi-frecuencia no se ha ejecutado.

### Strongest Counter-Argument

El título, el resumen, la pregunta de investigación y el objetivo general del documento comprometen la entrega de un **agente** que ajusta DVFS para reducir consumo energético sin degradar rendimiento, evaluado frente a gobernadores nativos. Tras la totalidad del trabajo reportado, no existe un solo punto de dato válido que relacione frecuencia con energía, tiempo o EDP. Lo que sí existe es un extenso diario de ingeniería —al menos ocho ciclos documentados de "bug encontrado → causa raíz → corrección → reverificación"— sobre un instrumento de telemetría construido desde cero en C++/Python, cuando herramientas maduras ya citadas en la bibliografía (LIKWID, PAPI) probablemente habrían evitado varias de esas fallas. Una explicación rival, más parsimoniosa que "aporte metodológico deliberado", es que la decisión de construir instrumentación propia no validada consumió el tiempo del proyecto de pregrado y desplazó las Fases 2–4 —el contenido que el título realmente promete—, y que el capítulo de Discusión reencuadra retrospectivamente ese consumo de tiempo como una contribución metodológica en sí misma. Esa contribución declarada no es nueva para el campo: es, en esencia, el mismo principio que Georges et al. (2007) —citado aprobatoriamente por los propios autores— ya formalizó hace casi veinte años.

### Issue List

#### CRITICAL

| # | Dimension | Issue Description | Evidence Anchor | Confidence |
|---|-----------|-------------------|-----------------|------------|
| C1 | Core Thesis Challenge / Data-Conclusion Mismatch | El título, RQ y objetivo general comprometen evidencia sobre el efecto de DVFS en energía/rendimiento; cero puntos de dato válidos abordan esa relación. | text: "esos datos no sustentan todavía comparaciones DVFS de tiempo, energía o EDP" | 4 |

#### MAJOR

| # | Dimension | Issue Description | Evidence Anchor | Confidence |
|---|-----------|-------------------|-----------------|------------|
| M1 | Cherry-Picking / Confirmation Bias | El catálogo GPU se depuró retirando kernels cerca del ridge point, y se amplió para cubrir el "régimen intermedio" — decisiones que sesgan el catálogo hacia ejemplos separables antes de la Fase 2. | text: "se retiró del catálogo"; "no cubría de forma suficiente el régimen intermedio" | 3 |
| M2 | Overgeneralization / Precision Theater | El punto de inflexión CPU varía 7.0–9.3 FLOP/byte (~33%), en fuerte contraste con el registro de precisión a dos decimales (0.29%, 7.48%) de la validación de FLOPs. | text: "un valor final en el rango 7.0--9.3 FLOP/byte" | 4 |
| M3 | "So What?" Test / Overgeneralization | El "aporte metodológico central" reencuadra como hallazgo propio un principio ya establecido (Georges et al. 2007), citado por los propios autores para otro fin en el mismo documento. | text: "El aporte metodológico central de esta fase..."; "siguiendo el principio... de Georges et al." | 4 |
| M4 | Ignored Alternative Explanation | La corrección de `-lgc` se apoya en pruebas bajo un driver NVIDIA distinto al de la prueba negativa anterior, sin control A/B que aísle la versión de driver como explicación alternativa. | text: "observó 765~MHz pese a solicitar cinco objetivos" | 3 |
| M5 | Logic Chain / Evidence Presentation | La narrativa de "bug → causa raíz → corrección → reverificación" se repite ~8 veces con estructura casi idéntica sin síntesis agregada, sustituyendo argumentación científica por bitácora de ingeniería. | text: múltiples secciones de Resultados | 3 |

#### MINOR

| # | Dimension | Issue Description | Confidence |
|---|-----------|-------------------|------------|
| m1 | Stakeholder Blind Spots | No se distingue la contribución individual de los dos autores. | 3 |
| m2 | Alternative Paths | No se justifica por qué se construyó un instrumento propio en vez de adoptar LIKWID/PAPI. | 3 |

### Ignored Alternative Explanations/Paths

1. Adopción de herramientas maduras (LIKWID/PAPI) en vez de instrumento propio — nunca se explica por qué no bastaban.
2. Dependencia de versión de driver como explicación de la reversión de `-lgc`, no investigada como hipótesis principal.
3. El "aporte metodológico central" como diligencia aplicada (aplicar principios ya establecidos a una plataforma nueva), no como hallazgo generalizable nuevo.

### Missing Stakeholder Perspectives

- Administradores del clúster compartido (privilegios elevados repetidos sobre infraestructura de producción).
- Comité evaluador del Trabajo de Grado (qué constituye entrega parcial aceptable cuando el título compromete un agente que no existe).
- Usuarios finales / investigadores HPC sin el mismo acceso administrativo privilegiado.

### Unexamined Premise

El documento asume implícitamente que la verificación empírica exhaustiva de un instrumento en **una** plataforma específica constituye una lección metodológica transferible, cuando la mayoría de los defectos encontrados son altamente específicos de esa microarquitectura y de ese estado de configuración del nodo en ese momento.

### Observations (Non-Defects)

- La honestidad explícita sobre la invalidez del eje F0–F4 de la campaña CPU es un manejo de resultados negativos poco común.
- El uso de identificadores ARC-xxx para trazar decisiones permite, en principio, auditar independientemente cada corrección narrada.
