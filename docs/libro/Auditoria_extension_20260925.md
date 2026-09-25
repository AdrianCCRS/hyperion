# Auditoría de extensión del libro — 25 de septiembre de 2026

## Criterio aplicado

Se conservaron las figuras principales en el cuerpo. Las tablas que permiten entender el diseño, los modelos y las decisiones permanecen allí; cuatro tablas extensas tienen ahora una síntesis en el capítulo y el detalle íntegro en anexos. La rejilla de frecuencias duplicada se eliminó, porque su versión única ya estaba en el Anexo A.

| Tabla | Ubicación | Función |
|---|---|---|
| `tab:met-plataforma` | Cuerpo | Hardware de referencia |
| `tab:agente-brazos` | Cuerpo | Diseño de brazos |
| `tab:met-parametros` | Cuerpo | Parámetros del control |
| `tab:agente-politica` | Cuerpo | Acciones por clase |
| `tab:agente-compuestas` | Cuerpo | Verdad de fase de las aplicaciones |
| `tab:fase4-matriz` | Cuerpo | Diseño inicial |
| `tab:fase4-mapa` | Cuerpo | Mapa breve de los escenarios |
| `tab:cpu-features-modelo` | Cuerpo | Contrato de entradas CPU |
| `tab:cpu-comparacion-modelos` | Cuerpo | Comparación que sustenta la selección |
| `tab:cpu-politica-ic` | Cuerpo | Síntesis de la decisión CPU |
| `tab:gpu-modelos-operativos` | Cuerpo | Selección del modelo desplegado |
| `tab:gpu-politica-ic` | Cuerpo | Síntesis de la decisión GPU |
| `tab:agente-latencias` | Cuerpo | Costo y tiempo de actuación |
| `tab:fase4-E-composicion` | Cuerpo | Composición resumida del escenario E |
| `tab:fase4-cloverleaf` | Cuerpo | Resultado confirmatorio externo |
| `tab:fase4-balance` | Cuerpo | Síntesis final de la Fase 4 |
| `tab:verificaciones-previas` | Anexo A | Desglose, control o sensibilidad para consulta |
| `tab:catalogo` | Anexo A | Desglose, control o sensibilidad para consulta |
| `tab:frecuencias` | Anexo A | Desglose, control o sensibilidad para consulta |
| `tab:met-senales` | Anexo A | Desglose, control o sensibilidad para consulta |
| `tab:met-granularidad` | Anexo A | Desglose, control o sensibilidad para consulta |
| `tab:met-amenazas` | Anexo A | Desglose, control o sensibilidad para consulta |
| `tab:fase4-escenarios` | Anexo A | Tabla íntegra detrás de una versión breve |
| `tab:cpu-variantes-entrada` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-protocolos` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-metricas-definicion` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-metricas-valores` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-resultado-frecuencia` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-umbral-rejilla` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-selectiva` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-contrafactual-frecuencia` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:gpu-modelos-2026-09-22` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:gpu-vif` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:agente-deteccion` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:fase4-apps` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:fase4-clasificacion` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:fase4-resultados` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:fase4-activo-sombra` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:fase4-costo` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:fase4-C` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:fase4-noturbo` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:fase4-E` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:fase4-clasificacion-CE` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:fase4-D` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-cobertura-conjunto` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-rechazos` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-flujo-modelo` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-tope-muestreo` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-configuracion-final` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-metricas-clase` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:agente-restauracion` | Anexo B | Desglose, control o sensibilidad para consulta |
| `tab:cpu-politica-detalle` | Anexo B | Tabla íntegra detrás de una versión breve |
| `tab:gpu-politica-detalle` | Anexo B | Tabla íntegra detrás de una versión breve |
| `tab:fase4-balance-detalle` | Anexo B | Tabla íntegra detrás de una versión breve |

## Figuras y extensión

Resultados conserva sus 38 figuras y Metodología sus 5 figuras. Las dos figuras metodológicas originales del Anexo A siguen allí. La tabla completa de protocolos que ocupaba una página quedó en el Anexo A y fue sustituida por un mapa compacto en Metodología. Las rejillas completas de política CPU y GPU y el balance amplio de la Fase 4 quedaron en el Anexo B; el cuerpo conserva sus conclusiones tabuladas.

El PDF anterior a esta revisión empezaba los anexos después de la página numerada 124. La nueva paginación debe verificarse cuando el usuario compile; se preservaron los gráficos aunque el cuerpo supere el objetivo aproximado de 90 páginas. No se tocaron el marco teórico ni las secciones anteriores.

Comprobación estática: etiquetas únicas, referencias existentes, imágenes presentes y entornos de tabla y figura cerrados. Se revisó además la ubicación de las 38 figuras de Resultados por subsección y se corrigió el gráfico de política GPU que había quedado en las métricas CPU, junto con el orden de las figuras de comparación GPU y CloverLeaf.
