# Correo para el profesor: hallazgo de turbo boost en paccaA100 y solicitud de permiso adicional

**Estado:** borrador, no enviado. Hallazgo registrado como ARC-136 en `docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md`; detalle completo en `docs/retoma/pacca/Informe_Preliminar_Campana_DVFS_CPU_20260813.md` (sección 0).

**Contexto:** al corregir un instrumento de medición de la Fase 1 (verificado en hardware real, no solo en teoría), esa misma verificación expuso que el candado de frecuencia de CPU en el nodo `paccaA100` no está reteniendo el reloj real bajo carga. GPU no presenta actualmente un bloqueo equivalente: `-lgc` ya era funcional y su actuación bajo carga quedó confirmada en ARC-137, mientras el problema de turbo CPU permanece abierto.

---

**Asunto:** paccaA100 — el candado de frecuencia de CPU no retiene el reloj bajo carga (turbo), necesito permiso adicional

Profe, buen día.

En paccaA100, `scaling_min_freq`/`scaling_max_freq` (el permiso que ya nos dieron) no retiene el reloj real bajo carga: con el candado en 2.2 GHz, el reloj medido durante la ejecución fue ~3.5 GHz de forma consistente — verificado con nuestro instrumento y también con un `cat` manual en shell, sin código del proyecto de por medio. Es turbo boost, activo a nivel de sistema.

`no_turbo` y `max_perf_pct` (los dos controles de `intel_pstate` que sí lo forzarían) dan `Permission denied` con nuestra cuenta — son globales, de root, distintos del permiso por-núcleo que ya tenemos.

Esto pone en duda el eje de frecuencia de toda la campaña de CPU corrida hasta ahora (las etiquetas de clasificación no dependen de frecuencia, así que esas siguen sirviendo).

¿Me ayuda a pedirle a la administración del clúster escritura sobre `no_turbo` o `max_perf_pct` (cualquiera de los dos alcanza)? Si prefiere que yo le escriba directo al admin, dígame y lo redacto.

Gracias,
