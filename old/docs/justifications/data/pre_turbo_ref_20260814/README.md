# Integración CPU REF previa al permiso de Turbo

Resumen de `pacca_pre_turbo_ref_20260814`, ejecutada en `paccaA100` bajo
asignación Slurm exclusiva con el pipeline vigente. Los datos crudos se
conservan en:

`~/hyperion-results/validation/pacca_pre_turbo_ref_20260814/`

La campaña contiene tres kernels por tres repeticiones: 9 aceptadas, 0
rechazadas, 0 omitidas y `frequency_restored_verified=true`. Es una prueba
de integración REF; no es parte del dataset final ni aporta evidencia sobre
la actuación de niveles fixed.

Comprobaciones agregadas sobre las 57.048 ventanas `ok`:

- ninguna sin frecuencia por ventana, FLOPs medidos o bytes uncore reales;
- ninguna con energía inválida;
- todas etiquetadas;
- `push_retries=0` en las nueve corridas y stderr vacío;
- temperatura real registrada antes de calibrar y por combinación
  (37–51 °C); carga externa normalizada igual a cero;
- una fila terminal `pmu_degraded` en 3MM (`delta_instructions=0`,
  `flops_measured_window=0`), excluida de las ventanas `ok`.

El ridge REF fue 8,60706 FLOP/byte (`P_pico=512,899 GFLOP/s`,
`BW_pico=59,5905 GB/s`) y pasó el chequeo de plausibilidad.
