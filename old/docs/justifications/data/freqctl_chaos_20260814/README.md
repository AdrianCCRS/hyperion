# FRQ-05/CAM-07 — prueba de caos en paccaA100

Artefactos resumidos de tres ejecuciones reales bajo asignación Slurm
exclusiva. Los artefactos originales permanecen en
`~/yacacerest/freqctl_chaos_*` de la cuenta `latorresn`.

- Job 5185 reprodujo el defecto: con `SIGINT` heredada como `SIG_IGN`, sysfs
  se restauró, pero el proceso ignoró la señal reenviada y terminó con 0.
- Job 5186 verificó la corrección con `SIGINT`: retorno 130 y restauración
  exacta de los seis CPU.
- Job 5187 verificó la misma corrección con `SIGTERM`: retorno 143 y
  restauración exacta de los seis CPU.

En los tres casos se observó primero el estado original
`performance, 800000–3600000 kHz`, después el nivel fijo
`performance, 2200000–2200000 kHz` y finalmente el estado original exacto.
La prueba valida restauración de las solicitudes de sysfs; no valida la
actuación física del reloj con Turbo activo.
