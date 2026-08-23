# Reporte para el admin: el binario `perf` no tiene CAP_PERFMON aplicado en paccaA100

**Estado: REABIERTO (2026-08-22, ARC-184) — ENVIAR.** La capability se perdió otra vez. `getcap /usr/bin/perf` vuelve a no devolver nada (`Current: =` vacío en `capsh --print`) y `perf stat -a -e uncore_imc_0/cas_count_read/ sleep 0.2` vuelve a dar `<not supported>` con el mismo sufijo `/u` de espacio de usuario -- idéntico al hallazgo original. Confirmado en vivo dentro de `--exclusive --gres=gpu:1` en paccaA100.

No fue un fallo ruidoso: `perf stat` sigue corriendo y emitiendo intervalos a su cadencia normal con los contadores vacíos, así que una campaña completa (`pacca_cpu_final_attempt03_20260820`) SÍ leyó uncore correctamente el 2026-08-20, y el pre-vuelo de fases del 2026-08-22 (job 6420, mismo nodo, sin reinicio de por medio -- uptime 11 días) terminó en 0 corridas aceptadas de 27 sin ningún error visible, solo `verdict.json` rechazando cada una por `I10: 0 ventanas 'ok'`. Se mitigó agregando un chequeo de preflight (E13, ARC-184) que sondea el PMU real antes de comprometer una campaña completa, pero el permiso en sí solo lo puede restablecer el administrador.

**Hipótesis de por qué se perdió:** `setcap` aplica la capability al *archivo* del binario, no de forma persistente al *nombre* del paquete -- una actualización o reinstalación de `linux-tools`/`perf` reemplaza el binario y con él cualquier `cap_perfmon=ep` asignado a mano, sin que quede rastro en ningún log de nuestro lado. Si el paquete se actualizó entre el 2026-08-12 y hoy, esa es la explicación más simple.

**Contexto original:** el correo de concesión de permisos indicó que se instaló `linux-tools` (`perf`) en el nodo y se asignó la capacidad `CAP_PERFMON` al binario `perf` específicamente, para permitir la lectura de los contadores uncore_imc (controlador de memoria) sin bajar `perf_event_paranoid` globalmente. Verificamos esto de dos formas independientes (un probe propio y una sesión interactiva manual) antes de escribir esto, y de nuevo ahora al reabrirlo.

---

**Asunto:** `perf` volvió a quedarse sin CAP_PERFMON en paccaA100 -- contadores uncore_imc inaccesibles otra vez

Hola,

El 12 de agosto nos confirmaron que se había aplicado `CAP_PERFMON` al binario `/usr/bin/perf` en `paccaA100`, y lo verificamos entonces con datos reales. Hoy (22 de agosto), verificando otra vez dentro de `--exclusive --gres=gpu:1`, la capability ya no está:

```
$ getcap /usr/bin/perf
```
(sin salida -- no hay ningún `cap_perfmon=ep` asignado al binario)

Consistente con esto:

```
$ perf stat -a -e uncore_imc_0/cas_count_read/ sleep 0.2
<not supported>,MiB,uncore_imc_0/cas_count_read/u
```

`capsh --print` muestra lo mismo: `cap_perfmon` en el *bounding set*, `Current:` vacío.

Entre el 12 y el 22 de agosto sí tuvimos una campaña que leyó uncore correctamente (20 de agosto), así que el permiso estuvo activo y se perdió en algún punto después. Nuestra sospecha es que una actualización o reinstalación del paquete `linux-tools`/`perf` reemplaza el binario y con él cualquier capability asignada a mano con `setcap` -- si hubo un cambio de paquete en esa ventana, encajaría.

¿Podrían volver a aplicar `setcap cap_perfmon,cap_sys_ptrace=ep /usr/bin/perf` en `paccaA100`? Si es viable de su lado, también preguntamos si existe alguna forma de que la capability sobreviva a futuras actualizaciones del paquete (por ejemplo, un hook de post-instalación), para no depender de detectarlo manualmente cada vez.

Quedamos atentos.

Saludos
