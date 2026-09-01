# Reporte para el admin: el binario `perf` no tiene CAP_PERFMON aplicado en paccaA100

**Estado: RESUELTO (2026-08-12, ARC-121) — no enviar.** El admin corrigió la asignación de `CAP_PERFMON` sobre `/usr/bin/perf` en `paccaA100`; verificado con `getcap` y con una corrida real de `telemetry_kernel_launcher --enable-uncore`, datos reales de principio a fin. Este borrador queda como registro histórico del hallazgo, no como pendiente de envío. Registrado en `docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md` como ARC-117/ARC-118/ARC-121.

**Contexto:** el correo de concesión de permisos indicó que se instaló `linux-tools` (`perf`) en el nodo y se asignó la capacidad `CAP_PERFMON` al binario `perf` específicamente, para permitir la lectura de los contadores uncore_imc (controlador de memoria) sin bajar `perf_event_paranoid` globalmente. Verificamos esto de dos formas independientes (un probe propio y una sesión interactiva manual) antes de escribir esto.

---

**Asunto:** `perf` no tiene CAP_PERFMON aplicado en paccaA100 -- contadores uncore_imc siguen inaccesibles

Hola,

Verificando en `paccaA100` (dentro de `--exclusive --gres=gpu:1`), `perf` no tiene la capability aplicada:

```
$ getcap /usr/bin/perf
```
(sin salida -- no hay ningún `cap_perfmon=ep` asignado al binario)

Consistente con esto:

```
$ perf stat -a -e uncore_imc_0/event=0x04,umask=0x0f/ sleep 1
   <not supported>      uncore_imc_0/event=0x04,umask=0x0f/u

$ perf stat -a -e uncore_imc_0/event=0x04,umask=0x0f/k sleep 1
Error:
Access to performance monitoring and observability operations is limited.
Consider adjusting /proc/sys/kernel/perf_event_paranoid setting to open
access to performance monitoring and observability operations for processes
without CAP_PERFMON, CAP_SYS_PTRACE or CAP_SYS_ADMIN Linux capability.
```

`capsh --print` muestra lo mismo: `cap_perfmon` en el *bounding set*, `Current:` vacío.

¿Podrían revisar si el `setcap` quedó aplicado en `paccaA100` específicamente? Quizás se aplicó en otro nodo o no se guardó.

Quedamos atentos.

Saludos
