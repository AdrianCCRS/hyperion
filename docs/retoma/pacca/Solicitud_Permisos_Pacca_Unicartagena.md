# Solicitud de permisos a administración HPC Unicartagena — justificación técnica

Este documento resume, para uso directo en el correo a administración del
clúster HPC de la Universidad de Cartagena, todos los permisos que el
proyecto necesita sobre el nodo `paccaA100` para completar la medición DVFS
descrita en `Guia_Maestra_Fase1_DVFS.md`. Cada ítem indica qué se pide, por
qué, y qué evidencia técnica ya se recolectó (de solo lectura, o con pruebas
mínimas propias compiladas in situ, sin modificar nada persistente del
nodo) que respalda la solicitud.

Estado del diagnóstico de base: auditoría de solo lectura ejecutada vía
`srun` en `paccaA100` el 2026-08-05, más una serie de pruebas empíricas
puntuales (compilación y ejecución de programas C mínimos que reproducen
exactamente el mecanismo de medición del proyecto) para confirmar qué
funciona hoy sin pedir nada. Ver `docs/retoma/Auditoria_PaccaA100_Unicartagena.md`
para el detalle completo.

**Nota de contexto para el lector interno del proyecto:** a diferencia de la
solicitud equivalente para SC3/felix (`Solicitud_Permisos_SC3.md`), aquí
**no se pide nada relacionado con energía** — `paccaA100` sí tiene RAPL y ya
confirmamos que se puede leer sin ningún permiso especial. Ver la sección
"Resueltos, no requieren solicitud" al final.

---

## P1. Delegación de escritura sobre cpufreq en cores asignados

**Qué se pide:** permiso de escritura para el usuario sobre los archivos
sysfs de control de frecuencia (`scaling_min_freq`, `scaling_max_freq`) de
los cores CPU que Slurm delegue a nuestros jobs en `paccaA100`. Confirmamos
que estos archivos existen pero no son escribibles por el usuario
(`scaling_min_freq` probado con `test -w`, resultado negativo — no es un
problema de driver de kernel faltante, es puramente de permisos).

**Por qué se necesita:** el proyecto mide el comportamiento
energía/rendimiento de cargas de cómputo en función de la frecuencia de CPU
(DVFS). Sin poder fijar la frecuencia, solo es posible medir en el punto
operativo por defecto (governor `performance`), lo cual reduce el dataset a
un único nivel en vez de la curva completa de frecuencias que el estudio
requiere.

**Detalle técnico importante — mecanismo distinto al de un cpufreq clásico:**
`paccaA100` usa el driver `intel_pstate`, cuyos governors disponibles son
únicamente `performance` y `powersave` (**no existe el governor `userspace`
que sí tiene un cpufreq clásico como `acpi-cpufreq`**). El control de
frecuencia en `intel_pstate` se hace fijando `scaling_min_freq` y
`scaling_max_freq` al mismo valor (el rango se colapsa a un punto), no
escribiendo un gobernador `userspace` + `scaling_setspeed`. Por eso la
solicitud es específicamente sobre esos dos archivos, no sobre
`scaling_governor`.

**Buena noticia para la administración — dominio de frecuencia por core, no
por socket:** a diferencia de otros nodos donde hemos trabajado, en
`paccaA100` cada CPU lógica tiene su **propio dominio de frecuencia
independiente** (confirmado: `related_cpus`/`affected_cpus` de cada core
probado — 0, 1, 8, 16 — devuelven solo esa misma CPU, nunca un grupo). Esto
significa que fijar la frecuencia de los cores que se nos asignen **no
puede afectar a otros procesos ni a otros usuarios**, incluso si el nodo
llegara a compartirse entre varios jobs simultáneos — no aplica aquí el
riesgo de "fuga de frecuencia entre usuarios" que sí existe en hardware más
viejo con dominios por socket. No se necesita ninguna condición adicional
de asignación de cores (como sí fue necesario pedir en SC3/felix).

**Evidencia de respaldo:** `scaling_driver=intel_pstate`,
`scaling_available_governors=performance powersave`,
`related_cpus`/`affected_cpus` de un solo elemento por CPU, y
`scaling_min_freq` no escribible — todo leído de sysfs vía `srun` de solo
lectura el 2026-08-05.

---

## P2. Confirmación de cuota individual de `$HOME` (o acceso a `/scratch`)

**Qué se pide:** confirmación de la cuota individual de `$HOME` para
nuestro usuario, o alternativamente un subdirectorio propio y escribible
dentro de `/scratch`.

**Por qué se necesita:** confirmamos que `/scratch` existe como punto de
montaje NFS pero no es escribible para el usuario. `$HOME` sí es escribible
y el volumen (montaje NFS compartido) reporta ~1.6 TB libres de 8 TB, pero
eso es el estado global del filesystem, no la cuota asignada a nuestra
cuenta específicamente — no pudimos confirmar ese número desde una sesión
de solo lectura. Necesitamos saberlo antes de planear el volumen de datos
de una campaña completa (ventanas de perf + metadata por cada combinación
kernel × nivel de frecuencia × repetición).

**Evidencia de respaldo:** `df -h /scratch` y `df -h $HOME` ejecutados
el 2026-08-05; intento de escritura en `/scratch` rechazado.

---

## P3. Acceso de lectura a contadores de uncore/memory controller

**Qué se pide:** bajar `perf_event_paranoid` (actualmente en `2`, más
restrictivo que en otros nodos donde hemos trabajado) a `0` o `-1`, o
alternativamente otorgar la capability `CAP_PERFMON` al usuario/binario
`perf` para `paccaA100`. Adicionalmente, si es sencillo desde su lado,
instalar el paquete `perf` (el CLI de `linux-tools`) — hoy no está
disponible en el nodo, lo que no nos bloquea (nuestro mecanismo de
medición no depende del CLI, solo del syscall `perf_event_open` que sí
funciona hoy) pero sí complica cualquier diagnóstico manual rápido de
nuestro lado o del suyo.

**Confirmado por auditoría (2026-08-05):** `paccaA100` expone una topología
de uncore completa y moderna en `/sys/bus/event_source/devices/`
(`uncore_cha`, `uncore_iio`, `uncore_imc` ×12 — los canales de memoria,
`uncore_irp`, `uncore_m2m`, `uncore_m2pcie`, `uncore_m3upi`, `uncore_pcu`,
`uncore_ubox`, `uncore_upi`), pero no confirmamos todavía si el usuario
puede abrirlos bajo el `perf_event_paranoid=2` actual (no se pudo probar
directamente por la ausencia del CLI `perf`, y estos eventos de uncore son
de alcance socket/sistema completo, la categoría que ese nivel de
`perf_event_paranoid` típicamente restringe).

**Por qué se necesita:** en una validación anterior de nuestro pipeline (en
otro nodo, hardware distinto) encontramos que el contador per-core genérico
que usamos hoy para estimar bytes movidos (`cache-misses`, vía
`perf_event_open`) subestima el tráfico real de memoria en ~30-34% frente
al valor analítico conocido en una carga de ancho de banda puro (STREAM).
Los contadores de **uncore/memory controller** sí ven ese tráfico
directamente (cuentan transacciones reales hacia DRAM), así que son la
validación cruzada correcta para confirmar o acotar ese sesgo en este
hardware específico. Esto afecta la calidad de las etiquetas
`compute_bound`/`memory_bound` del dataset para los kernels más cercanos al
punto de inflexión del modelo Roofline (los "intermedios"), no es un
capricho de instrumentación.

**Nota técnica:** a diferencia de los contadores por-PID que ya usamos hoy
sin ningún privilegio adicional (confirmado empíricamente, ver sección
"Resueltos" abajo), los contadores de uncore son *del socket completo*, no
por proceso — nuestro uso sería puntual, para validación de calibración con
el nodo en uso exclusivo por nuestro job (`--exclusive`), no para medición
continua por ventana durante campañas.

**Evidencia de respaldo:** desviación de -33.8% cuantificada en otro nodo,
documentada en `docs/retoma/Informe_Piloto_F3_2026-07-31.md`; listado de
dispositivos uncore de `paccaA100` obtenido el 2026-08-05.

---

## Resueltos, no requieren solicitud

Para que quede claro en el correo qué **no** hace falta pedir:

- **Medición de energía (RAPL):** `intel-rapl:0`/`intel-rapl:1` (uno por
  socket) están presentes y **confirmamos lectura directa de `energy_uj`
  como usuario normal, sin ningún permiso adicional**. A diferencia de
  otros nodos con los que hemos trabajado, aquí la medición de energía real
  ya funciona hoy.
- **GPU (A100):** confirmado accesible con `srun --gres=gpu:1` +
  `nvidia-smi` (A100-PCIe-40GB, driver 595.45.04, CUDA 13.2). No requiere
  ningún permiso adicional.
- **Contadores por-PID (el mecanismo central de medición del proyecto):**
  confirmamos empíricamente, compilando y ejecutando un programa C mínimo
  que abre `perf_event_open` sobre un proceso hijo con `inherit=1` (el
  mismo mecanismo exacto de nuestro harness), que funciona hoy bajo
  `perf_event_paranoid=2` sin ningún privilegio adicional. Esto es
  independiente de si el CLI `perf` está instalado o no.
- **Aislamiento de cores del job:** aunque la configuración de cgroups del
  clúster no delega un cgroup por-job en todos los controladores, `srun
  --exclusive` sí entrega el nodo completo (confirmado vía
  `/proc/self/status`, `Cpus_allowed_list` cubre las 32 CPUs lógicas) — no
  hace falta pedir nada adicional de aislamiento.

---

## Borrador de correo a administración HPC Unicartagena

> Asunto: Solicitud de permisos para proyecto de medición DVFS en paccaA100
>
> Estimados,
>
> Estamos desarrollando un proyecto de investigación que mide el
> comportamiento de energía y rendimiento de cargas de cómputo en función
> de la frecuencia de CPU (DVFS) sobre el nodo `paccaA100`. Ya hicimos un
> diagnóstico completo de solo lectura del nodo (sin modificar nada
> persistente) y necesitamos tres permisos puntuales para poder avanzar:
>
> **1. Escritura sobre control de frecuencia (cpufreq) en los cores que
> Slurm nos asigne en `paccaA100`.** El nodo usa el driver `intel_pstate`,
> así que el control se hace fijando `scaling_min_freq` y
> `scaling_max_freq` al mismo valor (no existe el governor `userspace` en
> este driver) — necesitamos permiso de escritura del usuario sobre esos
> dos archivos para los cores delegados a nuestros jobs.
>
> Un dato a favor: en `paccaA100` cada CPU lógica tiene su propio dominio
> de frecuencia independiente (no comparte con otras CPUs del mismo
> socket), así que fijar la frecuencia de nuestros cores asignados no
> puede afectar a otros procesos ni usuarios del nodo, sin necesidad de
> ninguna condición adicional sobre cómo se nos asignan los cores.
>
> **2. Confirmación de nuestra cuota individual en `$HOME`, o acceso a
> `/scratch`.** Notamos que `/scratch` no es escribible para nuestro
> usuario. ¿Podrían confirmarnos la cuota real que tenemos en `$HOME`, o
> habilitarnos un subdirectorio propio en `/scratch` si el volumen de datos
> de la campaña lo requiere?
>
> **3. Lectura de contadores de uncore** (memory controller, expuestos en
> `paccaA100` como `uncore_imc_0` a `uncore_imc_11` en
> `/sys/bus/event_source/devices/`). Hoy `perf_event_paranoid` está en `2`
> en el nodo; ¿podrían bajarlo a `0` (o `-1`), u otorgar `CAP_PERFMON` a
> nuestro usuario? En una validación anterior de nuestro pipeline
> detectamos que nuestro método actual de medir bytes movidos (contadores
> per-core de "cache miss") puede subestimar el tráfico real de memoria en
> cargas con acceso muy secuencial — los contadores de uncore nos
> permitirían confirmarlo y corregirlo en este hardware específico. Es un
> uso puntual de validación, no medición continua durante campañas. Si es
> sencillo de su lado, también agradeceríamos que instalaran el paquete
> `perf` (`linux-tools`) — hoy no está en el nodo, lo cual no nos bloquea
> pero sí dificulta cualquier diagnóstico rápido.
>
> Para que quede claro qué NO estamos pidiendo: ya confirmamos que la
> medición de energía (RAPL) funciona sin ningún permiso adicional, así
> como el mecanismo central de medición de rendimiento por proceso — ambos
> ya operativos hoy en `paccaA100`.
>
> Quedamos atentos y con gusto compartimos el diagnóstico técnico completo
> si es útil para evaluar la solicitud.
>
> Saludos,
> [nombre]
