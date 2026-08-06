# Auditoría de solo lectura: paccaA100 (HPC Universidad de Cartagena)

**Fecha:** 2026-08-05
**Metodología:** idéntica a la auditoría de felix/SC3 (`docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md`, ARC-29) — inspección de sysfs/procfs sin escritura, más una prueba empírica mínima del mecanismo real de medición (no solo del CLI `perf`).
**Acceso:** `ssh latorresn@hpc.unicartagena.edu.co` (alias `hpc-unicartagena`, resuelve internamente como `toctoc.unicartagena.edu.co`) → `ssh pacca` (resuelve como `pacca.unicartagena.edu.co`, ya autenticado por confianza desde el gateway) → `srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive --pty bash -i`. `$HOME` es NFS compartido entre gateway y `pacca`. Autenticación por llave dedicada (`~/.ssh/id_ed25519_pacca`), la contraseña entregada por el usuario se usó una sola vez para `ssh-copy-id` y no quedó persistida en ningún artefacto.

**Nota importante:** este nodo NO es el felix de SC3. Es un clúster distinto (Universidad de Cartagena), con hardware, kernel y configuración de Slurm propios. Ningún hecho de este documento debe asumirse válido para felix ni viceversa.

---

## 1. Veredicto ejecutivo

**paccaA100 es viable para el pipeline de medición**, y en varios aspectos es un entorno mejor que felix:

| Capacidad | felix (SC3) | paccaA100 (Unicartagena) |
|---|---|---|
| `perf_event_open` PID+inherit (mecanismo real del harness) | Funciona (`paranoid=1`) | **Confirmado empíricamente: funciona (`paranoid=2`)** |
| RAPL (energía) | Inexistente (CPU pre-RAPL) | **Presente y legible sin permisos especiales** |
| Dominio de frecuencia | Por socket (riesgo de fuga entre usuarios, mitigado con E10) | **Por core** (sin ese riesgo) |
| Escritura de frecuencia (`scaling_min/max_freq`) | No | No (aún sin probar solicitud) |
| GPU | GTX Titan X | **NVIDIA A100-PCIe-40GB** |
| `/scratch` escribible | No | No |
| LIKWID / MSR para validar uncore | Bloqueado (msr root-only) | Módulo LIKWID funcional; `msr` sigue siendo root-only (bloqueo probablemente igual) |

El bloqueador principal no es de arquitectura sino administrativo: **no hay ningún permiso especial otorgado todavía** (ni escritura de frecuencia, ni RAPL de escritura — aunque de lectura ya funciona —, ni acceso a `/dev/cpu/*/msr`). La ruta para pedirlos es la misma que se usó con SC3 (`Solicitud_Permisos_SC3.md` como plantilla).

---

## 2. Hardware

- **CPU:** 2× Intel Xeon Gold 5315Y ("Ice Lake-SP", 2021), 8 cores físicos por socket, 2 threads SMT por core → 32 procesadores lógicos totales. Frecuencia base 3.20 GHz, máxima con turbo ~3.6 GHz, mínima 0.8 GHz.
- **NUMA:** 2 nodos, uno por socket (coincide 1:1 con los sockets, sin sorpresas).
- **Caché (cpu0):** L1d 48K, L1i 32K, L2 1280K (1.25 MB) privados por core; L3 12288K (12 MB) **compartido por todo el socket** (`shared_cpu_list: 0-7,16-23` — los 8 cores físicos + sus 8 hilos SMT del socket 0). `coherency_line_size=64` (igual que felix, estándar x86 — el cálculo de `bytes_moved_window = delta_cache_misses × cache_line_size_bytes` no necesita ajuste por esto).
- **GPU:** NVIDIA A100-PCIe-40GB, driver 595.45.04, CUDA 13.2, sin procesos activos al momento de la auditoría. Solo visible con `--gres=gpu:1`.
- **Slurm:** versión 23.11.11. Partición `GPU` contiene un único nodo (`paccaA100`) — es decir, la partición GPU completa de este clúster es este nodo. Estado `ALLOC` durante la auditoría (ocupado por nuestra propia sesión `--exclusive`).

---

## 3. Control de frecuencia (cpufreq)

- **Driver:** `intel_pstate` (no `acpi-cpufreq` como felix). Esto es esperado en hardware Ice Lake/HWP moderno.
- **Governors disponibles:** `performance powersave` — **no aparece `userspace`**. Esto es coherente con `intel_pstate` en modo activo (no-passive): el control fino de frecuencia no se hace vía `governor=userspace` + `scaling_setspeed`, sino vía los límites `scaling_min_freq`/`scaling_max_freq` (o los nodos propios de `intel_pstate` bajo `/sys/devices/system/cpu/intel_pstate/`: `min_perf_pct`, `max_perf_pct`, `no_turbo`).
- **Nuestro código ya anticipa esto:** `freqctl.py` define una estrategia `bounded_range` específicamente para `intel_pstate` (fijar min=max=objetivo), distinta de `discrete_bounds` usada en felix. No se necesita ningún cambio estructural — es exactamente el caso para el que esa rama de código fue diseñada (ver nota "Adaptabilidad a un clúster más moderno" en la memoria de proyecto de felix).
- **Dominio de frecuencia: por core, no por socket.** `related_cpus`/`affected_cpus` de cada CPU probado (0, 1, 8, 16) devuelven solo esa misma CPU — a diferencia de felix, donde `freqdomain_cpus` de un core abarcaba las 16 CPUs de todo el socket. **Esto significa que el riesgo de "fuga de frecuencia entre usuarios/tareas" que motivó el check E10 en felix NO aplica de la misma forma aquí**: fijar la frecuencia de un core no afecta a los demás. E10 debería seguir corriendo (es agnóstico y barato), pero se espera que reporte domain=1 CPU y no bloquee nada.
- **Escritura:** `scaling_min_freq` **no es escribible** por el usuario actual (confirmado con `test -w`). Sin permisos otorgados todavía, igual que felix.

---

## 4. RAPL (energía)

- **Presente:** `/sys/class/powercap/intel-rapl:0` y `intel-rapl:1` (uno por socket, dominio `package`), con subdominios (`intel-rapl:0:0`, etc.).
- **Legible sin permisos especiales:** se confirmó lectura directa de `intel-rapl:0/energy_uj` (valor devuelto: ~22.37 mJ acumulados, dominio `package-0`) desde una sesión de usuario normal dentro del job de Slurm. **No fue necesario ningún permiso adicional.**
- Esto es una diferencia estructural mayor respecto a felix, que no tiene RAPL en absoluto (CPU anterior a esa tecnología). Si el proyecto llega a necesitar telemetría de energía real (más allá de lo que el harness ya soporta vía el tag `ENERGY`), **este nodo lo permite y felix nunca lo permitiría**, sin importar qué permisos se consigan ahí.
- No se probó escritura (no aplica — RAPL es de solo lectura por diseño para el "energy counter", la escritura relevante sería para límites de power capping, fuera de alcance de este proyecto).

---

## 5. `perf_event_open` y `perf_event_paranoid`

- `perf_event_paranoid = 2` (más restrictivo que felix, que está en 1).
- **El CLI `perf` no está instalado** en el nodo (`perf --version` → `command not found`, exit 127). Esto bloqueó la auditoría inicial basada en `perf stat`, que es el mismo enfoque usado en felix.
- **Prueba empírica alternativa, directa al syscall:** se escribió y compiló en el nodo un programa C mínimo (`test_perf_pid.c`) que reproduce exactamente el mecanismo real del harness (`telemetry/src/perf_reader.cpp`): `fork()` de un hijo detenido con `SIGSTOP`, `perf_event_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, pid=<hijo>, cpu=-1, inherit=1)`, luego `SIGCONT` y lectura del contador.
  - **Resultado: `PERF_EVENT_OPEN_PID_INHERIT: OK`, `INSTRUCTIONS_MEDIDAS: 1200001053`** (consistente con el bucle de 200M iteraciones del hijo). **El mecanismo central del proyecto funciona bajo `paranoid=2` sin ningún permiso adicional.**
  - Esto resuelve la pregunta más importante para viabilidad del nodo: `paranoid=2` bloquea eventos *system-wide*/CPU-wide para usuarios sin privilegios, pero el modo PID+inherit que el harness usa exclusivamente sigue permitido — coherente con la semántica documentada de `perf_event_paranoid` (nivel 2 = "disallow raw and ftrace function tracepoint access", no bloquea el monitoreo del propio proceso/hijos).
- **No se probó** el acceso a contadores uncore (bloqueado por la ausencia del CLI `perf`; se listaron los PMU disponibles en sysfs pero no se intentó una lectura directa vía syscall). Ver sección 7.

---

## 6. Contadores uncore

`/sys/bus/event_source/devices/` lista una topología uncore completa y moderna: `uncore_cha` (varios), `uncore_iio` (varios), `uncore_imc` (×12 — los canales de memoria), `uncore_irp`, `uncore_m2m`, `uncore_m2pcie`, `uncore_m3upi`, `uncore_pcu`, `uncore_ubox`, `uncore_upi`.

**No se confirmó si son legibles** por un usuario sin privilegios bajo `paranoid=2` (los eventos uncore son de alcance sistema/socket, la misma categoría que en felix bloqueó P4). Es plausible que estén bloqueados igual que en felix, pero no se puede afirmar sin una prueba directa (análoga a la que se hizo para PID+inherit, pero apuntando a un PMU uncore). Esto queda como el ítem más importante para revisitar la validación cruzada de `bytes_moved_window` (ver `Informe_Diagnostico_Beta_2026-08-05.md`, sección 6, sobre el sesgo de ~30-34% nunca validado).

**LIKWID 5.2.2 está disponible como módulo** (`module load likwid/5.2.2`) y `likwid-perfctr` corre y lista grupos de eventos sin error. Esto es una vía potencialmente más simple que instrumentar `perf` manualmente para probar acceso a uncore, aunque LIKWID normalmente necesita acceso a `/dev/cpu/*/msr` (ver sección 7) o su propio daemon `likwid-accessD` (no verificado si está desplegado con setuid en este nodo).

---

## 7. MSR / CPUID

`/dev/cpu/0/msr` y `/dev/cpu/0/cpuid` son accesibles solo por `root` (mismo patrón que felix). Esto bloquearía el acceso directo a MSR que algunas rutas de LIKWID requieren, salvo que exista un daemon `likwid-accessD` con setuid configurado por el administrador (no confirmado).

---

## 8. cgroups y aislamiento de CPUs del job

- **cgroup v1** (no v2 como felix), con controladores montados por separado (`cpu,cpuacct`, `cpuset`, `memory`, `devices`, `pids`, `freezer`, `perf_event`, etc., cada uno su propio punto de montaje — el patrón clásico de v1).
- La ruta de cgroup del job **no es uniforme entre controladores**: el controlador `freezer` sí muestra un path con scoping de job (`/slurm_paccaA100/uid_9999/job_4616/step_0`), pero otros (`cpuset`, `cpu`, `memory`, `devices`, `pids`) muestran `/system.slice/slurmd.service` o `/` — es decir, **el plugin de cgroups de Slurm en este clúster no delega un cgroup por-job en todos los controladores**, a diferencia de la jerarquía unificada y consistente de felix (cgroup v2).
- **Esto no es un problema práctico para nuestro caso:** se confirmó directamente vía `/proc/self/status` que `Cpus_allowed_list: 0-31` — es decir, con `--exclusive --ntasks=1` el job tiene acceso a las 32 CPUs lógicas del nodo completo. El aislamiento real lo da `--exclusive` a nivel de Slurm (nadie más puede correr en el nodo simultáneamente), no una restricción de `cpuset` fina. No se necesita ningún ajuste al pipeline por esto.

---

## 9. `/scratch` y `$HOME`

- `/scratch`: **no escribible** para el usuario (mismo patrón que felix), montado por NFS (`192.168.153.132`), 1.9 TB libres reportados a nivel de filesystem (irrelevante mientras no sea escribible).
- `$HOME`: NFS-montado, ~1.6 TB libres de 8 TB (81% usado a nivel de filesystem compartido — no es la cuota individual del usuario, es el estado global del volumen). Suficiente para el volumen de datos que produce el pipeline (comparado con felix, cuyo `$HOME` tiene ~20GB de cuota real por usuario — **aquí no se verificó la cuota individual**, solo el filesystem; confirmar antes de campañas grandes).

---

## 10. Toolchain y módulos

- `gcc`/`gfortran` 12.4.0 (sistema, sin necesidad de módulo — funcionalmente equivalente a felix aunque de versión distinta).
- `cmake` 3.24.2.
- Módulos relevantes disponibles: `openmpi4`, `mvapich2`, `mpich`, `openblas/0.3.21`, `likwid/5.2.2`, `cuda` (no confirmado el número de versión exacto en la lista filtrada).
- No se confirmó si `openblas/0.3.21` incluye headers de desarrollo (`-devel`) necesarios para compilar el kernel DGEMM del catálogo — a probar en el primer intento de compilación real.

---

## 11. Qué bloquea hoy ejecutar el pipeline real aquí

No hay bloqueadores de arquitectura. Los bloqueadores son puramente de permisos/pendientes-de-probar, en orden de urgencia:

1. **Escritura de frecuencia no otorgada** (`scaling_min_freq` no escribible). Sin esto no hay matriz DVFS multi-frecuencia — el mismo bloqueador que frenó felix (allí nunca llegó el permiso H1). Se necesita pedir al administrador de HPC Unicartagena acceso de escritura a `scaling_min_freq`/`scaling_max_freq` para el usuario del proyecto, o confirmar si hay un mecanismo de "reserva con control de frecuencia" en este Slurm.
2. **Repo del proyecto aún no clonado en este clúster** — todo lo probado fue con scripts sueltos, no con el checkout real. Siguiente paso mecánico: `git clone` de `hyperion` en `$HOME`, replicar la convención de `hyperion-kernels/`/`hyperion-results/` usada en felix, compilar kernels NPB/DGEMM, y correr `preflight.py` real (no solo la auditoría manual) para que los checks E01-E10 confirmen formalmente lo que esta auditoría ya adelantó a mano.
3. **Validación de uncore no probada** — no crítico para arrancar (el pipeline no depende de uncore para funcionar, solo para la validación cruzada de `bytes_moved_window` pendiente desde F3.4/ARC-33). Vale la pena una prueba dedicada (mismo patrón que la de PID+inherit, sección 5) antes de invertir tiempo en LIKWID.
4. **Cuota real de `$HOME` no confirmada** individualmente (solo se vio el estado global del filesystem).
5. **`openblas-devel` (headers)** no confirmado — verificar en la primera compilación de DGEMM.

Ninguno de estos requiere cambios de diseño en el orquestador: el código ya es agnóstico de nodo (`environment.py` detecta `scaling_driver` y elige la estrategia de `freqctl.py` correspondiente; el harness C++ no cambia). Es trabajo operativo (pedir permisos, clonar, compilar, correr preflight), no de ingeniería.

---

## 12. Próximos pasos concretos, en orden

1. Clonar el repo en `$HOME` de `pacca`/`paccaA100` y crear `hyperion-kernels/`, `hyperion-results/` siguiendo la misma convención que felix (ver memoria `sc3-cluster-felix-facts`).
2. Compilar el catálogo de kernels (NPB ×6 + DGEMM) y correr `preflight.py` real contra el nodo — confirmará formalmente E01-E10 en vez de la inspección manual de esta auditoría.
3. Enviar una solicitud de permisos al administrador de HPC Unicartagena pidiendo escritura de `scaling_min_freq`/`scaling_max_freq` (o el mecanismo equivalente para `intel_pstate`) — usar `Solicitud_Permisos_SC3.md` como plantilla, adaptando el lenguaje a `intel_pstate` en vez de `acpi-cpufreq`.
4. Escribir y correr una prueba empírica de acceso a un PMU uncore (mismo patrón que la prueba de PID+inherit de la sección 5) para saber si la validación cruzada de `bytes_moved_window` es viable aquí sin pedir permisos adicionales.
5. Correr una campaña piloto mínima (REF, un solo estado de frecuencia, pocos kernels) para confirmar el pipeline end-to-end antes de comprometerse a una matriz completa.
