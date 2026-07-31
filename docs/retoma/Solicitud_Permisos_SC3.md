# Solicitud de permisos a administración SC3 — justificación técnica

Este documento resume, para uso directo en el correo a administración de SC3,
todos los permisos que el proyecto necesita sobre el nodo `felix` (y el
clúster SC3 en general) para completar la medición DVFS descrita en
`Guia_Maestra_Fase1_DVFS.md` y `Plan_Implementacion_Medicion_SC3.md`. Cada
ítem indica qué se pide, por qué, y qué evidencia técnica ya se recolectó
(de solo lectura, sin haber modificado nada en el nodo) que respalda la
solicitud.

Estado del diagnóstico de base: auditorías de solo lectura ejecutadas vía
`srun` en `felix` el 2026-07-30 y el 2026-07-31 (ver `startup_diagnostic.json`
y las secciones "Ampliación" de `Plan_Implementacion_Medicion_SC3.md`).

---

## P1. Delegación de escritura sobre cpufreq en cores asignados

**Qué se pide:** permiso de escritura para el usuario sobre los archivos
sysfs de control de frecuencia (`scaling_governor`, `scaling_setspeed`) de
los cores CPU que Slurm delegue a nuestros jobs en `felix`. Hoy estos
archivos existen y el governor `userspace` está disponible en el sistema,
pero no son escribibles por el usuario (confirmado, no es un problema de
módulo de kernel faltante — es puramente de permisos).

**Por qué se necesita:** el proyecto mide el comportamiento
energía/rendimiento de cargas de cómputo en función de la frecuencia de CPU
(DVFS — Dynamic Voltage and Frequency Scaling). Sin poder fijar la
frecuencia, solo es posible medir en el punto operativo por defecto
(governor `performance`), lo cual reduce el dataset a un único nivel en vez
de la curva completa de frecuencias que el estudio requiere.

**Condición crítica que debe ir explícita en la solicitud:** en `felix`
(Xeon X7560 Nehalem-EX) el control de frecuencia con `acpi-cpufreq` opera
**por socket completo, no por core individual** — confirmado leyendo
`freqdomain_cpus` en sysfs: el dominio de cpu0 es `0-7,32-39` (los 8 cores
físicos del socket 0 más sus 8 hilos SMT). Esto significa que fijar la
frecuencia de un solo core delegado cambia la frecuencia de **todo el
socket**, incluyendo cores de otros jobs si la asignación Slurm no cubre el
socket completo. Ya detectamos un caso real: una asignación anterior de 4
CPUs sin `--hint=nomultithread` produjo el cpuset `0-1,32-33`, que es un
subconjunto estricto de ese dominio de 16 CPUs — ese escenario habría
compartido control de frecuencia con quien tuviera el resto del socket.

**Mitigación ya implementada de nuestro lado:** el pipeline de orquestación
ahora bloquea automáticamente (preflight check E10, bloqueante) cualquier
campaña cuya asignación de cores no cubra por completo el dominio de
frecuencia real leído de sysfs — es decir, nunca vamos a intentar escribir
frecuencia si eso arriesga afectar a otro usuario, aunque se conceda el
permiso. Aun así, para que esto funcione en la práctica se debe pedir a SC3
que garantice asignaciones de socket completo para nuestros jobs de
campaña DVFS (`--cpus-per-task=8 --hint=nomultithread` en la partición
`gpu_titan` produce exactamente el dominio completo `0-7,32-39`).

**Evidencia de respaldo:** `freqdomain_cpus`, `scaling_available_governors`
(incluye `userspace`), y el cpuset `0-1,32-33` de la asignación previa,
todos leídos de sysfs vía `srun` de solo lectura.

---

## P2. Medición de energía externa (PDU / IPMI del rack)

**Qué se pide:** acceso de lectura (o que SC3 nos entregue los datos) a
alguna fuente de medición de energía a nivel de nodo o rack — PDU
inteligente, IPMI/BMC del servidor, o cualquier telemetría de energía que
la infraestructura ya recolecte para `felix`.

**Por qué se necesita:** `felix` usa un Xeon X7560 (Nehalem-EX, 2010),
anterior a la introducción de RAPL (Running Average Power Limit) en Intel.
Confirmamos que `intel-rapl` **no existe físicamente** en este hardware —
no es un problema de permisos ni de kernel, es una limitación
irreversible del procesador. Sin RAPL ni una fuente externa de energía, el
proyecto no puede producir las columnas de energía/potencia
(`pkg_delta_uj`, `power_w`, EDP) del dataset, que son parte central del
objetivo de investigación (relación energía-rendimiento en función de la
frecuencia).

**Evidencia de respaldo:** ausencia confirmada de `/sys/class/powercap/intel-rapl*`
en `felix` durante la auditoría de solo lectura.

**Nota:** si SC3 no dispone de medición externa para `felix`, esto se
documenta como limitación de hardware y el dataset de esa fase queda sin
columnas de energía (features de rendimiento/IPC siguen siendo válidas).
No bloquea el resto del proyecto, pero vale la pena preguntarlo explícitamente
antes de asumir que no hay alternativa.

---

## P3. Espacio de trabajo escribible (subdirectorio en `/scratch` o cuota de `$HOME`)

**Qué se pide:** un subdirectorio propio y escribible dentro de `/scratch`
para el usuario, o confirmación de que la cuota de `$HOME` (actualmente
~20 GB, compartido por NFS) es suficiente para el volumen de datos
proyectado de la campaña completa (ventanas de perf + metadata por cada
combinación kernel × nivel de frecuencia × repetición).

**Por qué se necesita:** confirmamos que `/scratch` existe como punto de
montaje pero su directorio raíz es `root:root` con permisos `0755` — el
usuario no puede escribir ahí directamente (`Permission denied` al
intentar). El plan de despliegue (Fase 4) asumía tener `/scratch` disponible
como área de trabajo de alto volumen separada del `$HOME` compartido por
NFS; si esa cuenta no existe, el presupuesto de espacio debe recalcularse
contra la cuota real de `$HOME`.

**Evidencia de respaldo:** `touch /scratch/$USER-test` → `Permission
denied`; propietario/permisos del directorio raíz confirmados vía `stat`.

---

## Resueltos, no requieren solicitud

Para que quede claro en el correo qué **no** hace falta pedir:

- **GPU (Titan X):** confirmado accesible con `srun --gres=gpu:1` +
  `nvidia-smi` (GeForce GTX Titan X, driver 570.195.03, 12 GB). No requiere
  ningún permiso adicional, solo especificar `--gres=gpu:1` en los jobs que
  la usen.
- **Governor `userspace`:** ya está en `scaling_available_governors`; no
  hace falta que SC3 cargue ningún módulo de kernel, solo delegar el
  permiso de escritura (ver P1).
- **`perf` con eventos por PID:** funciona sin privilegios adicionales
  (`perf_event_paranoid=1`), ya verificado.

---

## Borrador de correo a administración SC3

> Asunto: Solicitud de permisos para proyecto de medición DVFS en felix (grupo [nombre del grupo/laboratorio])
>
> Estimados,
>
> Estamos desarrollando un proyecto de investigación que mide el
> comportamiento de energía y rendimiento de cargas de cómputo en función
> de la frecuencia de CPU (DVFS) sobre el nodo `felix`. Ya hicimos un
> diagnóstico completo de solo lectura del nodo (sin modificar nada) y
> necesitamos tres permisos puntuales para poder avanzar:
>
> **1. Escritura sobre control de frecuencia (cpufreq) en los cores que
> Slurm nos asigne en `felix`.** Confirmamos que el governor `userspace`
> ya está disponible en el sistema — solo falta el permiso de escritura
> sobre `scaling_governor`/`scaling_setspeed` para el usuario en los cores
> delegados a nuestros jobs.
>
> Un detalle importante que detectamos: en `felix` el control de
> frecuencia opera por socket completo (8 cores físicos + 8 hilos SMT
> comparten un mismo dominio de frecuencia), no por core individual. Para
> evitar cualquier riesgo de que nuestro control de frecuencia afecte a
> otro usuario, necesitamos que las asignaciones Slurm para nuestros jobs
> de esta campaña cubran siempre el socket completo (por ejemplo,
> `--cpus-per-task=8 --hint=nomultithread`). De nuestro lado ya
> implementamos una validación automática que bloquea cualquier ejecución
> si la asignación no cubre el dominio completo, así que el riesgo queda
> cubierto en ambos extremos.
>
> **2. Medición de energía externa para `felix` (PDU o IPMI/BMC), si está
> disponible.** El procesador de `felix` (Xeon X7560, 2010) es anterior a
> RAPL, así que no hay forma de leer energía desde el software del nodo.
> ¿Existe alguna fuente de telemetría de energía a nivel de rack/nodo a la
> que podamos acceder, o que ustedes puedan exportarnos?
>
> **3. Espacio de trabajo escribible.** Notamos que `/scratch` no es
> escribible para nuestro usuario. ¿Podrían habilitarnos un subdirectorio
> propio ahí, o confirmarnos que la cuota de `$HOME` (actualmente ~20 GB)
> es la vía recomendada para nuestro volumen de datos?
>
> Quedamos atentos y con gusto compartimos el diagnóstico técnico completo
> si es útil para evaluar la solicitud.
>
> Saludos,
> [nombre]
