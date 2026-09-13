# AGENTS.md

Contexto de orientación para cualquier agente de IA que trabaje en este repositorio, enfocado en la **operación de campañas en el clúster pacca** (Unicartagena). Para el estado del alcance del proyecto (objetivos, fases, decisiones metodológicas), la fuente de verdad es `Plan_Detallado_Realineacion_Hyperion.md` y `Seguimiento_Cambios_Plan_Director.md`, no este archivo. Para una descripción de la arquitectura del código (harness C++, orquestador Python), ver `old/AGENTS.md`, con la salvedad de que ese documento está desactualizado (fechado 2026-08-14) y debe verificarse contra el código real antes de confiar en cualquier detalle técnico que describa.

## Operación en pacca: cuenta compartida (`latorresn`)

La cuenta `latorresn` en pacca es compartida con al menos otros dos proyectos de tesis ajenos a este. Reglas no negociables al someter trabajo a Slurm:

- **Nunca usar `--time` en ningún `sbatch`/`srun`.** La partición `GPU` tiene `MaxTime=UNLIMITED`; un timeout propio solo sirve para que Slurm mate el job por reloj y otro usuario tome el nodo en el acto (ya pasó, job 7046). El control de cuánto dura algo es siempre manual, con `scancel`.
- **Todo `--job-name` empieza con `hyp_`.** Es la única forma de distinguir nuestros jobs de los de los otros dos proyectos en `squeue`/`sacct` de un vistazo.
- **Nunca cancelar jobs ajenos.** Investigar/diagnosticar jobs de otros usuarios (duración, causa de fallo) es aceptable y a veces necesario, pero es de solo lectura; nunca `scancel` sobre algo que no sometimos nosotros.
- **La cola es FIFO puro, sin preempción** (`PriorityWeightAge/FairShare/JobSize/QOS` todos en 0 — verificado con `sacctmgr show assoc/qos`, sin límites ocultos de tiempo/jobs en la cuenta). Un job en `PENDING (Dependency)` no reserva puesto frente a jobs ajenos que se sometan mientras la dependencia sigue sin resolverse: si una cadena `--dependency=afterok` se rompe (una etapa falla) y hay que reencolar, el nuevo job queda detrás de cualquier cosa que haya llegado mientras tanto. Ver el patrón "job contenedor" abajo, que es la mitigación vigente para esto.

## Patrón vigente para campañas largas: job contenedor ("holder")

Desde 2026-09-13, las campañas finales (CPU/GPU) no se someten como jobs `sbatch` separados encadenados con `afterok`. Se usa un único job contenedor (`scripts/pacca/final_campaign/hyp_final_holder.sbatch`) que:

- Toma el nodo en exclusiva una sola vez, sin `--time`.
- Corre las campañas como pasos internos del propio script (no como jobs de Slurm nuevos); si una falla, el contenedor **no se suelta**, queda vivo para diagnosticar y reintentar a mano.
- Al terminar, se queda indefinidamente en un loop de espera hasta recibir una señal explícita de liberación (`touch .../HOLDER_STOP` o `scancel`).
- Admite adjuntar una shell interactiva a la misma asignación en cualquier momento, sin volver a pasar por la cola: `srun --jobid=<id> --pty bash`.

Esto elimina el riesgo de reencolado detrás de jobs ajenos, pero introduce un riesgo distinto (ver siguiente sección).

## Regla dura: shells interactivas adjuntas NO deben tocar CPU/GPU mientras hay una medición activa

`srun --jobid=<id> --pty bash` corre dentro de la misma asignación que el harness de telemetría, y eso puede contaminar silenciosamente los datos que se están midiendo en ese instante:

- **RAPL de paquete (`rapl_pkg_total_delta_uj`/dram) mide energía de TODO el paquete de CPU, no solo de los cores reservados por el kernel bajo medición.** El *pinning* de cores del kernel (`sched_setaffinity`, ver `telemetry_kernel_launcher.cpp`) aísla contención de scheduler/caché, pero **no** aísla la lectura de energía: cualquier proceso que corra en el nodo durante la ventana de medición, sin importar en qué core caiga, se suma a esa lectura.
- **La GPU es un solo dispositivo compartido.** Cualquier trabajo de GPU lanzado desde una shell adjunta (un kernel de prueba, una compilación con `nvcc` que dispare trabajo en el device, etc.) aparece directamente en las muestras de `gpu_util_pct`/`gpu_power_mw`/`gpu_sm_clock_mhz` de la corrida activa.
- **Nada en el pipeline detecta esto automáticamente.** Las validaciones actuales (p. ej. D03, verificación de lock de frecuencia) comprueban que el reloj esté donde debe estar, no que no haya habido interferencia externa de energía/GPU. Una corrida contaminada así se acepta como válida sin ninguna alerta.

**Regla práctica:** mientras el status del contenedor indique una campaña activa, una shell adjunta se usa solo para inspección (leer logs, `cat` de status, `squeue`, editar un YAML que todavía no se leyó) — nunca para correr nada que use CPU real o GPU. Si hay que intervenir de verdad (matar un proceso colgado, por ejemplo), hacerlo, pero después verificar a mano si la corrida que estaba en curso en ese momento quedó aceptada y, si hay duda, forzar que se repita (el resume de campaña —CAM-03/CAM-11— solo salta las corridas ya aceptadas, así que basta con invalidarla para que se vuelva a medir).
