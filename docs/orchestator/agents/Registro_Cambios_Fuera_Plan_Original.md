# Registro de cambios fuera del plan original

## Propósito y regla de mantenimiento

Este documento es el registro normativo de decisiones que amplían, corrigen o
aclaran el plan original del orquestador. Evita que las condiciones reales del
nodo, los compromisos de despliegue y los contratos ya implementados queden
dispersos entre conversaciones, commits y ejemplos.

**Regla desde 2026-07-13:** antes de implementar un cambio que no esté descrito
en la Guía Técnica, el Checklist o el Plan de Tests, se añadirá aquí una entrada
con motivación, impacto, módulos afectados, estado y pruebas necesarias. Este
registro no sustituye los documentos base: los complementa y prevalece cuando
describe una adaptación explícita de plataforma.

## Estado de referencia

- Plan base: `Guia_Tecnica_Orquestador_Campanas.md`,
  `Checklist_Validaciones_Tecnicas_Orquestador.md` y
  `Plan_Tests_Orquestador.md`.
- Plataforma operativa actual confirmada: SC3, nodo `felix.sc3.uis.edu.co`.
- `smexa` y `exadell` están descartados para las pruebas actuales porque no
  cumplen el requisito de GPU NVIDIA. Sus diagnósticos se conservan únicamente
  como evidencia comparativa de infraestructura.
- Entorno de ejecución: Slurm + cgroup v2 + Conda con Python 3.11+.
- Política de seguridad: la detección y el diagnóstico son de solo lectura;
  ningún diagnóstico escribe sysfs ni requiere root.

## Handoff para el modelo que mantiene los documentos base

Este apartado permite reconciliar la documentación original con el código y la
infraestructura reales. Leerlo antes de modificar la Guía, Checklist, Plan de
Tests o de proponer módulos nuevos.

### Estado del proyecto y fuente de verdad

1. El plan original sigue definiendo la arquitectura final, pero **no describe
   completamente las restricciones observadas en SC3**. Las enmiendas ARC de
   este archivo son obligatorias para cualquier revisión del plan.
2. Los módulos con contrato ya implementado y pruebas unitarias son
   `catalog.py`, `config.py`, `environment.py`, `manifest.py`, `preflight.py`,
   `node_profile.py` y `diagnostics.py`. No declarar listo el pipeline completo:
   `freqctl.py`, calibración, runner, campaign, metadata, postproceso y GPU aún
   requieren integración completa con los contratos actualizados.
3. La rama de despliegue es `hpc-startup-diagnostic`. Hitos relevantes:
   `9dcd733` añadió diagnóstico/Conda/plantilla SC3 y `dc1047f` adaptó
   environment/preflight a las capacidades reales. La suite local asociada
   terminó con 69 pruebas aprobadas; una campaña real todavía necesita pruebas
   de hardware y de caos.
4. `startup_diagnostic.json` confirma carga estructural de manifest y catálogo,
   topología, RAPL, perf y cpuset. No reemplaza C01/C02, preflight de campaña,
   calibración ni una corrida del harness.
5. Para conflictos entre documentación y nodo real, prevalecen: (a) el cpuset
   efectivo del job, (b) `EnvironmentProfile`, (c) el preflight y (d) este
   registro. No inferir permisos ni control DVFS desde el modelo de CPU.

### Contratos vigentes que la documentación debe preservar

- `environment.py` es la única autoridad de detección. Otros módulos consumen
  su perfil; no reimplementan sondeos sysfs.
- `freq_control_capable` indica soporte de driver/niveles, mientras
  `frequency_write_capable` indica autoridad efectiva del usuario. Ambos son
  necesarios para producir datos de entrenamiento DVFS.
- `frequency_control_strategy` puede ser `discrete_bounds`, `bounded_range` o
  `unavailable`. `freqctl.py` debe usar esta estrategia y nunca asumir
  `scaling_setspeed`.
- `frequency_control_paths` contiene los atributos reales por CPU/policy que
  E09 y `freqctl.py` pueden comprobar. No hardcodear rutas en nuevos módulos.
- `rapl_domains_available` usa alias únicos: `package-0`, `core-package-0`,
  `package-1`, `core-package-1`. `rapl_domain_paths` conserva la ruta sysfs.
  Los manifests y artefactos futuros deben usar esos alias, no `dram` ni `core`
  ambiguo.
- MAN-10 continúa exigiendo exactamente un nivel `native_governor`, denominado
  actualmente `REF`; los niveles `fixed` son los únicos que activan E07/E09.
- Una campaña nativa puede validar mecánica y calibración, pero si no hay
  control de frecuencia debe registrar `not_eligible_for_training_dataset=true`.
- El manifest de auditoría SC3 solo es una sonda. Un manifest real se materializa
  dentro de la asignación Slurm con CPUs y cgroup de workload reales.

### Decisiones que no deben presentarse como resueltas

- `smexa` y `exadell` no son destinos operativos mientras el requisito de GPU
  NVIDIA siga vigente, independientemente de sus capacidades CPU/RAPL.
- Un cgroup de Slurm que contiene al orquestador no satisface E03. Se necesita
  un hijo vacío para el workload; no relajar E03 para ocultar esa diferencia.
- No adaptar GPU AMD MI210 a checks NVIDIA mediante nombres ficticios. La ruta
  GPU futura se valida exclusivamente contra la GPU NVIDIA que Slurm asigne en
  `felix`.
- Temperatura de core, presupuesto Slurm, bytes proyectados, PMCs y binarios no
  pueden aprobarse por omisión. Deben obtenerse de una fuente real o bloquear.

### Próxima revisión documental requerida

Cuando se implemente `freqctl.py` y la integración Slurm, actualizar de forma
coherente la Guía, Checklist y Plan de Tests con ARC-11 a ARC-23. Incluir tests
de sysfs mockeado para cada estrategia y un protocolo de validación en hardware
que cubra permisos, restauración tras SIGINT y cgroup hijo. No cambiar la
semántica de F0/REF sin resolver ARC-15 en los cuatro documentos y en código.

## Cambios incorporados al diseño

| ID | Cambio o decisión | Motivo | Módulos / artefactos | Estado |
|---|---|---|---|---|
| ARC-01 | Configuración externa en `orchestrator.toml`. | Separar rutas sysfs, flags del harness y detección de tier del código y de secretos. | `config.py`, `environment.py`, `catalog.py`, `preflight.py`. | Implementado. |
| ARC-02 | Capa sysfs inyectable para pruebas. | Probar detección y preflight sin hardware ni rutas `/sys` reales. | `SysfsPaths`, argumentos `config` o rutas inyectables. | Implementado parcialmente; completar todos los checks nuevos. |
| ARC-03 | `environment.py` es la fuente única de capacidades. | Evitar detección duplicada y decisiones contradictorias entre módulos. | `environment.py`, metadata y preflight. | Implementado; se refuerza en ARC-12. |
| ARC-04 | Manifest exige `smt_policy` explícita. | El plan exige registrar la política SMT en metadata, no solo en logs. | `manifest.py`, `environment.py`. | Implementado. |
| ARC-05 | Diagnóstico de arranque HPC. | Verificar carga de manifest/catálogo y recopilar contexto real antes de ejecutar kernels. | `diagnostics.py`, `environment-hpc.yml`, plantilla SC3, `HPC_STARTUP.md`. | Implementado. |
| ARC-06 | El diagnóstico usa los CPUs efectivos del proceso mediante `--use-allowed-cpus`. | Los CPUs declarados en un ejemplo no representan necesariamente el cpuset concedido por Slurm. | `diagnostics.py`. | Implementado. |
| ARC-07 | Entorno Conda reproducible con Python 3.11. | SC3 ofrece Python 3.9 nativo; el orquestador requiere Python 3.11+. | `environment-hpc.yml`. | Implementado y comprobado en SC3. |
| ARC-08 | Manifest de auditoría separado de manifest de campaña. | La auditoría no ejecuta kernels ni posee cgroup de carga real. | `campaign_sc3_audit.yaml`. | Implementado. |
| ARC-09 | Preflight reducido acepta rutas/lectores inyectables. | Mantener pruebas unitarias sin tocar host y evitar rutas mágicas. | `preflight.py`, tests. | Implementado. |
| ARC-10 | Se distingue auditoría de campaña. | Que manifest y catálogo carguen no prueba C01/C02 ni habilita la ejecución. | Flujo de despliegue y documentación. | Implementado como política. |

## Hechos observados en SC3

### Perfil operativo común del clúster SC3

Las pruebas finales se ejecutarán mediante **Slurm 24.11.5** en nodos Rocky
Linux 9.7, kernel `5.14.0-611.36.1.el9_7.x86_64`, arquitectura `x86_64` y
cgroup v2 gestionado por `proctrack/cgroup`. Esta información procede de jobs
interactivos reales, no de una máquina local.

- El Python del sistema es 3.9; el orquestador usa Conda `hyperion-hpc`, con
  Python 3.11.15 y PyYAML.
- El cpuset efectivo de `os.sched_getaffinity(0)` es la única fuente autorizada
  para los CPUs de campaña. No sustituirlo por la topología global ni por
  `SLURM_CPUS_ON_NODE`.
- La ruta de cgroup cambia por job y step, por ejemplo
  `/sys/fs/cgroup/system.slice/<nodo>_slurmstepd.scope/job_<id>/step_<id>/user/task_0`.
  Una campaña real requiere un cgroup hijo de workload; el manifest de auditoría
  no es ejecutable como campaña.
- `perf` expone eventos genéricos de CPU. RAPL usa la jerarquía de kernel
  `intel-rapl` aunque los nodos sean AMD; el nombre de la ruta no identifica al
  fabricante.
- Hay GPUs AMD MI210 visibles. Visibilidad DRM no implica asignación exclusiva;
  los nodos `smexa` y `exadell` quedan descartados como destino final porque el
  objetivo requiere GPUs NVIDIA. GPU queda deshabilitada para estas pruebas de
  CPU; no invertir trabajo en ROCm para el pipeline final sin una decisión nueva.

### Aclaración normativa: cgroups de Slurm, orquestador y workload

El campo histórico `cgroup_path` no puede interpretarse como una ruta estática
del manifest. En SC3, Slurm crea una jerarquía dinámica por job y step. El
orquestador vive en el cgroup de su step, por lo que ese cgroup contiene al
menos el propio proceso y **no satisface E03** si E03 significa "cgroup de
workload vacío antes de iniciar la corrida".

La jerarquía requerida para una campaña real es:

```text
cgroup de la step Slurm (padre, contiene al orquestador)
└── cgroup hijo de campaña/workload (delegado al orquestador)
    └── cgroup hijo por run_id, opcional pero recomendado
        └── proceso del kernel y sus descendientes
```

Contratos actualizados:

- **E03 (campaña):** antes de la primera corrida, el cgroup hijo de workload
  debe existir, pertenecer a la asignación y tener `cgroup.procs` vacío. No se
  consulta `cgroup.procs` del cgroup padre de la step.
- **E06 (por corrida):** antes de mover/lanzar el kernel, el cgroup del
  `run_id` debe estar vacío; después de la corrida y antes de la siguiente,
  debe volver a estar vacío. Los únicos PIDs permitidos durante la corrida son
  el proceso del kernel y sus descendientes trazables.
- **Manifest:** declara una política o nombre lógico de cgroup, no una ruta
  absoluta de job. La ruta efectiva se resuelve después de entrar a la step
  Slurm y se guarda en metadata/reportes para reproducibilidad.
- **Seguridad:** el orquestador no debe crear ni administrar cgroups fuera de
  su subárbol delegado. Nunca debe mover PIDs ajenos ni modificar el cgroup
  padre de Slurm.

La implementación depende de que SC3 delegue controladores cgroup v2 al usuario
o suministre un helper Slurm restringido. La petición al administrador debe
incluir: crear un subárbol hijo por job, delegar `cgroup.procs`, `cpuset` y los
controladores estrictamente necesarios, y limitar cualquier helper a los CPUs y
PIDs de la asignación actual. Sin esa delegación, el pipeline puede observar el
cgroup de la step, pero no cumplir E03/E06 de aislamiento de workload.

### Resultados del diagnóstico actualizado

La rama `hpc-startup-diagnostic` añadió descubrimiento RAPL recursivo, consulta
de `policyN`, rutas cpufreq y separación entre soporte y permiso efectivo.

| Nodo | Afinidad observada | Frecuencia | RAPL | Decisión actual |
|---|---|---|---|---|
| `smexa.sc3.uis.edu.co` | AMD EPYC 9534; CPUs `0-3`, NUMA 0, siblings `128-131` fuera de la job. | `acpi-cpufreq`; 1500/1900/2450 MHz; rutas governor/min/max; escritura denegada. | Package y core para ambos sockets. | **Descartado:** GPU AMD, no NVIDIA. Evidencia histórica. |
| `exadell.sc3.uis.edu.co` | CPUs `30-33`, NUMA 0, siblings `158-161` fuera de la job. | No expone `cpuN/cpufreq` ni `policyN`; estrategia no disponible. | Package y core para ambos sockets. | **Descartado:** GPU AMD, no NVIDIA. Evidencia histórica. |
| `felix.sc3.uis.edu.co` | CPUs `0-3`, NUMA 0, siblings `32-35` fuera de la job; nodo con 4 NUMA. | `acpi-cpufreq`; diez niveles 1064–2261 MHz; rutas visibles; escritura denegada. | No expone RAPL. | **Único destino operativo actual.** Requiere confirmar y solicitar GPU NVIDIA por Slurm. |

Manifest y catálogo cargan en ambos nodos, pero esto **no** verifica C01/C02:
los binarios reales deben desplegarse y sus checksums actualizarse antes de una
campaña.

### Estado de problemas de la versión anterior

| Problema | Resolución | Evidencia |
|---|---|---|
| RAPL solo detectaba primer nivel. | Búsqueda recursiva y alias únicos. | `core-package-0` y `core-package-1` en ambos nodos. |
| No distinguía soporte cpufreq de permiso. | Nuevos campos de soporte, permiso, estrategia y rutas. | `smexa` soporta pero no puede escribir; `exadell` no expone interfaz. |
| Solo se buscaba `cpuN/cpufreq`. | También se consulta `cpufreq/policyN`. | El resultado de `exadell` ya no es falso negativo por esa ruta. |
| Campaña nativa exigía userspace/permisos. | E07/E09 solo aplican a niveles `fixed`. | Auditoría `native_governor` es solo lectura. |
| Faltaba contexto Slurm/cgroup/cpuset. | El diagnóstico lo serializa. | `startup_diagnostic.json` incluye job, cgroup y CPUs efectivos. |

Las limitaciones activas requieren autorización cpufreq en `felix`, un cgroup
hijo, una fuente de energía alternativa a RAPL, GPU NVIDIA asignada por Slurm y
binarios reales; no son defectos de detección.

### Nodo `smexa`

**Estado:** descartado por no disponer de GPU NVIDIA para las pruebas actuales.

- AMD EPYC 9534, 2 sockets / 2 NUMA, 128 cores físicos y SMT x2.
- `acpi-cpufreq`, frecuencias discretas visibles: 1500, 1900 y 2450 MHz.
- La asignación de prueba concedió CPUs efectivos `0-3`; todos pertenecen a
  NUMA 0 y sus siblings SMT `128-131` quedaron fuera de la asignación.
- RAPL cambia entre lecturas y es utilizable para paquetes; el informe externo
  también mostró subdominios core.
- Perf expone los ocho eventos genéricos de CPU y el usuario puede usar perf de
  espacio de usuario.
- GPU AMD MI210 visible, pero su visibilidad no demuestra asignación exclusiva
  por Slurm.
- El usuario de campaña no tenía confirmados permisos de escritura cpufreq.
- Cgroup observado en cada job: ruta dinámica bajo
  `/sys/fs/cgroup/system.slice/smexa_slurmstepd.scope/job_<id>/step_<id>/...`.

### Nodo `exadell`

**Estado:** descartado por no disponer de GPU NVIDIA para las pruebas actuales.

- El diagnóstico no detectó una interfaz cpufreq por CPU ni por `policyN` para
  la job. En el diseño actual, no existe una interfaz estándar que el
  orquestador pueda controlar desde esa asignación.
- No se debe extrapolar la capacidad de frecuencia de `smexa` a `exadell`.

### Nodo `felix`

**Estado:** único nodo operativo actual. Todo manifest y preflight debe partir
de su perfil, no de las capacidades históricas de `smexa` o `exadell`.

- Cuatro nodos NUMA; la asignación observada `0-3` está correctamente contenida
  en NUMA 0 y no incluye los siblings SMT `32-35`.
- Expone `acpi-cpufreq` con diez niveles discretos entre 1064 y 2261 MHz y las
  rutas de governor/límites, pero el usuario no puede escribirlas.
- `rapl_capable=false` y no existen dominios de energía. Por tanto, aun con una
  futura delegación cpufreq, el pipeline actual no puede calcular energía ni EDP
  en este nodo. No inventar ni imputar energía.
- Perf expone eventos genéricos más `mem-loads`; sirve para validar telemetría,
  afinidad, NUMA, Roofline y el flujo de artefactos en modo nativo.

## Enmiendas pendientes al plan y checklist

Las siguientes decisiones fueron motivadas por los resultados observados y se
deben implementar antes de declarar listo el preflight para campañas reales.

| ID | Enmienda | Regla afectada | Estado requerido |
|---|---|---|---|
| ARC-11 | Descubrir RAPL recursivamente y conservar un identificador único por dominio, junto con su ruta. | ENV-03, ENV-08, I05, I08. | Implementado; falta incorporar `max_energy_range_uj` por dominio en metadata. |
| ARC-12 | Separar capacidad de niveles, permiso efectivo y estrategia de frecuencia: `frequency_levels_supported`, `frequency_write_capable` y `frequency_control_strategy`. `freq_control_capable` no debe ocultar esa distinción. | ENV-02, ENV-05, E09, FRQ-06. | Implementado en `EnvironmentProfile`; falta que `freqctl.py` consuma la estrategia. |
| ARC-13 | Soportar políticas cpufreq: interfaz por CPU y por `policyN`; `acpi-cpufreq` debe usar los atributos que realmente controle, no asumir `scaling_setspeed`. | E07, E09, FRQ-02, FRQ-03. | Implementado para descubrimiento y E09; pendiente de aplicación en `freqctl.py`. |
| ARC-14 | Aplicar E07/E09 solo cuando el manifest solicite algún nivel `fixed`. Una campaña solo `native_governor` (F0 nativo) es de lectura y no exige `userspace`. | E07, E09, FRQ-06. | Implementado. |
| ARC-15 | Normalizar la nomenclatura: `REF` es el único nivel nativo según MAN-10; `F0...Fn` son niveles fijos. Si se adopta F0 nativo, se actualizarán conjuntamente MAN-10, ejemplos, metadata y tests. | MAN-10, FRQ-07, CAM-*. | Decisión pendiente antes de campañas. |
| ARC-16 | Validar el cpuset efectivo de Slurm contra todos los CPUs de campaña (delegados, collector y consumer). | E04, E05, nueva validación Slurm. | Pendiente. |
| ARC-17 | Usar un cgroup hijo de workload, vacío antes de cada corrida; no usar el cgroup de la step que contiene al orquestador. Resolver su ruta en tiempo de job. | MAN-01, E03, E06. | Pendiente; contrato de jerarquía definido y requiere acuerdo con administración SC3. |
| ARC-18 | Hacer E01 agnóstico de fabricante: Intel Turbo/HWP y AMD CPB/CPPC. Si no hay interfaz legible/controlable, registrar el estado y degradar la campaña a nativa en vez de afirmar control fijo. | E01, FRQ-01, FRQ-07. | Implementado para lectura/snapshot; pendiente la política de degradación en `freqctl.py`. |
| ARC-19 | Temperatura y procesos ajenos deben ser observados, no campos declarativos del manifest. Temperatura no disponible es advertencia; procesos ajenos se inspeccionan en cgroup/afinidad reales. | E02, E06. | Pendiente. |
| ARC-20 | No aprobar D05, I09 ni OPS-01 por datos ausentes. La capacidad PMC, proyección de bytes y presupuesto deben estar presentes o marcar el check como bloqueante/no verificable según tier. | D05, I09, OPS-01. | Implementado; falta alimentar los valores desde el planificador y perfil de nodo. |
| ARC-21 | Normalizar E08 por CPUs efectivos/asignados y, cuando sea posible, medir el cgroup de campaña. | E08. | Implementado para CPUs delegados; pendiente medición específica del cgroup. |
| ARC-22 | Verificar GPU NVIDIA asignada por Slurm e implementar un inspector NVIDIA compatible. No usar ROCm ni los nodos AMD descartados mientras el requisito NVIDIA siga vigente. | G01, G02, G03. | Pendiente. |
| ARC-23 | C01/C02 se consideran verificados solo después de compilar/desplegar binarios en el nodo y actualizar checksums en catálogo. | CAT-01, CAT-02, C01, C02. | Pendiente operativo. |
| ARC-24 (2026-07-30) | F1.4 pide un mini INT-T11 local comparando `instructions` del launcher contra `perf stat -e instructions`; la máquina de desarrollo local no tiene el CLI `perf` instalado. Sustituto usado: comparar el total de `instructions` que reporta el launcher en modo `--exec` sobre un binario determinista (busy-loop de conteo fijo) contra una apertura independiente de `PerfReader` sobre el propio proceso (pid=0, mismo mecanismo PID+inherit) que ejecuta el mismo cómputo. Diferencia observada ≈0.05% (<5%, gate F1.4 cumplido). El INT-T11 real contra `perf stat` del sistema queda para F4.3 en felix, donde `perf 5.14` está confirmado disponible (Parte 0 del plan). | CPP-08, F1.4, F4.3 (INT-T11 real). | Implementado como sustituto local; INT-T11 real pendiente en felix. |
| ARC-25 (2026-07-30) | `runner.py` (F2.1) necesita saber dónde está el binario `telemetry_kernel_launcher` para construir el comando (RUN-01). `config.py` (ya "listo" según el plan) no tenía ese dato: se añadió `HarnessConfig.binary_path`, resuelto en `load_config()` relativo al propio `orchestrator.toml` si no es absoluto (mismo patrón que `manifest.load()` usa para `catalog_path`). Se agregó `harness.binary_path = "telemetry/build/telemetry_kernel_launcher"` a `orchestrator.toml`. Se actualizó el fixture TOML de `test_environment.py` para incluir la clave nueva (campo obligatorio). También se añadió `psutil>=5.9` a `environment-hpc.yml`, usado únicamente por `tests/orchestrator/test_runner.py` para verificar RUN-04 (ningún proceso hijo vivo tras timeout); `runner.py` en sí no depende de `psutil` en producción. | RUN-01, F2.1, config.py. | Implementado. |
| ARC-26 (2026-07-30) | F2.3 (`calibration.py`) requirió varias decisiones de diseño no explícitas en la Guía Maestra ni en `docs/orchestator/plan_v3/guia-tecnica.md`: (1) `catalog.py.KernelEntry` no tenía dónde declarar el regex de extracción de BW/FLOPs pedido por CAL-02/CAL-03 ("regex configurables en el catálogo"); se agregaron `bandwidth_stdout_pattern`/`flops_stdout_pattern`, validados en `__post_init__` (CAT-05 extendido) cuando el flag `reports_*_stdout` correspondiente es `True`. (2) El chequeo D03/CAL-04 necesita una "ficha técnica declarada en el manifest" que `manifest.py` tampoco tenía: se agregó `Manifest.hardware_datasheet` (opcional, `bw_pico_bytes_per_s`/`p_pico_flops_per_s`); si está ausente, D03 se trata como **fallo**, nunca como aprobado por falta de datos (consistente con ARC-20). (3) `run_calibration()` no hace conversión de unidades: el número que captura el regex se usa tal cual como `bw_pico_bytes_per_s`/`p_pico_flops_per_s`; la responsabilidad de que el regex capture el valor ya en esas unidades (o de documentar la conversión) queda en quien escribe la entrada del catálogo. (4) `CalibrationReferences` (guia-tecnica.md) define un único `cv_pct`/`accepted` para cuatro métricas (IPC/IPS/MPKI/MissRate); se implementó como el **máximo** de los cuatro coeficientes de variación individuales (criterio conservador: cualquier métrica inestable descalifica la referencia) por no haber una definición más específica en ninguna de las guías. `build_node_profile()` también fusiona el `NodeProfile` de la guía técnica (topología/cache) con el `pmc_count` que ya consumía `preflight.py` (D05), en vez de crear un dataclass paralelo. | CAL-01..11, CAT-05, MAN-00. | Implementado; decisiones documentadas para revisión humana. |
| ARC-27 (2026-07-30, confirmado 2026-07-30) | F2.5 (`postprocess.py`) requirió varias decisiones no explícitas en ninguna guía: (1) **Divergencia resuelta, confirmada por el director:** `docs/orchestator/plan_v3/guia-tecnica.md` (línea ~710) prorratea `flops_window_estimate` por duración (`run_flops_total * delta_t_ns/run_duration_ns`); `docs/retoma/Plan_Implementacion_Medicion_SC3.md` (F2.5, el documento ejecutable subordinado a la Guía Maestra) pide prorratear por `delta_instructions`. La Guía Maestra autoritativa (`Guia_Maestra_Fase1_DVFS.md`) no especifica el método, solo que el origen debe ser el stdout del binario (POST-09). Se implementó y se confirmó el prorrateo por `delta_instructions`: los FLOPs son un subconjunto de las instrucciones retiradas, mientras que el tiempo de reloj está confundido por los stalls de memoria — exactamente el efecto que `phase_label_train` necesita discriminar entre fases compute/memory-bound. Prorratear por tiempo asignaría FLOPs proporcionalmente altos a ventanas memory-bound (que duran mucho pero retiran pocas instrucciones), sesgando la intensidad operacional en la dirección equivocada. `guia-tecnica.md` queda documentado como método descartado, no como pendiente. (2) Los 7 valores de `quality_status` no alcanzan para cubrir todas las condiciones de POST-01..10 simultáneamente en una sola columna; se definió una prioridad propia (`first_sample_no_delta` > `pmu_degraded` > `warmup_excluded` > `intensity_undefined` > `energy_invalid` > `no_freq_reading` > `ok`), documentada en el código. (3) `quality_status='energy_invalid'` solo se activa si el manifest declaró `rapl_enabled=True` para esa campaña; en felix (sin RAPL para siempre) nunca se activa por defecto, evitando que todas las ventanas queden marcadas como inválidas solo por la ausencia estructural de RAPL — `energy_valid` (columna aparte) sigue siendo `False` en cada fila igualmente. (4) Nuevos campos requeridos: `catalog.KernelEntry.flops_total_stdout_pattern` (opcional, regex para el total de FLOPs de kernels dataset) y `NodeProfile.cache_line_size_bytes` (leído de `coherency_line_size`, POST-10). (5) El emparejamiento de filas ENERGY con la ventana CPU correspondiente se hace por barrido de dos punteros sobre timestamp (ambas listas ordenadas), no por índice posicional, para tolerar lecturas RAPL fallidas ocasionales sin desalinear el resto de la corrida. | POST-01..16, CAL-02/03. | Implementado; decisión de ARC-27(1) confirmada por el director. |
| ARC-28 (2026-07-31) | Hallazgo al revisar si el pipeline recorre los niveles de frecuencia propuestos (F0/F1/F2/REF): `campaign.build_matrix()` sí itera sobre todos los `frequency_levels` del manifest y `freqctl.apply_frequency()` sí implementa y verifica ambas estrategias (discrete_bounds/bounded_range), pero el resultado (`AppliedFrequency.requested_khz`/`applied_khz`, exigido literalmente por FRQ-03) se calculaba y se descartaba: `runner.py` invocaba `apply_frequency(...)` sin guardar el retorno, y `campaign.py` no pasaba `freq_khz_requested`/`freq_khz_applied`/`freq_khz_observed` a `postprocess.run_postprocess()`. En felix hoy es inofensivo (`frequency_write_capable=False`, `apply_frequency` nunca se invoca), pero habría producido `windows.csv` con esas columnas vacías en cuanto H1 conceda escritura de cpufreq. Corregido: `RunResult` gana el campo `applied_frequency`; `runner._merge_metadata()` lo funde en la metadata.json de la corrida; `campaign.py` lo propaga a `run_postprocess()` junto con una lectura de `freqctl.read_observed_frequency_khz()` tomada una vez por corrida (no por ventana individual — el harness C++ no muestrea `scaling_cur_freq` por tick de perf, ver nota FRQ-10). | FRQ-03, FRQ-10, RUN-06, MET-07. | Corregido. |
| ARC-29 (2026-07-31) | Auditoría adicional de solo lectura en felix (`ssh guane` + `srun`), motivada por la pregunta de si el control de frecuencia podría afectar reservas de otros usuarios. Hallazgo crítico: `freqdomain_cpus` de cpu0 = `0-7,32-39` — el dominio de frecuencia de `acpi-cpufreq` en felix es **por socket completo** (8 cores físicos + sus 8 SMT), no por core individual. Una asignación Slurm que no cubra el socket completo (ej. la de 4 CPUs auditada el 2026-07-30, cpuset `0-1,32-33`, subconjunto estricto del dominio) compartiría el dominio de frecuencia con cualquier otro job en el resto del socket — fijar `scaling_setspeed` en esos cores cambiaría la frecuencia de cores ajenos también. No hay ningún check de preflight (E01-E09 actuales) que verifique esto; todos verifican contaminación de *procesos*, no de *dominio de frecuencia compartido*. Se propone (no implementado) un nuevo check de preflight que compare `delegated_cpus` contra el `freqdomain_cpus`/`related_cpus` real leído de sysfs, bloqueante si `delegated_cpus` es un subconjunto estricto del dominio en cualquier CPU delegado. Se actualizó H1 para pedir explícitamente que la asignación cubra el dominio completo. Hallazgos adicionales de la misma auditoría: `userspace` SÍ está en `scaling_available_governors` (H1 es netamente un tema de permisos); `/scratch` NO es escribible para el usuario (contradice la suposición de F4.1); `~/hyperion` en el clúster está anidado en `~/hyperion/hyperion/` (artefacto de rsync) y desactualizado (último commit sincronizado 2026-07-13, no incluye Fase 1/Fase 2 de esta sesión); GPU confirmada (`--gres=gpu:1`) resolviendo H5; topología de cache/cpuinfo real coincide exactamente con lo que `node_profile.py` ya calculaba contra mocks. Todo esto es específico de hardware Nehalem-EX (2010); un clúster más moderno con `intel_pstate`/HWP probablemente tiene dominios por-core, pero el check propuesto debe seguir siendo genérico (leer el dominio real de sysfs, no asumir "todo el socket"). | Preflight (E01-E09, check nuevo propuesto), H1, H5, F4.1. | Hallazgo documentado; check de preflight nuevo pendiente de implementar. |
| ARC-30 (2026-07-31) | Implementado el check de preflight propuesto en ARC-29. `environment.py` gana `_frequency_domain_data()`, que lee `freqdomain_cpus` (o, si no existe, `related_cpus`/`affected_cpus`, en ese orden — cubre tanto el hardware antiguo de felix como drivers modernos que puedan exponer solo uno de los tres) por cada CPU delegado y lo expone como `EnvironmentProfile.frequency_domain_cpus: dict[int, list[int]]` (atributo dinámico, mismo patrón que `numa_cpu_map`/`delegated_cpu_numa_nodes`); se incluye en `environment_report.json`. `preflight.py` gana `check_frequency_domain()` (factor_id **E10**, nueva regla — no exigida por ninguna guía anterior), bloqueante, que falla si el dominio real de algún CPU delegado excede el conjunto de `delegated_cpus`; se ejecuta en `run_campaign_preflight()` únicamente cuando la campaña solicita algún nivel `fixed` (mismo gate que E07/E09). Decisión de diseño: si no hay datos de dominio disponibles (archivo ausente, ej. drivers modernos sin esa exposición), el check pasa sin bloquear — la ausencia de evidencia no debe impedir campañas en clústeres donde el problema no aplica; solo bloquea cuando el dominio leído demuestra explícitamente una fuga. Cubierto por `test_env_t11/t12/t13` (`test_environment.py`) y `test_e10_*` (`test_preflight.py`), 3 tests nuevos cada uno. Checklist actualizado: PRE-E10 (nuevo) marcado ☑ en §12.4 de la Guía Maestra, contador de la sección actualizado de 25 a 26 reglas. | E10 (nuevo), environment.py, preflight.py, Guía Maestra §12.4. | Implementado. |
| ARC-31 (2026-07-31) | F3.1 pedía compilar ERT en felix, con la alternativa explícita (CAL-03) de "un microbenchmark de FLOPs pico de suite reconocida" si el despliegue resultaba pesado, a condición de registrar la decisión como enmienda ARC. Se inspeccionó el repositorio oficial de ERT (Berkeley Lab CS Roofline Toolkit, `bitbucket.org/berkeleylab/cs-roofline-toolkit`): el driver completo (v1.1.0) requiere plantillas `Batch/`+`Config/` especificas por máquina (existen para NERSC/ANL/ORNL, ninguna para SC3/felix) y un driver Python (`roofline.py`) que orquesta un barrido de muchos jobs y post-procesa los resultados fuera de línea — desplegarlo correctamente en un Slurm que no administramos, sin iteración interactiva rápida (cada intento de config cuesta un round-trip de `srun`), se consideró desproporcionado para obtener un solo número de FLOPs pico por calibración. Se tomó la alternativa aprobada: `kernels/ert/ert_probe.c`, un driver propio y autocontenido que reimplementa la aritmética real de ERT (`Kernels/kernel1.c` v1.0.0, `ERT_FLOP=16`: `beta = beta*A[i] + alpha` desenrollado 8 veces) sobre un buffer que cabe en cache, barriendo tamaño de working set y repeticiones, e imprimiendo directamente `GFLOPs/sec: <pico>` por stdout — sin MPI/BGQ/QPX/intrinsics AVX ni el post-proceso Python. Compilado y verificado en felix (8 cores, `--hint=nomultithread`): `GFLOPs/sec: 31.939` con working set de 25294 doubles/hilo. | CAL-03, F3.1, ert\_probe. | Implementado. |
| ARC-32 (2026-07-31) | F3.1-F3.3 ejecutados de punta a punta en felix (vía `srun --partition=gpu_titan --nodelist=felix --cpus-per-task=8 --hint=nomultithread`, cubriendo el socket completo por E10/ARC-30). (1) **NPB**: descargado `NPB3.4.4.tar.gz` de nas.nasa.gov (sha256 registrado en `catalog.yaml`), compilado con el módulo `gnu14` y `config/make.def.template` sin modificar (`FFLAGS`/`CFLAGS` = `-O3 -fopenmp` ya son el default del suite). (2) **Selección de clase**: clase S (todas <0.1s) y clase A (todas <1.5s salvo LU en 7.1s) quedan muy por debajo del objetivo aspiracional de ≥60s del plan F3.1; clase B se midió real: ep=5.9s, mg=2.6s, cg=21.7-22.8s, is=0.8-1.6s, ft=11.1-11.9s, lu=35.5-35.9s. Ninguna clase estándar llega a 60s de forma uniforme en las 6, y clase C llevaría el total de la matriz de campaña muy por encima del presupuesto razonable sin necesidad real: con muestreo a 1ms (`--interval-ns 1000000`, confirmado en Guía Maestra §10.4) el requisito duro es ≥50 ventanas útiles tras warmup, y clase B ya da entre ~700 (is) y ~35000 (lu) ventanas por corrida, 10-700× el mínimo. Se eligió **clase B para las 6 (dataset)**, dejando clase C como opción futura si el director (H4) pide series más densas. (3) **STREAM**: `STREAM_ARRAY_SIZE=64000000` (~512 MB/arreglo, muy por encima de los 24 MB de L3 por socket de felix, con holgura amplia para no depender de si la campaña termina siendo mono- o multi-socket). (4) **Regex de éxito NPB**: las 6 suites comparten la misma línea final de resumen `Verification    =               SUCCESSFUL` (de la rutina común `print_results`) — se usó ese patrón uniforme en vez del sugerido originalmente en el plan ("VERIFICATION SUCCESSFUL", que solo aparece en el cuerpo de MG/CG, con formato distinto en FT/LU/IS). (5) **POST-09/FLOPs**: confirmado que NPB nunca imprime un total absoluto de FLOPs, solo `Mop/s total` (tasa) y `Time in seconds`; se agregaron los campos `flops_rate_stdout_pattern`/`runtime_seconds_stdout_pattern` a `KernelEntry` y el fallback correspondiente (tasa × 1e6 × tiempo) en `postprocess.extract_run_flops_total()`, activo cuando `flops_total_stdout_pattern` está ausente — cambio menor, ya anticipado por el propio texto del plan F3.2 ("Mop/s total × tiempo"). `orchestrator/schemas/kernels/catalog.yaml` quedó con los 8 checksums reales (2 calibration + 6 dataset) y las 6 clases CAT-01..08 pasaron a ☑. | CAT-01..08, POST-09, F3.1, F3.2, F3.3, catalog.py, postprocess.py. | Implementado; F3.4 (validación de bytes bajo el launcher C++) queda pendiente porque requiere compilar el harness en felix (se solapa con F4.1). |

## Flujo HPC actualizado

1. Crear/activar el entorno Conda Python 3.11 y solicitar una asignación Slurm
   con un único NUMA y `--hint=nomultithread`.
2. Ejecutar el diagnóstico de solo lectura con `--use-allowed-cpus` y conservar
   `startup_diagnostic.json`.
3. Resolver el manifest de campaña en tiempo de job usando cpuset y cgroup hijo
   reales; validar catalog y binarios antes de tocar frecuencia.
4. Ejecutar el preflight de campaña completo. Si no existe permiso de control,
   permitir únicamente una campaña explícitamente nativa/F0-only, etiquetada
   como no apta para entrenamiento DVFS.
5. Ejecutar calibración y matriz solo cuando C01/C02, RAPL, cgroup, NUMA/SMT y
   los checks de frecuencia correspondientes hayan pasado.
6. En `felix`, solicitar y verificar la GPU NVIDIA asignada por Slurm antes de
   habilitar GPU; implementar el inspector NVIDIA correspondiente.

## Criterio de cambio futuro

Cada nueva entrada añadida a este archivo debe indicar:

1. ID `ARC-NN` y fecha.
2. Motivo y evidencia observada.
3. Reglas/checklists/flujo que modifica.
4. Módulos y pruebas afectados.
5. Estado: propuesto, implementado, verificado en hardware o descartado.
