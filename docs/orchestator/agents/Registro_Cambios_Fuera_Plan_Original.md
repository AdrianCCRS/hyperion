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
