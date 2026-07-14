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
- Plataforma objetivo confirmada: SC3, nodo `smexa.sc3.uis.edu.co`.
- Entorno de ejecución: Slurm + cgroup v2 + Conda con Python 3.11+.
- Política de seguridad: la detección y el diagnóstico son de solo lectura;
  ningún diagnóstico escribe sysfs ni requiere root.

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

### Nodo `smexa`

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

- El diagnóstico no detectó una interfaz cpufreq por CPU. Esto no prueba que
  sea incapaz de controlar frecuencia: puede exponer una jerarquía `policyN` o
  tener cpufreq restringido por la plataforma.
- No se debe extrapolar la capacidad de frecuencia de `smexa` a `exadell`.

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
| ARC-17 | Usar un cgroup hijo de workload, vacío antes de cada corrida; no usar el cgroup de la step que contiene al orquestador. Resolver su ruta en tiempo de job. | MAN-01, E03, E06. | Pendiente; requiere acuerdo con administración SC3. |
| ARC-18 | Hacer E01 agnóstico de fabricante: Intel Turbo/HWP y AMD CPB/CPPC. Si no hay interfaz legible/controlable, registrar el estado y degradar la campaña a nativa en vez de afirmar control fijo. | E01, FRQ-01, FRQ-07. | Implementado para lectura/snapshot; pendiente la política de degradación en `freqctl.py`. |
| ARC-19 | Temperatura y procesos ajenos deben ser observados, no campos declarativos del manifest. Temperatura no disponible es advertencia; procesos ajenos se inspeccionan en cgroup/afinidad reales. | E02, E06. | Pendiente. |
| ARC-20 | No aprobar D05, I09 ni OPS-01 por datos ausentes. La capacidad PMC, proyección de bytes y presupuesto deben estar presentes o marcar el check como bloqueante/no verificable según tier. | D05, I09, OPS-01. | Implementado; falta alimentar los valores desde el planificador y perfil de nodo. |
| ARC-21 | Normalizar E08 por CPUs efectivos/asignados y, cuando sea posible, medir el cgroup de campaña. | E08. | Implementado para CPUs delegados; pendiente medición específica del cgroup. |
| ARC-22 | Adaptar G01-G03 a AMD ROCm o mantener GPU deshabilitada hasta tener un inspector compatible. NVML/MIG no describe MI210. | G01, G02, G03. | Pendiente. |
| ARC-23 | C01/C02 se consideran verificados solo después de compilar/desplegar binarios en el nodo y actualizar checksums en catálogo. | CAT-01, CAT-02, C01, C02. | Pendiente operativo. |

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
6. Antes de habilitar GPU, implementar y validar el inspector ROCm.

## Criterio de cambio futuro

Cada nueva entrada añadida a este archivo debe indicar:

1. ID `ARC-NN` y fecha.
2. Motivo y evidencia observada.
3. Reglas/checklists/flujo que modifica.
4. Módulos y pruebas afectados.
5. Estado: propuesto, implementado, verificado en hardware o descartado.
