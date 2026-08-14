# AGENTS.md

Contexto de orientación para cualquier agente de IA (Claude Code u otro) que trabaje en este repositorio. Léelo completo antes de tocar código. Resume el estado vigente y señala qué fuentes históricas siguen siendo normativas para cada tema.

## 1. Qué es este proyecto

Trabajo de grado: un agente en espacio de usuario que ajusta dinámicamente la frecuencia de CPU/GPU (DVFS) en sistemas heterogéneos, usando un clasificador ligero de Machine Learning que infiere si una aplicación está en régimen `compute_bound` o `memory_bound` a partir de telemetría de hardware, y actúa en consecuencia para optimizar el Producto Energía-Retardo (EDP).

**El repositorio se encuentra cerrando la Fase 1: la plataforma de recolección de telemetría**, no el clasificador ni el agente de control todavía. El instrumento CPU/GPU está implementado y ejercitado en hardware real; lo pendiente para consolidar el dataset DVFS no es la mecánica básica de adquisición, sino confirmar la actuación física de los niveles bajo carga y repetir las campañas que correspondan. Todo lo que hay aquí sirve para generar, de forma reproducible y auditable, el dataset con el que se entrenará el modelo en la Fase 2.

Estado de actuación al 2026-08-14: en CPU, los límites por núcleo se escriben y restauran, pero el turbo global impide todavía sostener bajo carga los niveles solicitados; esta divergencia se mide ahora por ventana y requiere resolución administrativa antes de repetir la campaña DVFS. En GPU, `nvidia-smi -lgc` ya era el mecanismo operativo de bloqueo de reloj y quedó verificado directamente bajo carga: 600 y 1200 MHz se sostuvieron exactamente. El controlador 610.57.04 es contexto de la verificación, no evidencia de que el mecanismo empezara a funcionar con esa versión. La campaña GPU multi-frecuencia completa sigue pendiente. No confundir permiso de escritura o código de retorno exitoso con actuación física: siempre verificar durante la carga.

## 2. Arquitectura en una imagen

```
harness C++17 (instrumento de producción, mantenido en este repositorio)
  telemetry_kernel_launcher    -> lanza baseline/telemetry, perf/RAPL/NVML, escribe samples.csv + metadata.json
  telemetry_kernel_workload    -> kernels sintéticos internos (SOLO para dev/testing, nunca para el dataset)

orquestador Python 3.11+ (implementado; continúa bajo validación experimental)
  envuelve al harness: manifest declarativo -> catálogo de kernels externos ->
  calibración -> control de frecuencia -> campaña -> validación -> post-procesamiento -> reporte
```

El catálogo mezcla cargas externas y microbenchmarks propios, pero **ninguna se inventa ni se selecciona desde el código del orquestador**. CPU usa NAS Parallel Benchmarks para el conjunto de datos y STREAM/ERT para calibración; GPU usa Rodinia, una DGEMM y microbenchmarks propios de calibración. La composición vigente, las rutas y las sumas de verificación están declaradas en `orchestrator/schemas/kernels/catalog.yaml` y se conectan al harness mediante el modo `--exec` del launcher.

La etiqueta de entrenamiento (`compute_bound` / `memory_bound`) **no se asume por el nombre del kernel**. En CPU se deriva empíricamente comparando contra el *ridge point* la intensidad medida a la granularidad real de cada intervalo `uncore_imc` (aproximadamente 10 ms), y se difunde a las ventanas CPU de aproximadamente 1 ms que ese intervalo cubre; nunca se vuelve al proxy `cache_misses × line_size` si falta cobertura. En GPU, NVML no mide FLOPs ni bytes: la intensidad se caracteriza fuera de línea con Nsight Compute y se compara contra el ridge específico de la precisión y frecuencia correspondientes. Cualquier expectativa de literatura se guarda aparte como `phase_label_hint`, solo para auditoría — nunca como la etiqueta que entrena el modelo.

## 3. Mapa de módulos (`orchestrator/`)

| Módulo | Responsabilidad | Depende de |
|---|---|---|
| `manifest.py` | Parsea y valida `campaign.yaml` | — |
| `environment.py` | Detecta, de solo lectura, qué puede controlarse realmente en este nodo (frecuencia, RAPL) | — |
| `preflight.py` | Verificaciones de solo lectura, bloqueantes o de advertencia, antes de campaña y por corrida | manifest, environment |
| `freqctl.py` | Control y **restauración garantizada** de frecuencia — el módulo más sensible del repo | environment |
| `catalog.py` | Valida binarios externos (existencia, checksum) desde `kernels/catalog.yaml` | manifest |
| `calibration.py` | Calibración Roofline: P_pico, BW_pico, I_ridge | catalog, runner, freqctl |
| `node_profile.py` | Perfil de hardware + referencias de estabilidad P95 (capa multinodo) | environment, runner |
| `runner.py` | Ejecuta una corrida individual del launcher (modo sintético y modo `--exec`) | manifest, preflight, catalog |
| `postprocess.py` | `samples.csv` → `windows.csv`: deltas, intensidad operacional, `phase_label_train`, features relativas | calibration, node_profile |
| `validation.py` | Acepta/rechaza cada corrida con un `factor_id` explícito | runner, catalog |
| `campaign.py` | El integrador: genera la matriz, aleatoriza, secuencia, reanuda | todos los anteriores |
| `metadata_schema.py` / `report.py` | Esquema de trazabilidad y reporte consolidado de campaña | validation, campaign |

Orden de construcción real (no el de la tabla): `manifest.py` y `environment.py` en paralelo → `preflight.py` → `runner.py` (modo sintético) → `freqctl.py` → `catalog.py` → `calibration.py` y `node_profile.py` en paralelo → `runner.py` (extensión `--exec`) → `postprocess.py` → `validation.py` → `campaign.py` → `metadata_schema.py`/`report.py`.

## 4. No negociables (léelos antes de escribir código)

- **`freqctl.py` nunca escribe fuera de `delegated_cpus`.** El governor userspace se aplica únicamente sobre los cores delegados a la campaña, nunca a nivel global del nodo — este es un clúster compartido con otros usuarios.
- **Toda escritura de frecuencia se verifica por relectura**, nunca se asume éxito porque la llamada no lanzó excepción.
- **La restauración de frecuencia es obligatoria e idempotente**, registrada en `atexit`, `SIGINT` y `SIGTERM`. Ningún cambio de diseño en `freqctl.py` se acepta sin una prueba de caos real (interrumpir a mitad de una corrida y confirmar por lectura de sysfs que todo volvió al estado previo).
- **Nunca inventes ni hardcodees un kernel de carga de trabajo.** Todo kernel de dataset viene de `kernels/catalog.yaml`, apunta a un binario real de NPB/STREAM/ERT, y se verifica por checksum antes de ejecutarse.
- **`phase_label_train` se calcula, nunca se asume.** Es siempre el resultado de comparar `operational_intensity` contra el ridge aplicable de la calibración. En CPU, `operational_intensity` solo se considera definida con bytes reales de `uncore_imc`; en GPU procede de la caracterización dinámica fuera de línea registrada en el catálogo. No copiar de `phase_label_hint`, no inferir estadísticamente.
- **Ninguna corrida rechazada se borra.** Se conserva en disco con `accepted: false` y su `rejection_factor_id` — es evidencia de auditoría.
- **Nunca dividir sin verificar el denominador.** `bytes_moved_window == 0` → `NaN` y `quality_status = "intensity_undefined"`, nunca una excepción no controlada ni un `0` silencioso.
- **RAPL y control de frecuencia reales casi nunca están disponibles en una VM cloud estándar** (el hipervisor no suele exponer los MSRs físicos). `environment.py` es la única fuente autorizada para decidir `rapl_capable` / `freq_control_capable` — ningún otro módulo repite esa detección por su cuenta.
- **La estrategia multinodo (varios nodos, transferencia entre ellos) está pendiente de decisión del director.** No comprometas tiempo de campaña en un segundo nodo sin confirmación explícita. Sí construye siempre `node_id`, `node_profile.json` y las referencias de calibración P95 — es la capa "sin arrepentimiento" que sirve a cualquier decisión futura.

## 5. Convención de código de rechazo (`factor_id`)

Todo lo que puede invalidar una corrida o una ventana tiene un ID de dos-tres letras + número, usado consistentemente en preflight, validación, metadata y reportes:

| Prefijo | Categoría |
|---|---|
| `E0x` | Entorno del nodo (térmica, NUMA, SMT, procesos ajenos, carga externa) |
| `I0x` | Implementación (PMU, jitter, RAPL, push_retries, run_id) |
| `M0x` | Metodología (orden aleatorio, warmup, etiquetado) |
| `C0x` | Catálogo y binarios externos |
| `D0x` | Calibración Roofline y multinodo |
| `G0x` | GPU (telemetría, aislamiento y actuación de frecuencia) |

Si agregas una verificación nueva, dale un `factor_id` del prefijo correspondiente y regístrala en el Checklist de Validaciones (documento de referencia, sección 7).

## 6. Entornos de prueba

| Tier | Qué es | Qué se puede validar |
|---|---|---|
| `local` | PC de un investigador, bare-metal, con root | Todo, incluida frecuencia y RAPL reales |
| `cloud_own` | Servidor cloud propio | Mecánica del pipeline; frecuencia/RAPL reales solo si es instancia bare-metal/dedicada |
| `hpc_sc3` | Tier HPC conservado por compatibilidad del esquema; la plataforma experimental vigente es paccaA100/Unicartagena | Instrumentación y campañas reales bajo asignación exclusiva de Slurm |

**Cuenta compartida en pacca (Unicartagena):** el login `latorresn` en `paccaA100` es una cuenta compartida entre varias personas/proyectos, no exclusiva de este trabajo de grado. Cualquier archivo de scratch, exploración, probes en C, scripts de verificación o diagnóstico generado por un agente de IA para este proyecto va dentro de `~/yacacerest/` en ese nodo, **nunca suelto en `$HOME`** — hay un `README_SCRATCH.txt` en el `$HOME` de esa cuenta que repite esta convención. Los directorios de instalación real del proyecto (`hyperion/`, `hyperion-kernels/`, `hyperion-results/`, `hyperion-venv/`) sí viven directamente en `$HOME`, son la excepción. Nunca borrar ni mover nada en esa cuenta que no se identifique con certeza como parte de este proyecto (hay directorios de otras personas ahí).

## 7. Tests y verificación

- Cada módulo se acompaña de sus tests en `tests/orchestrator/`, con los IDs definidos en el Plan de Tests (documento de referencia, sección 8).
- Los tests de manifest/environment/preflight/freqctl/catalog/calibration/runner/postprocess/validation/campaign/metadata usan mocks de sysfs y fixtures — deben poder correr en cualquier entorno Linux, incluido CI, sin hardware especial.
- Los tests de integración y las campañas de aceptación requieren hardware bare-metal y deben contrastar el efecto real durante la carga, no solo el éxito de una escritura. En particular, una relectura de `scaling_min_freq`/`scaling_max_freq` confirma que el kernel almacenó la solicitud, pero no que el reloj efectivo la haya respetado; desde ARC-135 esta última verificación se hace por ventana mediante `scaling_cur_freq`.
- Un módulo no se marca como completo solo porque sus tests pasan: cada regla del Checklist de Validaciones debe poder señalarse en una línea concreta del código.

## 8. Fuentes de verdad y precedencia

1. **Plan de trabajo de grado** — fuente normativa de objetivos, alcance académico y referencias base; no es una descripción actualizada del instrumento implementado.
2. **Código, catálogo y manifiestos vigentes** — fuente de verdad de lo que el instrumento ejecuta y mide hoy.
3. **Registro de Cambios Fuera del Plan Original**, dando precedencia a la entrada ARC más reciente sobre el mismo asunto — fuente de decisiones técnicas y evidencia de hardware posterior al plan.
4. **Artefactos de la campaña o prueba de hardware identificada** — fuente para cifras experimentales; comprobar fecha y `campaign_id`, porque se conservan campañas antiguas deliberadamente.
5. **Guía Técnica, Checklist y Plan de Tests** — contratos base útiles, pero pueden quedar rezagados hasta su reconciliación explícita con el código y las entradas ARC recientes.
6. **Guía Maestra y documentos de retoma** — material de orientación o fotografía histórica; no asumir que su estado operativo sigue vigente sin revisar su fecha y las notas de obsolescencia.

Ante una contradicción, no fuerces el código actual para hacerlo coincidir con una guía antigua. Señala la diferencia y aplica la precedencia anterior. Las decisiones reservadas al autor o al director siguen requiriendo confirmación explícita aunque un borrador histórico sugiera otra cosa.
