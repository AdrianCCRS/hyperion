# AGENTS.md

Contexto de orientación para cualquier agente de IA (Claude Code u otro) que trabaje en este repositorio. Léelo completo antes de tocar código. No reemplaza a los documentos de referencia listados en la sección 8 — los resume para que sepas cuál abrir según lo que estés haciendo.

## 1. Qué es este proyecto

Trabajo de grado: un agente en espacio de usuario que ajusta dinámicamente la frecuencia de CPU/GPU (DVFS) en sistemas heterogéneos, usando un clasificador ligero de Machine Learning que infiere si una aplicación está en régimen `compute_bound` o `memory_bound` a partir de telemetría de hardware, y actúa en consecuencia para optimizar el Producto Energía-Retardo (EDP).

**Lo que este repositorio construye ahora mismo es la Fase 1: la plataforma de recolección de telemetría**, no el clasificador ni el agente de control todavía. Todo lo que hay aquí sirve para generar, de forma reproducible y auditable, el dataset con el que se entrenará el modelo en la Fase 2.

## 2. Arquitectura en una imagen

```
harness C++17 (ya existe, no se toca salvo el modo --exec)
  telemetry_kernel_launcher    -> lanza baseline/telemetry, perf/RAPL/NVML, escribe samples.csv + metadata.json
  telemetry_kernel_workload    -> kernels sintéticos internos (SOLO para dev/testing, nunca para el dataset)

orquestador Python 3.11+ (lo que se está construyendo)
  envuelve al harness: manifest declarativo -> catálogo de kernels externos ->
  calibración -> control de frecuencia -> campaña -> validación -> post-procesamiento -> reporte
```

Los kernels de carga de trabajo **no se programan en este proyecto**. Se usan binarios pre-compilados de suites externas (NAS Parallel Benchmarks para dataset; STREAM y ERT para calibración), conectados al harness mediante un catálogo declarativo (`kernels/catalog.yaml`) y el modo `--exec` del launcher.

La etiqueta de entrenamiento (`compute_bound` / `memory_bound`) **no se asume por el nombre del kernel**. Se deriva empíricamente, por ventana, comparando la intensidad operacional medida contra el *ridge point* del modelo Roofline, calibrado por nodo en cada sesión (STREAM + ERT a frecuencia máxima). Cualquier expectativa de la literatura sobre el kernel se guarda aparte como `phase_label_hint`, solo para auditoría — nunca como la etiqueta que entrena el modelo.

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
- **`phase_label_train` se calcula, nunca se asume.** Es siempre el resultado de comparar `operational_intensity` contra `i_ridge_flops_per_byte` de la calibración de esa sesión. No copiar de `phase_label_hint`, no inferir estadísticamente.
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
| `G0x` | GPU (fuera de alcance por ahora) |

Si agregas una verificación nueva, dale un `factor_id` del prefijo correspondiente y regístrala en el Checklist de Validaciones (documento de referencia, sección 7).

## 6. Entornos de prueba

| Tier | Qué es | Qué se puede validar |
|---|---|---|
| `local` | PC de un investigador, bare-metal, con root | Todo, incluida frecuencia y RAPL reales |
| `cloud_own` | Servidor cloud propio | Mecánica del pipeline; frecuencia/RAPL reales solo si es instancia bare-metal/dedicada |
| `hpc_sc3` | Nodo real del clúster SC3 | Todo, pendiente de permisos formales — **no tocar hasta que el checklist de "listos para escalar" esté en verde** |

## 7. Tests y verificación

- Cada módulo se acompaña de sus tests en `tests/orchestrator/`, con los IDs definidos en el Plan de Tests (documento de referencia, sección 8).
- Los tests de manifest/environment/preflight/freqctl/catalog/calibration/runner/postprocess/validation/campaign/metadata usan mocks de sysfs y fixtures — deben poder correr en cualquier entorno Linux, incluido CI, sin hardware especial.
- Los tests de integración (campaña piloto real, prueba de caos de `freqctl.py`) requieren hardware bare-metal con root y **no son delegables a un agente de IA** — son el paso final antes de dar un módulo por terminado.
- Un módulo no se marca como completo solo porque sus tests pasan: cada regla del Checklist de Validaciones debe poder señalarse en una línea concreta del código.

## 8. Documentos de referencia (fuente de verdad — este archivo es solo el resumen de orientación)

1. **Plan de Implementación** — el porqué: metodología experimental, calibración Roofline, seguridad en el nodo compartido, estrategia multinodo.
2. **Guía Técnica del Orquestador** — el cómo: firmas de funciones, dataclasses, formato exacto de manifest/catálogo/artefactos JSON.
3. **Checklist de Validaciones Técnicas** — las reglas atómicas que el código debe cumplir, agrupadas por módulo.
4. **Plan de Tests** — los tests concretos que implementan y verifican cada regla del checklist.
5. **Guía de Desarrollo Asistido por IA** — el orden de construcción, prompt sugerido y riesgo típico por módulo.

Si una instrucción de una sesión de trabajo contradice alguno de estos documentos, el documento tiene prioridad — señala la contradicción en vez de resolverla por tu cuenta.