# Fase 1 — Recolección de telemetría y construcción del dataset

Cumple el **Objetivo 1** del trabajo de grado: caracterizar el comportamiento
computacional y energético de cargas representativas (CPU y GPU,
compute-bound y memory-bound) bajo distintos estados de frecuencia,
recolectando telemetría de bajo nivel vía Perf/RAPL (CPU) y NVML (GPU).

No decide nada por sí sola: produce la traza etiquetada (`windows.csv`) y el
dataset CPU agregado (`training_cpu_intervals.csv`) que consume
`fase2_clasificador/` para entrenar el modelo, y la política de
frecuencia que consulta `fase3_daemon/` en producción. Ver
`Plan_Detallado_Realineacion_Hyperion.md` §2 para el diseño completo y las
desviaciones justificadas frente al plan de trabajo de grado aprobado.

## Antes de correr nada: prerrequisitos de plataforma (§2.0 del plan)

1. **Chequeo de permisos de solo lectura**, primer paso siempre:
   ```bash
   python3 common/readiness/check_node_readiness.py
   ```
   Ver el README global (`/README.md`) para qué permisos exactos hacen
   falta (RAPL, `perf_event_paranoid`, NVML, cpuset/cgroup) y cómo
   obtenerlos en Rocky Linux/Fedora.
2. **Permiso de escritura de frecuencia de CPU** y **restricción real del
   reloj de GPU bajo carga** — verificar siguiendo §2.0.1/§2.0.2 del plan de
   realineación antes de lanzar cualquier barrido real. Si algún prerrequisito
   no se puede confirmar, el resultado se documenta como bloqueo de
   infraestructura — nunca se fabrica un dato de frecuencia no verificado.

## Convención de directorios (importante: el cwd del proceso sí importa)

```
~/hyperion-kernels/          # EXTERNO al repositorio, no versionado aquí
  ├── src/                   # fuente de terceros (NPB, Rodinia, RAJAPerf...)
  ├── bin/                   # binarios compilados -- lo que exec_path del
  │                          #   catálogo referencia, relativo a este cwd
  └── checksums.sha256
~/hyperion-results/
  ├── validation/            # corridas puntuales de validación
  └── campaigns/             # output_dir de campañas reales
```

`run_campaign.py` debe invocarse **con `~/hyperion-kernels` como directorio
de trabajo** (no la raíz del repo) porque `exec_path` en `catalog.yaml` es
relativo a ese cwd — a propósito, para que el mismo catálogo sirva en otro
clúster sin editar cada ruta. El binario propio del harness
(`telemetry_kernel_launcher`) no depende de este cwd: se resuelve aparte,
relativo a la raíz del repositorio, vía `common/hpc_config.toml`.

```bash
cd ~/hyperion-kernels
python3 /ruta/al/repo/fase1_telemetria/run_campaign.py --help
```

## Uso

Para ejecutar el flujo completo de selección hasta el informe de utilidad de
los kernels tentativos (sin entrenar ni lanzar todavía la rejilla fina), ver
[`SCREENING_TO_REPORT.md`](SCREENING_TO_REPORT.md) y usar
`bash run_screening_to_report.sh`. Ese flujo aplica obligatoriamente el gate
`ncu` antes del cribado GPU.

`run_campaign.py` es una envoltura delgada sobre `cli.py`, que expone 5
subcomandos (ver `--help` de cada uno para la lista completa de flags):

| Subcomando | Qué hace |
|---|---|
| `diagnose` | Diagnóstico de arranque de solo lectura (sin escribir nada) antes de comprometer una campaña real. |
| `calibrate` | Calibración Roofline (P_pico, BW_pico, I_ridge) + `node_profile` + referencias de estabilidad. |
| `run-campaign` | Corre la campaña completa: calibración + matriz kernel×nivel_frecuencia×repetición. |
| `postprocess` | `samples.csv` → `windows.csv` auditable + `training_cpu_intervals.csv` para entrenamiento CPU (deltas, intensidad operacional, `phase_label_train`, features relativas). |
| `report` | Reporte consolidado de campaña (tabla por `factor_id` de aceptación/rechazo). |

Ejemplo mínimo (campaña de humo local, sin hardware real — ver
`catalog/campaigns/campaign_example.yaml`):

```bash
cd ~/hyperion-kernels
python3 /ruta/al/repo/fase1_telemetria/run_campaign.py run-campaign \
    --manifest /ruta/al/repo/fase1_telemetria/catalog/campaigns/campaign_example.yaml \
    --node-id nodo01 --reference-kernel-ref stream_official
```

## El catálogo (`catalog/`)

`catalog/catalog.yaml`: **232 kernels**, fusión verificada del catálogo
original de este proyecto (23 kernels: NPB, STREAM/ERT, DGEMM, RAJAPerf 3MM,
Rodinia) con el catálogo ampliado construido en la rama `origin/fase-02`
(familias `dual_gemm/fft/axpy/stencil/cholesky/spmv`, NPB adicionales,
GAP, LULESH, HPCG, CHOLMOD, etc.) — necesario para tener suficientes
familias algorítmicas distintas para la validación leave-one-familia-out de
Fase 2 (ver §2.1.1 del plan). `catalog/campaigns/` trae los 53 manifiestos de
campaña asociados, también fusionados.

⚠️ **Decisión tomada durante la fusión, con evidencia, no arbitraria**: para
las 23 entradas presentes en ambos catálogos, se adoptaron los valores de
`fase-02` en 11 casos donde divergían:

- **9 checksums de binarios GPU** (`gpu_dgemm_calibration`, `gpu_dgemm_n4096`,
  `gpu_ert_probe_fp32/fp64`, `gpu_stream_bw`, `rodinia_backprop/gaussian/
  heartwall/lavamd/lud`) — `fase-02` los recompiló para checksums
  reproducibles (commit `f4dadc9`, "despojar los 8 binarios GPU restantes"),
  un fix real que el catálogo original nunca recibió. Si vas a ejecutar una
  campaña GPU contra estos kernels, **hay que recompilarlos y verificar el
  checksum de nuevo en el nodo real** antes de confiar en la campaña — un
  checksum del catálogo no es evidencia de que el binario ya esté compilado
  en tu clúster.
- **`rodinia_lavamd_omp`**: `fase-02` añadió `runtime_seconds_stdout_pattern`
  (commit `ab5b13a`) para medir el tiempo de ejecución parseando el propio
  stdout del kernel en vez de depender únicamente del tiempo de pared medido
  por el harness externo.

Dos manifiestos de ejemplo/históricos (`campaign_example.yaml`,
`campaign_felix_ref*.yaml`, `campaign_sc3_audit.yaml`) referenciaban
`npb_ep`/`npb_is`, kernels reemplazados hace tiempo por `npb_bt`/`npb_sp`
(no hacían punto flotante real, ver el catálogo — este era un problema ya
presente en ambas ramas de origen, no introducido por esta fusión).
Corregidos para que los 53 manifiestos carguen limpio.

**Lo que NO se portó del catálogo de `fase-02`** (queda solo en
`old/`/rama `fase-02`, fuera de alcance de este proyecto según §7.2 del
plan): `classifier/selector/`, y el campo opcional `config_id` de
`catalog.yaml` (solo lo consume el análisis del selector, nunca el
orquestador).

## Etiquetado (`phase_label_train`)

La etiqueta `compute_bound`/`memory_bound` **nunca se asume por el nombre
del kernel**. En CPU se deriva comparando la intensidad operacional medida
(FLOPs reales / bytes reales de `uncore_imc`) contra el punto de ridge de la
calibración Roofline del propio nodo; en GPU, contra el ridge específico de
precisión (fp32/fp64) y frecuencia, medido offline con `ncu`. Cualquier
expectativa de la literatura se guarda aparte como `phase_label_hint`, solo
para auditoría — nunca entrena el modelo. Ver `postprocess.py` y §2.3 del
plan para el detalle completo.

## Tests

```bash
python3 -m pytest fase1_telemetria/tests/ -q
```

328 tests, herméticos (mocks de sysfs/subprocess, sin requerir hardware
real) — igual que `common/tests/`, deben poder correr en cualquier entorno
Linux, incluido CI.

## Limitaciones conocidas (heredadas del plan de realineación)

- La campaña GPU multi-frecuencia completa sobre el catálogo ampliado
  todavía no se ha ejecutado de punta a punta con los binarios GPU
  recompilados de `fase-02`.
- `T_transición_gpu` (latencia de conmutación de reloj de GPU) **aún no
  está medido en hardware**, pero la infraestructura ya existe (F1-GPU-002):
  el probe `common/telemetry/experiments/gpu_clock_transition_probe.cpp`
  (build `-DWITH_GPU=ON`) y el agregador
  `fase1_telemetria/gpu_transition/aggregate_transition_matrix.py`. Falta
  ejecutarlo en paccaA100 con NVML real y alimentar
  `--t-transicion-gpu-ns` a `fase3_daemon/policy/derive_policy_table.py`.
  Procedimiento: `fase1_telemetria/gpu_transition/README.md`. Hasta entonces
  la actuación de frecuencia GPU por fase sigue bloqueada (ver §2.4.1 del
  plan).
- Herramientas de diagnóstico ad hoc de investigaciones puntuales
  (`orchestrator/schemas/scripts/`, `orchestrator/schemas/tools/`,
  `orchestrator/schemas/kernels/class_c_stress/` en `old/`) no se portaron
  — eran scripts de un solo uso para depurar problemas ya resueltos
  (p. ej. la saga CAL-07), no parte del pipeline reutilizable.
