# Scripts operativos para felix (SC3)

Scripts usados para compilar y validar los kernels del catálogo
(`orchestrator/schemas/kernels/catalog.yaml`) contra hardware real. Viven
versionados aquí (no sueltos en el `$HOME` del clúster) para que la Fase 3
sea reproducible en otro clúster sin tener que reconstruir cada script de
memoria — ver `docs/retoma/Propuesta_Seleccion_Kernels_Dataset.md` y el
objetivo de portabilidad discutido en `docs/retoma/Plan_Implementacion_Medicion_SC3.md`.

## Convención de directorios en `$HOME` (mientras `/scratch` no esté disponible — ver P3 en `Solicitud_Permisos_SC3.md`)

```
~/hyperion/                      # este repo (git clone)
~/hyperion-kernels/
  ├── src/                       # fuente de terceros descargada (NPB), NO versionada aquí
  ├── bin/                       # binarios compilados -- lo que usa catalog.yaml (exec_path relativo)
  └── checksums.sha256           # salida de sha256sum de bin/, registro de la última compilación
~/hyperion-results/
  ├── validation/                 # corridas puntuales de validación (F3.4, F4.2...), no campañas reales
  └── campaigns/                  # output_dir de campañas reales del orquestador (F4.4+)
```

El orquestador (`orchestrator.cli`) debe lanzarse con `~/hyperion-kernels`
como directorio de trabajo, porque `catalog.yaml` usa `exec_path` relativo
(`bin/stream_c`, no una ruta absoluta) a propósito, para que el mismo
catálogo sirva en otro clúster sin editarlo.

## Scripts

| Script | Qué hace |
|---|---|
| `build_npb.sh` | Compila NPB3.4-OMP (requiere el tarball ya en `~/hyperion-kernels/src/`, ver procedencia en `catalog.yaml`) para las clases pedidas por argumento. Uso: `./build_npb.sh S W A B` |
| `build_dgemm.sh` | Compila `kernels/dgemm/dgemm_bench.c` contra OpenBLAS del sistema y corre un smoke test de tamaños. |
| `build_stream_ert.sh` | Compila `kernels/stream/stream.c` y `kernels/ert/ert_probe.c`. |
| `run_f34_validation.sh` | Compila el harness C++ (cmake+gnu14), corre CTest, y ejecuta STREAM bajo `telemetry_kernel_launcher --exec` para la validación de bytes de F3.4. |
| `run_preflight_check.py` | Corre `environment.detect_environment()` + `preflight.run_campaign_preflight()` reales contra el nodo (F4.2), sin escribir nada en sysfs. |
| `audit_readonly.sh` | Auditoría de solo lectura del nodo (topología, cpufreq, RAPL, GPU, etc.) — el script original de la Parte 0 del plan. |

Todos se corren dentro de un `srun` en la partición `gpu_titan` sobre
`felix`, cubriendo el socket completo (`--cpus-per-task=8
--hint=nomultithread`) por el requisito de dominio de frecuencia (E10,
ver ARC-30).
