"""Genera los dos manifiestos de la campana real del selector CPU/GPU:
campaign_pacca_dual_cpu_full.yaml (68 kernels CPU, 8 niveles finos de CPU) y
campaign_pacca_dual_gpu_full.yaml (68 kernels GPU, 4 niveles reducidos de
CPU x 8 niveles finos de GPU).

POR QUE DOS ARCHIVOS: ver el docstring de gen_dual_full_catalog.py --
frequency_levels es una sola lista por manifiesto, aplicada a TODOS sus
kernels; no se puede dar 8 niveles finos a los kernels CPU-solo y 4
reducidos a los kernels GPU dentro del mismo archivo.

NIVELES CPU-DURANTE-GPU (4: REF/F0/F3/F6): el MISMO grid reducido que ya
uso y valido el smoke (job 6668, 2026-08-27) -- confirmo con datos reales
que el reloj de CPU afecta el despacho GPU (hasta 95% mas lento en F6),
con forma de meseta REF~F0 y penalizacion creciente hacia F6. Reusar ese
grid, ya probado, evita diseñar uno nuevo sin evidencia.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from gen_dual_full_catalog import OP_META  # noqa: E402

REPS = 3
TARGET_WINDOWS_PER_REP = 5

CPU_LEVELS_FULL = [
    ("REF", "native_governor", None),
    ("F0", "fixed", 1.0),
    ("F1", "fixed", 0.833),
    ("F2", "fixed", 0.667),
    ("F3", "fixed", 0.5),
    ("F4", "fixed", 0.333),
    ("F5", "fixed", 0.167),
    ("F6", "fixed", 0.0),
]
CPU_LEVELS_REDUCED = [
    ("REF", "native_governor", None),
    ("F0", "fixed", 1.0),
    ("F3", "fixed", 0.5),
    ("F6", "fixed", 0.0),
]
GPU_LEVELS_FULL = [
    ("REF", "native_governor", None),
    ("F0", "fixed", 1.0),
    ("F1", "fixed", 0.833),
    ("F2", "fixed", 0.667),
    ("F3", "fixed", 0.5),
    ("F4", "fixed", 0.333),
    ("F5", "fixed", 0.167),
    ("F6", "fixed", 0.0),
]


def fmt_levels(levels):
    out = []
    for id_, mode, frac in levels:
        if mode == "native_governor":
            out.append(f"  - {{id: {id_}, mode: native_governor}}")
        else:
            out.append(f"  - {{id: {id_}, mode: fixed, fraction: {frac}}}")
    return "\n".join(out)


def all_config_ids():
    ids = []
    for op, meta in OP_META.items():
        for n in meta["grid"]:
            ids.append(f"dual_{op}_{{device}}_N{n}")
    return ids


def kernels_block(device):
    lines = []
    for op, meta in OP_META.items():
        for n in meta["grid"]:
            lines.append(f"  - kernel_ref: dual_{op}_{device}_N{n}")
    return "\n".join(lines)


CPU_MANIFEST = f"""# Campana real del selector CPU/GPU -- eje CPU (2026-08-27). 68 config_id
# x 8 niveles finos de CPU x {REPS} rep = {68*8*REPS} corridas. Ver
# scripts/pacca/gen_dual_campaign_manifests.py -- NO editar a mano.
#
# Rejilla de tamaños y --iterations calibrados por operacion:
# scripts/pacca/gen_dual_full_catalog.py (68 config_id, 6 operaciones).
# warmup_seconds=0.05 en las 136 entradas del catalogo (corrige el bug del
# smoke 6668/6657: 0.5s excluia TODAS las ventanas de corridas mas cortas).

campaign_id: pacca_dual_cpu_full_20260827
environment_tier: hpc_sc3
seed: 20260827

output_dir: /home/latorresn/hyperion-results/campaigns/pacca_dual_cpu_full_20260827
overwrite: true

catalog_path: ../kernels/catalog.yaml

calibration:
  - kernel_ref: stream_official
  - kernel_ref: ert_probe

kernels:
{kernels_block("cpu")}

frequency_levels:
{fmt_levels(CPU_LEVELS_FULL)}

repetitions_per_combination: {REPS}
target_windows_per_repetition: {TARGET_WINDOWS_PER_REP}
interval_ns: 1000000  # 1 ms

running_ratio_min: 0.90

frequency_validation:
  require_per_window: true
  tolerance_fraction: 0.05
  grace_seconds: 0.05

cores:
  delegated_cpus: [0, 1, 2, 3, 4, 5]
  collector_cpu: 6
  consumer_cpu: 7
  numa_node_pin: 0

smt_policy: one_thread_per_physical_core

cgroup_path: null

perf_enabled: true

frequency_settle:
  enabled: true
  timeout_seconds: 30.0
  tolerance_fraction: 0.05
  poll_interval_seconds: 0.5

rapl:
  enabled: true

uncore:
  enabled: true

turbo:
  require_disabled: true

temperature:
  require_package_sensor: true
  minimum_c: 0
  maximum_c: 90

gpu:
  enabled: false

timeouts_seconds:
  ready: 15
  run: 90
  shutdown: 15

hardware_datasheet:
  # p_pico medido real 2026-08-27 (post-reparacion de la corrupcion de
  # frecuencia, job 6651->6668): ert_probe nativo a 6 hilos da 509.083
  # GFLOP/s. bw_pico heredado de campaign_pacca_gpu_dvfs.yaml (ya paso D03
  # en ese valor, no contaminado).
  bw_pico_bytes_per_s: 59500000000
  p_pico_flops_per_s: 509083000000

projected_campaign_bytes: 500000000
remaining_core_hours: 1000.0
projected_core_hours: 10.0
"""

GPU_MANIFEST = f"""# Campana real del selector CPU/GPU -- eje GPU (2026-08-27). 68 config_id
# x 4 niveles reducidos de CPU x 8 niveles finos de GPU x {REPS} rep =
# {68*4*8*REPS} corridas. Ver scripts/pacca/gen_dual_campaign_manifests.py --
# NO editar a mano.
#
# Niveles de CPU reducidos a 4 (REF/F0/F3/F6, no los 8 de la campaña de
# CPU-solo): mismo grid que ya uso y valido el smoke (job 6668) -- el
# reloj de CPU SI afecta el despacho GPU (hasta 95% mas lento en F6), con
# forma de meseta REF~F0 + penalizacion creciente hacia F6, capturada con
# estos 4 puntos sin pagar el costo combinatorio de 8x8.

campaign_id: pacca_dual_gpu_full_20260827
environment_tier: hpc_sc3
seed: 20260827

output_dir: /home/latorresn/hyperion-results/campaigns/pacca_dual_gpu_full_20260827
overwrite: true

catalog_path: ../kernels/catalog.yaml

calibration:
  - kernel_ref: stream_official
  - kernel_ref: ert_probe

kernels:
{kernels_block("gpu")}

frequency_levels:
{fmt_levels(CPU_LEVELS_REDUCED)}

gpu_frequency_levels:
{fmt_levels(GPU_LEVELS_FULL)}

repetitions_per_combination: {REPS}
target_windows_per_repetition: {TARGET_WINDOWS_PER_REP}
interval_ns: 1000000  # 1 ms
gpu_interval_ns: 5000000  # 5 ms

running_ratio_min: 0.90

frequency_validation:
  require_per_window: true
  tolerance_fraction: 0.05
  grace_seconds: 0.05

cores:
  delegated_cpus: [0, 1, 2, 3, 4, 5]
  collector_cpu: 6
  consumer_cpu: 7
  numa_node_pin: 0

smt_policy: one_thread_per_physical_core

cgroup_path: null

perf_enabled: true

frequency_settle:
  enabled: true
  timeout_seconds: 30.0
  tolerance_fraction: 0.05
  poll_interval_seconds: 0.5

rapl:
  enabled: true

uncore:
  enabled: true

turbo:
  require_disabled: true

temperature:
  require_package_sensor: true
  minimum_c: 0
  maximum_c: 90

gpu:
  enabled: true
  calibration:
    - gpu_stream_bw
    - gpu_ert_probe_fp32
    - gpu_ert_probe_fp64

timeouts_seconds:
  ready: 15
  run: 90
  shutdown: 15

hardware_datasheet:
  bw_pico_bytes_per_s: 59500000000
  p_pico_flops_per_s: 509083000000

projected_campaign_bytes: 2000000000
remaining_core_hours: 1000.0
projected_core_hours: 40.0
"""

if __name__ == "__main__":
    base = __import__("pathlib").Path(__file__).parent.parent.parent / "orchestrator/schemas/campaigns"
    (base / "campaign_pacca_dual_cpu_full.yaml").write_text(CPU_MANIFEST)
    (base / "campaign_pacca_dual_gpu_full.yaml").write_text(GPU_MANIFEST)
    print(f"CPU: 68 config_id x 8 niveles x {REPS} rep = {68*8*REPS} corridas")
    print(f"GPU: 68 config_id x 4 niveles CPU x 8 niveles GPU x {REPS} rep = {68*4*8*REPS} corridas")
    print(f"TOTAL: {68*8*REPS + 68*4*8*REPS} corridas")
