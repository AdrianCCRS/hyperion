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

BASELINE_REPETITION_INDICES=[1] (2026-08-27, auditoria pre-lanzamiento):
sin esto, campaign.py empareja baseline+telemetry en CADA repeticion
(CAM-04), duplicando el total de lanzamientos de proceso. El overhead de
instrumentacion ya esta caracterizado (media 1.95%, estable por nivel de
frecuencia, ver Estrategia_CPU_Fase2.md) sobre 540 pares previos -- no hace
falta re-medirlo en cada una de las 8160 combinaciones nuevas. Restringir
el baseline a la repeticion 1 de cada combinacion (spot-check) recorta
~33% de los lanzamientos totales sin perder cobertura de deteccion de
deriva silenciosa.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from gen_dual_full_catalog import OP_META  # noqa: E402

REPS = 3
TARGET_WINDOWS_PER_REP = 5
CAMPAIGN_DATE = "20260828"

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


CPU_MANIFEST = f"""# Primer dataset cold/warm del selector CPU/GPU -- eje CPU ({CAMPAIGN_DATE}). 68 config_id
# x 8 niveles finos de CPU x {REPS} rep = {68*8*REPS} corridas. Ver
# scripts/pacca/gen_dual_campaign_manifests.py -- NO editar a mano.
#
# Rejilla de tamaños y --iterations calibrados por operacion:
# scripts/pacca/gen_dual_full_catalog.py (68 config_id, 6 operaciones).
# warmup_seconds=0.05 en las 136 entradas del catalogo (corrige el bug del
# smoke 6668/6657: 0.5s excluia TODAS las ventanas de corridas mas cortas).

campaign_id: pacca_dual_cpu_full_{CAMPAIGN_DATE}
environment_tier: hpc_sc3
seed: {CAMPAIGN_DATE}

output_dir: /home/latorresn/hyperion-results/campaigns/pacca_dual_cpu_full_{CAMPAIGN_DATE}
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
baseline_repetition_indices: [1]
target_windows_per_repetition: {TARGET_WINDOWS_PER_REP}
interval_ns: 1000000  # 1 ms
gpu_interval_ns: 5000000  # 5 ms; GPU ociosa entra al mismo subtotal energetico

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
  # Mismo alcance energetico que el lado GPU: RAPL package+DRAM + NVML.
  # En corridas CPU la contribucion GPU es la potencia ociosa, no cero.
  enabled: true

timeouts_seconds:
  ready: 15
  run: 180
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

GPU_MANIFEST = f"""# Primer dataset cold/warm del selector CPU/GPU -- eje GPU ({CAMPAIGN_DATE}). 68 config_id
# x 4 niveles reducidos de CPU x 8 niveles finos de GPU x {REPS} rep =
# {68*4*8*REPS} corridas. Ver scripts/pacca/gen_dual_campaign_manifests.py --
# NO editar a mano.
#
# Niveles de CPU reducidos a 4 (REF/F0/F3/F6, no los 8 de la campaña de
# CPU-solo): mismo grid que ya uso y valido el smoke (job 6668) -- el
# reloj de CPU SI afecta el despacho GPU (hasta 95% mas lento en F6), con
# forma de meseta REF~F0 + penalizacion creciente hacia F6, capturada con
# estos 4 puntos sin pagar el costo combinatorio de 8x8.

campaign_id: pacca_dual_gpu_full_{CAMPAIGN_DATE}
environment_tier: hpc_sc3
seed: {CAMPAIGN_DATE}

output_dir: /home/latorresn/hyperion-results/campaigns/pacca_dual_gpu_full_{CAMPAIGN_DATE}
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
baseline_repetition_indices: [1]
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
  # Reposo medido en la rejilla exacta (job 6714: 300 muestras/nivel,
  # 60 s, reloj observado == solicitado). Los margenes REF/F0/F3/F6
  # conservan los anclajes activos de ARC-194; F1/F2/F4/F5 son
  # interpolaciones lineales por MHz entre esos anclajes, no nuevas
  # mediciones de carga. El smoke completo debe validar el criterio antes
  # de lanzar este manifiesto.
  idle_power_mw_by_level:
    REF: 34837.9
    F0: 56565.5
    F1: 45052.4
    F2: 38466.0
    F3: 36941.6
    F4: 35369.0
    F5: 34837.9
    F6: 34368.9
  active_power_margin_mw:
    REF: 800.0
    F0: 4000.0
    F1: 2700.0   # interpolado entre 1230/1170 MHz de ARC-194
    F2: 1720.0   # interpolado entre 1110/810 MHz de ARC-194
    F3: 1200.0
    F4: 1005.0   # interpolado entre 810/510 MHz de ARC-194
    F5: 865.0    # interpolado entre 510/210 MHz de ARC-194
    F6: 800.0

timeouts_seconds:
  ready: 15
  run: 180
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
    cpu_combos = 68 * 8 * REPS
    gpu_combos = 68 * 4 * 8 * REPS
    # baseline_repetition_indices=[1]: solo 1 de cada REPS repeticiones empareja
    # baseline+telemetry (2 lanzamientos); el resto es solo telemetry (1 lanzamiento).
    cpu_launches = (cpu_combos // REPS) * 2 + (cpu_combos - cpu_combos // REPS)
    gpu_launches = (gpu_combos // REPS) * 2 + (gpu_combos - gpu_combos // REPS)
    print(f"CPU: 68 config_id x 8 niveles x {REPS} rep = {cpu_combos} corridas ({cpu_launches} lanzamientos de proceso)")
    print(f"GPU: 68 config_id x 4 niveles CPU x 8 niveles GPU x {REPS} rep = {gpu_combos} corridas ({gpu_launches} lanzamientos de proceso)")
    print(f"TOTAL: {cpu_combos + gpu_combos} corridas, {cpu_launches + gpu_launches} lanzamientos de proceso "
          f"(vs {2*(cpu_combos+gpu_combos)} sin baseline_repetition_indices)")
