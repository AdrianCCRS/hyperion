import csv

paths = {
    "s192 (sweep)": "/home/latorresn/hyperion-results/campaigns/pacca_gpu_dwt2d_size_sweep_20260823/pacca_gpu_dwt2d_size_sweep_20260823__rodinia_dwt2d_s192__REF__gpuF0__rep01/windows.csv",
    "s8192 (sweep)": "/home/latorresn/hyperion-results/campaigns/pacca_gpu_dwt2d_size_sweep_20260823/pacca_gpu_dwt2d_size_sweep_20260823__rodinia_dwt2d_s8192__REF__gpuF0__rep01/windows.csv",
    "original (alpha_screening)": "/home/latorresn/hyperion-results/campaigns/pacca_gpu_alpha_screening_20260823/pacca_gpu_alpha_screening_20260823__rodinia_dwt2d__REF__gpuF0__rep01/windows.csv",
}

for label, path in paths.items():
    with open(path) as f:
        rows = list(csv.DictReader(f))
    gpu_rows = [r for r in rows if r["quality_status"] == "gpu_telemetry"]
    print(f"{label:<28} total_filas={len(rows)}  gpu_telemetry={len(gpu_rows)}")
