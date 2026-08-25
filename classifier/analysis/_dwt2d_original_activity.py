import csv

candidates = [
    ("/home/latorresn/hyperion-results/campaigns/pacca_gpu_alpha_screening_20260823/"
     "pacca_gpu_alpha_screening_20260823__rodinia_dwt2d__REF__gpuF0__rep01/windows.csv"),
    ("/home/latorresn/hyperion-results/campaigns/pacca_gpu_fine_grid_dataset_20260823/"
     "pacca_gpu_fine_grid_dataset_20260823__rodinia_dwt2d__REF__gpuF0__rep01/windows.csv"),
]

for path in candidates:
    print(f"=== {path} ===")
    try:
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r["quality_status"] == "gpu_telemetry"]
    except FileNotFoundError:
        print("  NO EXISTE")
        continue
    if not rows:
        print("  0 filas gpu_telemetry")
        continue
    utils = [int(r["gpu_util_pct"]) for r in rows if r["gpu_util_pct"] not in ("", None)]
    powers = [int(r["gpu_power_mw"]) for r in rows if r["gpu_power_mw"] not in ("", None)]
    rows.sort(key=lambda r: int(r["t_end_ns"]))
    t0 = int(rows[0]["t_end_ns"])
    t_last = int(rows[-1]["t_end_ns"])
    print(f"  n={len(rows)}  total={(t_last-t0)/1e9:.3f}s  "
          f"max_util={max(utils)}%  max_power={max(powers)/1000:.1f}W  "
          f"n_util_ge_50={sum(1 for u in utils if u>=50)}")
