import csv

path = ("/home/latorresn/hyperion-results/campaigns/pacca_gpu_alpha_screening_20260823/"
        "pacca_gpu_alpha_screening_20260823__rodinia_dwt2d__REF__gpuF0__rep01/windows.csv")
with open(path) as f:
    rows = [r for r in csv.DictReader(f) if r["quality_status"] == "gpu_telemetry"]
rows.sort(key=lambda r: int(r["t_end_ns"]))
t0 = int(rows[0]["t_end_ns"])
t_last = int(rows[-1]["t_end_ns"])
print(f"total_elapsed_gpu_window={(t_last - t0)/1e9:.3f}s  n_rows={len(rows)}")

utils = [int(r["gpu_util_pct"]) for r in rows if r["gpu_util_pct"] not in ("", None)]
active = [(int(r["t_end_ns"]) - t0) / 1e9 for r, u in zip(rows, utils) if u >= 50]
if active:
    print(f"ventana util>=50%: [{active[0]:.3f}s .. {active[-1]:.3f}s]  duracion={active[-1]-active[0]:.3f}s")
