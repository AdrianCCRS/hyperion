import csv

BASE = "/home/latorresn/hyperion-results/campaigns/pacca_gpu_dwt2d_size_sweep_20260823"
CID = "pacca_gpu_dwt2d_size_sweep_20260823"

for kernel in ["rodinia_dwt2d_s192", "rodinia_dwt2d_s8192"]:
    path = f"{BASE}/{CID}__{kernel}__REF__gpuF0__rep01/windows.csv"
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["quality_status"] == "gpu_telemetry"]
    rows.sort(key=lambda r: int(r["t_end_ns"]))
    t0 = int(rows[0]["t_end_ns"])
    print(f"=== {kernel} ({len(rows)} filas) ===")
    for r in rows:
        off = (int(r["t_end_ns"]) - t0) / 1e9
        util = r["gpu_util_pct"]
        power = r["gpu_power_mw"]
        print(f"  t={off:6.3f}s util={util:>4s}% power={power:>7s}mW")
