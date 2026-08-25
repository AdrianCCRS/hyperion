import csv

BASE = "/home/latorresn/hyperion-results/campaigns/pacca_gpu_dwt2d_size_sweep_20260823"
CID = "pacca_gpu_dwt2d_size_sweep_20260823"

for kernel in ["rodinia_dwt2d_s192", "rodinia_dwt2d_s2048", "rodinia_dwt2d_s4096", "rodinia_dwt2d_s8192"]:
    path = f"{BASE}/{CID}__{kernel}__REF__gpuF0__rep01/windows.csv"
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["quality_status"] == "gpu_telemetry"]
    rows.sort(key=lambda r: int(r["t_end_ns"]))
    t0 = int(rows[0]["t_end_ns"])
    t_last = int(rows[-1]["t_end_ns"])
    total_s = (t_last - t0) / 1e9

    active = [
        (int(r["t_end_ns"]) - t0) / 1e9
        for r in rows
        if r["gpu_util_pct"] not in ("", None) and int(r["gpu_util_pct"]) >= 50
    ]
    max_util = max((int(r["gpu_util_pct"]) for r in rows if r["gpu_util_pct"] not in ("", None)), default=0)
    max_power = max((int(r["gpu_power_mw"]) for r in rows if r["gpu_power_mw"] not in ("", None)), default=0)

    if active:
        print(f"{kernel:<24} total={total_s:6.3f}s  "
              f"ventana_util>=50%: [{active[0]:.3f}s .. {active[-1]:.3f}s] "
              f"({active[-1]-active[0]:.3f}s, {len(active)} muestras)  "
              f"max_util={max_util}%  max_power={max_power/1000:.1f}W")
    else:
        print(f"{kernel:<24} total={total_s:6.3f}s  NUNCA llego a util>=50%  "
              f"max_util={max_util}%  max_power={max_power/1000:.1f}W")
