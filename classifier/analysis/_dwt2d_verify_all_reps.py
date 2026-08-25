import csv

BASE = "/home/latorresn/hyperion-results/campaigns/pacca_gpu_dwt2d_size_sweep_20260823"
CID = "pacca_gpu_dwt2d_size_sweep_20260823"
ORIG_BASE = "/home/latorresn/hyperion-results/campaigns/pacca_gpu_alpha_screening_20260823"
ORIG_CID = "pacca_gpu_alpha_screening_20260823"

cases = [
    (BASE, CID, "rodinia_dwt2d_s192"),
    (BASE, CID, "rodinia_dwt2d_s2048"),
    (BASE, CID, "rodinia_dwt2d_s4096"),
    (BASE, CID, "rodinia_dwt2d_s8192"),
    (ORIG_BASE, ORIG_CID, "rodinia_dwt2d"),
]

for base, cid, kernel in cases:
    for rep in (1, 2, 3):
        path = f"{base}/{cid}__{kernel}__REF__gpuF0__rep0{rep}/windows.csv"
        try:
            with open(path) as f:
                rows = [r for r in csv.DictReader(f) if r["quality_status"] == "gpu_telemetry"]
        except FileNotFoundError:
            print(f"{kernel:<22} rep{rep}  NO EXISTE ({path})")
            continue
        if not rows:
            print(f"{kernel:<22} rep{rep}  0 filas")
            continue
        rows.sort(key=lambda r: int(r["t_end_ns"]))
        t0 = int(rows[0]["t_end_ns"])
        t_last = int(rows[-1]["t_end_ns"])
        utils = [int(r["gpu_util_pct"]) for r in rows if r["gpu_util_pct"] not in ("", None)]
        active = [(int(r["t_end_ns"]) - t0) / 1e9 for r, u in zip(rows, utils) if u >= 50]
        window = f"[{active[0]:.3f}..{active[-1]:.3f}]" if active else "ninguna>=50%"
        print(f"{kernel:<22} rep{rep}  total={(t_last-t0)/1e9:5.3f}s  n={len(rows):4d}  "
              f"max_util={max(utils):3d}%  ventana_activa={window}")
