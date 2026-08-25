from classifier.analysis import gpu_dwt2d_size_diversity as m

runs = m.load(3)
for k in m.KERNELS:
    f0 = runs.get((k, "F0"))
    ref = runs.get((k, "REF"))
    if f0:
        print(f"{k:<24} F0: t={f0['elapsed_s']:.3f}s E={f0['gpu_j']:.1f}J mhz={f0.get('gpu_mhz')}")
    if ref:
        print(f"{k:<24} REF: t={ref['elapsed_s']:.3f}s E={ref['gpu_j']:.1f}J")
