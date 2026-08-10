import sys, os, subprocess
sys.path.insert(0, "/home/latorresn/hyperion")
sys.path.insert(0, "/home/latorresn/hyperion/docs/justifications/scripts")
os.chdir("/home/latorresn/yacacerest")
from orchestrator.gpu_shim import cuda_lib_dirs
from ncu_gpu_precision import ALL_METRICS, compute_gpu_precision_result, parse_ncu_csv_totals, is_mixed_precision

env = dict(os.environ)
d = cuda_lib_dirs()
if d:
    env["LD_LIBRARY_PATH"] = ":".join(str(x) for x in d) + ":" + env.get("LD_LIBRARY_PATH", "")

cmd = ["ncu", "--metrics", ",".join(ALL_METRICS), "--launch-count", "20", "--csv",
       "./gaussian_build_test", "-s", "4096"]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
print("RETURNCODE:", r.returncode)
if r.returncode != 0:
    print("STDOUT:", r.stdout[-3000:])
    print("STDERR:", r.stderr[-3000:])
    sys.exit(1)

with open("/home/latorresn/yacacerest/ncu_gaussian_probe.csv", "w") as f:
    f.write(r.stdout)

totals, n = parse_ncu_csv_totals(r.stdout)
result = compute_gpu_precision_result(totals, n)
print(f"n_launches={n}")
print(f"flops_fp32={result.flops_fp32:.6e} flops_fp64={result.flops_fp64:.6e}")
print(f"dram_bytes={result.dram_bytes:.6e}")
print(f"fraction_fp32={result.fraction_fp32} fraction_fp64={result.fraction_fp64}")
print(f"operational_intensity={result.operational_intensity}")
print(f"mixed={is_mixed_precision(result)}")
