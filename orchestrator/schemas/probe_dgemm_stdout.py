import sys, os, subprocess
sys.path.insert(0, "/home/latorresn/hyperion")
os.chdir("/home/latorresn/hyperion-kernels")
from orchestrator.gpu_shim import cuda_lib_dirs

env = dict(os.environ)
d = cuda_lib_dirs()
if d:
    env["LD_LIBRARY_PATH"] = ":".join(str(x) for x in d) + ":" + env.get("LD_LIBRARY_PATH", "")
r = subprocess.run(["bin/cublas_dgemm_bench", "--size", "4096", "--iterations", "5"],
                    capture_output=True, text=True, env=env, timeout=60)
print(r.stdout)
print("STDERR:", r.stderr[-500:])
