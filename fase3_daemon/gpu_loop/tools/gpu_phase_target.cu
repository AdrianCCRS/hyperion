// Carga objetivo de GPU para el preflight del daemon (Bloque D). Un solo
// proceso, un solo stream, TODO en el dispositivo. Modos:
//   alternate [fase_s=3] [total_s=1e9] [hueco_s=2]
//                                       alterna fases memory (triad) y compute
//                                       (cadenas FMA fp64 en registros) e imprime
//                                       "<ns CLOCK_MONOTONIC> <M|C>" en cada frontera,
//                                       para puntuar la clasificacion contra fronteras
//                                       conocidas (mismo reloj que el registro de decisiones).
//                                       Entre fases deja la GPU ociosa `hueco_s`: el daemon
//                                       detecta fases por transiciones idle->activo del sondeo
//                                       NVML, sin hueco no veria la frontera memory->compute.
//   bench_memory  [segundos]            imprime GB/s del triad
//   bench_compute [segundos]            imprime GFLOP/s de las cadenas FMA
// bench_* sirve para medir el EFECTO REAL del candado de reloj: el rendimiento
// compute debe escalar con el reloj SM; el de memoria, casi no.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

static int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s en %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(2); } } while (0)

__global__ void triad(double* a, const double* b, const double* c, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        a[i] = b[i] + 3.0 * c[i];
}

__global__ void fma_chain(double* out, int iters) {
    double x = 1.0 + threadIdx.x * 1e-3, y = x + 0.1, z = x + 0.2, w = x + 0.3;
    for (int k = 0; k < iters; ++k) {
        x = fma(x, 1.0000001, 1e-9); y = fma(y, 1.0000001, 1e-9);
        z = fma(z, 1.0000001, 1e-9); w = fma(w, 1.0000001, 1e-9);
    }
    if (x + y + z + w == 12345.0) out[0] = x;  // evita que el compilador elimine el bucle
}

static const size_t N = 128u * 1024u * 1024u;  // 3 arreglos fp64 de 1 GiB: muy por encima del L2 (40 MB)
static const int FMA_ITERS = 20000;
static const int FMA_BLOCKS = 1728, FMA_THREADS = 256;  // 108 SM x 16 bloques

struct Work { double* a; double* b; double* c; double* out; };

// Ejecuta una fase de `seconds`; devuelve (bytes movidos | flops) segun el tipo.
static double run_phase(Work& w, bool memory, double seconds) {
    const int64_t t_end = now_ns() + (int64_t)(seconds * 1e9);
    double work = 0.0;
    while (now_ns() < t_end) {
        if (memory) { triad<<<4096, 256>>>(w.a, w.b, w.c, N); work += 24.0 * N; }
        else        { fma_chain<<<FMA_BLOCKS, FMA_THREADS>>>(w.out, FMA_ITERS); work += 8.0 * FMA_ITERS * FMA_BLOCKS * FMA_THREADS; }
        CK(cudaDeviceSynchronize());
    }
    return work;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "alternate";
    Work w{};
    CK(cudaMalloc(&w.a, N * sizeof(double))); CK(cudaMalloc(&w.b, N * sizeof(double)));
    CK(cudaMalloc(&w.c, N * sizeof(double))); CK(cudaMalloc(&w.out, sizeof(double)));
    CK(cudaMemset(w.a, 0, N * sizeof(double))); CK(cudaMemset(w.b, 0, N * sizeof(double))); CK(cudaMemset(w.c, 0, N * sizeof(double)));
    run_phase(w, true, 0.2); run_phase(w, false, 0.2);  // calentamiento

    if (!std::strcmp(mode, "bench_memory") || !std::strcmp(mode, "bench_compute")) {
        const bool mem = !std::strcmp(mode, "bench_memory");
        const double secs = argc > 2 ? std::atof(argv[2]) : 3.0;
        const int64_t t0 = now_ns();
        const double work = run_phase(w, mem, secs);
        const double dt = (now_ns() - t0) / 1e9;
        std::printf("%s %.2f %s\n", mode, work / dt / 1e9, mem ? "GB/s" : "GFLOP/s");
        return 0;
    }
    const double phase_s = argc > 2 ? std::atof(argv[2]) : 3.0;
    const double total_s = argc > 3 ? std::atof(argv[3]) : 1e9;
    const double gap_s = argc > 4 ? std::atof(argv[4]) : 2.0;
    const int64_t t_end = now_ns() + (int64_t)(total_s * 1e9);
    bool memory = true;
    while (now_ns() < t_end) {
        std::printf("%lld %c\n", (long long)now_ns(), memory ? 'M' : 'C');
        std::fflush(stdout);
        run_phase(w, memory, phase_s);
        memory = !memory;
        CK(cudaDeviceSynchronize());
        const int64_t t_gap = now_ns() + (int64_t)(gap_s * 1e9);
        while (now_ns() < t_gap) usleep(10000);
    }
    return 0;
}
