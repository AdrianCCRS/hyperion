/*
 * stencil_gpu_dispatch.cu -- Jacobi 2D de 5 puntos en GPU (kernel CUDA
 * propio, un thread por celda), midiendo H2D + kernel + D2H dentro de la
 * ventana. Contraparte exacta de stencil_cpu_bench.c: mismo patron de
 * vecinos, mismo criterio de "un despacho = un paso completo desde el mismo
 * estado inicial".
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <vector>

#include "dispatch_timing.h"

/* ARC: sello absoluto de la region medida (CLOCK_MONOTONIC, mismo reloj que
 * usa la telemetria -- telemetry/include/telemetry/metrics.hpp). Permite al
 * constructor del dataset filtrar las ventanas al bucle realmente medido, en
 * vez de promediar sobre todo el proceso (que en GPU es ~85% inicializacion
 * de contexto CUDA). Ver docs/general/metodologia_selector_cpu_gpu_20260827.md
 * seccion 6.9. */
static long long now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static double next_uniform() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) / (double)(1ULL << 53);
}

static void fill(std::vector<double>& v) {
    for (double& value : v) value = next_uniform() * 2.0 - 1.0;
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(err), __FILE__, __LINE__);         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

__global__ void jacobi_kernel(const double* in, double* out, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    int idx = i * n + j;
    if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
        out[idx] = in[idx];
    } else {
        out[idx] = 0.25 * (in[idx - n] + in[idx + n] + in[idx - 1] + in[idx + 1]);
    }
}

int main(int argc, char** argv) {
    long n = 1024;
    int iterations = 10;
    int verify_samples = 64;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            n = std::strtol(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = (int)std::strtol(argv[++i], nullptr, 10);
        }
    }
    if (n <= 2 || iterations <= 0) {
        std::fprintf(stderr, "uso: %s [--size N>2] [--iterations M]\n", argv[0]);
        return 2;
    }

    const size_t elems = (size_t)n * (size_t)n;
    const size_t bytes = elems * sizeof(double);
    std::vector<double> h_in(elems), h_out(elems), h_original(elems);
    fill(h_in);
    h_original = h_in;

    long long cold_t0_ns = now_ns();
    double *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    dim3 block(16, 16);
    dim3 grid((unsigned)((n + block.x - 1) / block.x),
              (unsigned)((n + block.y - 1) / block.y));
    long long setup_complete_ns = now_ns();

    /* Primer despacho en frio: incluye carga/JIT/cache de la primera
     * ejecucion del kernel. */
    CUDA_CHECK(cudaMemcpy(d_in, h_original.data(), bytes, cudaMemcpyHostToDevice));
    jacobi_kernel<<<grid, block>>>(d_in, d_out, (int)n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    long long cold_t1_ns = now_ns();

    long long t0_ns = now_ns();

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        CUDA_CHECK(cudaMemcpy(d_in, h_original.data(), bytes, cudaMemcpyHostToDevice));
        jacobi_kernel<<<grid, block>>>(d_in, d_out, (int)n);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    long long t1_ns = now_ns();
    double seconds = t1 - t0;

    double max_abs_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        long i = 1 + (long)(next_uniform() * (double)(n - 2));
        long j = 1 + (long)(next_uniform() * (double)(n - 2));
        if (i >= n - 1) i = n - 2;
        if (j >= n - 1) j = n - 2;
        long idx = i * n + j;
        double expected = 0.25 * (h_original[idx - n] + h_original[idx + n]
                                 + h_original[idx - 1] + h_original[idx + 1]);
        double got = h_out[idx];
        if (!std::isfinite(got)) { max_abs_error = INFINITY; break; }
        double abs_error = std::fabs(got - expected);
        if (abs_error > max_abs_error) max_abs_error = abs_error;
    }
    const bool ok = max_abs_error < 1e-9;

    double interior_cells = (double)(n - 2) * (double)(n - 2);
    double total_flops = (double)iterations * 5.0 * interior_cells;
    double mops_total = total_flops / 1e6 / seconds;
    double moved_bytes = (double)iterations * 2.0 * (double)bytes;

    std::printf("\n Jacobi 2D stencil dispatch benchmark (GPU / CUDA, transferencias incluidas)\n\n");
    std::printf(" Grid size (N)         =                %8ld\n", n);
    std::printf(" Iterations            =                %8d\n", iterations);
    std::printf(" Bytes transferred     =        %16.0f\n", moved_bytes);
    std::printf("\n");
    std::printf(" Time in seconds =    %12.6f\n", seconds);
    print_dispatch_timing(cold_t0_ns, setup_complete_ns, cold_t1_ns, t0_ns, t1_ns);
    std::printf(" Mop/s total     =    %12.2f\n", mops_total);
    std::printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
}
