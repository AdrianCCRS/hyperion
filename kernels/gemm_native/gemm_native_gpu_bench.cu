/*
 * gemm_native_gpu_bench.cu -- GEMM denso C = A*B (NxN, doble precision) en
 * GPU con un kernel CUDA propio (tiled, memoria compartida), SIN cuBLAS.
 *
 * Por que un GEMM propio y no reusar dual_gemm_gpu_dispatch.cu o
 * kernels/gpu/cublas_dgemm_bench.cu: ambos miden OI implausible via ncu
 * (log2(OI/ridge) muy por debajo de lo esperado para GEMM denso -- ver
 * Seguimiento_Cambios_Plan_Director.md, F1-GPU-007 y su continuacion). Se
 * confirmo que el codigo fuente de dual_gemm_gpu_dispatch.cu llama a
 * cublasDgemm con parametros correctos (M=N=K=n, datos reales, sin
 * degeneracion) -- la causa NO es un bug de la llamada, sino que el
 * kernel interno y propietario que cuBLAS despacha para DGEMM en esta
 * combinacion de GPU/driver no emite las instrucciones SASS que el
 * metodo de conteo de FLOPs de ncu reconoce (sm__sass_thread_inst_
 * executed_op_dfma_pred_on.sum da CERO en ambos binarios basados en
 * cuBLAS, aunque el resultado numerico es correcto). Los demas kernels
 * `dual_*_gpu` (stencil, cholesky, spmv, axpy, fft) SI miden OI plausible
 * con ncu -- y todos son kernels CUDA escritos a mano, no cuBLAS. Este
 * archivo sigue ese mismo patron para evitar el problema de raiz en vez
 * de intentar parchearlo (no se puede: no se puede cambiar que
 * instrucciones emite el binario propietario de cuBLAS).
 *
 * Kernel: tiled GEMM clasico con memoria compartida (bloque TILE x TILE),
 * sin ninguna optimizacion agresiva -- prioridad en corrección y en que
 * las instrucciones aritmeticas sean FMA/MUL/ADD estandar, reconocibles
 * por ncu. Layout row-major (mas simple de razonar que el column-major de
 * cuBLAS).
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <vector>

#include "../dual/dispatch_timing.h"

#define TILE 16

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

static void fill_matrix(std::vector<double>& matrix) {
    for (double& value : matrix) value = next_uniform() * 2.0 - 1.0;
}

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(err), __FILE__, __LINE__);        \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

/* C = A*B, row-major NxN. Un hilo por elemento de C, bloque TILE x TILE,
 * acumulacion por tiles en memoria compartida. */
__global__ void gemm_tiled_kernel(const double* __restrict__ a,
                                   const double* __restrict__ b,
                                   double* __restrict__ c, int n) {
    __shared__ double a_tile[TILE][TILE];
    __shared__ double b_tile[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    double acc = 0.0;

    int num_tiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            (row < n && a_col < n) ? a[(size_t)row * n + a_col] : 0.0;
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < n && col < n) ? b[(size_t)b_row * n + col] : 0.0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n) c[(size_t)row * n + col] = acc;
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
    if (n <= 0 || iterations <= 0) {
        std::fprintf(stderr, "uso: %s [--size N] [--iterations M]\n", argv[0]);
        return 2;
    }

    const size_t elems = (size_t)n * (size_t)n;
    const size_t bytes = elems * sizeof(double);
    std::vector<double> h_a(elems), h_b(elems), h_c(elems, 0.0);
    fill_matrix(h_a);
    fill_matrix(h_b);

    long long cold_t0_ns = now_ns();
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    dim3 block(TILE, TILE);
    dim3 grid((unsigned)((n + TILE - 1) / TILE), (unsigned)((n + TILE - 1) / TILE));
    long long setup_complete_ns = now_ns();

    /* Primer despacho completo en frio. */
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    gemm_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, (int)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    long long cold_t1_ns = now_ns();

    long long t0_ns = now_ns();
    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
        gemm_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, (int)n);
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    long long t1_ns = now_ns();
    double seconds = t1 - t0;

    /* Verificacion por muestreo, row-major (mismo criterio que el resto
     * del catalogo: O(n) por muestra en vez de recalcular todo C). */
    double max_rel_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        long i = (long)(next_uniform() * (double)n);
        long j = (long)(next_uniform() * (double)n);
        if (i >= n) i = n - 1;
        if (j >= n) j = n - 1;
        double expected = 0.0;
        for (long k = 0; k < n; ++k) {
            expected += h_a[(size_t)i * n + k] * h_b[(size_t)k * n + j];
        }
        double got = h_c[(size_t)i * n + j];
        if (!std::isfinite(got)) {
            max_rel_error = INFINITY;
            break;
        }
        double denom = std::fabs(expected) > 1e-9 ? std::fabs(expected) : 1.0;
        double rel_error = std::fabs(got - expected) / denom;
        if (rel_error > max_rel_error) max_rel_error = rel_error;
    }
    const bool ok = max_rel_error < 1e-6;

    double total_flops = (double)iterations * 2.0 * (double)n * (double)n * (double)n;
    double mops_total = total_flops / 1e6 / seconds;

    std::printf("\n GEMM tiled benchmark (GPU / CUDA propio, sin cuBLAS)\n\n");
    std::printf(" Matrix size (N)       =                %8ld\n", n);
    std::printf(" Iterations            =                %8d\n", iterations);
    std::printf("\n");
    std::printf(" Time in seconds =    %12.6f\n", seconds);
    print_dispatch_timing(cold_t0_ns, setup_complete_ns, cold_t1_ns, t0_ns, t1_ns);
    std::printf(" Mop/s total     =    %12.2f\n", mops_total);
    std::printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return ok ? 0 : 1;
}
