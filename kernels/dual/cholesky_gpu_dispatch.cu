/*
 * cholesky_gpu_dispatch.cu -- factorizacion de Cholesky en GPU via
 * cusolverDnDpotrf, midiendo H2D + factorizacion + D2H dentro de la ventana.
 * Contraparte exacta de cholesky_cpu_bench.c: misma matriz SPD sintetica
 * diagonal-dominante, mismo PRNG/semilla y mismo criterio de despacho.
 */
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <algorithm>
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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(err), __FILE__, __LINE__);         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CUSOLVER_CHECK(call)                                                   \
    do {                                                                       \
        cusolverStatus_t status = (call);                                      \
        if (status != CUSOLVER_STATUS_SUCCESS) {                               \
            std::fprintf(stderr, "cuSOLVER error %d at %s:%d\n", (int)status,  \
                         __FILE__, __LINE__);                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)status,    \
                         __FILE__, __LINE__);                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

/* Matriz simetrica estrictamente diagonal-dominante y positiva: SPD por
 * Gershgorin. Es O(N^2), evita que la generacion excluida de la medida haga
 * un GEMM O(N^3) y cause timeouts antes de llegar al primer marcador. */
static void make_spd_matrix_host(std::vector<double>& A, long n) {
    std::fill(A.begin(), A.end(), 0.0);
    for (long i = 0; i < n; ++i) {
        for (long j = i + 1; j < n; ++j) {
            double value = next_uniform() * 2.0 - 1.0;
            A[(size_t)i * n + j] = value;
            A[(size_t)j * n + i] = value;
        }
    }
    for (long i = 0; i < n; ++i) {
        double radius = 0.0;
        for (long j = 0; j < n; ++j) {
            if (i != j) radius += std::fabs(A[(size_t)i * n + j]);
        }
        A[(size_t)i * n + i] = radius + 1.0;
    }
}

int main(int argc, char** argv) {
    long n = 512;
    int iterations = 10;
    int verify_samples = 32;

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
    std::vector<double> h_original(elems), h_work(elems);
    make_spd_matrix_host(h_original, n);

    long long cold_t0_ns = now_ns();
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    double* d_A;
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    int lwork = 0;
    /* cuSOLVER espera column-major; A es simetrica asi que row-major ==
     * column-major para efectos de este layout (A[i][j]==A[j][i]). */
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER,
                                                (int)n, d_A, (int)n, &lwork));
    double* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(double)));
    long long setup_complete_ns = now_ns();

    int info_host = 0;
    /* Primer despacho completo en frio, incluida la carga perezosa de
     * kernels de cuSOLVER. */
    CUDA_CHECK(cudaMemcpy(d_A, h_original.data(), bytes, cudaMemcpyHostToDevice));
    CUSOLVER_CHECK(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, (int)n,
                                     d_A, (int)n, d_work, lwork, d_info));
    CUDA_CHECK(cudaMemcpy(&info_host, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_work.data(), d_A, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    long long cold_t1_ns = now_ns();

    long long t0_ns = now_ns();

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        CUDA_CHECK(cudaMemcpy(d_A, h_original.data(), bytes, cudaMemcpyHostToDevice));
        CUSOLVER_CHECK(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, (int)n,
                                         d_A, (int)n, d_work, lwork, d_info));
        CUDA_CHECK(cudaMemcpy(&info_host, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_work.data(), d_A, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    long long t1_ns = now_ns();
    double seconds = t1 - t0;

    double max_abs_error = 0.0;
    if (info_host == 0) {
        for (int s = 0; s < verify_samples; ++s) {
            long i = (long)(next_uniform() * (double)n);
            long j = (long)(next_uniform() * (double)n);
            if (i >= n) i = n - 1;
            if (j >= n) j = n - 1;
            long lo = i < j ? i : j, hi = i < j ? j : i;
            double sum = 0.0;
            /* cuSOLVER es SIEMPRE column-major (a diferencia de LAPACKE, que
             * tiene el flag LAPACK_ROW_MAJOR y transpone por dentro) -- L(fila,
             * columna) vive en el offset columna*n+fila, no fila*n+columna
             * como en la contraparte de CPU. Sin este ajuste se lee L
             * transpuesta y la verificacion falla aunque la factorizacion
             * (info=0) sea correcta -- diagnosticado 2026-08-27 imprimiendo
             * info_host directamente: la factorizacion siempre dio 0. */
            for (long k = 0; k <= lo; ++k) {
                sum += h_work[(size_t)k * n + hi] * h_work[(size_t)k * n + lo];
            }
            double expected = h_original[(size_t)i * n + j];
            if (!std::isfinite(sum)) { max_abs_error = INFINITY; break; }
            double abs_error = std::fabs(sum - expected);
            if (abs_error > max_abs_error) max_abs_error = abs_error;
        }
    } else {
        max_abs_error = INFINITY;
    }
    const bool ok = max_abs_error < 1e-6 * (double)n;

    double total_flops = (double)iterations * (1.0 / 3.0) * (double)n * (double)n * (double)n;
    double mops_total = total_flops / 1e6 / seconds;
    double moved_bytes = (double)iterations * 2.0 * (double)bytes;

    std::printf("\n Cholesky factorization dispatch benchmark (GPU / cuSOLVER, transferencias incluidas)\n\n");
    std::printf(" Matrix size (N)       =                %8ld\n", n);
    std::printf(" Iterations            =                %8d\n", iterations);
    std::printf(" Bytes transferred     =        %16.0f\n", moved_bytes);
    std::printf("\n");
    std::printf(" Time in seconds =    %12.6f\n", seconds);
    print_dispatch_timing(cold_t0_ns, setup_complete_ns, cold_t1_ns, t0_ns, t1_ns);
    std::printf(" Mop/s total     =    %12.2f\n", mops_total);
    std::printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    cudaFree(d_A);
    cudaFree(d_work);
    cudaFree(d_info);
    cusolverDnDestroy(handle);
    return ok ? 0 : 1;
}
