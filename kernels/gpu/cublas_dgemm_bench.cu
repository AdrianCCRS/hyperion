/*
 * cublas_dgemm_bench.cu -- multiplicacion de matrices densas (C = alpha*A*B
 * + beta*C) en GPU via cuBLAS DGEMM, para calibrar P_pico de la calibracion
 * Roofline de GPU (ver docs/retoma/pacca/Diseno_Politica_DVFS_CPU_GPU.md
 * seccion 5). Analogo directo de kernels/dgemm/dgemm_bench.c (CPU/OpenBLAS),
 * mismo formato de salida a proposito -- reusa los mismos
 * flops_rate_stdout_pattern/runtime_seconds_stdout_pattern/success_check
 * que ya estan validados en catalog.yaml, no hace falta inventar nuevos.
 *
 * El tamano N se pasa por CLI (--size), igual que el bench de CPU.
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* PRNG determinista (xorshift64), igual que el bench de CPU -- reproducible
 * sin depender de curandGenerate*. */
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

#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                          cudaGetErrorString(err), __FILE__, __LINE__);           \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                         \
        cublasStatus_t status = (call);                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)status,      \
                          __FILE__, __LINE__);                                   \
            std::exit(1);                                                       \
        }                                                                        \
    } while (0)

int main(int argc, char** argv) {
    long n = 4096;
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
    std::vector<double> h_a(elems), h_b(elems), h_c(elems, 0.0);
    fill_matrix(h_a);
    fill_matrix(h_b);

    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c, elems * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), elems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), elems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, elems * sizeof(double)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const double alpha = 1.0, beta = 0.0;

    // Warmup fuera de la ventana medida -- la primera llamada cuBLAS incluye
    // costo de autotuning/JIT que no es representativo del estado estable.
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)n, (int)n,
                             &alpha, d_a, (int)n, d_b, (int)n, &beta, d_c, (int)n));
    CUDA_CHECK(cudaDeviceSynchronize());

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)n, (int)n,
                                 &alpha, d_a, (int)n, d_b, (int)n, &beta, d_c, (int)n));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    double seconds = t1 - t0;

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, elems * sizeof(double), cudaMemcpyDeviceToHost));

    // Verificacion por muestreo, igual criterio que el bench de CPU: verificar
    // C completo costaria lo mismo que recalcularlo (O(n^3)).
    double max_rel_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        long i = (long)(next_uniform() * (double)n);
        long j = (long)(next_uniform() * (double)n);
        if (i >= n) i = n - 1;
        if (j >= n) j = n - 1;
        double expected = 0.0;
        for (long k = 0; k < n; ++k) {
            // cuBLAS column-major: C(i,j) = sum_k A(i,k)*B(k,j), almacenado
            // column-major -> A[i + k*n], B[k + j*n], C[i + j*n].
            expected += h_a[(size_t)i + (size_t)k * n] * h_b[(size_t)k + (size_t)j * n];
        }
        double got = h_c[(size_t)i + (size_t)j * n];
        double denom = std::fabs(expected) > 1e-9 ? std::fabs(expected) : 1.0;
        double rel_error = std::fabs(got - expected) / denom;
        if (rel_error > max_rel_error) max_rel_error = rel_error;
    }
    // Tolerancia mas laxa que el bench de CPU: acumulacion FP64 en GPU con
    // reduccion en paralelo (no secuencial como la verificacion en host)
    // introduce mas error de redondeo para el mismo N grande.
    const bool ok = max_rel_error < 1e-6;

    double total_flops = (double)iterations * 2.0 * (double)n * (double)n * (double)n;
    double mops_total = total_flops / 1e6 / seconds;

    std::printf("\n cuBLAS DGEMM Benchmark (GPU)\n\n");
    std::printf(" Matrix size (N)       =                %8ld\n", n);
    std::printf(" Iterations            =                %8d\n", iterations);
    std::printf("\n");
    std::printf(" Time in seconds =    %12.6f\n", seconds);
    std::printf(" Mop/s total     =    %12.2f\n", mops_total);
    std::printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return ok ? 0 : 1;
}
