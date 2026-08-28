/*
 * axpy_gpu_dispatch.cu -- y = alpha*x + y (BLAS-1) en GPU via cublasDaxpy,
 * midiendo H2D(x,y) + daxpy + D2H(y) dentro de la ventana medida.
 *
 * Misma logica de "costo de despacho completo" que gemm_gpu_dispatch.cu, pero
 * aqui el efecto es mucho mas fuerte: AXPY mueve 3 vectores de N elementos
 * (lee x,y; escribe y) para hacer solo 2*N FLOPs -- intensidad operacional
 * O(1). Con transferencia incluida, la GPU no tiene forma de ganar salvo que
 * el ancho de banda PCIe supere al de memoria de CPU, lo cual no ocurre en
 * esta maquina. Es la ancla del extremo "siempre CPU" del catalogo.
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <vector>

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

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)status,    \
                         __FILE__, __LINE__);                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

int main(int argc, char** argv) {
    long n = 1000000;
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

    const size_t bytes = (size_t)n * sizeof(double);
    std::vector<double> h_x(n), h_y(n), h_y_original(n);
    fill(h_x);
    fill(h_y);
    h_y_original = h_y;

    double *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const double alpha = 2.5;

    /* Warmup fuera de ventana. */
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y_original.data(), bytes, cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasDaxpy(handle, (int)n, &alpha, d_x, 1, d_y, 1));
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    long long t0_ns = now_ns();

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y_original.data(), bytes, cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasDaxpy(handle, (int)n, &alpha, d_x, 1, d_y, 1));
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    long long t1_ns = now_ns();
    double seconds = t1 - t0;

    double max_abs_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        long idx = (long)(next_uniform() * (double)n);
        if (idx >= n) idx = n - 1;
        double expected = alpha * h_x[idx] + h_y_original[idx];
        double got = h_y[idx];
        if (!std::isfinite(got)) { max_abs_error = INFINITY; break; }
        double abs_error = std::fabs(got - expected);
        if (abs_error > max_abs_error) max_abs_error = abs_error;
    }
    const bool ok = max_abs_error < 1e-9;

    double total_flops = (double)iterations * 2.0 * (double)n;
    double mops_total = total_flops / 1e6 / seconds;
    double moved_bytes = (double)iterations * 3.0 * (double)bytes;

    std::printf("\n AXPY dispatch benchmark (GPU / cuBLAS, transferencias incluidas)\n\n");
    std::printf(" Vector size (N)       =                %8ld\n", n);
    std::printf(" Iterations            =                %8d\n", iterations);
    std::printf(" Bytes transferred     =        %16.0f\n", moved_bytes);
    std::printf("\n");
    std::printf(" Time in seconds =    %12.6f\n", seconds);
    std::printf(" Measured region t0_ns = %lld\n", t0_ns);
    std::printf(" Measured region t1_ns = %lld\n", t1_ns);
    std::printf(" Mop/s total     =    %12.2f\n", mops_total);
    std::printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y);
    return ok ? 0 : 1;
}
