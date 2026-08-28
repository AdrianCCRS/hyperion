/*
 * fft_gpu_dispatch.cu -- FFT compleja 2D (N x N, doble precision) en GPU via
 * cuFFT, midiendo el costo COMPLETO de despachar: H2D + cufftExecZ2Z + D2H
 * dentro de la ventana medida.
 *
 * Misma decision de diseño que gemm_gpu_dispatch.cu y por la misma razon: el
 * selector CPU/GPU tiene que ver el costo de transferencia, porque es lo que
 * decide la frontera a tamaño pequeño. Aqui pesa aun mas que en GEMM: la FFT
 * mueve los mismos O(N^2) bytes pero hace solo O(N^2 log N) trabajo, contra
 * O(N^3) de GEMM, asi que la transferencia domina en un rango de tamaños
 * mucho mas amplio.
 *
 * La creacion del contexto y del plan cuFFT forma parte de la region cold;
 * la region warm separada conserva el caso de reutilizacion.
 */
#include <cufft.h>
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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(err), __FILE__, __LINE__);         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult status = (call);                                           \
        if (status != CUFFT_SUCCESS) {                                         \
            std::fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)status,     \
                         __FILE__, __LINE__);                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

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
    const size_t bytes = elems * sizeof(cufftDoubleComplex);
    std::vector<cufftDoubleComplex> h_data(elems), h_original(elems);
    for (size_t i = 0; i < elems; ++i) {
        h_data[i].x = next_uniform() * 2.0 - 1.0;
        h_data[i].y = next_uniform() * 2.0 - 1.0;
        h_original[i] = h_data[i];
    }

    long long cold_t0_ns = now_ns();
    cufftDoubleComplex* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, (int)n, (int)n, CUFFT_Z2Z));
    long long setup_complete_ns = now_ns();

    /* Primer despacho en frio: incluye la carga perezosa de kernels cuFFT. */
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
    CUFFT_CHECK(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    long long cold_t1_ns = now_ns();

    long long t0_ns = now_ns();

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        /* Despacho completo: subir, transformar, bajar. */
        CUDA_CHECK(cudaMemcpy(d_data, h_original.data(), bytes, cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    long long t1_ns = now_ns();
    double seconds = t1 - t0;

    /* Verificacion fuera de ventana: la inversa del resultado debe devolver el
     * input salvo el factor N^2 que cuFFT no normaliza (misma convencion que
     * FFTW en la contraparte de CPU). */
    CUFFT_CHECK(cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE));
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    const double norm = (double)elems;
    /* Mismo criterio que la contraparte de CPU (fft_cpu_bench.c): error
     * absoluto sobre datos de magnitud O(1), y chequeo isfinite explicito --
     * sin el, un inf produce NaN en la resta y toda comparacion con NaN es
     * falsa, con lo que el bench reportaria SUCCESSFUL en el caso mas roto. */
    double max_abs_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        size_t idx = (size_t)(next_uniform() * (double)elems);
        if (idx >= elems) idx = elems - 1;
        double got_re = h_data[idx].x / norm;
        double got_im = h_data[idx].y / norm;
        if (!std::isfinite(got_re) || !std::isfinite(got_im)) {
            max_abs_error = INFINITY;
            break;
        }
        double abs_error = std::fabs(got_re - h_original[idx].x)
                         + std::fabs(got_im - h_original[idx].y);
        if (abs_error > max_abs_error) max_abs_error = abs_error;
    }
    const bool ok = max_abs_error < 1e-9;

    double m = (double)elems;
    double total_flops = (double)iterations * 5.0 * m * std::log2(m);
    double mops_total = total_flops / 1e6 / seconds;
    double moved_bytes = (double)iterations * 2.0 * (double)bytes;

    std::printf("\n FFT 2D complex dispatch benchmark (GPU / cuFFT, transferencias incluidas)\n\n");
    std::printf(" Grid size (N)         =                %8ld\n", n);
    std::printf(" Iterations            =                %8d\n", iterations);
    std::printf(" Bytes transferred     =        %16.0f\n", moved_bytes);
    std::printf("\n");
    std::printf(" Time in seconds =    %12.6f\n", seconds);
    print_dispatch_timing(cold_t0_ns, setup_complete_ns, cold_t1_ns, t0_ns, t1_ns);
    std::printf(" Mop/s total     =    %12.2f\n", mops_total);
    std::printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    cufftDestroy(plan);
    cudaFree(d_data);
    return ok ? 0 : 1;
}
