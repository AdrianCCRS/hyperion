/*
 * gemm_gpu_dispatch.cu -- GEMM en GPU midiendo el costo COMPLETO de despachar
 * la operacion al device: H2D + cublasDgemm + D2H, todo DENTRO de la ventana
 * medida.
 *
 * POR QUE NO SE REUSA kernels/gpu/cublas_dgemm_bench.cu. Ese bench deja las
 * transferencias FUERA de la ventana (t0..t1 solo rodea el bucle de dgemm), y
 * eso es correcto para su proposito: calibrar P_pico del Roofline de GPU, donde
 * se quiere el pico puro de computo.
 *
 * Pero este binario alimenta la DECISION CPU-vs-GPU del selector, y ahi el
 * costo de transferencia no es ruido: es la razon fisica principal por la que
 * a N pequeño conviene quedarse en CPU. Medir sin transferencias haria que la
 * GPU ganara siempre y el modelo aprenderia una frontera que no existe.
 *
 * Semantica de una iteracion = un despacho completo e independiente
 * (H2D -> gemm -> D2H), que es como una aplicacion real llama a la operacion
 * con datos nuevos. No se amortiza la transferencia entre iteraciones a
 * proposito.
 *
 * Formato de salida identico al del bench de CPU (kernels/dgemm/dgemm_bench.c)
 * para reusar los mismos regex ya validados en catalog.yaml.
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

static void fill_matrix(std::vector<double>& matrix) {
    for (double& value : matrix) value = next_uniform() * 2.0 - 1.0;
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

    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const double alpha = 1.0, beta = 0.0;

    /* Warmup fuera de la ventana: la primera llamada cuBLAS carga kernels y
     * hace autotuning, costo que no se repite en estado estable. La creacion
     * del contexto CUDA tambien queda fuera -- el runtime del selector lo
     * tendria ya inicializado al decidir. */
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)n, (int)n,
                             &alpha, d_a, (int)n, d_b, (int)n, &beta, d_c, (int)n));
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    long long t0_ns = now_ns();

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        /* Despacho completo: subir operandos, computar, bajar resultado. */
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)n, (int)n,
                                 &alpha, d_a, (int)n, d_b, (int)n, &beta, d_c, (int)n));
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    long long t1_ns = now_ns();
    double seconds = t1 - t0;

    /* Verificacion por muestreo (verificar C completo costaria O(n^3), lo
     * mismo que recalcularlo). cuBLAS es column-major. */
    double max_rel_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        long i = (long)(next_uniform() * (double)n);
        long j = (long)(next_uniform() * (double)n);
        if (i >= n) i = n - 1;
        if (j >= n) j = n - 1;
        double expected = 0.0;
        for (long k = 0; k < n; ++k) {
            expected += h_a[(size_t)i + (size_t)k * n] * h_b[(size_t)k + (size_t)j * n];
        }
        double got = h_c[(size_t)i + (size_t)j * n];
        /* isfinite explicito: con NaN toda comparacion es falsa, incluida
         * `rel_error > max_rel_error`, asi que sin este chequeo el bench
         * reportaria SUCCESSFUL precisamente cuando el resultado esta roto. */
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
    double moved_bytes = (double)iterations * 3.0 * (double)bytes;

    std::printf("\n GEMM dispatch benchmark (GPU / cuBLAS, transferencias incluidas)\n\n");
    std::printf(" Matrix size (N)       =                %8ld\n", n);
    std::printf(" Iterations            =                %8d\n", iterations);
    std::printf(" Bytes transferred     =        %16.0f\n", moved_bytes);
    std::printf("\n");
    std::printf(" Time in seconds =    %12.6f\n", seconds);
    std::printf(" Measured region t0_ns = %lld\n", t0_ns);
    std::printf(" Measured region t1_ns = %lld\n", t1_ns);
    std::printf(" Mop/s total     =    %12.2f\n", mops_total);
    std::printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return ok ? 0 : 1;
}
