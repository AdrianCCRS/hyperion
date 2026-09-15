/*
 * cutlass_simt_dgemm_bench.cu -- DGEMM (C = alpha*A*B + beta*C) via CUTLASS
 * con OperatorClass=Simt explicito (CUDA cores, NO Tensor Core), candidato
 * de la revision externa (Nota_Candidatos_GPU_Compute_Bound_20260914.md,
 * F1-GPU-012/013) para recuperar un GEMM FP64 optimizado por el proveedor
 * despues de que CUBLAS_PEDANTIC_MATH NO logro desactivar la ruta Tensor
 * Core de cuBLAS en este entorno (confirmado con ncu, mismo kernel
 * cutlass_80_tensorop_d884gemm con o sin pedantic).
 *
 * La instanciacion del template sigue exactamente
 * cutlass/test/unit/gemm/device/simt_dgemm_nn_sm50.cu (ya en el repo de
 * CUTLASS en pacca, ~/cutlass, v2.11.0) -- tile 128x32x8/32x16x8,
 * InstructionShape 1x1x1, ColumnMajor/ColumnMajor/RowMajor, la
 * configuracion SIMT double ya validada por los tests propios de CUTLASS,
 * no una eleccion propia sin precedente.
 *
 * Formato de I/O y verificacion identicos a cublas_dgemm_pedantic_bench.cu
 * (mismo PRNG xorshift64, mismo criterio de muestreo) para comparabilidad
 * directa.
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "cutlass/gemm/device/gemm.h"

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

#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                          cudaGetErrorString(err), __FILE__, __LINE__);           \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

// Instanciacion SIMT double, identica a
// test/unit/gemm/device/simt_dgemm_nn_sm50.cu (config "128x32x8").
using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 8>;
using WarpShape = cutlass::gemm::GemmShape<32, 16, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    double, 1, double, double>;

using CutlassSimtDgemm = cutlass::gemm::device::Gemm<
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2 // Stages
>;

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

    const double alpha = 1.0, beta = 0.0;
    CutlassSimtDgemm gemm_op;
    CutlassSimtDgemm::Arguments args(
        {(int)n, (int)n, (int)n},
        {d_a, (int)n}, {d_b, (int)n}, {d_c, (int)n}, {d_c, (int)n},
        {alpha, beta});

    // Warmup fuera de la ventana medida.
    cutlass::Status st = gemm_op(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS GEMM error (warmup): %d\n", (int)st);
        return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        st = gemm_op(args);
        if (st != cutlass::Status::kSuccess) {
            std::fprintf(stderr, "CUTLASS GEMM error (rep %d): %d\n", rep, (int)st);
            return 1;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    double seconds = t1 - t0;

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, elems * sizeof(double), cudaMemcpyDeviceToHost));

    // C es RowMajor aqui (distinto de cublas_dgemm_pedantic_bench.cu, que es
    // ColumnMajor) -- el layout de salida se eligio para que coincida con
    // la firma de Gemm mas simple de CUTLASS; se ajusta la verificacion.
    double max_rel_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        long i = (long)(next_uniform() * (double)n);
        long j = (long)(next_uniform() * (double)n);
        if (i >= n) i = n - 1;
        if (j >= n) j = n - 1;
        double expected = 0.0;
        for (long k = 0; k < n; ++k) {
            // A, B column-major: A(i,k)=A[i+k*n], B(k,j)=B[k+j*n].
            expected += h_a[(size_t)i + (size_t)k * n] * h_b[(size_t)k + (size_t)j * n];
        }
        // C row-major: C(i,j) = C[i*n + j].
        double got = h_c[(size_t)i * n + (size_t)j];
        double denom = std::fabs(expected) > 1e-9 ? std::fabs(expected) : 1.0;
        double rel_error = std::fabs(got - expected) / denom;
        if (rel_error > max_rel_error) max_rel_error = rel_error;
    }
    const bool ok = max_rel_error < 1e-6;

    double total_flops = (double)iterations * 2.0 * (double)n * (double)n * (double)n;
    double mops_total = total_flops / 1e6 / seconds;

    std::printf("\n CUTLASS SIMT DGEMM Benchmark (GPU)\n\n");
    std::printf(" Matrix size (N)       =                %8ld\n", n);
    std::printf(" Iterations            =                %8d\n", iterations);
    std::printf("\n");
    std::printf(" Time in seconds =    %12.6f\n", seconds);
    std::printf(" Mop/s total     =    %12.2f\n", mops_total);
    std::printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return ok ? 0 : 1;
}
