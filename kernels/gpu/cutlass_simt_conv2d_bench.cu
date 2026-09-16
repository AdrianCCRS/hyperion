/*
 * cutlass_simt_conv2d_bench.cu -- Conv2d forward (NHWC, FP32) via CUTLASS
 * ImplicitGemm con OperatorClass=Simt explicito (CUDA cores, NO Tensor
 * Core), candidato de prioridad "opcional" de la revision externa
 * (Nota_Candidatos_GPU_Compute_Bound_20260914.md) para ampliar el
 * catalogo con una segunda familia compute_bound (la primera fue
 * cutlass_simt_dgemm_bench.cu, F1-GPU-013).
 *
 * La instanciacion del kernel sigue exactamente
 * cutlass/test/unit/conv/device/conv2d_fprop_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32_sm80.cu
 * (tile 128x128x8/64x32x8, IteratorAlgorithm::kOptimized, Sm80) -- una
 * configuracion ya validada por los tests propios de CUTLASS, no una
 * eleccion propia sin precedente. Mismo patron de benchmark standalone
 * (compilado directo con nvcc -I cutlass/include, sin CMake) que
 * cutlass_simt_dgemm_bench.cu.
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

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

static void fill_tensor(std::vector<float>& t) {
    for (float& value : t) value = (float)(next_uniform() * 2.0 - 1.0);
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

// Instanciacion SIMT fp32, identica a
// test/unit/conv/device/conv2d_fprop_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32_sm80.cu
// (segundo TEST del archivo, IteratorAlgorithm::kOptimized).
using ElementA           = float;
using ElementB           = float;
using ElementC           = float;
using ElementAccumulator = float;
using ElementCompute     = float;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA,
    cutlass::layout::TensorNHWC,
    ElementB,
    cutlass::layout::TensorNHWC,
    ElementC,
    cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 1, ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

int main(int argc, char** argv) {
    // NHWC de entrada / KRSC de filtro -- tamano por defecto elegido para
    // dar un GEMM-equivalente (M=N*P*Q, N_gemm=K, K_gemm=C*R*S) del mismo
    // orden de magnitud que el DGEMM N=4096 ya en el catalogo, con reuso
    // de datos real (recorte 3x3) que un GEMM plano no tiene.
    int n = 8, h = 64, w = 64, c = 512;
    int k = 512, r = 3, s = 3;
    int pad_h = 1, pad_w = 1, stride_h = 1, stride_w = 1, dilation_h = 1, dilation_w = 1;
    int iterations = 10;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--h") == 0 && i + 1 < argc) h = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--w") == 0 && i + 1 < argc) w = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--c") == 0 && i + 1 < argc) c = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
    }

    cutlass::conv::Conv2dProblemSize problem_size(
        {n, h, w, c},
        {k, r, s, c},
        {pad_h, pad_h, pad_w, pad_w},
        {stride_h, stride_w},
        {dilation_h, dilation_w});

    int p = problem_size.P;
    int q = problem_size.Q;

    const size_t elems_a = (size_t)n * h * w * c;
    const size_t elems_b = (size_t)k * r * s * c;
    const size_t elems_c = (size_t)n * p * q * k;

    std::vector<float> h_a(elems_a), h_b(elems_b), h_c(elems_c, 0.0f);
    fill_tensor(h_a);
    fill_tensor(h_b);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, elems_a * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, elems_b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, elems_c * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), elems_a * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), elems_b * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, elems_c * sizeof(float)));

    cutlass::layout::TensorNHWC layout_a = cutlass::layout::TensorNHWC::packed({n, h, w, c});
    cutlass::layout::TensorNHWC layout_b = cutlass::layout::TensorNHWC::packed({k, r, s, c});
    cutlass::layout::TensorNHWC layout_c = cutlass::layout::TensorNHWC::packed({n, p, q, k});

    cutlass::TensorRef<ElementA, cutlass::layout::TensorNHWC> ref_a(d_a, layout_a);
    cutlass::TensorRef<ElementB, cutlass::layout::TensorNHWC> ref_b(d_b, layout_b);
    cutlass::TensorRef<ElementC, cutlass::layout::TensorNHWC> ref_c(d_c, layout_c);

    const float alpha = 1.0f, beta = 0.0f;
    Conv2dFprop conv_op;
    Conv2dFprop::Arguments args(
        problem_size, ref_a, ref_b, ref_c, ref_c,
        {alpha, beta},
        cutlass::conv::SplitKMode::kSerial);

    size_t workspace_size = Conv2dFprop::get_workspace_size(args);
    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    cutlass::Status st = conv_op.initialize(args, d_workspace);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS Conv2d initialize error: %d\n", (int)st);
        return 1;
    }

    // Warmup fuera de la ventana medida.
    st = conv_op();
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS Conv2d error (warmup): %d\n", (int)st);
        return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        st = conv_op();
        if (st != cutlass::Status::kSuccess) {
            std::fprintf(stderr, "CUTLASS Conv2d error (rep %d): %d\n", rep, (int)st);
            return 1;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    double seconds = t1 - t0;

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, elems_c * sizeof(float), cudaMemcpyDeviceToHost));

    // Verificacion: recompute un muestreo de salidas en CPU (convolucion
    // directa NHWC) y compara contra el resultado de CUTLASS.
    int verify_samples = 32;
    double max_rel_error = 0.0;
    for (int sidx = 0; sidx < verify_samples; ++sidx) {
        int ni = (int)(next_uniform() * n); if (ni >= n) ni = n - 1;
        int pi = (int)(next_uniform() * p); if (pi >= p) pi = p - 1;
        int qi = (int)(next_uniform() * q); if (qi >= q) qi = q - 1;
        int ki = (int)(next_uniform() * k); if (ki >= k) ki = k - 1;
        double acc = 0.0;
        for (int ri = 0; ri < r; ++ri) {
            int hi = pi * stride_h - pad_h + ri * dilation_h;
            if (hi < 0 || hi >= h) continue;
            for (int si = 0; si < s; ++si) {
                int wi = qi * stride_w - pad_w + si * dilation_w;
                if (wi < 0 || wi >= w) continue;
                for (int ci = 0; ci < c; ++ci) {
                    size_t a_idx = ((size_t)ni * h + hi) * w * c + (size_t)wi * c + ci;
                    size_t b_idx = ((size_t)ki * r + ri) * s * c + (size_t)si * c + ci;
                    acc += (double)h_a[a_idx] * (double)h_b[b_idx];
                }
            }
        }
        size_t c_idx = ((size_t)ni * p + pi) * q * k + (size_t)qi * k + ki;
        double got = (double)h_c[c_idx];
        double denom = std::fabs(acc) > 1e-4 ? std::fabs(acc) : 1.0;
        double rel_error = std::fabs(got - acc) / denom;
        if (rel_error > max_rel_error) max_rel_error = rel_error;
    }
    const bool ok = max_rel_error < 1e-2; // fp32 acumulado sobre C*R*S grande

    double flops_per_iter = 2.0 * (double)n * p * q * k * r * s * c;
    double total_flops = (double)iterations * flops_per_iter;
    double mops_total = total_flops / 1e6 / seconds;

    std::printf("\n CUTLASS SIMT Conv2d Fprop Benchmark (GPU)\n\n");
    std::printf(" N,H,W,C = %d,%d,%d,%d   K,R,S = %d,%d,%d   P,Q = %d,%d\n", n, h, w, c, k, r, s, p, q);
    std::printf(" Iterations            =                %8d\n", iterations);
    std::printf(" FLOPs per iteration   =              %10.3e\n", flops_per_iter);
    std::printf("\n");
    std::printf(" Time in seconds =    %12.6f\n", seconds);
    std::printf(" Mop/s total     =    %12.2f\n", mops_total);
    std::printf(" Max rel error   =    %12.6e\n", max_rel_error);
    std::printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    if (d_workspace) cudaFree(d_workspace);
    return ok ? 0 : 1;
}
