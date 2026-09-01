/*
 * ert_probe_gpu.cu -- microbenchmark de FLOPs pico de GPU (regimen
 * compute-bound), equivalente en CUDA de kernels/ert/ert_probe.c.
 *
 * ARC-75/76: la calibracion de P_pico de GPU usaba `cublas_dgemm_bench`
 * (cublasDgemm) -- pero cuBLAS elige internamente, sin que se le pida, una
 * ruta acelerada por Tensor Cores (confirmado con `ncu`:
 * cutlass_80_tensorop_d884gemm) que los demas kernels del catalogo (Rodinia,
 * CUDA escrito a mano, sin ninguna libreria de terceros) no pueden alcanzar.
 * Usar ese numero como techo universal de Roofline compara una ruta de
 * hardware que casi nadie usa contra kernels que nunca pasan por ahi -- el
 * mismo error de "unidades incompatibles" que ya se corrigio en CPU con
 * npb_ep/npb_is (ARC-57), aplicado aqui al eje de cómputo en vez de al de
 * FLOPs reportados.
 *
 * Mismo criterio que ert_probe.c (CAL-03: microbenchmark propio de FLOPs
 * pico, sin depender del driver Python/plantillas Batch de ERT completo,
 * que tampoco existen para GPU en este cluster): un bucle de operaciones
 * tipo FMA (beta = beta*val + alpha, desenrollado 8 veces = 16 FLOPs por
 * iteracion) que cada hilo corre enteramente en un registro -- cada hilo
 * lee un elemento de memoria una sola vez al principio y escribe una sola
 * vez al final, así que el trafico de memoria es fijo (no crece con el
 * numero de iteraciones) y la intensidad operacional del propio probe
 * queda deliberadamente sin techo -- es el extremo compute-bound puro que
 * un calibrador de P_pico necesita, sin pasar por ninguna libreria de
 * NVIDIA que pueda elegir una ruta de ejecucion distinta a la de los
 * kernels que se van a clasificar despues.
 *
 * Sin cuBLAS, sin CUTLASS, sin intrinsics de Tensor Cores (wmma/mma) --
 * aritmetica float/double corriente, un solo __global__ por precision.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define FLOPS_PER_ITER 16

template <typename T>
__global__ void ert_kernel(T *buf, uint64_t ntrials, T alpha_init) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    T val = buf[idx];
    T alpha = alpha_init;
    for (uint64_t trial = 0; trial < ntrials; ++trial) {
        T beta = (T)0.8;
        beta = beta * val + alpha;
        beta = beta * val + alpha;
        beta = beta * val + alpha;
        beta = beta * val + alpha;
        beta = beta * val + alpha;
        beta = beta * val + alpha;
        beta = beta * val + alpha;
        beta = beta * val + alpha;
        val = beta;
        alpha = alpha * ((T)1.0 - (T)1e-8);
    }
    buf[idx] = val;
}

static void die_on_cuda_error(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

template <typename T>
static double run_precision(uint64_t nelems, int threads_per_block) {
    T *d_buf = nullptr;
    die_on_cuda_error(cudaMalloc(&d_buf, nelems * sizeof(T)), "cudaMalloc");

    T *h_buf = (T *)malloc(nelems * sizeof(T));
    for (uint64_t i = 0; i < nelems; ++i) {
        h_buf[i] = (T)(1.0 + (double)(i % 7) * 1e-3);
    }
    die_on_cuda_error(cudaMemcpy(d_buf, h_buf, nelems * sizeof(T), cudaMemcpyHostToDevice),
                       "cudaMemcpy H2D");
    free(h_buf);

    int blocks = (int)((nelems + threads_per_block - 1) / threads_per_block);

    double best_gflops = 0.0;
    for (uint64_t trials = 1; trials <= (1ULL << 24); trials *= 4) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        ert_kernel<T><<<blocks, threads_per_block>>>(d_buf, trials, (T)0.5);
        cudaEventRecord(stop);
        die_on_cuda_error(cudaEventSynchronize(stop), "kernel launch/sync");

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        double seconds = ms / 1000.0;
        if (seconds <= 0.0) {
            continue;
        }
        double total_flops = (double)nelems * (double)trials * FLOPS_PER_ITER;
        double gflops = total_flops / seconds / 1e9;
        if (gflops > best_gflops) {
            best_gflops = gflops;
        }
        if (seconds > 3.0) {
            break;
        }
    }

    cudaFree(d_buf);
    return best_gflops;
}

int main(int argc, char **argv) {
    const char *precision = (argc > 1) ? argv[1] : "fp64";
    uint64_t nelems = (argc > 2) ? strtoull(argv[2], nullptr, 10) : (32ULL << 20); // 32M elems
    int threads_per_block = 256;

    double best_gflops;
    if (strcmp(precision, "fp32") == 0) {
        best_gflops = run_precision<float>(nelems, threads_per_block);
    } else if (strcmp(precision, "fp64") == 0) {
        best_gflops = run_precision<double>(nelems, threads_per_block);
    } else {
        fprintf(stderr, "precision desconocida: %s (use fp32 o fp64)\n", precision);
        return 1;
    }

    printf("ERT_PROBE_GPU_RESULT\n");
    printf("PRECISION %s\n", precision);
    printf("ELEMENTS %llu\n", (unsigned long long)nelems);
    printf("GFLOPs/sec: %.3f\n", best_gflops);
    return 0;
}
