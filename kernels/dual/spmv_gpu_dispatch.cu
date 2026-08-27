/*
 * spmv_gpu_dispatch.cu -- y = A*x para la misma matriz CSR banda (7
 * no-ceros/fila) que spmv_cpu_bench.c, en GPU via la API generica de
 * cuSPARSE (cusparseSpMV), midiendo H2D + SpMV + D2H dentro de la ventana.
 */
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <vector>

#define NNZ_PER_ROW 7

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

#define CUSPARSE_CHECK(call)                                                   \
    do {                                                                       \
        cusparseStatus_t status = (call);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            std::fprintf(stderr, "cuSPARSE error %d at %s:%d\n", (int)status,  \
                         __FILE__, __LINE__);                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

/* Mismo algoritmo y misma secuencia de PRNG que build_csr() en
 * spmv_cpu_bench.c -- misma matriz para la misma N. cuSPARSE usa
 * row_ptr de tipo int (indexado 0), col_idx de tipo int (no long, a
 * diferencia del lado CPU -- se convierte al construir). */
static void build_csr(long n, std::vector<int>& row_ptr, std::vector<int>& col_idx,
                       std::vector<double>& values) {
    row_ptr.resize(n + 1);
    col_idx.resize((size_t)n * NNZ_PER_ROW);
    values.resize((size_t)n * NNZ_PER_ROW);
    row_ptr[0] = 0;
    for (long i = 0; i < n; ++i) {
        for (int k = 0; k < NNZ_PER_ROW; ++k) {
            long offset = k - NNZ_PER_ROW / 2;
            long col = ((i + offset) % n + n) % n;
            col_idx[i * NNZ_PER_ROW + k] = (int)col;
            values[i * NNZ_PER_ROW + k] = next_uniform() * 2.0 - 1.0;
        }
        row_ptr[i + 1] = (int)((i + 1) * NNZ_PER_ROW);
    }
}

static void fill_vector(std::vector<double>& v) {
    for (double& value : v) value = next_uniform() * 2.0 - 1.0;
}

int main(int argc, char** argv) {
    long n = 1000000;
    int iterations = 200;
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

    std::vector<int> h_row_ptr, h_col_idx;
    std::vector<double> h_values;
    build_csr(n, h_row_ptr, h_col_idx, h_values);
    long nnz = (long)h_values.size();

    std::vector<double> h_x(n), h_y(n);
    fill_vector(h_x);

    int *d_row_ptr, *d_col_idx;
    double *d_values, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, h_row_ptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, h_col_idx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, h_values.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)n * sizeof(double)));
    /* La topologia (row_ptr/col_idx/values) no cambia entre despachos -- solo
     * se sube una vez, fuera del bucle medido, igual que un runtime real que
     * factoriza/ensambla la matriz una sola vez y la reutiliza. */
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), h_row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx.data(), h_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), h_values.size() * sizeof(double), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, n, n, nnz, d_row_ptr, d_col_idx, d_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, n, d_y, CUDA_R_64F));

    const double alpha = 1.0, beta = 0.0;
    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                            CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
    void* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));

    /* Warmup fuera de ventana. */
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                 vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        /* Despacho: subir x, multiplicar, bajar y -- la matriz (subida antes
         * del bucle) se reutiliza igual que en un despliegue real. */
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                     vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    double seconds = t1 - t0;

    double max_abs_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        long i = (long)(next_uniform() * (double)n);
        if (i >= n) i = n - 1;
        double expected = 0.0;
        for (int k = h_row_ptr[i]; k < h_row_ptr[i + 1]; ++k) expected += h_values[k] * h_x[h_col_idx[k]];
        double got = h_y[i];
        if (!std::isfinite(got)) { max_abs_error = INFINITY; break; }
        double abs_error = std::fabs(got - expected);
        if (abs_error > max_abs_error) max_abs_error = abs_error;
    }
    const bool ok = max_abs_error < 1e-9;

    double total_flops = (double)iterations * 2.0 * (double)nnz;
    double mops_total = total_flops / 1e6 / seconds;
    double moved_bytes = (double)iterations * 2.0 * (double)n * sizeof(double);

    std::printf("\n SpMV CSR dispatch benchmark (GPU / cuSPARSE, transferencias incluidas)\n\n");
    std::printf(" Matrix rows (N)       =                %8ld\n", n);
    std::printf(" Iterations            =                %8d\n", iterations);
    std::printf(" Bytes transferred     =        %16.0f\n", moved_bytes);
    std::printf("\n");
    std::printf(" Time in seconds =    %12.6f\n", seconds);
    std::printf(" Mop/s total     =    %12.2f\n", mops_total);
    std::printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_values);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_buffer);
    return ok ? 0 : 1;
}
