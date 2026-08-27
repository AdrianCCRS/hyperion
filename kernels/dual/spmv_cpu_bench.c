/*
 * spmv_cpu_bench.c -- y = A*x para una matriz dispersa NxN en formato CSR,
 * 7 no-ceros por fila (patron de banda, columnas {i-3..i+3} mod N -- misma
 * conectividad que un stencil 1D de 7 puntos, generada sinteticamente),
 * en CPU con OpenMP.
 *
 * Sexta y ultima operacion del catalogo dual: memory-bound IRREGULAR --
 * a diferencia de stencil_cpu (acceso predecible, offsets fijos conocidos
 * en tiempo de compilacion), SpMV indirecciona por col_idx[]: el patron de
 * acceso a x[] no es secuencial ni prefetcheable por hardware de la misma
 * manera. Contraparte de spmv_gpu_dispatch.cu (cuSPARSE).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define NNZ_PER_ROW 7

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static double next_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) / (double)(1ULL << 53);
}

/* CSR con NNZ_PER_ROW fijo por fila -- construccion determinista, misma
 * secuencia de PRNG que la contraparte GPU para la misma N. */
static void build_csr(long n, int **row_ptr, long **col_idx, double **values) {
    *row_ptr = malloc((size_t)(n + 1) * sizeof(int));
    *col_idx = malloc((size_t)n * NNZ_PER_ROW * sizeof(long));
    *values = malloc((size_t)n * NNZ_PER_ROW * sizeof(double));
    (*row_ptr)[0] = 0;
    for (long i = 0; i < n; ++i) {
        for (int k = 0; k < NNZ_PER_ROW; ++k) {
            long offset = k - NNZ_PER_ROW / 2;
            long col = ((i + offset) % n + n) % n;
            (*col_idx)[i * NNZ_PER_ROW + k] = col;
            (*values)[i * NNZ_PER_ROW + k] = next_uniform() * 2.0 - 1.0;
        }
        (*row_ptr)[i + 1] = (int)((i + 1) * NNZ_PER_ROW);
    }
}

static void fill_vector(double *v, long n) {
    for (long i = 0; i < n; ++i) v[i] = next_uniform() * 2.0 - 1.0;
}

int main(int argc, char **argv) {
    long n = 1000000;
    int iterations = 200;
    int verify_samples = 64;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            n = strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = (int)strtol(argv[++i], NULL, 10);
        }
    }
    if (n <= 0 || iterations <= 0) {
        fprintf(stderr, "uso: %s [--size N] [--iterations M]\n", argv[0]);
        return 2;
    }

    int *row_ptr; long *col_idx; double *values;
    build_csr(n, &row_ptr, &col_idx, &values);

    double *x = malloc((size_t)n * sizeof(double));
    double *y = malloc((size_t)n * sizeof(double));
    fill_vector(x, n);

    /* Warmup fuera de ventana. */
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) sum += values[k] * x[col_idx[k]];
        y[i] = sum;
    }

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) sum += values[k] * x[col_idx[k]];
            y[i] = sum;
        }
    }
    double t1 = now_seconds();
    double seconds = t1 - t0;

    double max_abs_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        long i = (long)(next_uniform() * (double)n);
        if (i >= n) i = n - 1;
        double expected = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) expected += values[k] * x[col_idx[k]];
        double got = y[i];
        if (!isfinite(got)) { max_abs_error = INFINITY; break; }
        double abs_error = fabs(got - expected);
        if (abs_error > max_abs_error) max_abs_error = abs_error;
    }
    const int ok = max_abs_error < 1e-9;

    double total_flops = (double)iterations * 2.0 * (double)n * NNZ_PER_ROW;
    double mops_total = total_flops / 1e6 / seconds;

    printf("\n SpMV CSR benchmark (CPU / OpenMP)\n\n");
    printf(" Matrix rows (N)       =                %8ld\n", n);
    printf(" Iterations            =                %8d\n", iterations);
    printf("\n");
    printf(" Time in seconds =    %12.6f\n", seconds);
    printf(" Mop/s total     =    %12.2f\n", mops_total);
    printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    free(row_ptr); free(col_idx); free(values); free(x); free(y);
    return ok ? 0 : 1;
}
