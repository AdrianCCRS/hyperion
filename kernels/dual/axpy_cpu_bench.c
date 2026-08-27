/*
 * axpy_cpu_bench.c -- y = alpha*x + y (BLAS-1) en CPU via cblas_daxpy.
 *
 * Ancla del extremo memory-bound del catalogo dual: AXPY hace UNA operacion
 * de punto flotante por CADA elemento tocado (O(N) trabajo, O(N) bytes),
 * frente a O(N^3)/O(N^2) de GEMM. Con el costo de transferencia incluido a
 * proposito (ver cabecera de axpy_gpu_dispatch.cu), deberia perder contra
 * CPU en TODO el rango de N -- si eso no ocurre, algo esta mal medido.
 *
 * Formato de salida identico al resto del catalogo dual.
 */
#include <cblas.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

static void fill(double *v, long n) {
    for (long i = 0; i < n; ++i) v[i] = next_uniform() * 2.0 - 1.0;
}

int main(int argc, char **argv) {
    long n = 1000000;
    int iterations = 10;
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

    double *x = malloc((size_t)n * sizeof(double));
    double *y = malloc((size_t)n * sizeof(double));
    double *y_original = malloc((size_t)n * sizeof(double));
    if (!x || !y || !y_original) {
        fprintf(stderr, "fallo de asignacion para N=%ld\n", n);
        return 2;
    }
    fill(x, n);
    fill(y, n);
    memcpy(y_original, y, (size_t)n * sizeof(double));
    const double alpha = 2.5;

    /* Warmup fuera de ventana. */
    cblas_daxpy((int)n, alpha, x, 1, y, 1);
    memcpy(y, y_original, (size_t)n * sizeof(double));

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        /* Cada iteracion restaura y antes de acumular -- si no, y crece sin
         * limite y despues de pocas iteraciones el resultado deja de ser
         * comparable entre reps. Restaurar es tambien lo simetrico de lo que
         * hace el lado GPU (vuelve a subir y_original en cada despacho). */
        memcpy(y, y_original, (size_t)n * sizeof(double));
        cblas_daxpy((int)n, alpha, x, 1, y, 1);
    }
    double t1 = now_seconds();
    double seconds = t1 - t0;

    double max_abs_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        long idx = (long)(next_uniform() * (double)n);
        if (idx >= n) idx = n - 1;
        double expected = alpha * x[idx] + y_original[idx];
        double got = y[idx];
        if (!isfinite(got)) { max_abs_error = INFINITY; break; }
        double abs_error = fabs(got - expected);
        if (abs_error > max_abs_error) max_abs_error = abs_error;
    }
    const int ok = max_abs_error < 1e-9;

    double total_flops = (double)iterations * 2.0 * (double)n;
    double mops_total = total_flops / 1e6 / seconds;

    printf("\n AXPY benchmark (CPU / OpenBLAS)\n\n");
    printf(" Vector size (N)       =                %8ld\n", n);
    printf(" Iterations            =                %8d\n", iterations);
    printf("\n");
    printf(" Time in seconds =    %12.6f\n", seconds);
    printf(" Mop/s total     =    %12.2f\n", mops_total);
    printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    free(x); free(y); free(y_original);
    return ok ? 0 : 1;
}
