/*
 * dgemm_bench.c -- multiplicacion de matrices densas (C = alpha*A*B + beta*C)
 * via OpenBLAS cblas_dgemm, para poblar la zona compute-bound del plano
 * Roofline cerca del ridge point (ver docs/retoma/Propuesta_Seleccion_Kernels_Dataset.md,
 * seccion 4: el hueco que EP no cubre porque casi no toca memoria).
 *
 * El tamano de matriz N se pasa por CLI (--size), no se fija en tiempo de
 * compilacion -- una sola compilacion puede alimentar varios size_variant
 * del catalogo (mas cerca o mas lejos del ridge point) sin recompilar.
 *
 * Salida en el mismo formato que usan los kernels NPB en el catalogo
 * (catalog.yaml), para reutilizar exactamente los mismos regex de
 * flops_rate_stdout_pattern/runtime_seconds_stdout_pattern/success_check
 * ya validados contra stdout real (ARC-32):
 *   Time in seconds =    X.XXX
 *   Mop/s total     =    Y.YY
 *   Verification    =               SUCCESSFUL
 */
#include <cblas.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

/* PRNG determinista (xorshift64) -- reproducible sin depender de /dev/urandom. */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static double next_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) / (double)(1ULL << 53);
}

static void fill_matrix(double *matrix, long n) {
    for (long i = 0; i < n * n; ++i) {
        matrix[i] = next_uniform() * 2.0 - 1.0;
    }
}

/* Verifica una muestra de entradas de C contra el producto punto directo
 * -- verificar C completo costaria lo mismo que recalcularlo (O(n^3)), asi
 * que se toma una muestra aleatoria de pares (i, j), suficiente para
 * detectar un error real de BLAS/enlazado sin duplicar el costo de computo. */
static int verify_sample(const double *a, const double *b, const double *c, long n,
                          double alpha, double beta, int samples) {
    double max_rel_error = 0.0;
    for (int s = 0; s < samples; ++s) {
        long i = (long)(next_uniform() * (double)n);
        long j = (long)(next_uniform() * (double)n);
        if (i >= n) i = n - 1;
        if (j >= n) j = n - 1;

        double expected = 0.0;
        for (long k = 0; k < n; ++k) {
            expected += a[i * n + k] * b[k * n + j];
        }
        expected = alpha * expected + beta * 0.0; /* beta*C_inicial, C se inicializa en 0 */

        double got = c[i * n + j];
        double denom = fabs(expected) > 1e-12 ? fabs(expected) : 1.0;
        double rel_error = fabs(got - expected) / denom;
        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
        }
    }
    return max_rel_error < 1e-8;
}

int main(int argc, char **argv) {
    long n = 2048;
    int iterations = 5;
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

    double *a = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *b = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *c = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!a || !b || !c) {
        fprintf(stderr, "Out Of Memory (n=%ld)\n", n);
        return 1;
    }

    fill_matrix(a, n);
    fill_matrix(b, n);
    memset(c, 0, (size_t)n * (size_t)n * sizeof(double));

    const double alpha = 1.0, beta = 0.0;

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)n, (int)n, (int)n, alpha, a, (int)n, b, (int)n, beta, c, (int)n);
    }
    double t1 = now_seconds();
    double seconds = t1 - t0;

    int ok = verify_sample(a, b, c, n, alpha, beta, verify_samples);

    double total_flops = (double)iterations * 2.0 * (double)n * (double)n * (double)n;
    double mops_total = total_flops / 1e6 / seconds;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    const char *threads_env = getenv("OPENBLAS_NUM_THREADS");

    printf("\n DGEMM Benchmark (OpenBLAS)\n\n");
    printf(" Matrix size (N)       =                %8ld\n", n);
    printf(" Iterations            =                %8d\n", iterations);
    printf(" OPENBLAS_NUM_THREADS  =                %8s\n", threads_env ? threads_env : "(default)");
    printf(" OMP max threads       =                %8d\n", nthreads);
    printf("\n");
    printf(" Time in seconds =    %12.6f\n", seconds);
    printf(" Mop/s total     =    %12.2f\n", mops_total);
    printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    free(a);
    free(b);
    free(c);
    return ok ? 0 : 1;
}
