/*
 * stencil_cpu_bench.c -- Jacobi 2D de 5 puntos (promedio de 4 vecinos) sobre
 * una malla NxN, en CPU con OpenMP.
 *
 * Sexta operacion del catalogo dual: memory-bound ESTRUCTURADO -- a
 * diferencia de AXPY (acceso puramente secuencial), cada celda toca 4
 * vecinos con patron de acceso predecible mediante 2D blocking/prefetching de
 * hardware, asi que su comportamiento frente al cache es distinto al de AXPY
 * aunque ambos sean memory-bound. Intensidad operacional ~5 FLOPs / 5 loads +
 * 1 store por celda (bytes reales dependen de cuanto capture el cache de la
 * fila anterior).
 *
 * Frontera con doble buffer (in/out) para que el orden de escritura no
 * dependa de como se recorra la malla -- necesario para paralelizar con
 * OpenMP sin condicion de carrera.
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

/* ARC: sello absoluto de la region medida (CLOCK_MONOTONIC, mismo reloj que
 * usa la telemetria -- telemetry/include/telemetry/metrics.hpp). Permite al
 * constructor del dataset filtrar las ventanas al bucle realmente medido, en
 * vez de promediar sobre todo el proceso (que en GPU es ~85% inicializacion
 * de contexto CUDA). Ver docs/general/metodologia_selector_cpu_gpu_20260827.md
 * seccion 6.9. */
static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

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

/* Un paso de Jacobi 2D: bordes se copian sin cambio (frontera fija), interior
 * se promedia con sus 4 vecinos. */
static void jacobi_step(const double *in, double *out, long n) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            long idx = i * n + j;
            if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
                out[idx] = in[idx];
            } else {
                out[idx] = 0.25 * (in[idx - n] + in[idx + n] + in[idx - 1] + in[idx + 1]);
            }
        }
    }
}

int main(int argc, char **argv) {
    long n = 1024;
    int iterations = 10;
    int verify_samples = 64;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            n = strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = (int)strtol(argv[++i], NULL, 10);
        }
    }
    if (n <= 2 || iterations <= 0) {
        fprintf(stderr, "uso: %s [--size N>2] [--iterations M]\n", argv[0]);
        return 2;
    }

    const size_t elems = (size_t)n * (size_t)n;
    double *a = malloc(elems * sizeof(double));
    double *b = malloc(elems * sizeof(double));
    double *original = malloc(elems * sizeof(double));
    if (!a || !b || !original) {
        fprintf(stderr, "fallo de asignacion para N=%ld\n", n);
        return 2;
    }
    fill(a, (long)elems);
    memcpy(original, a, elems * sizeof(double));

    /* Warmup fuera de ventana: un paso completo (ida y vuelta) para tocar
     * ambos buffers antes de medir. */
    jacobi_step(a, b, n);
    jacobi_step(b, a, n);
    memcpy(a, original, elems * sizeof(double));

    long long t0_ns = now_ns();

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        /* Cada "despacho" es UN paso de Jacobi completo, partiendo siempre del
         * mismo estado -- mismo criterio que el resto del catalogo dual: cada
         * iteracion es una llamada independiente, no una simulacion
         * acumulada de muchos pasos. */
        memcpy(a, original, elems * sizeof(double));
        jacobi_step(a, b, n);
    }
    double t1 = now_seconds();
    long long t1_ns = now_ns();
    double seconds = t1 - t0;

    double max_abs_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        long i = 1 + (long)(next_uniform() * (double)(n - 2));
        long j = 1 + (long)(next_uniform() * (double)(n - 2));
        if (i >= n - 1) i = n - 2;
        if (j >= n - 1) j = n - 2;
        long idx = i * n + j;
        double expected = 0.25 * (original[idx - n] + original[idx + n]
                                 + original[idx - 1] + original[idx + 1]);
        double got = b[idx];
        if (!isfinite(got)) { max_abs_error = INFINITY; break; }
        double abs_error = fabs(got - expected);
        if (abs_error > max_abs_error) max_abs_error = abs_error;
    }
    const int ok = max_abs_error < 1e-9;

    /* 4 sumas + 1 multiplicacion por celda interior. */
    double interior_cells = (double)(n - 2) * (double)(n - 2);
    double total_flops = (double)iterations * 5.0 * interior_cells;
    double mops_total = total_flops / 1e6 / seconds;

    printf("\n Jacobi 2D stencil benchmark (CPU / OpenMP)\n\n");
    printf(" Grid size (N)         =                %8ld\n", n);
    printf(" Iterations            =                %8d\n", iterations);
    printf("\n");
    printf(" Time in seconds =    %12.6f\n", seconds);
    printf(" Measured region t0_ns = %lld\n", t0_ns);
    printf(" Measured region t1_ns = %lld\n", t1_ns);
    printf(" Mop/s total     =    %12.2f\n", mops_total);
    printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    free(a); free(b); free(original);
    return ok ? 0 : 1;
}
