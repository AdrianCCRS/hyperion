/*
 * cholesky_cpu_bench.c -- factorizacion de Cholesky (A = L*L^T) de una
 * matriz SPD densa NxN, en CPU via LAPACKE_dpotrf (OpenBLAS).
 *
 * Quinta operacion del catalogo dual, contraparte de cholesky_gpu_dispatch.cu
 * (cuSOLVER). Perfil compute-bound CON ESTRUCTURA: a diferencia de GEMM
 * (denso, sin dependencias entre bloques), Cholesky tiene dependencias
 * secuenciales entre columnas/paneles -- menos paralelismo disponible que
 * GEMM al mismo tamaño, asi que su frontera CPU/GPU deberia caer en un N
 * mayor que el de GEMM.
 *
 * Matriz SPD diagonal-dominante generada en tiempo de ejecucion, sin
 * depender de un archivo externo y sin esconder otro kernel cubico antes de
 * la region medida.
 */
#include <lapacke.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dispatch_timing.h"

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

/* Matriz simetrica estrictamente diagonal-dominante con diagonal positiva.
 * Por Gershgorin todos sus autovalores son positivos: es SPD. A diferencia
 * de B^T*B, se construye en O(N^2), no ejecuta un GEMM oculto O(N^3) antes
 * de medir y no inicializa OpenBLAS por efecto lateral fuera de cold_t0. */
static void make_spd_matrix(double *A, long n) {
    memset(A, 0, (size_t)n * (size_t)n * sizeof(double));
    for (long i = 0; i < n; ++i) {
        for (long j = i + 1; j < n; ++j) {
            double value = next_uniform() * 2.0 - 1.0;
            A[(size_t)i * n + j] = value;
            A[(size_t)j * n + i] = value;
        }
    }
    for (long i = 0; i < n; ++i) {
        double radius = 0.0;
        for (long j = 0; j < n; ++j) {
            if (i != j) radius += fabs(A[(size_t)i * n + j]);
        }
        A[(size_t)i * n + i] = radius + 1.0;
    }
}

int main(int argc, char **argv) {
    long n = 512;
    int iterations = 10;
    int verify_samples = 32;

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

    const size_t elems = (size_t)n * (size_t)n;
    double *original = malloc(elems * sizeof(double));
    double *work = malloc(elems * sizeof(double));
    if (!original || !work) {
        fprintf(stderr, "fallo de asignacion para N=%ld\n", n);
        return 2;
    }
    make_spd_matrix(original, n);

    /* Primer despacho en frio: LAPACKE/OpenBLAS no expone un handle aqui;
     * su inicializacion perezosa queda incluida en la primera llamada. */
    long long cold_t0_ns = now_ns();
    long long setup_complete_ns = cold_t0_ns;
    memcpy(work, original, elems * sizeof(double));
    int first_info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', (int)n, work, (int)n);
    long long cold_t1_ns = now_ns();

    long long t0_ns = now_ns();

    double t0 = now_seconds();
    int lapack_ok = first_info == 0;
    for (int rep = 0; rep < iterations; ++rep) {
        /* Cada despacho parte de la matriz original -- dpotrf es in-place y
         * destruye su entrada, igual criterio que el resto del catalogo
         * dual (cada iteracion es un despacho independiente). */
        memcpy(work, original, elems * sizeof(double));
        int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', (int)n, work, (int)n);
        if (info != 0) lapack_ok = 0;
    }
    double t1 = now_seconds();
    long long t1_ns = now_ns();
    double seconds = t1 - t0;

    /* Verificacion: reconstruir L*L^T en un muestreo de entradas y comparar
     * contra la matriz original. Solo el triangulo inferior de work es L. */
    double max_abs_error = 0.0;
    if (lapack_ok) {
        for (int s = 0; s < verify_samples; ++s) {
            long i = (long)(next_uniform() * (double)n);
            long j = (long)(next_uniform() * (double)n);
            if (i >= n) i = n - 1;
            if (j >= n) j = n - 1;
            long lo = i < j ? i : j, hi = i < j ? j : i;
            double sum = 0.0;
            for (long k = 0; k <= lo; ++k) {
                sum += work[hi * n + k] * work[lo * n + k];
            }
            double got = (i <= j) ? sum : sum; /* simetrico: L*L^T(i,j)==L*L^T(j,i) */
            double expected = original[i * n + j];
            if (!isfinite(got)) { max_abs_error = INFINITY; break; }
            double abs_error = fabs(got - expected);
            if (abs_error > max_abs_error) max_abs_error = abs_error;
        }
    } else {
        max_abs_error = INFINITY;
    }
    /* Tolerancia mas laxa que GEMM: acumulacion de N terminos por entrada
     * reconstruida, mismo orden que un producto denso NxN. */
    const int ok = max_abs_error < 1e-6 * (double)n;

    /* 1/3*N^3 flops teoricos para Cholesky denso (LAPACK dpotrf). */
    double total_flops = (double)iterations * (1.0 / 3.0) * (double)n * (double)n * (double)n;
    double mops_total = total_flops / 1e6 / seconds;

    printf("\n Cholesky factorization benchmark (CPU / LAPACKE dpotrf)\n\n");
    printf(" Matrix size (N)       =                %8ld\n", n);
    printf(" Iterations            =                %8d\n", iterations);
    printf("\n");
    printf(" Time in seconds =    %12.6f\n", seconds);
    print_dispatch_timing(cold_t0_ns, setup_complete_ns, cold_t1_ns, t0_ns, t1_ns);
    printf(" Mop/s total     =    %12.2f\n", mops_total);
    printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    free(original); free(work);
    return ok ? 0 : 1;
}
