/*
 * hpccg_cg_cpu_bench.c -- Gradiente Conjugado (CG) sobre el Laplaciano
 * discreto de 7 puntos en una malla estructurada NxNxN, formato CSR, en
 * CPU con OpenMP.
 *
 * Familia NUEVA (no dual_*, no lbm_cpu). El paper de Littman & Deakin
 * (SC25 poster) cita "HPCCG (MiniFE)" como benchmark memory-bound sobre
 * malla 50x50x50 -- el mini-app original es Mantevo/HPCCG (Sandia). No se
 * vendoriza (mismo criterio que lbm_cpu: evitar dependencias de
 * procedencia no verificada para una tesis); en su lugar, implementacion
 * propia del metodo CG estandar (Hestenes-Stiefel) sobre una matriz de
 * Laplaciano 3D de 7 puntos, construida explicitamente en CSR (mismo
 * formato que kernels/dual/spmv_cpu_bench.c).
 *
 * Por que esta familia y no otro tamano de lbm_cpu/dual_spmv: a diferencia
 * de LBM (barrido de un paso, sin reutilizacion entre iteraciones) y de
 * SpMV puro (una sola multiplicacion por despacho), CG hace VARIAS
 * pasadas (cg_steps) sobre la MISMA matriz y las MISMAS estructuras de
 * vectores dentro de un mismo despacho -- si la matriz cabe en cache, esa
 * reutilizacion entre iteraciones puede dar una transicion mas gradual
 * conforme el tamano de malla crece, en vez del salto de cache abrupto
 * que se documento en F1-CPU-008 para lbm_cpu. Esto es una hipotesis a
 * verificar empiricamente, no un resultado asumido -- de ahi el barrido
 * de tamanos que sigue a este kernel en el catalogo.
 *
 * Matriz: diagonal fija en 6.1 (estrictamente mayor que la suma de
 * |fuera-diagonal| <= 6 de cualquier fila) para garantizar diagonal
 * estrictamente dominante -> SPD garantizado -> CG converge de forma
 * monotona sin necesidad de ajuste fino. Fuera-diagonal -1.0 para cada
 * vecino de malla que exista (menos en las caras/aristas/esquinas del
 * dominio).
 *
 * Verificacion: con A SPD, CG reduce la norma del residuo de forma
 * monotona en cada iteracion -- se compara ||b - A*x_final|| contra
 * ||b - A*x0|| = ||b|| (x0 = 0). Un residuo final estrictamente menor
 * (y finito) confirma que el solver esta funcionando, sin depender de
 * cuantas iteraciones bastan para converger del todo (no se ejecuta hasta
 * tolerancia, cg_steps es fijo para que el trabajo por despacho sea
 * determinista).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../dual/dispatch_timing.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define CG_STEPS 20
#define DIAG_VALUE 6.1

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

/* Laplaciano 3D de 7 puntos sobre malla n x n x n, CSR. cells = n^3.
 * cota superior de nnz = 7*cells (diagonal + hasta 6 vecinos). */
static long build_csr_laplacian3d(long n, long **row_ptr, long **col_idx, double **values) {
    long cells = n * n * n;
    *row_ptr = malloc((size_t)(cells + 1) * sizeof(long));
    long cap = 7 * cells;
    *col_idx = malloc((size_t)cap * sizeof(long));
    *values = malloc((size_t)cap * sizeof(double));

    long nnz = 0;
    (*row_ptr)[0] = 0;
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            for (long k = 0; k < n; ++k) {
                long row = (i * n + j) * n + k;
                (*col_idx)[nnz] = row;
                (*values)[nnz] = DIAG_VALUE;
                ++nnz;
                static const long di[6] = {-1, 1, 0, 0, 0, 0};
                static const long dj[6] = {0, 0, -1, 1, 0, 0};
                static const long dk[6] = {0, 0, 0, 0, -1, 1};
                for (int d = 0; d < 6; ++d) {
                    long ni = i + di[d], nj = j + dj[d], nk = k + dk[d];
                    if (ni < 0 || ni >= n || nj < 0 || nj >= n || nk < 0 || nk >= n) continue;
                    long col = (ni * n + nj) * n + nk;
                    (*col_idx)[nnz] = col;
                    (*values)[nnz] = -1.0;
                    ++nnz;
                }
                (*row_ptr)[row + 1] = nnz;
            }
        }
    }
    return nnz;
}

static void spmv(long cells, const long *row_ptr, const long *col_idx,
                  const double *values, const double *x, double *y) {
    #pragma omp parallel for schedule(static)
    for (long r = 0; r < cells; ++r) {
        double sum = 0.0;
        for (long p = row_ptr[r]; p < row_ptr[r + 1]; ++p) sum += values[p] * x[col_idx[p]];
        y[r] = sum;
    }
}

static double dot(long cells, const double *a, const double *b) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = 0; i < cells; ++i) sum += a[i] * b[i];
    return sum;
}

/* Un solve completo de CG_STEPS iteraciones, partiendo siempre de x=0. */
static void cg_solve(long cells, const long *row_ptr, const long *col_idx,
                      const double *values, const double *b,
                      double *x, double *r, double *p, double *ap) {
    memset(x, 0, (size_t)cells * sizeof(double));
    memcpy(r, b, (size_t)cells * sizeof(double));
    memcpy(p, b, (size_t)cells * sizeof(double));
    double rsold = dot(cells, r, r);

    for (int step = 0; step < CG_STEPS; ++step) {
        spmv(cells, row_ptr, col_idx, values, p, ap);
        double pap = dot(cells, p, ap);
        double alpha = rsold / pap;

        #pragma omp parallel for schedule(static)
        for (long i = 0; i < cells; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        double rsnew = dot(cells, r, r);
        double beta = rsnew / rsold;
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < cells; ++i) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }
}

int main(int argc, char **argv) {
    long n = 32; /* malla n x n x n */
    int iterations = 10;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            n = strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = (int)strtol(argv[++i], NULL, 10);
        }
    }
    if (n <= 1 || iterations <= 0) {
        fprintf(stderr, "uso: %s [--size N>1] [--iterations M]\n", argv[0]);
        return 2;
    }

    long *row_ptr; long *col_idx; double *values;
    long nnz = build_csr_laplacian3d(n, &row_ptr, &col_idx, &values);
    long cells = n * n * n;

    double *b = malloc((size_t)cells * sizeof(double));
    double *x = malloc((size_t)cells * sizeof(double));
    double *r = malloc((size_t)cells * sizeof(double));
    double *p = malloc((size_t)cells * sizeof(double));
    double *ap = malloc((size_t)cells * sizeof(double));
    if (!b || !x || !r || !p || !ap) {
        fprintf(stderr, "fallo de asignacion para N=%ld (cells=%ld)\n", n, cells);
        return 2;
    }
    for (long i = 0; i < cells; ++i) b[i] = next_uniform() * 2.0 - 1.0;
    double b_norm = sqrt(dot(cells, b, b));

    /* Primer despacho en frio. */
    long long cold_t0_ns = now_ns();
    long long setup_complete_ns = cold_t0_ns;
    cg_solve(cells, row_ptr, col_idx, values, b, x, r, p, ap);
    long long cold_t1_ns = now_ns();

    long long t0_ns = now_ns();
    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        cg_solve(cells, row_ptr, col_idx, values, b, x, r, p, ap);
    }
    double t1 = now_seconds();
    long long t1_ns = now_ns();
    double seconds = t1 - t0;

    double final_residual = sqrt(dot(cells, r, r));
    const int ok = isfinite(final_residual) && final_residual < b_norm;

    /* Por iteracion CG: SpMV (2*nnz) + 2 dot (2*2*cells) + axpy/combine
     * (2*3*cells) FLOPs. */
    double flops_per_cg_iter = 2.0 * (double)nnz + 10.0 * (double)cells;
    double total_flops = (double)iterations * (double)CG_STEPS * flops_per_cg_iter;
    double mops_total = total_flops / 1e6 / seconds;

    printf("\n HPCCG-style CG benchmark (CPU / OpenMP)\n\n");
    printf(" Grid size (N)         =                %8ld\n", n);
    printf(" Cells (N^3)           =                %8ld\n", cells);
    printf(" Nonzeros              =                %8ld\n", nnz);
    printf(" Iterations            =                %8d\n", iterations);
    printf(" CG steps/solve        =                %8d\n", CG_STEPS);
    printf(" Residual: b_norm=%.6e final=%.6e\n", b_norm, final_residual);
    printf("\n");
    printf(" Time in seconds =    %12.6f\n", seconds);
    print_dispatch_timing(cold_t0_ns, setup_complete_ns, cold_t1_ns, t0_ns, t1_ns);
    printf(" Mop/s total     =    %12.2f\n", mops_total);
    printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    free(row_ptr); free(col_idx); free(values);
    free(b); free(x); free(r); free(p); free(ap);
    return ok ? 0 : 1;
}
