/*
 * lbm_d2q9_cpu_bench.c -- Lattice Boltzmann D2Q9 BGK (Bhatnagar-Gross-Krook),
 * malla NxN con frontera periodica, en CPU con OpenMP.
 *
 * Familia NUEVA (no dual_*, no vendorizada de ningun repo externo). El
 * benchmark "LBM D2Q9" del paper de Littman & Deakin (SC25 poster,
 * clasificado alli como "LLC cache or main memory bandwidth bound") usa
 * como fuente UoB-HPC/advanced-hpc-lbm, que es material de entrega de un
 * curso ("Advanced HPC Coursework") de la Universidad de Bristol SIN
 * archivo LICENSE (404 en /license de la API de GitHub, verificado
 * 2026-09-14) -- no se vendoriza por falta de procedencia clara. En su
 * lugar, esta es una implementacion propia del metodo LBM D2Q9-BGK a partir
 * de su formulacion numerica estandar (lattice de 9 velocidades, operador
 * de colision BGK, funcion de equilibrio de Maxwell-Boltzmann truncada a
 * segundo orden -- ver p.ej. Kruger et al., "The Lattice Boltzmann Method",
 * 2017), sin copiar ninguna base de codigo especifica.
 *
 * Por que esta familia y no otro tamano mas de dual_stencil: LBM D2Q9 tiene
 * una intensidad operacional NATURAL (~1.2 FLOP/byte tocado, ver comentario
 * en collide_stream) mayor que un stencil Jacobi de 5 puntos (~1.0) y mucho
 * mayor que AXPY (~0.06), y un patron de acceso distinto (9 poblaciones por
 * celda, streaming con desplazamiento por direccion) -- otra familia
 * algoritmica real, no otro tamano de la misma. Igual que con dual_*, el OI
 * MEDIDO (trafico real a DRAM via uncore_imc, no el "tocado" de este
 * comentario) depende de si la malla cabe en LLC; hace falta un barrido de
 * tamanos para localizar el punto cerca del ridge -- no se asume a priori.
 *
 * Frontera periodica (sin obstaculos, sin bounce-back) para no depender de
 * archivos de entrada externos, igual que el resto del catalogo dual_*
 * (datos sinteticos generados en el propio binario). Cada "despacho"
 * (iteracion medida) es UN paso completo de colision+streaming partiendo
 * siempre del mismo estado inicial -- mismo criterio que dual_stencil.
 *
 * Verificacion: bajo frontera periodica y sin forzamiento externo, la masa
 * total (suma de rho en todas las celdas) es un invariante exacto del
 * esquema LBM -- el streaming es una permutacion (no crea ni destruye masa)
 * y la colision BGK conserva el momento de orden 0 por celda
 * (sum_i feq_i = rho = sum_i f_i). Se compara masa total antes/despues.
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

#define NDIR 9
static const int EX[NDIR] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static const int EY[NDIR] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static const double W[NDIR] = {
    4.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

static void feq_all(double rho, double ux, double uy, double *feq) {
    double usqr = ux * ux + uy * uy;
    for (int d = 0; d < NDIR; ++d) {
        double eu = (double)EX[d] * ux + (double)EY[d] * uy;
        feq[d] = W[d] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usqr);
    }
}

/* Un paso completo de colision (BGK) + streaming. f_in y f_out son
 * NDIR*N*N doblones, layout [dir][fila][columna] (dir mas lento).
 *
 * Costo por celda (aprox., sin contar el bucle de reduccion inicial):
 * rho ~8 flops, ux/uy ~2*18 flops, usqr ~3, feq (9 direcciones) ~99,
 * colision ~27 => ~173 FLOPs/celda. Trafico "tocado": 9 loads + 9 stores de
 * 8 bytes = 144 bytes/celda (cota inferior; el trafico REAL a DRAM depende
 * de cuanto de la malla siga viva en LLC). OI tocado ~= 1.2 FLOP/byte. */
static void collide_stream(const double *f_in, double *f_out, long n, double omega) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        double feq[NDIR];
        for (long j = 0; j < n; ++j) {
            long c = i * n + j;
            double rho = 0.0;
            for (int d = 0; d < NDIR; ++d) rho += f_in[(long)d * n * n + c];

            double ux = 0.0, uy = 0.0;
            for (int d = 0; d < NDIR; ++d) {
                double fd = f_in[(long)d * n * n + c];
                ux += fd * (double)EX[d];
                uy += fd * (double)EY[d];
            }
            ux /= rho;
            uy /= rho;

            feq_all(rho, ux, uy, feq);

            for (int d = 0; d < NDIR; ++d) {
                double fd = f_in[(long)d * n * n + c];
                double f_post = fd + omega * (feq[d] - fd);
                long ni = (i + EY[d] + n) % n;
                long nj = (j + EX[d] + n) % n;
                long dest = (long)d * n * n + ni * n + nj;
                f_out[dest] = f_post;
            }
        }
    }
}

static double total_mass(const double *f, long n) {
    double sum = 0.0;
    long cells = n * n;
    for (int d = 0; d < NDIR; ++d) {
        const double *plane = f + (long)d * cells;
        for (long c = 0; c < cells; ++c) sum += plane[c];
    }
    return sum;
}

int main(int argc, char **argv) {
    long n = 256;
    int iterations = 10;
    double tau = 1.0; /* omega = 1/tau; tau>=0.5 requerido por estabilidad. */

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

    const double omega = 1.0 / tau;
    const size_t cells = (size_t)n * (size_t)n;
    const size_t elems = cells * (size_t)NDIR;

    double *f_original = malloc(elems * sizeof(double));
    double *f_a = malloc(elems * sizeof(double));
    double *f_b = malloc(elems * sizeof(double));
    if (!f_original || !f_a || !f_b) {
        fprintf(stderr, "fallo de asignacion para N=%ld\n", n);
        return 2;
    }

    /* Estado inicial: rho cerca de 1 con perturbacion pequena, velocidad
     * pequena aleatoria -- f_i = feq_i(rho, u) para que el estado de
     * arranque ya sea consistente con el esquema (evita un transitorio de
     * relajacion inicial artificial). */
    for (long c = 0; c < (long)cells; ++c) {
        double rho0 = 1.0 + 0.01 * (next_uniform() * 2.0 - 1.0);
        double ux0 = 0.02 * (next_uniform() * 2.0 - 1.0);
        double uy0 = 0.02 * (next_uniform() * 2.0 - 1.0);
        double feq[NDIR];
        feq_all(rho0, ux0, uy0, feq);
        for (int d = 0; d < NDIR; ++d) f_original[(long)d * (long)cells + c] = feq[d];
    }

    /* Primer despacho en frio. */
    long long cold_t0_ns = now_ns();
    long long setup_complete_ns = cold_t0_ns;
    memcpy(f_a, f_original, elems * sizeof(double));
    collide_stream(f_a, f_b, n, omega);
    long long cold_t1_ns = now_ns();

    double mass_before = total_mass(f_original, n);

    long long t0_ns = now_ns();
    double t0 = now_seconds();
    double mass_after = 0.0;
    for (int rep = 0; rep < iterations; ++rep) {
        /* Cada despacho parte del mismo estado inicial -- un paso de
         * colision+streaming aislado, no una simulacion acumulada. */
        memcpy(f_a, f_original, elems * sizeof(double));
        collide_stream(f_a, f_b, n, omega);
        if (rep == iterations - 1) mass_after = total_mass(f_b, n);
    }
    double t1 = now_seconds();
    long long t1_ns = now_ns();
    double seconds = t1 - t0;

    double mass_rel_error = fabs(mass_after - mass_before) / fabs(mass_before);
    const int ok = mass_rel_error < 1e-9;

    double interior_cells = (double)cells;
    /* ~173 FLOPs/celda (colision+equilibrio+macroscopicas), ver comentario
     * de collide_stream. */
    double total_flops = (double)iterations * 173.0 * interior_cells;
    double mops_total = total_flops / 1e6 / seconds;

    printf("\n LBM D2Q9-BGK benchmark (CPU / OpenMP)\n\n");
    printf(" Grid size (N)         =                %8ld\n", n);
    printf(" Iterations            =                %8d\n", iterations);
    printf(" Mass rel. error       =            %12.3e\n", mass_rel_error);
    printf("\n");
    printf(" Time in seconds =    %12.6f\n", seconds);
    print_dispatch_timing(cold_t0_ns, setup_complete_ns, cold_t1_ns, t0_ns, t1_ns);
    printf(" Mop/s total     =    %12.2f\n", mops_total);
    printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    free(f_original); free(f_a); free(f_b);
    return ok ? 0 : 1;
}
