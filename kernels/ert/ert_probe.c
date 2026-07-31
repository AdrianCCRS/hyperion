/*
 * ert_probe.c -- microbenchmark de FLOPs pico (regimen compute-bound).
 *
 * Metodo tomado de ERT (Empirical Roofline Toolkit, Berkeley Lab CS
 * Roofline Toolkit, Kernels/kernel1.c + Drivers/driver1.c, version 1.0.0):
 * un bucle de operaciones tipo FMA (beta = beta*A[i] + alpha, desenrollado
 * 8 veces = 16 FLOPs por elemento) sobre un buffer que cabe en cache,
 * barriendo tamano de working set y numero de repeticiones, reportando el
 * maximo GFLOP/s observado.
 *
 * Reescrito standalone -- sin MPI/BGQ/QPX/intrinsics AVX y sin el driver
 * Python de post-proceso (roofline.py) de ERT, que requiere plantillas
 * Batch/Config especificas por cluster que SC3 no tiene publicadas -- por
 * decision documentada en ARC-31 (Registro_Cambios_Fuera_Plan_Original.md),
 * amparada por CAL-03 ("microbenchmark de FLOPs pico de suite reconocida
 * que reporte GFLOPs por stdout"). La aritmetica del kernel es la misma;
 * el driver que barre tamanos/repeticiones y calcula el pico es propio.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define FLOPS_PER_ELEM 16
#define ALIGN_BYTES 64

static double now_seconds(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
#endif
}

/* Kernel identico en espiritu a ERT Kernels/kernel1.c (ERT_FLOP == 16). */
static double run_once(double *array, uint64_t nsize, uint64_t ntrials) {
    double alpha = 0.5;
    for (uint64_t trial = 0; trial < ntrials; ++trial) {
        for (uint64_t i = 0; i < nsize; ++i) {
            double beta = 0.8;
            beta = beta * array[i] + alpha;
            beta = beta * array[i] + alpha;
            beta = beta * array[i] + alpha;
            beta = beta * array[i] + alpha;
            beta = beta * array[i] + alpha;
            beta = beta * array[i] + alpha;
            beta = beta * array[i] + alpha;
            beta = beta * array[i] + alpha;
            array[i] = beta;
        }
        alpha = alpha * (1.0 - 1e-8);
    }
    return alpha;
}

int main(void) {
    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp master
        nthreads = omp_get_num_threads();
    }
#endif

    double best_gflops = 0.0;
    uint64_t best_n = 0, best_trials = 0;
    double checksum = 0.0;

    /* Working set por hilo: 1 KiB -> 4 MiB (cabe en L1/L2, evita L3/DRAM). */
    for (uint64_t bytes = 1024; bytes <= (4UL << 20); bytes = (uint64_t)(bytes * 1.5) + 8) {
        uint64_t n = bytes / sizeof(double);
        if (n < 8) {
            continue;
        }

        double *buf = NULL;
        if (posix_memalign((void **)&buf, ALIGN_BYTES, n * sizeof(double) * (uint64_t)nthreads) != 0) {
            continue;
        }
        for (uint64_t i = 0; i < n * (uint64_t)nthreads; ++i) {
            buf[i] = 1.0 + (double)(i % 7) * 1e-3;
        }

        uint64_t max_trials = (bytes < 65536) ? 64 : 8;
        for (uint64_t trials = 1; trials <= max_trials; trials *= 2) {
            double sink = 0.0;
            double t0 = now_seconds();
#ifdef _OPENMP
#pragma omp parallel reduction(+ : sink)
            {
                int id = omp_get_thread_num();
                sink += run_once(&buf[(uint64_t)id * n], n, trials);
            }
#else
            sink += run_once(buf, n, trials);
#endif
            double t1 = now_seconds();
            double seconds = t1 - t0;
            if (seconds <= 0.0) {
                continue;
            }

            checksum += sink;
            double total_flops = (double)n * (double)nthreads * (double)trials * FLOPS_PER_ELEM;
            double gflops = total_flops / seconds / 1e9;
            if (gflops > best_gflops) {
                best_gflops = gflops;
                best_n = n;
                best_trials = trials;
            }
        }
        free(buf);
    }

    printf("ERT_PROBE_RESULT\n");
    printf("OPENMP_THREADS %d\n", nthreads);
    printf("BEST_WORKING_SET_DOUBLES %" PRIu64 "\n", best_n);
    printf("BEST_TRIALS %" PRIu64 "\n", best_trials);
    printf("CHECKSUM %.6f\n", checksum);
    printf("GFLOPs/sec: %.3f\n", best_gflops);
    return 0;
}
