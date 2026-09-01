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
/* ARC-158: por debajo de este umbral, el tiempo medido queda dominado por
 * el overhead de arranque/union de la region paralela de OpenMP (y por la
 * resolucion del reloj) en vez de computo real sostenido -- confirmado en
 * paccaA100: la medicion "mejor" que el driver reportaba correspondia a
 * ventanas de 31-63 microsegundos (BEST_WORKING_SET_DOUBLES=4994,
 * BEST_TRIALS=32-64), y el GFLOPs/sec resultante no variaba con la
 * frecuencia de nucleo fijada (485-510 GFLOP/s identico en F0=3.2GHz,
 * F2=2.2GHz, F4=0.8GHz, con Turbo confirmado desactivado) -- la firma de
 * estar midiendo ruido de sincronizacion, no FMA real. 1ms es dos ordenes
 * de magnitud por encima de esas ventanas espurias, sin acercarse al
 * tiempo total del barrido completo (cientos de ms).
 */
#define MIN_TIMED_SECONDS 1e-3
/* ARC-158 (segunda vuelta): tope de seguridad para el numero de
 * repeticiones al buscar una medicion de al menos MIN_TIMED_SECONDS --
 * nunca alcanzado en la practica (con seis nucleos y el tamano de trabajo
 * mas chico del barrido ya se cruza el umbral en pocas duplicaciones),
 * solo evita un bucle sin fin ante un caso patologico. */
#define MAX_TRIALS_CAP (1ULL << 24)

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

        /* ARC-158: en vez de barrer un numero fijo de repeticiones y
         * descartar las que resultan demasiado cortas (lo que sesgaba la
         * medicion "mejor" hacia tamanos de trabajo grandes -- que ya no
         * caben en L1/L2 y pasan a estar limitados por ancho de banda, no
         * por computo, invalidando el propósito del probe), se duplican
         * las repeticiones hasta alcanzar una duracion fiable para ESTE
         * tamano de trabajo especificamente, preservando que se mida
         * dentro de la cache mientras se obtiene una senal por encima del
         * overhead de sincronizacion de OpenMP y de la resolucion del
         * reloj. Una sola medicion fiable por tamano, no varias. */
        for (uint64_t trials = 1; trials <= MAX_TRIALS_CAP; trials *= 2) {
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
            if (seconds < MIN_TIMED_SECONDS) {
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
            break; /* medicion fiable obtenida para este tamano -- siguiente bytes. */
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
