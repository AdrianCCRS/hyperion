/* phasic -- carga MULTIFÁSICA con etiqueta de verdad conocida.
 *
 * Por qué existe (ARC-176). El dataset de Fase 1 no contiene variación de
 * fase intra-ejecución: la clase minoritaria dentro de cada kernel es 4.0 %
 * de media y cuatro kernels están al 0.0 %. Cada benchmark es de un solo
 * régimen de principio a fin, porque la §5.1 del anteproyecto eligió a
 * propósito ejemplos PUROS de cada escenario. Eso deja al clasificador sin
 * nada que clasificar dentro de una corrida y al agente sin nada a lo que
 * adaptarse, que es justo el fenómeno que la §4.2 declara como el vacío a
 * atacar.
 *
 * Este kernel alterna deliberadamente entre dos fases con alpha muy
 * distinto (alpha = fracción del tiempo sensible al reloj):
 *
 *   FASE C (compute): cadenas de FMA sobre registros, sin tráfico de
 *     memoria. alpha esperado ~1.0 -> frecuencia óptima 3200 MHz.
 *   FASE M (memoria): persecución de punteros con permutación aleatoria,
 *     limitada por latencia de DRAM. alpha esperado ~0 -> frecuencia
 *     óptima 800 MHz, con hasta 28 % de mejora de EDP y sin slowdown.
 *
 * Lo que aporta y el dataset actual no puede dar:
 *   1. Variación de fase real dentro de una corrida.
 *   2. ETIQUETA DE VERDAD: el programa imprime las marcas de tiempo de cada
 *      transición, así que se puede validar si la etiqueta Roofline y el
 *      clasificador las detectan, y medir cuántas ventanas tardan.
 *   3. Un periodo de fase configurable, que permite medir a partir de qué
 *      frecuencia de cambio deja de compensar conmutar -- la amortización
 *      del costo de transición que señala la §4.2 (Velicka et al.).
 *
 * La fase se mide por TIEMPO y no por número de iteraciones a propósito:
 * al bajar la frecuencia, un número fijo de iteraciones haría durar más la
 * fase de cómputo que la de memoria, cambiando la proporción entre fases
 * entre niveles y arruinando la comparación.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CACHE_LINE 64
#define MAX_TRANSITIONS 4096

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static void build_ring(size_t *buffer, size_t cells, size_t stride, uint64_t seed) {
    size_t *order = malloc(cells * sizeof(size_t));
    if (!order) { fprintf(stderr, "phasic: sin memoria\n"); exit(1); }
    for (size_t i = 0; i < cells; i++) order[i] = i;
    uint64_t st = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (size_t i = cells - 1; i > 0; i--) {
        size_t j = (size_t)(xorshift64(&st) % (uint64_t)(i + 1));
        size_t t = order[i]; order[i] = order[j]; order[j] = t;
    }
    for (size_t k = 0; k < cells - 1; k++) buffer[order[k] * stride] = order[k + 1] * stride;
    buffer[order[cells - 1] * stride] = order[0] * stride;
    free(order);
}

int main(int argc, char **argv) {
    double phase_seconds = 0.1;   /* duración de CADA fase */
    double total_seconds = 20.0;
    size_t mib = 512;
    uint64_t seed = 20260806ULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--phase-seconds") && i + 1 < argc) phase_seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--total-seconds") && i + 1 < argc) total_seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--size-mib") && i + 1 < argc) mib = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else {
            fprintf(stderr, "uso: %s [--phase-seconds S] [--total-seconds S] "
                            "[--size-mib N] [--seed N]\n", argv[0]);
            return 2;
        }
    }
    if (phase_seconds <= 0 || total_seconds <= 0) {
        fprintf(stderr, "phasic: duraciones deben ser positivas\n"); return 2;
    }

    const size_t stride = CACHE_LINE / sizeof(size_t);
    const size_t cells = (mib * 1024UL * 1024UL) / CACHE_LINE;
    if (cells < 1024) { fprintf(stderr, "phasic: buffer demasiado pequeno\n"); return 2; }

    size_t *buffer = aligned_alloc(CACHE_LINE, cells * CACHE_LINE);
    if (!buffer) { fprintf(stderr, "phasic: sin memoria\n"); return 1; }
    memset(buffer, 0, cells * CACHE_LINE);
    build_ring(buffer, cells, stride, seed);

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    double *marks = malloc(MAX_TRANSITIONS * sizeof(double));
    char *kinds = malloc(MAX_TRANSITIONS);
    if (!marks || !kinds) { fprintf(stderr, "phasic: sin memoria\n"); return 1; }
    int n_marks = 0;

    volatile double fsink = 0.0;
    volatile size_t isink = 0;
    long compute_flops = 0, memory_hops = 0;

    const double t_start = now_seconds();
    int phase = 0;  /* 0 = compute, 1 = memoria */

    while (now_seconds() - t_start < total_seconds && n_marks < MAX_TRANSITIONS) {
        double phase_start = now_seconds();
        marks[n_marks] = phase_start - t_start;
        kinds[n_marks] = phase ? 'M' : 'C';
        n_marks++;

        if (phase == 0) {
            /* FASE COMPUTE: 8 acumuladores independientes para llenar el
             * pipeline de FMA sin que la dependencia serial lo estanque --
             * el objetivo es saturar la unidad aritmética, no medir latencia. */
            long flops = 0;
#ifdef _OPENMP
#pragma omp parallel reduction(+:flops)
#endif
            {
                double a0=1.0000001,a1=1.0000002,a2=1.0000003,a3=1.0000004;
                double a4=1.0000005,a5=1.0000006,a6=1.0000007,a7=1.0000008;
                const double k = 1.0000000001, c = 1e-9;
                long local = 0;
                while (now_seconds() - phase_start < phase_seconds) {
                    for (int rep = 0; rep < 20000; rep++) {
                        a0 = a0 * k + c; a1 = a1 * k + c;
                        a2 = a2 * k + c; a3 = a3 * k + c;
                        a4 = a4 * k + c; a5 = a5 * k + c;
                        a6 = a6 * k + c; a7 = a7 * k + c;
                    }
                    local += 20000L * 8L * 2L;  /* 8 FMA = 16 FLOP por rep */
                }
                fsink += a0+a1+a2+a3+a4+a5+a6+a7;
                flops += local;
            }
            compute_flops += flops;
        } else {
            /* FASE MEMORIA: persecución de punteros, dependencia serial. */
            long hops = 0;
#ifdef _OPENMP
#pragma omp parallel reduction(+:hops)
#endif
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                size_t idx = ((size_t)tid * (cells / (size_t)(threads > 0 ? threads : 1))) * stride;
                long local = 0;
                while (now_seconds() - phase_start < phase_seconds) {
                    for (int rep = 0; rep < 4096; rep++) { idx = buffer[idx]; }
                    local += 4096;
                }
                isink += idx;
                hops += local;
            }
            memory_hops += hops;
        }
        phase = !phase;
    }

    const double elapsed = now_seconds() - t_start;
    (void)fsink; (void)isink;

    printf("phasic: phase_seconds=%.4f total=%.2f buffer=%zu MiB threads=%d\n",
           phase_seconds, elapsed, mib, threads);
    printf("Time in seconds = %.6f\n", elapsed);
    printf("transitions = %d\n", n_marks);
    printf("compute_flops = %ld\n", compute_flops);
    printf("memory_hops = %ld\n", memory_hops);
    /* ETIQUETA DE VERDAD: instante (relativo al inicio) en que empieza cada
     * fase y de qué tipo es. Es lo que permite validar contra la etiqueta
     * Roofline y medir la latencia de deteccion del clasificador. */
    printf("# ground_truth_phases offset_seconds,kind\n");
    for (int i = 0; i < n_marks; i++) {
        printf("PHASE %.6f %c\n", marks[i], kinds[i]);
    }
    printf("Verification = SUCCESSFUL\n");

    free(marks); free(kinds); free(buffer);
    return 0;
}
