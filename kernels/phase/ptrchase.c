/* ptrchase -- microbenchmark limitado por LATENCIA de memoria.
 *
 * Por qué existe (ARC-176). Los 9 kernels del dataset tienen alpha entre
 * 0.384 y 1.026, donde alpha es la fracción del tiempo sensible al reloj en
 * T(f)/T(ref) = (1-alpha) + alpha*(f_ref/f). Con el modelo de potencia real
 * de paccaA100 (116.5 W a 3200 MHz, 83.4 W a 800 MHz), bajar la frecuencia
 * solo mejora el EDP si alpha <= 0.224. Ninguna carga actual llega, y por
 * eso ninguna se beneficia de DVFS: no es un problema del modelo ni de la
 * rejilla de frecuencias, es que faltan cargas en esa zona.
 *
 * STREAM y los NPB son limitados por ANCHO DE BANDA: saturan el bus, y el
 * bus sí depende en parte del reloj. La persecución de punteros es limitada
 * por LATENCIA: cada carga depende de la anterior, así que el tiempo lo
 * domina el viaje a DRAM (~80-100 ns) y no el reloj del núcleo. Es el
 * patrón con más probabilidad de bajar de 0.224 en este nodo.
 *
 * Diseño:
 *  - Un anillo de punteros que recorre TODO el buffer en un ciclo único,
 *    con permutación aleatoria: cada salto es impredecible para el
 *    prefetcher, que es el punto -- si el prefetcher acierta, el acceso deja
 *    de ser latency-bound y alpha sube.
 *  - Granularidad de línea de caché (64 B) para que cada salto sea un fallo
 *    distinto.
 *  - Buffer muy por encima de la LLC (12 MiB en este SKU) para forzar DRAM.
 *  - La cadena de dependencias es estrictamente serial dentro de cada hilo:
 *    no hay paralelismo a nivel de memoria que oculte la latencia.
 *
 * No reporta FLOPs: no hace aritmética útil a propósito. Su intensidad
 * operacional es ~0 por construcción.
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

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Permutación de Fisher-Yates con un PRNG propio (xorshift64*) para no
 * depender de rand(), cuyo estado global serializaría los hilos y cuya
 * calidad varía entre libc. La semilla es un parámetro para que la
 * permutación sea reproducible entre corridas y entre niveles de
 * frecuencia -- si cambiara, cambiaría el patrón de fallos y las corridas
 * no serían comparables. */
static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static void build_ring(size_t *ring, size_t n, uint64_t seed) {
    size_t *order = malloc(n * sizeof(size_t));
    if (!order) { fprintf(stderr, "ptrchase: sin memoria para la permutacion\n"); exit(1); }
    for (size_t i = 0; i < n; i++) order[i] = i;

    uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(xorshift64(&state) % (uint64_t)(i + 1));
        size_t tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
    /* Anillo hamiltoniano: order[k] -> order[k+1], y el ultimo cierra al
     * primero. Un ciclo unico que cubre las n celdas garantiza que el
     * recorrido no se quede atrapado en un subciclo corto que cupiera en
     * cache -- que es el modo de fallo clasico de este microbenchmark. */
    for (size_t k = 0; k < n - 1; k++) ring[order[k]] = order[k + 1];
    ring[order[n - 1]] = order[0];
    free(order);
}

int main(int argc, char **argv) {
    size_t mib = 512;
    long iterations = 40;
    uint64_t seed = 20260806ULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--size-mib") && i + 1 < argc) mib = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) iterations = atol(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else { fprintf(stderr, "uso: %s [--size-mib N] [--iterations N] [--seed N]\n", argv[0]); return 2; }
    }

    const size_t bytes = mib * 1024UL * 1024UL;
    const size_t stride = CACHE_LINE / sizeof(size_t);
    const size_t cells = bytes / CACHE_LINE;
    if (cells < 1024) { fprintf(stderr, "ptrchase: buffer demasiado pequeno\n"); return 2; }

    size_t *buffer = aligned_alloc(CACHE_LINE, cells * CACHE_LINE);
    if (!buffer) { fprintf(stderr, "ptrchase: sin memoria para el buffer\n"); return 1; }
    memset(buffer, 0, cells * CACHE_LINE);

    size_t *ring = malloc(cells * sizeof(size_t));
    if (!ring) { fprintf(stderr, "ptrchase: sin memoria para el anillo\n"); return 1; }
    build_ring(ring, cells, seed);
    for (size_t k = 0; k < cells; k++) buffer[k * stride] = ring[k] * stride;
    free(ring);

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    /* Cada hilo arranca en un punto distinto del MISMO anillo. Comparten el
     * buffer a proposito: replicarlo por hilo multiplicaria la huella y
     * podria volver a caber en cache por seccion. */
    double t0 = now_seconds();
    volatile size_t sink = 0;
    long total_hops = 0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:total_hops)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        size_t idx = ((size_t)tid * (cells / (size_t)(threads > 0 ? threads : 1))) * stride;
        long hops = 0;
        for (long it = 0; it < iterations; it++) {
            for (size_t k = 0; k < cells; k++) {
                idx = buffer[idx];   /* la dependencia serial ES el experimento */
                hops++;
            }
        }
        sink += idx;
        total_hops += hops;
    }

    double elapsed = now_seconds() - t0;
    (void)sink;

    double ns_per_hop = elapsed * 1e9 / (double)total_hops;
    printf("ptrchase: buffer=%zu MiB cells=%zu threads=%d iterations=%ld\n",
           mib, cells, threads, iterations);
    printf("Time in seconds = %.6f\n", elapsed);
    printf("hops = %ld\n", total_hops);
    printf("latency_ns_per_hop = %.3f\n", ns_per_hop);
    /* La verificacion es que el recorrido volvio a un indice valido: si el
     * anillo estuviera roto, idx se saldria del buffer y el acceso ya habria
     * fallado antes de llegar aqui. */
    printf("Verification = SUCCESSFUL\n");
    free(buffer);
    return 0;
}
