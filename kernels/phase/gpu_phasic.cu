/* gpu_phasic -- carga GPU MULTIFÁSICA con etiqueta de verdad conocida.
 *
 * Análogo en GPU de kernels/phase/phasic.c (ARC-176). Alterna dos fases con
 * sensibilidad muy distinta al reloj de SM:
 *
 *   FASE C (compute): cadenas de FMA sobre registros, sin tráfico de
 *     memoria. El tiempo escala con el reloj de SM -> alpha esperado ~1.
 *   FASE M (memoria): persecución de punteros independiente por hilo sobre
 *     un buffer muy superior a la L2. El tiempo lo fija la latencia y el
 *     ancho de banda de HBM, cuyo reloj es INDEPENDIENTE del de SM y no lo
 *     toca `nvidia-smi -lgc` -> alpha esperado ~0.
 *
 * Por qué muchos hilos y no pocos. Una persecución de punteros con baja
 * ocupación deja la GPU casi inactiva, que es exactamente el modo de fallo
 * de rodinia_lud en el catálogo actual (0% de utilización, potencia de
 * reposo, datos inservibles). Aquí cada hilo recorre su PROPIA cadena
 * independiente: la ocupación se mantiene alta --- utilización y potencia
 * reales, como rodinia_lavamd, que sostiene 247 W con alpha = 0.201 ---
 * mientras el avance de cada hilo sigue limitado por memoria.
 *
 * La fase se mide por TIEMPO y no por número de lanzamientos, por el mismo
 * motivo que en la versión de CPU: con un recuento fijo, bajar el reloj
 * alargaría la fase de cómputo más que la de memoria y cambiaría la
 * proporción entre ambas entre niveles del barrido.
 *
 * Valida CUDA en cada llamada. Dos kernels del catálogo actual
 * (rodinia_lud y rodinia_heartwall) salen con código 0 en una máquina sin
 * GPU, de modo que su verificación de éxito no distingue una corrida real
 * de una degradada; este programa falla ruidosamente en ese caso.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "gpu_phasic: %s:%d: %s -> %s\n", __FILE__,        \
                    __LINE__, #call, cudaGetErrorString(_e));                 \
            return EXIT_FAILURE;                                              \
        }                                                                     \
    } while (0)

static double now_seconds() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

/* FASE C: ocho acumuladores independientes para llenar el pipeline de FMA
 * sin que la dependencia serial lo estanque. El objetivo es saturar la
 * unidad aritmética, no medir latencia. */
__global__ void compute_phase(double *sink, long inner) {
    double a0 = 1.0000001, a1 = 1.0000002, a2 = 1.0000003, a3 = 1.0000004;
    double a4 = 1.0000005, a5 = 1.0000006, a6 = 1.0000007, a7 = 1.0000008;
    const double k = 1.0000000001, c = 1e-9;
    for (long i = 0; i < inner; ++i) {
        a0 = fma(a0, k, c); a1 = fma(a1, k, c);
        a2 = fma(a2, k, c); a3 = fma(a3, k, c);
        a4 = fma(a4, k, c); a5 = fma(a5, k, c);
        a6 = fma(a6, k, c); a7 = fma(a7, k, c);
    }
    /* El sumidero impide que el compilador elimine el bucle; la condición
     * es siempre falsa en la práctica pero el compilador no puede probarlo. */
    if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 < 0.0) sink[threadIdx.x] = a0;
}

/* FASE M: cada hilo recorre su propia cadena. `chain` contiene índices
 * pre-permutados; la dependencia idx = chain[idx] impide al hardware
 * adelantar el siguiente acceso. */
__global__ void memory_phase(const uint64_t *chain, uint64_t n,
                             uint64_t *sink, long hops) {
    uint64_t tid = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    uint64_t idx = (tid * 2654435761ULL) % n;   /* arranques dispersos */
    for (long i = 0; i < hops; ++i) idx = chain[idx];
    if (idx == 0xFFFFFFFFFFFFFFFFULL) sink[tid] = idx;
}

int main(int argc, char **argv) {
    double phase_seconds = 0.1;
    double total_seconds = 20.0;
    size_t mib = 2048;                 /* muy por encima de los 40 MiB de L2 */
    unsigned long long seed = 20260806ULL;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--phase-seconds") && i + 1 < argc) phase_seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--total-seconds") && i + 1 < argc) total_seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--size-mib") && i + 1 < argc) mib = size_t(atol(argv[++i]));
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        else {
            fprintf(stderr, "uso: %s [--phase-seconds S] [--total-seconds S] "
                            "[--size-mib N] [--seed N]\n", argv[0]);
            return 2;
        }
    }

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices < 1) { fprintf(stderr, "gpu_phasic: no hay GPU disponible\n"); return 1; }

    const uint64_t n = (uint64_t(mib) * 1024ULL * 1024ULL) / sizeof(uint64_t);

    /* Permutación construida en el host con un PRNG propio (xorshift64*)
     * para que sea reproducible entre corridas y entre niveles de
     * frecuencia: si cambiara, cambiaría el patrón de fallos y las corridas
     * no serían comparables. */
    uint64_t *h_chain = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!h_chain) { fprintf(stderr, "gpu_phasic: sin memoria en host\n"); return 1; }
    for (uint64_t i = 0; i < n; ++i) h_chain[i] = i;
    uint64_t st = seed;
    for (uint64_t i = n - 1; i > 0; --i) {
        st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
        uint64_t j = (st * 0x2545F4914F6CDD1DULL) % (i + 1);
        uint64_t t = h_chain[i]; h_chain[i] = h_chain[j]; h_chain[j] = t;
    }

    uint64_t *d_chain = nullptr, *d_isink = nullptr;
    double *d_dsink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chain, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_chain, h_chain, n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    free(h_chain);
    CUDA_CHECK(cudaMalloc(&d_isink, 1024 * 1024 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_dsink, 1024 * sizeof(double)));

    const int block = 256;
    const int grid_compute = 2048;
    const int grid_memory = 4096;   /* ocupación alta: evita el modo lud */

    printf("gpu_phasic: phase_seconds=%.4f total=%.1f buffer=%zu MiB "
           "grid_c=%d grid_m=%d block=%d\n",
           phase_seconds, total_seconds, mib, grid_compute, grid_memory, block);

    const double t0 = now_seconds();
    int phase = 0;
    int transitions = 0;
    double *marks = (double *)malloc(8192 * sizeof(double));
    char *kinds = (char *)malloc(8192);

    while (now_seconds() - t0 < total_seconds && transitions < 8192) {
        const double phase_start = now_seconds();
        marks[transitions] = phase_start - t0;
        kinds[transitions] = phase ? 'M' : 'C';
        ++transitions;

        while (now_seconds() - phase_start < phase_seconds) {
            if (phase == 0) compute_phase<<<grid_compute, block>>>(d_dsink, 20000L);
            else            memory_phase<<<grid_memory, block>>>(d_chain, n, d_isink, 2048L);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        phase = !phase;
    }

    const double elapsed = now_seconds() - t0;
    printf("Time in seconds = %.6f\n", elapsed);
    printf("transitions = %d\n", transitions);
    /* ANCLA DE RELOJ. Sin esto las marcas de abajo son offsets sin origen y
     * NO se pueden cruzar con windows.csv: el colector estampa t_start_ns /
     * t_end_ns con CLOCK_MONOTONIC absoluto (telemetry metrics.hpp), no con
     * el t0 de este proceso. Publicar t0 en el MISMO reloj es lo que hace
     * alineable la etiqueta de verdad. */
    printf("T0_MONOTONIC_NS %lld\n", (long long)(t0 * 1e9));
    /* ETIQUETA DE VERDAD: instante (relativo a T0_MONOTONIC_NS) en que
     * empieza cada fase y de qué tipo. Permite validar la etiqueta Roofline
     * y medir el retardo de detección del clasificador en ventanas. */
    printf("# ground_truth_phases offset_seconds,kind\n");
    for (int i = 0; i < transitions; ++i) printf("PHASE %.6f %c\n", marks[i], kinds[i]);
    printf("Verification = SUCCESSFUL\n");

    free(marks); free(kinds);
    CUDA_CHECK(cudaFree(d_chain));
    CUDA_CHECK(cudaFree(d_isink));
    CUDA_CHECK(cudaFree(d_dsink));
    return EXIT_SUCCESS;
}
