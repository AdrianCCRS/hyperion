/* Carga objetivo de prueba (Bloque C8): alterna fases memory_bound y
 * compute_bound de un solo hilo, TODO en espacio de usuario, e imprime cada
 * transicion como "<ns CLOCK_MONOTONIC> <M|C>". Sirve para (1) probar el
 * daemon activo con un target realista y (2) puntuar la clasificacion contra
 * fronteras de fase conocidas: `steady_clock` de C++ en Linux es el mismo
 * CLOCK_MONOTONIC que usa el registro de decisiones.
 *
 * Por que existe: el primer target del preflight fue `dd if=/dev/zero
 * of=/dev/null`, que gasta casi todo su tiempo en el kernel; el collector abre
 * los contadores con exclude_kernel=1 (perf_reader.cpp), asi que veia ~0
 * ciclos y el 96% de las ventanas fallaban con zero_cycles. No era un fallo
 * del collector sino un target inadecuado.
 *
 * Uso: phase_target [segundos_por_fase=3] [segundos_totales=1e9]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

#define N (32u * 1024u * 1024u) /* 3 arreglos de 256 MB: mucho mas que L3 (12 MB/socket) */

int main(int argc, char** argv) {
    const double phase_s = argc > 1 ? atof(argv[1]) : 3.0;
    const double total_s = argc > 2 ? atof(argv[2]) : 1e9;
    double* a = malloc(N * sizeof(double));
    double* b = malloc(N * sizeof(double));
    double* c = malloc(N * sizeof(double));
    if (!a || !b || !c) return 1;
    for (size_t i = 0; i < N; ++i) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; }

    const int64_t t_end = now_ns() + (int64_t)(total_s * 1e9);
    volatile double sink = 0.0;
    int memory = 1;
    while (now_ns() < t_end) {
        const int64_t t0 = now_ns();
        printf("%lld %c\n", (long long)t0, memory ? 'M' : 'C');
        fflush(stdout);
        const int64_t t_phase = t0 + (int64_t)(phase_s * 1e9);
        if (memory) {  /* triad: ~2 lecturas + 1 escritura por elemento, sin reutilizacion */
            while (now_ns() < t_phase) {
                for (size_t i = 0; i < N; ++i) a[i] = b[i] + 3.0 * c[i];
                sink += a[N / 2];
            }
        } else {       /* cadenas de FMA independientes en registros: sin trafico de memoria */
            double x0 = 1.0, x1 = 1.1, x2 = 1.2, x3 = 1.3, x4 = 1.4, x5 = 1.5, x6 = 1.6, x7 = 1.7;
            while (now_ns() < t_phase) {
                for (int k = 0; k < 200000; ++k) {
                    x0 = x0 * 1.0000001 + 1e-9; x1 = x1 * 1.0000001 + 1e-9;
                    x2 = x2 * 1.0000001 + 1e-9; x3 = x3 * 1.0000001 + 1e-9;
                    x4 = x4 * 1.0000001 + 1e-9; x5 = x5 * 1.0000001 + 1e-9;
                    x6 = x6 * 1.0000001 + 1e-9; x7 = x7 * 1.0000001 + 1e-9;
                }
                sink += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
            }
        }
        memory = !memory;
    }
    return 0;
}
