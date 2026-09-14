#ifndef HYPERION_DUAL_DISPATCH_TIMING_H
#define HYPERION_DUAL_DISPATCH_TIMING_H

#include <stdio.h>

/* Contrato comun de tiempo para el catalogo CPU/GPU.
 *
 * cold_t0_ns: datos de entrada listos en RAM del host; todavia no se ha
 * creado contexto, handle, plan, workspace ni estado explicito de biblioteca.
 * setup_complete_ns: recursos del dispositivo/biblioteca listos, pero aun no
 * se ha transferido ni ejecutado la primera operacion.
 * cold_t1_ns: resultado de la primera operacion listo otra vez en RAM host.
 * warm_t0/t1_ns: repeticiones posteriores, reutilizando contexto/handles/
 * planes/asignaciones pero pagando de nuevo todos los operandos H2D y todos
 * los resultados D2H en GPU.
 *
 * Generacion de entradas y verificacion quedan fuera de ambas regiones.
 */
static inline void print_dispatch_timing(
    long long cold_t0_ns,
    long long setup_complete_ns,
    long long cold_t1_ns,
    long long warm_t0_ns,
    long long warm_t1_ns
) {
    printf(" Cold region t0_ns = %lld\n", cold_t0_ns);
    printf(" Setup complete t_ns = %lld\n", setup_complete_ns);
    printf(" Cold region t1_ns = %lld\n", cold_t1_ns);
    printf(" Measured region t0_ns = %lld\n", warm_t0_ns);
    printf(" Measured region t1_ns = %lld\n", warm_t1_ns);
    printf(" Cold time in seconds = %.9f\n", (cold_t1_ns - cold_t0_ns) / 1e9);
    printf(" Setup time in seconds = %.9f\n", (setup_complete_ns - cold_t0_ns) / 1e9);
    printf(" First dispatch time in seconds = %.9f\n", (cold_t1_ns - setup_complete_ns) / 1e9);
}

#endif
