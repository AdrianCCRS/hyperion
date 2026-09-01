/* ARC-53: mismo criterio de D05 (environment.probe_pmc_count) sin
 * depender del CLI `perf`, ausente en paccaA100 (confirmado en la
 * auditoria: no hay linux-tools instalado, aunque el mecanismo real de
 * medicion -- perf_event_open, PID + inherit=1 -- si funciona bajo
 * perf_event_paranoid=2). Se usa en vez de reimplementar el struct
 * perf_event_attr en Python/ctypes: un ABI de kernel mal replicado de
 * memoria fallaria silenciosamente distinto (el mismo riesgo que ya
 * confirmamos con la codificacion cruda de stalled cycles, ARC-52) --
 * este C reusa exactamente el mismo patron ya validado en
 * telemetry/src/perf_reader.cpp.
 *
 * Uso: pmc_multiplex_probe <evento0> [evento1 ...]  (hasta 10 nombres de
 * orchestrator.environment._GENERIC_HARDWARE_EVENTS, ya traducidos a
 * PERF_COUNT_HW_* por config_for_name()). Abre un evento POR CADA
 * argumento sobre un hijo corriendo `sleep 0.2` (mismo patron que
 * `perf stat -e ... -- sleep 0.2`), inherit=1. Si algun evento tiene
 * time_running < time_enabled (evidencia de multiplexado de PMCs),
 * imprime "(50.00%)" a stdout -- el mismo formato que el marcador de
 * porcentaje que `perf stat` imprime en su stderr, para que
 * environment._MULTIPLEXING_MARKER lo reconozca sin cambios. Si no hay
 * evidencia, no imprime nada. */
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>

static long perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, flags);
}

/* Codigos estables de linux/perf_event.h, PERF_TYPE_HARDWARE. Debe
 * coincidir exactamente con el orden/nombres de
 * environment._GENERIC_HARDWARE_EVENTS. */
static int config_for_name(const char* name, uint64_t* out) {
    static const struct { const char* name; uint64_t config; } table[] = {
        {"instructions", PERF_COUNT_HW_INSTRUCTIONS},
        {"cycles", PERF_COUNT_HW_CPU_CYCLES},
        {"cache-references", PERF_COUNT_HW_CACHE_REFERENCES},
        {"cache-misses", PERF_COUNT_HW_CACHE_MISSES},
        {"branch-instructions", PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
        {"branch-misses", PERF_COUNT_HW_BRANCH_MISSES},
        {"bus-cycles", PERF_COUNT_HW_BUS_CYCLES},
        {"ref-cycles", PERF_COUNT_HW_REF_CPU_CYCLES},
        {"stalled-cycles-frontend", PERF_COUNT_HW_STALLED_CYCLES_FRONTEND},
        {"stalled-cycles-backend", PERF_COUNT_HW_STALLED_CYCLES_BACKEND},
    };
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
        if (strcmp(table[i].name, name) == 0) { *out = table[i].config; return 1; }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s evento0 [evento1 ...]\n", argv[0]); return 2; }
    const int n = argc - 1;

    pid_t child = fork();
    if (child == 0) {
        execlp("sleep", "sleep", "0.2", (char*)NULL);
        _exit(127);
    }
    int status = 0;
    /* No hace falta STOP/CONT aqui: a diferencia del harness real, D05
     * solo necesita saber cuantos PMCs caben simultaneos, no atribuir
     * eventos a instrucciones precisas del hijo. */

    int fds[10];
    for (int i = 0; i < n; i++) {
        uint64_t config;
        if (!config_for_name(argv[i + 1], &config)) {
            fprintf(stderr, "evento desconocido: %s\n", argv[i + 1]);
            kill(child, SIGKILL);
            waitpid(child, &status, 0);
            return 2;
        }
        struct perf_event_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.type = PERF_TYPE_HARDWARE;
        attr.size = sizeof(attr);
        attr.config = config;
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.inherit = 1;
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        fds[i] = (int) perf_event_open(&attr, child, -1, -1, 0);
        if (fds[i] < 0) {
            /* Evento no abrible (permiso o no soportado): se trata igual
             * que "perf" ausente -- el llamador Python ya interpreta
             * cualquier fallo de apertura como evidencia de que este N no
             * cabe, deteniendo el barrido ahi. */
            fprintf(stderr, "perf_event_open fallo para %s: errno=%d\n", argv[i + 1], errno);
            kill(child, SIGKILL);
            waitpid(child, &status, 0);
            return 1;
        }
        ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
        ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
    }

    waitpid(child, &status, 0);

    int multiplexed = 0;
    for (int i = 0; i < n; i++) {
        struct { uint64_t value, time_enabled, time_running; } rf;
        if (read(fds[i], &rf, sizeof(rf)) == sizeof(rf)) {
            if (rf.time_running < rf.time_enabled) multiplexed = 1;
        }
        close(fds[i]);
    }

    if (multiplexed) printf("(50.00%%)\n");
    return 0;
}
