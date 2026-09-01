/* Extiende common/hpc/native/blocking_sync_shim.cpp (ARC-70) para el loop de
 * GPU del daemon (§4.1/§4.3 punto 6 del plan de realineación). Conserva
 * exactamente el mecanismo original -- fuerza cudaDeviceScheduleBlockingSync
 * vía un constructor, sin tocar el binario de terceros -- e INTENTA añadir
 * la detección de fronteras de fase que gpu_clock_controller.hpp necesitaba.
 *
 * ⚠️⚠️ HALLAZGO CRÍTICO CONFIRMADO (no una limitación de entorno como se
 * documentó en una versión anterior de este comentario) -- léase antes de
 * usar este archivo: **la intercepción de `cudaLaunchKernel` NO FUNCIONA
 * para kernels lanzados con la sintaxis estándar `<<<>>>`**, verificado
 * compilando y corriendo este shim contra un kernel CUDA real (nvcc 13.3,
 * GPU NVIDIA real, driver 610.57.04) tras conseguir acceso a un entorno con
 * CUDA toolkit. Evidencia exacta:
 *
 *   1. `cudaDeviceSynchronize`/`cudaStreamSynchronize` SÍ se interceptan
 *      correctamente (confirmado con una build de depuración: el
 *      `fprintf` de diagnóstico se imprime dos veces para dos llamadas
 *      reales) -- pero SOLO cuando el binario objetivo enlaza cudart de
 *      forma dinámica (`nvcc -cudart shared`).
 *   2. `cudaLaunchKernel` NUNCA se intercepta, en NINGÚN modo de enlace
 *      (ni `-cudart static`, el default de nvcc moderno, ni `-cudart
 *      shared`). Confirmado con `nm -D` sobre el binario compilado: la
 *      sintaxis `kernel<<<grid,block>>>(args)` NO genera una llamada
 *      dinámica a `cudaLaunchKernel` en absoluto -- nvcc compila esa
 *      sintaxis en un stub de lanzamiento (`__device_stub__...` +
 *      `__cudaPushCallConfiguration`) que resuelve la llamada real en
 *      tiempo de COMPILACIÓN/ENLACE, nunca a través de la tabla de
 *      símbolos dinámicos que `LD_PRELOAD` puede interceptar -- con
 *      `-cudart shared`, el binario ni siquiera importa `cudaLaunchKernel`
 *      como símbolo dinámico (verificado, `nm -D` no lo lista).
 *
 * Consecuencia directa: el mecanismo de "BEGIN en el primer
 * cudaLaunchKernel tras una sincronización" descrito más abajo **nunca se
 * dispara** contra un kernel real compilado normalmente -- ningún evento
 * BEGIN llega jamás a fase3_daemon/shim/event_listener.py, verificado con
 * un socket Unix real y un cliente de prueba: el listener recibe eventos
 * enviados manualmente sin problema, pero cero eventos del shim real
 * corriendo contra un kernel CUDA real.
 *
 * Qué SÍ sigue funcionando, sin cambios: el mecanismo original de ARC-70
 * (forzar cudaDeviceScheduleBlockingSync vía el constructor) no depende de
 * interceptar cudaLaunchKernel en absoluto -- es una llamada directa y
 * proactiva antes de main(), no una intercepción. Nada en esta
 * investigación pone en duda esa parte.
 *
 * Caminos de arreglo reales, ninguno implementado todavía (decisión de
 * diseño pendiente, no una corrección menor):
 *   (a) Interceptar en la API de driver (`cuLaunchKernel` de libcuda.so)
 *       en vez de la API de runtime -- pero cudart resuelve esa llamada
 *       internamente vía `dlsym()` sobre un handle propio, no vía la tabla
 *       de símbolos global que LD_PRELOAD puede alterar, así que esto
 *       probablemente tenga el mismo problema sin una técnica adicional
 *       (hookear el propio dlsym(), o usar CUDA_INJECTION64_PATH).
 *   (b) Usar el mecanismo oficial de NVIDIA para esto (`CUDA_INJECTION64_PATH`
 *       + CUPTI callback API), diseñado exactamente para interceptar
 *       lanzamientos de kernel de forma confiable -- más pesado, pero es
 *       la vía que usan las herramientas de profiling reales de NVIDIA.
 *   (c) Abandonar la detección de fase basada en intercepción y detectar
 *       actividad GPU sondeando `gpu_util_pct` desde el daemon mismo (NVML,
 *       ya disponible vía fase3_daemon/shim/event_listener.py::query_gpu_features)
 *       -- evita el problema por completo, a costa de perder la frontera
 *       exacta de "primer lanzamiento" (la transición se detectaría con la
 *       latencia del muestreo NVML, no al instante).
 *
 * fase3_daemon/gpu_loop/loop.py y shim/event_listener.py (el lado Python
 * del canal) siguen siendo correctos y están probados de punta a punta con
 * un socket real -- el problema está exclusivamente en que ESTE archivo no
 * logra emitir los eventos que ellos esperan consumir.
 *
 * Granularidad de fase pretendida (documentado igual, aunque el mecanismo
 * no la logre hoy): un kernel real puede hacer miles de llamadas de
 * lanzamiento entre sincronizaciones (rodinia_gaussian: ~8190 por corrida,
 * ARC-110) -- tratar cada llamada como una fase nueva violaría el
 * principio de gpu_clock_controller.hpp ("decidir cada ventana gastaría
 * más en overhead de transición de lo que ahorra"). Una fase debería ser
 * un PERIODO DE ACTIVIDAD GPU: BEGIN en el primer lanzamiento después de
 * una sincronización, END en la sincronización que le sigue.
 *
 * Transporte (la parte que sí se verificó end-to-end con un socket real):
 * socket de dominio Unix, no bloqueante para el hilo que corre el kernel
 * real. Ruta del socket: variable de entorno HYPERION_GPU_PHASE_SOCKET; si
 * no está definida, el shim sigue forzando blocking-sync (comportamiento
 * original intacto) pero no intenta emitir eventos. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <atomic>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

using CudaLaunchKernelFn = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
using CudaDeviceSynchronizeFn = cudaError_t (*)();
using CudaStreamSynchronizeFn = cudaError_t (*)(cudaStream_t);

CudaLaunchKernelFn real_cuda_launch_kernel = nullptr;
CudaDeviceSynchronizeFn real_cuda_device_synchronize = nullptr;
CudaStreamSynchronizeFn real_cuda_stream_synchronize = nullptr;

// true mientras hay lanzamientos sin sincronizar todavía -- ver la nota de
// granularidad de fase arriba. std::atomic porque el binario de terceros
// puede lanzar desde varios hilos (poco común pero no descartado); no
// intenta ser una máquina de estados por-stream, a propósito: el daemon ve
// "la GPU está ocupada" como una sola señal, igual que
// gpu_clock_controller.hpp la trata como una sola fase a la vez.
std::atomic<bool> phase_active{false};

int phase_socket_fd = -1;
bool phase_socket_attempted = false;

int64_t monotonic_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}

// Conecta (una sola vez, perezosamente) al socket de eventos de fase. Si
// HYPERION_GPU_PHASE_SOCKET no está definida, o la conexión falla, deja
// phase_socket_fd en -1 permanentemente -- nunca reintenta ni bloquea:
// esta función se llama desde el camino caliente de lanzamiento de kernel.
int phase_socket() {
    if (phase_socket_attempted) return phase_socket_fd;
    phase_socket_attempted = true;

    const char* path = std::getenv("HYPERION_GPU_PHASE_SOCKET");
    if (path == nullptr || path[0] == '\0') return -1;

    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    // SOCK_DGRAM + connect(): el "connect" de un socket de datagrama solo
    // fija el destino por defecto para send(), no abre una conexión con
    // estado -- si el daemon todavía no está escuchando, el connect en sí
    // puede seguir teniendo éxito (o fallar de inmediato según el sistema);
    // send() más abajo es el punto real donde un evento puede perderse sin
    // bloquear, que es la garantía que este shim necesita.
    if (connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    phase_socket_fd = fd;
    return phase_socket_fd;
}

// Formato del mensaje: "BEGIN,<ns>\n" / "END,<ns>\n" -- texto plano
// deliberado (no un formato binario propio) para poder depurar con
// `socat`/`nc` sin herramientas adicionales durante la prueba dirigida que
// pide el plan.
void emit_phase_event(const char* kind, int64_t now_ns) {
    int fd = phase_socket();
    if (fd < 0) return;
    char message[64];
    int len = std::snprintf(message, sizeof(message), "%s,%lld\n", kind,
                             static_cast<long long>(now_ns));
    if (len <= 0) return;
    // MSG_DONTWAIT: nunca bloquea el hilo que corre el kernel real, ni
    // siquiera si el buffer del socket está lleno -- perder un evento es
    // aceptable (el daemon simplemente no actúa esa fase), bloquear el
    // binario medido no lo es.
    send(fd, message, static_cast<size_t>(len), MSG_DONTWAIT);
}

void ensure_real_symbols_resolved() {
    if (real_cuda_launch_kernel == nullptr) {
        real_cuda_launch_kernel =
            reinterpret_cast<CudaLaunchKernelFn>(dlsym(RTLD_NEXT, "cudaLaunchKernel"));
    }
    if (real_cuda_device_synchronize == nullptr) {
        real_cuda_device_synchronize =
            reinterpret_cast<CudaDeviceSynchronizeFn>(dlsym(RTLD_NEXT, "cudaDeviceSynchronize"));
    }
    if (real_cuda_stream_synchronize == nullptr) {
        real_cuda_stream_synchronize =
            reinterpret_cast<CudaStreamSynchronizeFn>(dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
    }
}

}  // namespace

extern "C" {

__attribute__((constructor))
static void force_blocking_sync() {
    // Mecanismo original de ARC-70, sin cambios -- ver
    // common/hpc/native/blocking_sync_shim.cpp para el porqué completo.
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[blocking_sync_shim] cudaSetDeviceFlags failed: %s\n",
                     cudaGetErrorString(err));
    }
}

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                              void** args, size_t sharedMem, cudaStream_t stream) {
    ensure_real_symbols_resolved();

    // BEGIN solo en la transición idle -> ocupado, nunca en cada llamada
    // (ver la nota de granularidad de fase al inicio del archivo).
    bool was_active = phase_active.exchange(true);
    if (!was_active) {
        emit_phase_event("BEGIN", monotonic_now_ns());
    }

    return real_cuda_launch_kernel(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t cudaDeviceSynchronize(void) {
    ensure_real_symbols_resolved();
    // La sincronización REAL ocurre primero -- el evento END debe reflejar
    // el instante en que la GPU de verdad terminó, no cuándo se decidió
    // notificar. cudaDeviceScheduleBlockingSync (forzado en el constructor
    // de arriba) es lo que hace que esta espera sea un bloqueo real y no
    // spin, igual que en el shim original.
    cudaError_t result = real_cuda_device_synchronize();

    bool was_active = phase_active.exchange(false);
    if (was_active) {
        emit_phase_event("END", monotonic_now_ns());
    }
    return result;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    ensure_real_symbols_resolved();
    cudaError_t result = real_cuda_stream_synchronize(stream);

    // Nota: con múltiples streams, sincronizar UNO no implica que la GPU
    // esté ociosa en general -- pero el catálogo de este proyecto (kernels
    // Rodinia/NPB de terceros) no usa streams concurrentes propios más
    // allá del default, así que tratar cualquier sincronización (device o
    // stream) como fin de fase es una simplificación deliberada, documentada
    // aquí, no un descuido. Si algún kernel del catálogo llega a usar
    // múltiples streams reales, esta simplificación debe revisarse antes de
    // confiar en sus eventos de fase.
    bool was_active = phase_active.exchange(false);
    if (was_active) {
        emit_phase_event("END", monotonic_now_ns());
    }
    return result;
}

}  // extern "C"
