/* ARC-70: fuerza cudaDeviceScheduleBlockingSync en un binario CUDA de
 * terceros (Rodinia u otros del catalogo GPU) sin tocar su codigo fuente --
 * mismo principio del proyecto que ya aplica a NPB/STREAM en CPU (medir
 * binarios de referencia sin modificarlos).
 *
 * Por que hace falta: cudaDeviceSynchronize() hace spin por defecto
 * (cudaDeviceScheduleAuto), lo que hace que el hilo de CPU que espera a la
 * GPU aparezca con IPC alto y casi cero cache-misses -- el clasificador de
 * CPU lo veria como compute_bound cuando en realidad no hace nada util. Ver
 * docs/retoma/pacca/Diseno_Politica_DVFS_CPU_GPU.md seccion 3.5.a/4.1.
 *
 * Mecanismo: se carga con LD_PRELOAD antes de ejecutar el binario objetivo.
 * Un constructor de libreria (__attribute__((constructor))) corre antes de
 * main() del binario objetivo -- antes de que pueda haber hecho ninguna
 * llamada CUDA propia -- y fija el flag de contexto ahi, con
 * cudaSetDeviceFlags() (solo valido antes de que exista un contexto CUDA
 * activo, que es exactamente la garantia que da correr esto en un
 * constructor).
 *
 * Solo necesita el API C de cudart (cudaSetDeviceFlags/cudaGetErrorString),
 * sin codigo de dispositivo -- compila con g++ normal, no hace falta nvcc.
 * Verificado empiricamente en paccaA100 con un probe dedicado
 * (sync_probe.cu, no comprometido al repo) que mide el % de tiempo de CPU
 * en tiempo de usuario durante un cudaDeviceSynchronize() deliberadamente
 * largo: sin este shim, 99.8% (spin real); con el shim cargado via
 * LD_PRELOAD, 0.0% (bloqueo real). Confirmado ademas que no altera la
 * salida de hotspot ni pathfinder (los dos kernels Rodinia del catalogo,
 * ARC-70) corridos con y sin el shim -- diff identico. */
#include <cuda_runtime.h>
#include <cstdio>

__attribute__((constructor))
static void force_blocking_sync() {
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[blocking_sync_shim] cudaSetDeviceFlags failed: %s\n",
                     cudaGetErrorString(err));
    }
}
