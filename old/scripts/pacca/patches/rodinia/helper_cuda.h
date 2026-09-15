// Reemplazo minimo de helper_cuda.h (CUDA Samples de NVIDIA, no incluido
// en el toolkit desde hace anios) -- euler3d_double.cu (Rodinia/cfd) solo
// usa checkCudaErrors() como wrapper de verificacion de error, nada mas de
// este header. Implementacion equivalente, sin dependencias externas.
#ifndef HELPER_CUDA_H_SHIM
#define HELPER_CUDA_H_SHIM
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void checkCudaErrorsImpl(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) checkCudaErrorsImpl((val), __FILE__, __LINE__)

#endif
