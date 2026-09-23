// Lee el contador de energia acumulada de la GPU (NVML, mJ) y lo imprime. Se llama antes y despues de una corrida para
// obtener su energia de GPU exacta (el delta), sin muestrear ni lanzar procesos durante la corrida. Uso: gpu_energy_read [indice]
#include <nvml.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    unsigned int idx = argc > 1 ? static_cast<unsigned int>(std::atoi(argv[1])) : 0;
    if (nvmlInit_v2() != NVML_SUCCESS) return 2;
    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex_v2(idx, &dev) != NVML_SUCCESS) return 3;
    unsigned long long mj = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(dev, &mj) != NVML_SUCCESS) return 4;
    std::printf("%llu\n", mj);
    nvmlShutdown();
    return 0;
}
