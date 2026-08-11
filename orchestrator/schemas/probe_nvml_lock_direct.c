#include <nvml.h>
#include <stdio.h>
#include <unistd.h>

static void check(nvmlReturn_t r, const char *what) {
    if (r != NVML_SUCCESS) {
        fprintf(stderr, "%s FALLO: %s\n", what, nvmlErrorString(r));
    } else {
        printf("%s OK\n", what);
    }
}

int main(void) {
    check(nvmlInit_v2(), "nvmlInit_v2");

    nvmlDevice_t dev;
    check(nvmlDeviceGetHandleByIndex_v2(0, &dev), "nvmlDeviceGetHandleByIndex_v2");

    unsigned int before = 0;
    check(nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &before), "nvmlDeviceGetClockInfo(antes)");
    printf("reloj SM antes: %u MHz\n", before);

    unsigned int target = 1200;
    nvmlReturn_t lock_result = nvmlDeviceSetGpuLockedClocks(dev, target, target);
    check(lock_result, "nvmlDeviceSetGpuLockedClocks(1200,1200)");

    for (int i = 0; i < 5; i++) {
        usleep(300000);
        unsigned int cur = 0;
        nvmlUtilization_t util;
        nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &cur);
        nvmlDeviceGetUtilizationRates(dev, &util);
        printf("t+%dms: reloj SM=%u MHz, util=%u%%\n", (i + 1) * 300, cur, util.gpu);
    }

    check(nvmlDeviceResetGpuLockedClocks(dev), "nvmlDeviceResetGpuLockedClocks");
    check(nvmlShutdown(), "nvmlShutdown");
    return 0;
}
