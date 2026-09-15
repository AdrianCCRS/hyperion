// Reemplazo minimo de helper_timer.h (CUDA Samples de NVIDIA) --
// euler3d_double.cu solo usa StopWatchInterface + sdkCreateTimer/
// sdkStartTimer/sdkStopTimer. Implementacion equivalente con
// std::chrono, sin dependencias externas.
#ifndef HELPER_TIMER_H_SHIM
#define HELPER_TIMER_H_SHIM
#include <chrono>

class StopWatchInterface {
public:
    void start() { start_ = std::chrono::steady_clock::now(); running_ = true; }
    void stop() {
        if (running_) {
            elapsed_ += std::chrono::steady_clock::now() - start_;
            running_ = false;
        }
    }
    void reset() { elapsed_ = std::chrono::duration<double, std::milli>::zero(); running_ = false; }
    float getTime() const { return static_cast<float>(elapsed_.count()); }
private:
    std::chrono::steady_clock::time_point start_;
    std::chrono::duration<double, std::milli> elapsed_{0};
    bool running_ = false;
};

inline void sdkCreateTimer(StopWatchInterface **timer) { *timer = new StopWatchInterface(); }
inline void sdkStartTimer(StopWatchInterface **timer) { (*timer)->start(); }
inline void sdkStopTimer(StopWatchInterface **timer) { (*timer)->stop(); }
inline void sdkResetTimer(StopWatchInterface **timer) { (*timer)->reset(); }
inline float sdkGetTimerValue(StopWatchInterface **timer) { return (*timer)->getTime(); }
inline float sdkGetAverageTimerValue(StopWatchInterface **timer) { return (*timer)->getTime(); }
inline void sdkDeleteTimer(StopWatchInterface **timer) { delete *timer; *timer = nullptr; }

#endif
