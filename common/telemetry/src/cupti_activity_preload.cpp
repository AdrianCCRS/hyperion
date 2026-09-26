#include <cupti.h>
#include <cupti_activity.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::size_t kBufferSize = 4 * 1024 * 1024;

struct ActivityRecord {
    std::string kind;
    std::string name;
    std::uint64_t start_ns = 0;
    std::uint64_t end_ns = 0;
    std::uint32_t device_id = 0;
    std::uint32_t context_id = 0;
    std::uint32_t stream_id = 0;
    std::uint32_t correlation_id = 0;
    std::int32_t grid_x = 0;
    std::int32_t grid_y = 0;
    std::int32_t grid_z = 0;
    std::int32_t block_x = 0;
    std::int32_t block_y = 0;
    std::int32_t block_z = 0;
    std::uint64_t transfer_bytes = 0;
};

struct State {
    std::mutex mutex;
    std::vector<ActivityRecord> records;
    std::string output_path;
    bool enabled = false;
};

State& state() {
    static State instance;
    return instance;
}

bool is_shell_interpreter() {
    // El catálogo ejecuta varios `bin/*` que son wrappers Bash. LD_PRELOAD
    // también entra en el intérprete, que no crea contexto CUDA y luego
    // engendra el binario real con el mismo preload. Inicializar CUPTI en
    // ambos procesos provoca dos propietarios de la misma traza y puede
    // bloquear el flush del padre. El hijo CUDA hace un exec separado y
    // vuelve a cargar esta biblioteca, así que saltar Bash no pierde datos.
    char path[4096]{};
    const auto count = ::readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (count <= 0) return false;  // fail open: una ruta desconocida se traza.
    path[count] = '\0';
    const std::string executable(path);
    const auto slash = executable.find_last_of('/');
    const std::string basename = executable.substr(slash == std::string::npos ? 0 : slash + 1);
    return basename == "bash" || basename == "sh" || basename == "dash";
}

std::uint64_t CUPTIAPI monotonic_timestamp() {
    timespec ts{};
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL
        + static_cast<std::uint64_t>(ts.tv_nsec);
}

void CUPTIAPI request_buffer(std::uint8_t** buffer, std::size_t* size, std::size_t* max_records) {
    void* allocation = nullptr;
    if (::posix_memalign(&allocation, 8, kBufferSize) != 0) {
        *buffer = nullptr;
        *size = 0;
        *max_records = 0;
        return;
    }
    *buffer = static_cast<std::uint8_t*>(allocation);
    *size = kBufferSize;
    *max_records = 0;
}

void CUPTIAPI complete_buffer(
    CUcontext context,
    std::uint32_t stream_id,
    std::uint8_t* buffer,
    std::size_t,
    std::size_t valid_size
) {
    if (buffer == nullptr) return;
    if (valid_size > 0) {
        CUpti_Activity* activity = nullptr;
        while (cuptiActivityGetNextRecord(buffer, valid_size, &activity) == CUPTI_SUCCESS) {
            ActivityRecord record;
            if (activity->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                const auto* kernel = reinterpret_cast<const CUpti_ActivityKernel9*>(activity);
                if (kernel->start == 0 || kernel->end <= kernel->start) continue;
                record.kind = "kernel";
                record.name = kernel->name == nullptr ? "" : kernel->name;
                record.start_ns = kernel->start;
                record.end_ns = kernel->end;
                record.device_id = kernel->deviceId;
                record.context_id = kernel->contextId;
                record.stream_id = kernel->streamId;
                record.correlation_id = kernel->correlationId;
                record.grid_x = kernel->gridX;
                record.grid_y = kernel->gridY;
                record.grid_z = kernel->gridZ;
                record.block_x = kernel->blockX;
                record.block_y = kernel->blockY;
                record.block_z = kernel->blockZ;
            } else if (activity->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
                const auto* copy = reinterpret_cast<const CUpti_ActivityMemcpy5*>(activity);
                if (copy->start == 0 || copy->end <= copy->start) continue;
                record.kind = "memcpy";
                record.name = "[memcpy]";
                record.start_ns = copy->start;
                record.end_ns = copy->end;
                record.device_id = copy->deviceId;
                record.context_id = copy->contextId;
                record.stream_id = copy->streamId;
                record.correlation_id = copy->correlationId;
                record.transfer_bytes = copy->bytes;
            } else {
                continue;
            }
            std::lock_guard<std::mutex> lock(state().mutex);
            state().records.push_back(std::move(record));
        }
    }
    std::free(buffer);
    (void)context;
    (void)stream_id;
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '\"') escaped += '\"';
        escaped += c;
    }
    escaped += '\"';
    return escaped;
}

void shutdown_trace() {
    State& s = state();
    if (!s.enabled) return;
    // The measured programs synchronize before normal termination. A forced
    // flush delivers the final, possibly non-full activity buffer.
    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);

    std::vector<ActivityRecord> records;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        records = s.records;
    }
    // LD_PRELOAD se hereda a los procesos auxiliares de un wrapper (rm, awk,
    // sha256sum...). Sin registros propios no deben truncar la traza del
    // proceso que si uso CUDA: un archivo ausente es un fallo explicito.
    if (records.empty()) {
        s.enabled = false;
        return;
    }
    std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
        if (a.start_ns != b.start_ns) return a.start_ns < b.start_ns;
        return a.end_ns < b.end_ns;
    });

    std::ofstream out(s.output_path, std::ios::trunc);
    if (!out) return;
    out << "launch_index,kernel_name,start_ns,end_ns,duration_ns,device_id,context_id,stream_id,"
           "correlation_id,grid_x,grid_y,grid_z,block_x,block_y,block_z,activity_kind,transfer_bytes\n";
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        out << i << ',' << csv_escape(r.name) << ',' << r.start_ns << ',' << r.end_ns << ','
            << (r.end_ns - r.start_ns) << ',' << r.device_id << ',' << r.context_id << ','
            << r.stream_id << ',' << r.correlation_id << ',' << r.grid_x << ',' << r.grid_y << ','
            << r.grid_z << ',' << r.block_x << ',' << r.block_y << ',' << r.block_z << ','
            << r.kind << ',' << r.transfer_bytes << '\n';
    }
    s.enabled = false;
}

__attribute__((constructor)) void initialize_trace() {
    const char* output = std::getenv("HYPERION_CUPTI_ACTIVITY_FILE");
    if (output == nullptr || *output == '\0') return;
    if (is_shell_interpreter()) return;
    State& s = state();
    s.output_path = output;
    if (cuptiActivityRegisterTimestampCallback(monotonic_timestamp) != CUPTI_SUCCESS) return;
    if (cuptiActivityRegisterCallbacks(request_buffer, complete_buffer) != CUPTI_SUCCESS) return;
    if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS) return;
    if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY) != CUPTI_SUCCESS) {
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        return;
    }
    s.enabled = true;
    std::atexit(shutdown_trace);
}

}  // namespace
