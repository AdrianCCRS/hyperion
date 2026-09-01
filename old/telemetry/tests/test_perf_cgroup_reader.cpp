#include "telemetry/perf_cgroup_reader.hpp"

#include <stdexcept>

int main() {
    {
        telemetry::PerfCgroupReader reader("", {0});
        bool threw = false;
        try {
            reader.open();
        } catch(const std::invalid_argument&) {
            threw = true;
        }
        if(!threw) return 1;
    }

    {
        telemetry::PerfCgroupReader reader("/definitely/not/a/tg/cgroup", {0});
        bool threw = false;
        try {
            reader.open();
        } catch(const std::runtime_error&) {
            threw = true;
        }
        if(!threw) return 1;
        if(reader.is_open()) return 1;
    }

    {
        telemetry::PerfCgroupReader reader("/tmp", {});
        bool threw = false;
        try {
            reader.open();
        } catch(const std::invalid_argument&) {
            threw = true;
        }
        if(!threw) return 1;
    }

    return 0;
}
