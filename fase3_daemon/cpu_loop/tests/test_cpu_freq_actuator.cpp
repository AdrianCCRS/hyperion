// Pruebas del actuador nativo de frecuencia (Bloque C8) sobre un sysfs
// simulado en un directorio temporal: no toca hardware ni requiere permisos.
// Usa CHECK propio (no assert): el build por defecto es RelWithDebInfo, que
// define NDEBUG y volvería un assert() un no-op silencioso.
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "cpu_freq_actuator.hpp"

using namespace hyperion::cpu_loop;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "CHECK fallo %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

namespace {

void put(const std::string& path, const std::string& v) {
    std::ofstream(path, std::ios::trunc) << v << "\n";
}
std::string get(const std::string& path) {
    auto t = detail::read_text(path);
    return t ? *t : "<ausente>";
}
void mkdirs(const std::string& p) {
    std::string cmd = "mkdir -p '" + p + "'";
    CHECK(std::system(cmd.c_str()) == 0);
}

struct FakeSysfs {
    std::string root;
    FakeSysfs() {
        char tmpl[] = "/tmp/hyp_actuator_XXXXXX";
        CHECK(mkdtemp(tmpl) != nullptr);
        root = tmpl;
        mkdirs(root + "/intel_pstate");
        put(root + "/intel_pstate/no_turbo", "0");
        // cpu0 y cpu6 son hermanos SMT; cpu1 es independiente.
        for (int cpu : {0, 1, 6}) {
            std::string d = root + "/cpu" + std::to_string(cpu);
            mkdirs(d + "/cpufreq");
            mkdirs(d + "/topology");
            put(d + "/cpufreq/scaling_min_freq", "800000");
            put(d + "/cpufreq/scaling_max_freq", "3600000");
            put(d + "/cpufreq/scaling_governor", "powersave");
            put(d + "/cpufreq/cpuinfo_min_freq", "800000");
            put(d + "/cpufreq/cpuinfo_max_freq", "3600000");
        }
        put(root + "/cpu0/topology/thread_siblings_list", "0,6");
        put(root + "/cpu6/topology/thread_siblings_list", "0,6");
        put(root + "/cpu1/topology/thread_siblings_list", "1");
        // Wrapper falso: escribe su primer argumento en no_turbo, como el real.
        put(root + "/set_turbo_ok.sh", "echo \"$1\" > " + root + "/intel_pstate/no_turbo");
        put(root + "/set_turbo_fail.sh", "exit 1");
        put(root + "/set_turbo_noop.sh", "exit 0");
    }
    ~FakeSysfs() {
        std::string cmd = "rm -rf '" + root + "'";
        (void)std::system(cmd.c_str());
    }
    std::string f(int cpu, const char* a) const { return root + "/cpu" + std::to_string(cpu) + "/cpufreq/" + a; }
    CpuFreqActuatorConfig cfg(const char* script = "set_turbo_ok.sh") const {
        CpuFreqActuatorConfig c;
        c.sysfs_cpu_root = root;
        c.cpus = {0, 1};
        c.turbo_command = {"/bin/sh", root + "/" + script};
        c.write_verify_retry_delay_ms = 1;
        return c;
    }
};

void test_snapshot_expande_hermanos_smt() {
    FakeSysfs fs;
    auto c = fs.cfg();
    c.cpus = {0};
    CpuFreqActuator a(c);
    CHECK(a.snapshot());
    CHECK((a.cpus() == std::vector<int>{0, 6}));  // cpu6 entra por ser hermano de cpu0
}

void test_ciclo_completo_fija_y_restaura() {
    FakeSysfs fs;
    {
        CpuFreqActuator a(fs.cfg());
        CHECK(a.snapshot());
        CHECK(a.enter());
        CHECK(get(fs.root + "/intel_pstate/no_turbo") == "1");  // turbo apagado y verificado
        CHECK(a.set_khz(3200000));
        for (int cpu : {0, 1, 6}) {
            CHECK(get(fs.f(cpu, "scaling_min_freq")) == "3200000");
            CHECK(get(fs.f(cpu, "scaling_max_freq")) == "3200000");
        }
        // Bajar: el techo nuevo queda bajo el piso vigente (camino min-primero).
        CHECK(a.set_khz(2900000));
        CHECK(get(fs.f(0, "scaling_min_freq")) == "2900000");
        CHECK(get(fs.f(0, "scaling_max_freq")) == "2900000");
        // Subir de nuevo (camino max-primero).
        CHECK(a.set_khz(3200000));
        CHECK(get(fs.f(6, "scaling_max_freq")) == "3200000");
        CHECK(a.restore());
        CHECK(a.restore());  // idempotente
    }
    for (int cpu : {0, 1, 6}) {
        CHECK(get(fs.f(cpu, "scaling_min_freq")) == "800000");
        CHECK(get(fs.f(cpu, "scaling_max_freq")) == "3600000");
    }
    CHECK(get(fs.root + "/intel_pstate/no_turbo") == "0");  // turbo devuelto a su valor original
}

void test_destructor_restaura_sin_llamada_explicita() {
    FakeSysfs fs;
    {
        CpuFreqActuator a(fs.cfg());
        CHECK(a.snapshot() && a.enter() && a.set_khz(2900000));
    }  // sale de alcance sin restore(): la ruta de "el proceso terminó"
    CHECK(get(fs.f(0, "scaling_max_freq")) == "3600000");
    CHECK(get(fs.root + "/intel_pstate/no_turbo") == "0");
}

void test_clamp_a_limites_fisicos() {
    FakeSysfs fs;
    CpuFreqActuator a(fs.cfg());
    CHECK(a.snapshot() && a.enter());
    CHECK(a.set_khz(9000000));  // por encima de cpuinfo_max_freq
    CHECK(get(fs.f(0, "scaling_max_freq")) == "3600000");
    CHECK(a.set_khz(100));  // por debajo de cpuinfo_min_freq
    CHECK(get(fs.f(0, "scaling_min_freq")) == "800000");
}

void test_orden_de_escritura_del_rango() {
    CHECK(detail::min_write_first(3200000, 2900000));   // techo nuevo bajo el piso vigente
    CHECK(!detail::min_write_first(2900000, 3200000));  // subiendo: techo primero
    CHECK(!detail::min_write_first(3200000, 3200000));  // igual: no viola min<=max
    CHECK(!detail::min_write_first(std::nullopt, 2900000));
}

void test_turbo_wrapper_falla_no_toca_frecuencia() {
    FakeSysfs fs;
    CpuFreqActuator a(fs.cfg("set_turbo_fail.sh"));
    CHECK(a.snapshot());
    CHECK(!a.enter());
    CHECK(a.disabled());
    CHECK(!a.set_khz(3200000));
    CHECK(get(fs.f(0, "scaling_max_freq")) == "3600000");  // nunca se escribió
}

void test_turbo_no_verificado_por_relectura() {
    FakeSysfs fs;
    CpuFreqActuator a(fs.cfg("set_turbo_noop.sh"));  // sale con 0 pero no cambia no_turbo
    CHECK(a.snapshot());
    CHECK(!a.enter());  // el exit 0 del wrapper no basta: se relee no_turbo
    CHECK(a.disabled());
}

void test_falla_cerrado_ante_escritura_no_verificable() {
    FakeSysfs fs;
    CpuFreqActuator a(fs.cfg());
    CHECK(a.snapshot() && a.enter());
    CHECK(a.set_khz(3200000));
    if (geteuid() == 0) return;  // como root chmod no impide escribir; el resto no aplica
    // cpu6 pierde permiso de escritura en max: la relectura no coincide.
    CHECK(chmod(fs.f(6, "scaling_max_freq").c_str(), 0444) == 0);
    CHECK(!a.set_khz(2900000));
    CHECK(a.disabled());
    CHECK(!a.set_khz(3200000));  // deshabilitado: ya no toca sysfs
    // El intento de restauración dejó cpu0 y cpu1 en su estado original.
    CHECK(get(fs.f(0, "scaling_max_freq")) == "3600000");
    CHECK(get(fs.f(1, "scaling_max_freq")) == "3600000");
    CHECK(get(fs.root + "/intel_pstate/no_turbo") == "0");
    chmod(fs.f(6, "scaling_max_freq").c_str(), 0644);
}

void test_sin_cpufreq_falla_el_snapshot() {
    FakeSysfs fs;
    auto c = fs.cfg();
    c.cpus = {42};  // no existe
    CpuFreqActuator a(c);
    CHECK(!a.snapshot());
    CHECK(!a.enter());
}

void test_sin_administrar_turbo_no_lo_toca() {
    FakeSysfs fs;
    auto c = fs.cfg("set_turbo_fail.sh");  // si se invocara, fallaría
    c.manage_turbo = false;
    CpuFreqActuator a(c);
    CHECK(a.snapshot() && a.enter() && a.set_khz(3200000));
    CHECK(get(fs.root + "/intel_pstate/no_turbo") == "0");
}

void test_paralelo_da_el_mismo_estado_que_secuencial() {
    for (bool parallel : {false, true}) {
        FakeSysfs fs;
        auto c = fs.cfg();
        c.parallel = parallel;
        CpuFreqActuator a(c);
        CHECK(a.snapshot() && a.enter());
        CHECK(a.set_khz(3200000));
        CHECK(a.set_khz(2900000));
        for (int cpu : {0, 1, 6}) {
            CHECK(get(fs.f(cpu, "scaling_min_freq")) == "2900000");
            CHECK(get(fs.f(cpu, "scaling_max_freq")) == "2900000");
        }
        CHECK(a.restore());
        for (int cpu : {0, 1, 6}) CHECK(get(fs.f(cpu, "scaling_max_freq")) == "3600000");
    }
}

void test_paralelo_falla_cerrado_ante_un_cpu_no_escribible() {
    if (geteuid() == 0) return;
    FakeSysfs fs;
    auto c = fs.cfg();
    c.parallel = true;
    CpuFreqActuator a(c);
    CHECK(a.snapshot() && a.enter());
    CHECK(chmod(fs.f(6, "scaling_max_freq").c_str(), 0444) == 0);
    CHECK(!a.set_khz(2900000));
    CHECK(a.disabled());
    CHECK(get(fs.f(0, "scaling_max_freq")) == "3600000");  // los demas volvieron a su estado
    chmod(fs.f(6, "scaling_max_freq").c_str(), 0644);
}

void test_solo_techo_no_toca_el_piso() {
    FakeSysfs fs;
    auto c = fs.cfg();
    c.pin_min = false;
    CpuFreqActuator a(c);
    CHECK(a.snapshot() && a.enter());
    CHECK(a.set_khz(3200000));
    CHECK(get(fs.f(0, "scaling_max_freq")) == "3200000");
    CHECK(get(fs.f(0, "scaling_min_freq")) == "800000");  // piso original intacto
    CHECK(a.set_khz(2900000));
    CHECK(get(fs.f(6, "scaling_max_freq")) == "2900000");
    CHECK(get(fs.f(6, "scaling_min_freq")) == "800000");
    CHECK(a.restore());
    CHECK(get(fs.f(0, "scaling_max_freq")) == "3600000");
    CHECK(get(fs.f(0, "scaling_min_freq")) == "800000");
}

void test_solo_techo_baja_el_piso_si_hiciera_falta() {
    FakeSysfs fs;
    put(fs.f(0, "scaling_min_freq"), "3200000");  // piso vigente por encima del objetivo
    auto c = fs.cfg();
    c.pin_min = false;
    CpuFreqActuator a(c);
    CHECK(a.snapshot() && a.enter());
    CHECK(a.set_khz(2900000));
    CHECK(get(fs.f(0, "scaling_min_freq")) == "2900000");  // min<=max se respeta
    CHECK(get(fs.f(0, "scaling_max_freq")) == "2900000");
}

}  // namespace

int main() {
    test_snapshot_expande_hermanos_smt();
    test_ciclo_completo_fija_y_restaura();
    test_destructor_restaura_sin_llamada_explicita();
    test_clamp_a_limites_fisicos();
    test_orden_de_escritura_del_rango();
    test_turbo_wrapper_falla_no_toca_frecuencia();
    test_turbo_no_verificado_por_relectura();
    test_falla_cerrado_ante_escritura_no_verificable();
    test_sin_cpufreq_falla_el_snapshot();
    test_sin_administrar_turbo_no_lo_toca();
    test_paralelo_da_el_mismo_estado_que_secuencial();
    test_paralelo_falla_cerrado_ante_un_cpu_no_escribible();
    test_solo_techo_no_toca_el_piso();
    test_solo_techo_baja_el_piso_si_hiciera_falta();
    std::printf("cpu_freq_actuator_test: OK\n");
    return 0;
}
