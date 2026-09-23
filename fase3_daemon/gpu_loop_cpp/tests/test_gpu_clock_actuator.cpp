// Pruebas del actuador de reloj de GPU con un `nvidia-smi` FALSO (un script que registra sus argumentos):
// no toca la GPU ni requiere permisos.
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <string>
#include <vector>

#include "check.hpp"
#include "gpu_clock_actuator.hpp"

using namespace hyperion::gpu_loop;

static std::string slurp(const std::string& p) {
    std::ifstream in(p);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

struct Fake {
    std::string dir, log, script_ok, script_fail;
    Fake() {
        char tmpl[] = "/tmp/hyp_gpu_actuator_XXXXXX";
        CHECK(mkdtemp(tmpl) != nullptr);
        dir = tmpl; log = dir + "/calls.log"; script_ok = dir + "/smi_ok.sh"; script_fail = dir + "/smi_fail.sh";
        std::ofstream(script_ok) << "echo \"$@\" >> " << log << "\n";
        std::ofstream(script_fail) << "echo \"$@\" >> " << log << "\nexit 1\n";
    }
    ~Fake() { std::string c = "rm -rf " + dir; (void)std::system(c.c_str()); }
    GpuClockActuatorConfig cfg(const std::string& script, double settle = 0.0) const {
        GpuClockActuatorConfig c;
        c.nvidia_smi = {"/bin/sh", script};
        c.settle_s = settle;
        return c;
    }
};

static void test_fija_el_reloj_con_lgc_y_restaura_con_rgc() {
    Fake f;
    {
        GpuClockActuator a(f.cfg(f.script_ok));
        CHECK(a.set_clock_mhz(1260));
        CHECK(a.dirty());
        CHECK(slurp(f.log) == "-i 0 -lgc 1260,1260\n");
        CHECK(a.restore());
        CHECK(!a.dirty());
        CHECK(a.restore());  // idempotente: sin candado no vuelve a llamar
    }
    CHECK(slurp(f.log) == "-i 0 -lgc 1260,1260\n-i 0 -rgc\n");
}

static void test_cero_significa_liberar_el_candado_no_fijar_el_minimo() {
    Fake f;
    GpuClockActuator a(f.cfg(f.script_ok));
    CHECK(a.set_clock_mhz(1260));
    CHECK(a.set_clock_mhz(0));           // fase compute_bound: no_actuar
    CHECK(!a.dirty());
    CHECK(slurp(f.log) == "-i 0 -lgc 1260,1260\n-i 0 -rgc\n");  // nunca -lgc con un minimo
}

static void test_destructor_libera_el_candado() {
    Fake f;
    { GpuClockActuator a(f.cfg(f.script_ok)); CHECK(a.set_clock_mhz(810)); }
    CHECK(slurp(f.log) == "-i 0 -lgc 810,810\n-i 0 -rgc\n");
}

static void test_falla_del_comando_devuelve_false() {
    Fake f;
    GpuClockActuator a(f.cfg(f.script_fail));
    CHECK(!a.set_clock_mhz(1260));
    CHECK(!a.dirty());  // -lgc fallo: nunca se dio por puesto
}

static void test_arc112_observado_sobre_el_techo_con_carga_falla_y_restaura() {
    Fake f;
    auto c = f.cfg(f.script_ok);
    c.query_sm_clock_mhz = [] { return std::optional<int>(1410); };
    c.query_util_pct = [] { return std::optional<int>(100); };
    GpuClockActuator a(c);
    CHECK(!a.set_clock_mhz(1260));       // 1410 > 1260 con la GPU cargada: el candado no se aplico
    CHECK(!a.dirty());                   // se restauro antes de devolver
    CHECK(slurp(f.log) == "-i 0 -lgc 1260,1260\n-i 0 -rgc\n");
}

static void test_arc112_con_la_gpu_ociosa_el_exceso_es_inconcluyente() {
    Fake f;
    auto c = f.cfg(f.script_ok);
    c.query_sm_clock_mhz = [] { return std::optional<int>(765); };  // reposo, por encima de un techo de 210
    c.query_util_pct = [] { return std::optional<int>(0); };
    GpuClockActuator a(c);
    CHECK(a.set_clock_mhz(210));
    CHECK(a.dirty());
    CHECK(a.last_observed_mhz().value() == 765);
}

static void test_observado_igual_al_pedido_es_exito() {
    Fake f;
    auto c = f.cfg(f.script_ok);
    c.query_sm_clock_mhz = [] { return std::optional<int>(1260); };
    c.query_util_pct = [] { return std::optional<int>(100); };
    GpuClockActuator a(c);
    CHECK(a.set_clock_mhz(1260));
}

static void test_indice_de_gpu_se_pasa_a_nvidia_smi() {
    Fake f;
    auto c = f.cfg(f.script_ok);
    c.gpu_index = "GPU-abc";
    GpuClockActuator a(c);
    CHECK(a.set_clock_mhz(510));
    CHECK(slurp(f.log) == "-i GPU-abc -lgc 510,510\n");
}

int main() {
    test_fija_el_reloj_con_lgc_y_restaura_con_rgc();
    test_cero_significa_liberar_el_candado_no_fijar_el_minimo();
    test_destructor_libera_el_candado();
    test_falla_del_comando_devuelve_false();
    test_arc112_observado_sobre_el_techo_con_carga_falla_y_restaura();
    test_arc112_con_la_gpu_ociosa_el_exceso_es_inconcluyente();
    test_observado_igual_al_pedido_es_exito();
    test_indice_de_gpu_se_pasa_a_nvidia_smi();
    std::printf("gpu_clock_actuator_test: OK\n");
}
