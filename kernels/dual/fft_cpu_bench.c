/*
 * fft_cpu_bench.c -- FFT compleja 2D (N x N, doble precision) en CPU via FFTW,
 * contraparte de kernels/dual/fft_gpu_dispatch.cu.
 *
 * Segunda operacion del selector CPU/GPU, junto a GEMM. Se eligio FFT a
 * proposito porque su perfil es el OPUESTO al de GEMM: intensidad operacional
 * baja (O(log N) operaciones por elemento frente a O(N) de GEMM), asi que la
 * frontera CPU/GPU cae en otro sitio. Con una sola operacion el modelo solo
 * podria aprender "tamaño -> device"; con dos de perfil opuesto tiene que
 * aprender tambien "que operacion es".
 *
 * La planificacion FFTW queda FUERA de la ventana medida: es costo de setup
 * que el runtime del selector pagaria una sola vez, no por despacho. Se usa
 * FFTW_ESTIMATE y no FFTW_MEASURE para que el plan sea determinista y no
 * dependa de un autotuning que varia entre corridas (lo que meteria varianza
 * en el eje de frecuencia, que es justo lo que se quiere medir limpio).
 *
 * Formato de salida identico al resto del catalogo (catalog.yaml).
 */
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* Mismo PRNG determinista que el resto de benches del repo. */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static double next_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) / (double)(1ULL << 53);
}

int main(int argc, char **argv) {
    long n = 1024;
    int iterations = 10;
    int verify_samples = 64;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            n = strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = (int)strtol(argv[++i], NULL, 10);
        }
    }
    if (n <= 0 || iterations <= 0) {
        fprintf(stderr, "uso: %s [--size N] [--iterations M]\n", argv[0]);
        return 2;
    }

#ifdef _OPENMP
    if (fftw_init_threads()) {
        int nthreads = 1;
        const char *env = getenv("OMP_NUM_THREADS");
        if (env) nthreads = atoi(env);
        if (nthreads < 1) nthreads = 1;
        fftw_plan_with_nthreads(nthreads);
    }
#endif

    const size_t elems = (size_t)n * (size_t)n;
    fftw_complex *data = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * elems);
    fftw_complex *original = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * elems);
    if (!data || !original) {
        fprintf(stderr, "fallo de asignacion para N=%ld\n", n);
        return 2;
    }

    for (size_t i = 0; i < elems; ++i) {
        double re = next_uniform() * 2.0 - 1.0;
        double im = next_uniform() * 2.0 - 1.0;
        data[i] = re + im * I;
        original[i] = data[i];
    }

    /* Planificacion fuera de la ventana medida: es setup, no despacho. */
    fftw_plan forward = fftw_plan_dft_2d((int)n, (int)n, data, data,
                                         FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan backward = fftw_plan_dft_2d((int)n, (int)n, data, data,
                                          FFTW_BACKWARD, FFTW_ESTIMATE);
    if (!forward || !backward) {
        fprintf(stderr, "fallo al planificar FFTW para N=%ld\n", n);
        return 2;
    }

    /* Warmup fuera de ventana, para igualar el trato que recibe el bench de GPU
     * (que descarta su primera llamada por autotuning/carga de kernels). */
    fftw_execute(forward);
    fftw_execute(backward);
    for (size_t i = 0; i < elems; ++i) data[i] = original[i];

    double t0 = now_seconds();
    for (int rep = 0; rep < iterations; ++rep) {
        /* Cada iteracion parte de datos frescos, por dos razones. La primera
         * es de correctitud: el plan es in-place, asi que encadenar directas
         * sobre el mismo buffer calcularia FFT(FFT(...)) y los valores
         * desbordarian. La segunda es de simetria con el lado GPU, que en cada
         * despacho vuelve a subir h_original -- si aqui no se pagara el costo
         * de traer los datos, la comparacion CPU/GPU estaria sesgada a favor
         * de la CPU justo en el termino que decide la frontera. */
        memcpy(data, original, sizeof(fftw_complex) * elems);
        fftw_execute(forward);
    }
    double t1 = now_seconds();
    double seconds = t1 - t0;

    /* Verificacion fuera de la ventana: una inversa sobre el resultado de la
     * ultima directa debe devolver el input original salvo el factor N^2 que
     * FFTW no normaliza. */
    fftw_execute(backward);
    const double norm = (double)elems;
    /* Error ABSOLUTO, no relativo: los datos son uniformes en [-1,1], asi que
     * la magnitud tipica es O(1) y un error absoluto es directamente
     * interpretable. El error relativo era una trampa -- al dividir por un
     * |valor| que puede caer cerca de cero, un error de redondeo normal se
     * inflaba por encima de la tolerancia.
     *
     * El chequeo isfinite es obligatorio y NO redundante: si un valor sale
     * inf, la resta da NaN, y toda comparacion con NaN es falsa, incluida
     * `rel_error > max_error`. Sin este chequeo max_error se quedaba en 0 y el
     * bench reportaba SUCCESSFUL justo en los casos mas rotos. */
    double max_abs_error = 0.0;
    for (int s = 0; s < verify_samples; ++s) {
        size_t idx = (size_t)(next_uniform() * (double)elems);
        if (idx >= elems) idx = elems - 1;
        double got_re = creal(data[idx]) / norm;
        double got_im = cimag(data[idx]) / norm;
        if (!isfinite(got_re) || !isfinite(got_im)) {
            max_abs_error = INFINITY;
            break;
        }
        double abs_error = fabs(got_re - creal(original[idx]))
                         + fabs(got_im - cimag(original[idx]));
        if (abs_error > max_abs_error) max_abs_error = abs_error;
    }
    /* FFT de M puntos acumula error ~ eps*log2(M); con M=16.7M (N=4096) eso es
     * ~5e-15. La tolerancia de 1e-9 deja seis ordenes de margen. */
    const int ok = max_abs_error < 1e-9;

    /* Estimacion estandar para FFT compleja: 5*M*log2(M) con M = N*N. */
    double m = (double)elems;
    double total_flops = (double)iterations * 5.0 * m * log2(m);
    double mops_total = total_flops / 1e6 / seconds;

    printf("\n FFT 2D complex benchmark (CPU / FFTW)\n\n");
    printf(" Grid size (N)         =                %8ld\n", n);
    printf(" Iterations            =                %8d\n", iterations);
    printf("\n");
    printf(" Time in seconds =    %12.6f\n", seconds);
    printf(" Mop/s total     =    %12.2f\n", mops_total);
    printf(" Verification    =               %s\n", ok ? "SUCCESSFUL" : "FAILED");

    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(data);
    fftw_free(original);
    return ok ? 0 : 1;
}
