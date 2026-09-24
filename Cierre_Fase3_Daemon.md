# Cierre de la Fase 3 (daemon de control DVFS)

Fecha de cierre: 2026-09-23. Documento de resumen. El detalle cronológico, los
números de job y los logs están en `Plan_Fase3_Daemon.md` (secciones indicadas entre
paréntesis). Donde este resumen y el plan discrepen, manda el plan.

## 1. Qué se construyó

| Pieza | Ubicación | Estado |
|---|---|---|
| Daemon de CPU (C++) | `fase3_daemon/cpu_loop/` | Cerrado. Tres brazos: base (daemon apagado, gobernador nativo con turbo), sombra (decide y no escribe), activo (escribe). |
| Daemon de GPU (C++) | `fase3_daemon/gpu_loop_cpp/` | Cerrado. Mismos tres brazos. NVML directo, modelo en ONNX, abstención. |
| Tabla de política | `fase3_daemon/policy/build_policy_table.py`, `policy_table.yaml` | Cerrado. Un único derivador, consumido por ambos lanzadores. |
| Lanzadores | `cpu_loop/launch_cpu_daemon.py`, `gpu_loop_cpp/launch_gpu_daemon.py` | Cerrado. Traducen la tabla a banderas de cada binario. |
| Aplicaciones compuestas | `fase3_daemon/composite_apps/` | CPU A y B, GPU A y GPU B (BabelStream memory + `rodinia_lavamd -boxes1d 100` compute, un kernel inédito por fase). |
| Pruebas | `fase3_daemon/tests/`, `cpu_loop/tests/`, `gpu_loop_cpp/tests/` | Python y C++ en verde. |

Se decidió mantener **dos binarios** (uno por dispositivo) y no fusionarlos: se evita
acoplar NVML y `perf` en un mismo proceso, se conserva el aislamiento de fallos (cada
uno restaura lo suyo) y no se rehace la verificación de caos y coexistencia. Fusionarlos
solo ahorraba la señal por archivo.

## 2. Qué cambió respecto al plan original, y por qué

### 2.1 Modelo de GPU: de regresión logística con reloj a random forest sin reloj

- **Antes:** candidato de regresión logística con la variable de reloj SM, umbral de
  abstención 0.70.
- **Ahora:** random forest (`gpu_random_forest_historical_20260923_sin_reloj`), exportado
  a ONNX, sin reloj ni potencia. Variables: `util_median`, `mem_util_median`,
  `mem_util_std`. Abstención con umbral 0.90 (si no llega, decide "revisar" y libera el
  reloj).
- **Por qué:** la auditoría (`fase2_clasificador/analysis/gpu_clock_feature_audit.py`)
  mostró que la regresión logística dependía del reloj: al cambiar solo el reloj de forma
  contrafactual, el 26.7% de las predicciones cambiaba de clase, y la exactitud por
  familia dejada fuera caía de 0.792 a 0.654. Es una fuga grave para un daemon, porque el
  propio daemon mueve el reloj y el modelo reaccionaría a su propia acción. El random
  forest no mostró esa dependencia.
- **Cómo se reentrenó:** solo se quitó esa variable sobre el mismo conjunto histórico
  (`docs/libro/datos/gpu_calidad_20260922/historical_gpu_relaxed020_20260922.csv`). No se
  remidió ninguna campaña ni se usó CUPTI. Se probó además un reentrenamiento por
  ventanas (`gpu_window_quality.py`) y no se adoptó.

### 2.2 Modelo de CPU: vector de 6 variables y auditoría contrafactual

- XGBoost en ONNX con 6 variables, incluida `freq_khz_observed`, y umbral selectivo
  (recalculo de 2026-09-21, umbral por exactitud por celda).
- Como el daemon también cambia la frecuencia de CPU, se hizo la misma auditoría
  contrafactual: pasar de F0 a F1 altera solo el 0.42% de las predicciones
  (`cpu_freq_feature_counterfactual.py`). No hay lazo de realimentación relevante.

### 2.3 Daemon de GPU: de Python a C++

- El primer daemon de GPU se hizo en Python sin consultarlo. Costaba 30.6% de un núcleo
  frente a 0.0-0.1% en C++, y perturbaba la medición de energía. Se rehízo en C++ como
  binario aparte, con el modelo en ONNX, y el de Python se **retiró** del repositorio
  (`run_daemon.py`, `gpu_loop/`, `decision_log.py` y sus pruebas).
- Regla adoptada: las decisiones de diseño (lenguaje, arquitectura, modelo, política) se
  consultan antes de decidir.

### 2.4 Política

- **GPU:** memory_bound baja a 1260 MHz (F1). compute_bound no actúa: libera el candado
  y deja el DVFS nativo del controlador (`nvidia-smi -rgc`, sin usar el reloj mínimo).
  El "nativo" de GPU es el DVFS por defecto del controlador.
- **CPU:** la medición de Fase 2 dice "no actuar" (ningún nivel mejora el EDP frente a
  REF de forma significativa). Se decidió, aun así, **experimentar**: F0 (3.2 GHz, turbo
  apagado) como base y F1 (2.9 GHz) en memory_bound con confianza mayor o igual a 0.85.
  Esto queda marcado en la tabla como `actuar_experimental` y conserva la acción
  medida (`measured_action`), para que nadie lo lea como política ganadora. La
  comparación es siempre contra el gobernador nativo (REF), no contra F0.
- **Variantes que se llevan a la Fase 4:** piso de CPU con la GPU activa (F0 frente a
  ninguno). La aplicación B de GPU ya se resolvió (ver §1 y §4.2).

### 2.5 Actuador de CPU y coordinación

- Actuador en C++ que reutiliza las reglas de `freqctl.py` (verificado por una prueba de
  paridad), con turbo por `sudo -n /usr/local/bin/set_turbo_state` (1 apaga el turbo),
  falla cerrado y restaura el estado inicial. Conmutación 2.85 ms.
- Histéresis de 50 ventanas consecutivas para cambiar de nivel.
- Coordinación CPU-GPU por archivo de señal escrito de forma atómica: con la GPU activa,
  la CPU no baja de 3.2 GHz. Subir al piso no espera a la histéresis.
- Errores corregidos en el camino: el consumidor ignoraba SIGTERM con cola acumulada; el
  objetivo `dd` corría en modo kernel y daba 96% de ciclos en cero (se usa
  `phase_target.c`); `set_clock(0)` fijaba el reloj mínimo (ahora libera al nativo); la
  señal `gpu_active` se duplicaba (ahora sube solo al inicio de la actividad).

## 3. Resultados verificados (todos en pacca, paccaA100)

- Candado de reloj de GPU funcional (H1 resuelto, jobs 7599 y 7600). Transición de reloj
  ~50 ms por comando, ~80 ms observada; permanencia mínima 3.7 s (10 veces la transición).
- Restauración con prueba de caos: CPU (job 7598), GPU (SIGTERM y SIGINT con el candado
  puesto), y ambos daemons a la vez (job 7632: código de salida 0, turbo y rango de CPU
  idénticos al inicial, reloj de GPU liberado).
- Piso de CPU respetado el 99.2% del tiempo con la GPU activa (job 7632).
- Sobrecarga: CPU p99 19.3 microsegundos por inferencia; GPU 0.1% de un núcleo.
- Ahorro posible al bajar la CPU durante cargas de GPU (jobs 7628 y 7633, mediana de 3):
  dgemm F1 EDP x0.983 (mejor x0.975 a 2.0 GHz); triad F1 EDP x1.046 (peor); F0 equivale a
  REF. El ahorro es pequeño y depende de cuánto de la aplicación sea CPU.

## 4. Limitaciones conocidas

1. **Clasificación de GPU en línea débil.** El modelo se entrenó con agregados por
   corrida (`mem_util_std` incluye arranque e inactividad), y es sensible a la cadencia de
   muestreo (a 5 ms todo sale compute; se mantiene 50 ms). La abstención elimina los
   errores blandos y con kernels reales dio 0 errores en las fases principales, pero en
   muestras pequeñas. No arregla la exactitud de fondo.
2. **Aplicación B de GPU:** repetir lanzamientos cortos no sirve (la actividad cae bajo el umbral entre procesos y el daemon no decide, job 7634). Se resolvió con un kernel por fase con argumentos escalados (jobs 7635-7637). En la primera corrida en sombra, `lavamd` 3 de 3 correctas y BabelStream 1 abstención y 2 ciclos sin decisión (hipótesis: fases pegadas sin bajar del umbral); se resuelve o documenta en la Fase 4.
3. **Ahorro de CPU posible pequeño** y nulo o negativo con partes de CPU en la aplicación.
4. **El libro** sigue describiendo el candidato anterior de GPU (regresión logística con
   reloj, umbral 0.70). Se actualiza cuando se retome el libro.
5. La campaña CUPTI está pausada (~188 de 570 corridas) y no forma parte de la Fase 3.

## 5. Estado del checklist de cierre

6 de 8 cumplidos (ver `Plan_Fase3_Daemon.md` §2). Los 2 restantes son corridas de la
Fase 4 y no bloquean el cierre del desarrollo: los tres brazos sobre las compuestas A y B
con registro por decisión, y la validación de generalización de la aplicación B.

## 6. Entrada a la Fase 4

- Tres brazos (base, sombra, activo) más activo-F0 sobre las compuestas, con al menos 3
  repeticiones.
- Variantes de piso de CPU (`--gpu-active-floor-khz` 3200000 frente a 0).
- Medir la sobrecarga sombra menos base con RAPL.
- Para lanzar: sbatch en `scripts/pacca/`, desde el clon `~/hyperion_c8` con
  `HYPERION_ROOT` (nunca `git pull` en `~/hyperion`), sin `--time`, prefijo `hyp_`.
