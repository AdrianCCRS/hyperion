# Cribado de Fase 1 hasta el informe de utilidad

El punto de entrada es `run_screening_to_report.sh`. Es distinto de
`run_all.sh`: no entrena, no construye una política y no lanza la rejilla fina.
Su única meta es producir evidencia suficiente para decidir, de forma
revisable, qué kernels deben entrar en la campaña definitiva.

Debe ejecutarse dentro de una asignación Slurm exclusiva de paccaA100. Los
binarios y datos de entrada permanecen en `~/hyperion-kernels`; el script no
los descarga ni los recompila silenciosamente.

```bash
cd /ruta/a/hyperion
bash run_screening_to_report.sh \
  --tag pacca_screen_20260904 \
  --kernel-root "$HOME/hyperion-kernels" \
  --results-root "$HOME/hyperion-results/screening"
```

El flujo completo es reanudable por etapas con `--stage`. Es recomendable
ejecutarlas por separado la primera vez para revisar cada gate antes de gastar
la siguiente porción de la reserva.

## Orden y significado de las etapas

### 1. `prepare`

Crea `<results-root>/<tag>/workflow.json`, una copia de trabajo de
`catalog.yaml` y manifiestos con rutas propias de la cuenta que ejecuta. El
catálogo versionado del repositorio no se modifica: las OI y warmups medidos
se aplican primero a la copia y luego se revisa el diff.

### 2. `validate`

Ejecuta el chequeo de permisos, compila el harness con `-DWITH_GPU=ON`, corre
CTest y genera diagnósticos CPU/GPU. Falla antes de medir si faltan Perf,
uncore IMC, RAPL, NVML, control de reloj, exclusividad o binarios.

### 3. `screen-cpu` (independiente)

Ejecuta los 33 candidatos CPU, tres frecuencias y tres repeticiones. La OI se
calcula durante cada corrida con FLOPs PMU y bytes reales de `uncore_imc`; no
depende de `ncu` ni de una OI guardada. Puede lanzarse en una reserva distinta
mientras avanza la caracterización GPU. En `all` se ejecuta antes de ocupar la
GPU y de forma serial para evitar interferencia dentro de un solo nodo.

### 4. `transition`

Ejecuta F1-GPU-002 antes del cribado GPU:

1. mide las cadencias solicitadas de 5, 10, 50 y 100 ms;
2. selecciona `q_produccion` según la retención de escalones observables de
   potencia, utilización GPU/memoria y relojes SM/gráfico; temperatura y
   energía también se informan, pero no gobiernan la decisión porque son una
   señal lenta y una magnitud acumulativa, respectivamente;
3. escribe ese intervalo en el manifiesto GPU de trabajo;
4. mide, con tres réplicas, `REF→F3`, `REF→F6`, `F3→F6` y `F6→F3`;
5. conserva la cota máxima como `T_transicion_gpu`.

Los niveles F3/F6 se resuelven contra los clocks detectados en el nodo. La
carga por defecto es `gpu_phasic` durante 20 s; puede sustituirse con
`--transition-workload '<comando CUDA sostenido>'`.

### 5. `ncu`

Este gate ocurre **antes** de cualquier cribado GPU. Para cada uno de los 23
candidatos del YAML:

- ejecuta el comando real declarado en el catálogo;
- usa `ncu --launch-count 5/20/50` como límites crecientes —no cambia el
  tamaño del kernel sustituyendo `{N}`—;
- parsea el CSV largo real (`ID`, `Metric Name`, `Metric Value`);
- cuenta launches distintos, FLOPs FP32/FP64 y bytes DRAM;
- exige convergencia de OI inferior al 1 % entre los dos puntos finales;
- acepta tanto límites alcanzados como una aplicación que terminó y saturó
  en el mismo número total de launches;
- bloquea precisión mixta, ausencia de FLOPs y actividad Tensor Core sin una
  regla explícita de conversión;
- verifica el checksum del binario antes de perfilar;
- actualiza la OI y precisión únicamente en la copia de trabajo del catálogo.

Finalmente genera `gpu_eligible.yaml`. Sólo los reportes con
`roofline_label_eligible=true` aparecen allí. Por tanto, el cribado GPU nunca
puede usar silenciosamente una OI heredada.

Si un kernel necesita más puntos:

```bash
bash run_screening_to_report.sh \
  --stage ncu --tag pacca_screen_20260904 \
  --ncu-kernel rodinia_lud --force-ncu \
  --ncu-launch-counts 5,20,50,100,150,200
```

### 6. `screen-gpu`

Ejecuta sólo los candidatos GPU que superaron el gate `ncu`, tres frecuencias
y tres repeticiones. El valor de OI usado para la etiqueta es el recién medido
en la copia de trabajo del catálogo. La etapa falla si no existe
`gpu_eligible.yaml`. Como atajo, `--stage screen` ejecuta `screen-cpu` y luego
`screen-gpu`, pero presupone que la etapa `ncu` ya terminó.

Las campañas CPU y GPU conservan el transitorio con
`warmup_seconds_override: 0.0`. Sus `windows.csv` iniciales son provisionales.

### 7. `warmup`

Analiza las mismas corridas, exige señal y tres repeticiones, aplica los
warmups medidos al catálogo de trabajo y re-postprocesa `samples.csv` sin
relanzar kernels. También recalcula `verdict.json`. Desde este punto los CSV
derivados reflejan el warmup real.

### 8. `report`

Produce diagnósticos de cobertura CPU y GPU. GPU se analiza desde
`training_gpu_phases.csv` —una fila agregada por corrida—, nunca desde cada
muestra NVML. Las corridas cuyo `verdict.json` no sea aceptado no cuentan.

El resultado principal está en `kernel_utility/`:

- `tentative_kernel_utility.csv`: una fila por candidato;
- `tentative_kernel_utility.json`: versión estructurada;
- `tentative_kernel_utility.md`: tabla para revisión humana.

Para cada kernel informa familia, categoría metodológica, estado `ncu`, OI
medida, cambio frente al catálogo histórico, warmup, repeticiones, niveles,
calidad, clases observadas, posición `log2(OI/I_ridge)` y acción recomendada.
También verifica el objetivo mínimo de cinco familias por clase y dispositivo.
Como diagnóstico de señal —no como evaluación ML— compara además las medianas
compute/memory de los cinco proxies NVML y normaliza su diferencia por el IQR
global.

`candidate_for_final_campaign` no congela automáticamente el catálogo. Los
controles sintéticos, anclas de infraestructura y `dual_*` en cuarentena se
reportan por separado y no cuentan para la cobertura mínima.

## Lo que el script no hace

- No incorpora los diez candidatos del plan que todavía carecen de binario,
  checksum y caracterización; los cinco Rodinia GPU siguen siendo prioritarios
  si el informe muestra falta de familias compute-bound.
- No inventa un fallback de warmup ni una OI para un kernel no convergente.
- No genera la lista congelada ni el YAML fino: esas son decisiones humanas
  posteriores al informe.
- No entrena ni balancea el dataset.

## Reanudación

Cada etapa conserva sus artefactos bajo el mismo `--tag`. Para continuar:

```bash
bash run_screening_to_report.sh --stage ncu    --tag pacca_screen_20260904
bash run_screening_to_report.sh --stage screen-gpu --tag pacca_screen_20260904
bash run_screening_to_report.sh --stage warmup --tag pacca_screen_20260904
bash run_screening_to_report.sh --stage report --tag pacca_screen_20260904
```

No se debe cambiar `--kernel-root`, `--results-root`, `--node-id` ni el
contenido de los binarios entre etapas. `workflow.json` y los checksums dejan
traza de esa identidad.
