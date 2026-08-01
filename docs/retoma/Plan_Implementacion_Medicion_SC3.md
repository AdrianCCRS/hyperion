# Plan de Implementación — Dejar todo listo para medir en SC3

**Documento ejecutable por un agente de IA (Sonnet 5).**
Subordinado a `docs/retoma/Guia_Maestra_Fase1_DVFS.md` (la "Guía Maestra"): si algo aquí la contradice, señalar la contradicción y detenerse. Los IDs de reglas (CPP-NN, RUN-NN, POST-NN, …) refieren a las tablas de la sección 12 de la Guía Maestra.

**Objetivo:** al terminar este plan, el proyecto puede lanzar en el nodo `felix` del clúster SC3 una campaña piloto REF (governor nativo) que produzca `samples.csv` → `windows.csv` con `phase_label_train` derivado por Roofline — el **ground truth** confiable para entrenar los modelos de la Fase 2 — con el subsistema C++ y el orquestador verificados extremo a extremo.

---

## Parte 0 — Hechos confirmados del clúster (no re-derivar; auditados 2026-07-30 por SSH)

El acceso es `ssh guane` (llaves ya configuradas, usuario `yacacerest01`). Estos hechos se recopilaron con `sinfo`, `scontrol` y un `srun` de solo lectura sobre `felix`:

### Slurm y particiones

| Campo | Valor |
|---|---|
| Slurm | 24.11.5, cgroup v2, proctrack/cgroup |
| Partición para felix | `gpu_titan` (felix + yaje; yaje está `down`) |
| felix | `State=IDLE`, sin límite de tiempo (`MaxTime=UNLIMITED`), `DefMemPerCPU=3750M` |
| Otras particiones | `legacy` (guane01-08, 24 CPUs), `fat` (thor, inval), `amd` (exadell/smexa — **descartados**, GPU AMD) |

### Nodo felix (`felix.sc3.uis.edu.co`)

| Campo | Valor confirmado |
|---|---|
| CPU | Intel Xeon X7560 @ 2.27 GHz (**Nehalem-EX**, 2010). 4 sockets × 8 cores × 2 SMT = 64 CPUs lógicas |
| NUMA | 4 nodos. node0=`0-7,32-39`, node1=`8-15,40-47`, node2=`16-23,48-55`, node3=`24-31,56-63`. Siblings SMT: par `(k, k+32)` |
| Caché | L3 24 MB por socket (4 instancias, 96 MB total), línea de caché **64 bytes** (`coherency_line_size`) |
| cpufreq | `acpi-cpufreq`, 10 niveles discretos: 2261, 2128, 1995, 1862, 1729, 1596, 1463, 1330, 1197, 1064 MHz. `scaling_setspeed` existe. **Governor actual: `performance`** (≈ f_max fija). **NOT-WRITABLE** para el usuario |
| RAPL | `/sys/class/powercap/` **vacío**. Nehalem-EX es **anterior a RAPL** (Sandy Bridge, 2011): la energía por RAPL es **físicamente imposible en felix, no es un problema de permisos**. Ninguna gestión administrativa lo habilitará |
| perf | `perf_event_paranoid=1` → **perf por PID funciona para el usuario sin privilegios** (verificado con `perf stat` real). Eventos hardware genéricos (instructions, cycles, cache-references, cache-misses, branches, ref-cycles, stalled-cycles-*) **más eventos de caché con calificador: `LLC-loads`, `LLC-load-misses`, `LLC-stores`, `LLC-store-misses`, `LLC-prefetches`, `LLC-prefetch-misses`, L1-dcache-\*** |
| Temperatura | `coretemp` disponible en hwmon2-5 → check E02 factible |
| GPU | `Gres=gpu:nvidia_geforce_gtx_titan_x:2` (Maxwell, CC 5.2). `nvidia-smi` **no ve dispositivos sin `--gres=gpu:N`** en el srun. Módulo `cuda/11.8` disponible |
| Toolchain (en el nodo) | gcc/gfortran **14.2.0** (módulo `gnu14`, cargado por defecto), make, cmake 3.29 (módulo), perf 5.14 |
| SO | Rocky/EL9, kernel 5.14.0-611 |
| cgroup del step | `cpuset.cpus.effective` de un srun de 4 CPUs **sin** `--hint=nomultithread` fue `0-1,32-33` (incluye siblings SMT). Con `--hint=nomultithread` se esperan solo cores físicos |
| Almacenamiento | `$HOME` NFS con **quota 20 GB** (usados 3.6 GB). `/tmp` del nodo: tmpfs 110 GB (volátil). Existe `/scratch` (verificar escritura/quota en Fase 4) |
| Repo en clúster | Ya existe `~/hyperion` en el home del clúster (verificar frescura antes de usar; ver F4.1) |

### Consecuencias directas (asumirlas, no re-discutirlas)

1. **Sin energía/EDP en felix, para siempre.** Las columnas `pkg_delta_uj`, `dram_delta_uj`, `power_w` van con `energy_valid=False`. No imputar. La pregunta abierta de energía se responde: RAPL nunca; alternativa sería PDU/IPMI (gestión humana, Parte H).
2. **Campañas solo REF por ahora** (sin escritura cpufreq). Nota favorable: el governor es `performance`, así que REF ≈ frecuencia máxima fija → la calibración Roofline a "frecuencia máxima" (CAL-01) **es válida en REF** tal cual.
3. **PID+inherit está confirmado viable** (`paranoid=1` + `perf stat` por PID funcionó). La estrategia de la sección 3 de la Guía Maestra no tiene bloqueo de permisos.
4. **`bytes_moved` puede medirse mejor que con `cache-misses` genérico:** felix expone `LLC-load-misses`/`LLC-store-misses`/`LLC-prefetch-misses`. Ver F3.4 (validación de bytes contra STREAM).
5. Los FLOPs vienen del stdout de las suites (POST-09/CAL-02/CAL-03) — NPB reporta `Mop/s total` y verificación interna `VERIFICATION SUCCESSFUL`; STREAM reporta bandwidth; ERT reporta GFLOPs. Nunca de PMU.

### Ampliación 2026-07-31 (srun de solo lectura adicional en felix)

| Campo | Valor confirmado |
|---|---|
| **Dominio de frecuencia** | `freqdomain_cpus` de cpu0 = `0-7,32-39` (todo el socket 0: 8 cores físicos + sus 8 hilos SMT). **El control de cpufreq es POR SOCKET, no por core.** |
| Governors disponibles | `conservative ondemand userspace powersave performance schedutil` — **`userspace` sí está disponible**, H1 es puramente un tema de permisos de escritura, no de módulos faltantes |
| `scaling_setspeed` | Existe como archivo ya con `governor=performance` activo (no solo bajo `userspace`) |
| `/scratch` | **NO escribible** (`root:root 0755`, `touch` → Permission denied). Corrige la nota anterior ("verificar escritura") |
| `~/hyperion` en clúster | El checkout real (con `.git`, último commit 2026-07-13) está anidado en `~/hyperion/hyperion/`, no en `~/hyperion/` — artefacto de un rsync sin barra final. **Desactualizado**: no incluye ningún commit de Fase 1/Fase 2 (hasta `4e84b8a`) |
| GPU | Confirmada con `--gres=gpu:1`: GTX TITAN X, driver 570.195.03, 12288 MiB (resuelve H5) |
| cache/cpuinfo | `coherency_line_size=64` en los 4 niveles, L3 compartida por todo el socket, 32 núcleos físicos × 2 SMT = 64 lógicos — validado contra hardware real, coincide con lo que `node_profile.py` ya calcula |
| Módulos Fase 3 | `cmake/3.29.3` existe como módulo pero el `cmake` activo por defecto es 3.26.5 (cargar módulo explícito si se necesita 3.29.3); `openmpi5/5.0.7`, `mpi/openmpi-gcc11/4.1.6` disponibles por si NPB-MZ/MPI los requiere |

**Consecuencia crítica nueva (6):** como el dominio de frecuencia es por socket, fijar la frecuencia de un core con `scaling_setspeed` cambia la de **todo el socket**, incluidos cores que pudieran pertenecer a la asignación Slurm de otro usuario si la reserva de la campaña no cubre el socket completo. Una asignación de 4 CPUs sin `--hint=nomultithread` (auditada 2026-07-30) dio un cpuset (`0-1,32-33`) que es subconjunto estricto del dominio del socket — ese escenario habría compartido el dominio de frecuencia con quien tuviera el resto del socket 0. **Toda campaña DVFS real debe reservar el socket completo** (`--cpus-per-task=8 --hint=nomultithread` dio exactamente `0-7,32-39` = el dominio entero). Esto debe ir explícito en la solicitud H1 (no solo "delegar escritura", también "confirmar que la asignación cubre el dominio de frecuencia completo") y **ya está implementado** como nuevo check de preflight (2026-07-31, ARC-30): `preflight.check_frequency_domain()` (factor_id E10, bloqueante) verifica que `delegated_cpus` cubre por completo el `freqdomain_cpus`/`related_cpus`/`affected_cpus` real leído de sysfs para cada core delegado; sin datos de dominio no bloquea.

---

## Parte A — Reglas de ejecución para el agente

1. **Un módulo por sesión** (sección 14 de la Guía Maestra). Cada fase de abajo lista sus tareas en orden; no adelantar tareas de otra fase salvo que se indique paralelismo.
2. Cada tarea termina con: (a) tests en verde, (b) reglas del checklist marcables línea a línea, (c) commit con mensaje `feat(scope): …` en la rama de trabajo.
3. **Todo acceso al clúster es de solo lectura** salvo: compilar/copiar archivos en `$HOME`/`/scratch` del usuario y lanzar jobs Slurm. **Nunca** escribir en sysfs, nunca `sudo`, nunca procesos fuera de asignaciones Slurm.
4. Comandos al clúster: `ssh guane '…'` (no interactivo) y `srun --partition=gpu_titan --nodelist=felix …` para todo lo que corre en el nodo. Plantilla estándar de asignación de medición:
   ```bash
   srun --partition=gpu_titan --nodelist=felix --nodes=1 --ntasks=1 \
        --cpus-per-task=8 --hint=nomultithread --mem=16G --time=HH:MM:SS bash -lc '…'
   ```
   y verificar siempre el cpuset efectivo real (`cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/cpuset.cpus.effective`) antes de medir — el manifest se resuelve contra ese cpuset, no contra números asumidos.
5. Sincronización local → clúster: `rsync -az --delete --exclude .git --exclude build --exclude __pycache__ ./ guane:hyperion/` desde la raíz del repo local (confirmar con el usuario antes del primer `--delete` si `~/hyperion` remoto tiene contenido no versionado).
6. Lo que requiere humano (Parte H) se reporta como bloqueado, no se simula.

---

## Fase 1 — Subsistema C++: migración a PID+inherit y `--exec` sin cgroup

**Meta:** el launcher mide por `perf_event_open(pid_hijo, inherit=1)` con secuencia stop→open→resume; `--cgroup-path` pasa a opcional. Reglas CPP-01…CPP-08. Especificación completa: secciones 3 y 4 de la Guía Maestra (leerlas antes de tocar código).

Estado actual del código: `--exec/--exec-args` ya existen en [telemetry_kernel_launcher.cpp](../../telemetry/experiments/telemetry_kernel_launcher.cpp), pero perf sigue exigiendo `--cgroup-path` (lanza `invalid_argument` si falta) y usa `PerfCgroupReader`. Eso es lo que se migra.

### F1.1 — `PerfReader` con PID externo (CPP-01)
- Modificar `telemetry/src/perf_reader.cpp` / header: constructor/`open()` que reciba `pid_t target_pid`; `pe.inherit=1`; un fd por evento (instructions, cycles, cache-references, cache-misses); **no** `PERF_FORMAT_GROUP` (incompatible con inherit); mantener `PERF_FORMAT_TOTAL_TIME_ENABLED|TOTAL_TIME_RUNNING` por fd para `running_ratio`.
- No romper el uso existente (los 9 tests CTest deben seguir en verde — CPP-04/CPP-06).
- Test unitario nuevo (CPP-08): fork de un hijo trivial (busy-loop corto, no `sleep`, para generar cuentas), abrir perf sobre su PID, muestrear ≥5 veces mientras vive y **afirmar que los deltas intermedios son > 0** (lecturas en vivo, no planas — es el error silencioso descrito en la sección 3.2 de la Guía Maestra).

### F1.2 — Launcher: secuencia stop→open→resume (CPP-02, CPP-03, CPP-05)
- Implementar en `telemetry_kernel_launcher.cpp` la secuencia exacta de la sección 4.2 de la Guía Maestra: fork → hijo `raise(SIGSTOP)` → padre `waitpid(WUNTRACED)` → padre abre PerfReader(pid hijo) + ioctl RESET/ENABLE → arranca collector/consumer → `kill(child, SIGCONT)` → hijo hace `execvp` (modo `--exec`) o setup interno (modo `--kernel`).
- `--cgroup-path` pasa a **opcional**: si se pasa, solo mueve el PID al cgroup como aislamiento adicional; nunca se usa para abrir perf. Eliminar el `throw` de "required when perf is enabled".
- Modo `--kernel` sin regresión (handshake ready/go se conserva); modo `--exec` sin handshake.
- `metadata.json` del launcher registra `perf_attach_mode: "pid_inherit"` y el PID medido.

### F1.3 — Collector y deprecación (CPP-06, CPP-07)
- `collector.cpp` acepta el PerfReader por PID sin cambios en la ruta caliente (`clock_gettime → read(fd) → try_push → flush_producer → clock_nanosleep`).
- Marcar `perf_cgroup_reader.*` como deprecated (comentario + no ser ruta del launcher); no eliminarlo.

### F1.4 — Verificación local (gate de la fase)
- `cmake --build build && ctest`: 9 tests previos + nuevos en verde.
- Smoke local con kernel sintético en ambos modos (`--kernel stream_triad` y `--exec /bin/stress-ng` o un busy-loop compilado ad hoc): `samples.csv` con deltas incrementales reales, no un salto final.
- Mini INT-T11 local: correr el launcher sobre un binario OpenMP multihilo y comparar `instructions` totales contra `perf stat -e instructions -- <mismo binario>`; diferencia < 5%.

**Criterio de salida Fase 1:** CPP-01…CPP-08 marcables; ctest verde; INT-T11 local < 5%.

---

## Fase 2 — Orquestador Python: módulos restantes

**Meta:** completar los módulos vacíos siguiendo el orden de construcción (sección 9 de la Guía Maestra) y las firmas de la sección 10. Los ya listos (config, manifest, environment, diagnostics, preflight, catalog, node_profile parcial) no se reescriben; se extienden solo donde una tarea lo pida. Tests con mocks de sysfs/subprocess — deben correr en cualquier Linux (los 69 actuales siguen en verde).

Cada módulo: implementar + tests + marcar checklist. Una sesión por módulo.

### F2.1 — `runner.py` modo sintético (RUN-01…RUN-08)
- Construye el comando **siempre** desde `KernelEntry` (RUN-01); `run_id` determinista `{campaign_id}__{kernel_ref}__{freq_level.id}__rep{n:02d}` (RUN-02); timeout `expected_runtime_seconds × ≥3` con kill efectivo y verificación de que el PID murió (RUN-03/04, verificar con psutil en el test); `success_check` (RUN-05); fusión de metadata launcher+orquestador (RUN-06, usar el merge de F2.8); `stdout.txt`/`stderr.txt` completos (RUN-07); si `frequency_write_capable=False` **no** invocar freqctl (RUN-08).
- Tests con un launcher falso (script que emite samples.csv/metadata.json de fixture) — sin hardware.

### F2.2 — `freqctl.py` (FRQ-01…FRQ-10)
- Consume `frequency_control_strategy` del `EnvironmentProfile` (enmiendas ARC-12/13/18 del `Registro_Cambios_Fuera_Plan_Original.md`): `discrete_bounds` | `bounded_range` | `unavailable`. En felix hoy: **`unavailable`** → la implementación debe ser un no-op verificable que registra `"unavailable"` en metadata (FRQ-06) y nunca toca sysfs.
- Implementar completo de todos modos (snapshot único, apply con relectura, restore idempotente verificado, handlers atexit/SIGINT/SIGTERM, solo `delegated_cpus`) — se probará con sysfs mockeado (tmpdir con árbol `cpufreq` falso).
- **La prueba de caos INT-T03 (bare-metal local con root) queda pendiente y es humana**; el módulo no se declara terminado sin ella (FRQ-08). Para felix REF-only no es bloqueante porque freqctl opera en `unavailable`.

### F2.3 — `calibration.py` + extensión de `node_profile.py` (CAL-01…CAL-11)
- `run_roofline_calibration()`: ejecuta (vía runner) los kernels `role=calibration`, parsea BW_pico del stdout de STREAM y P_pico del stdout de ERT (regex configurables en el catálogo), `I_ridge = P_pico / BW_pico`, check D03 (±40% de ficha técnica declarada en el manifest) en la misma función con excepción bloqueante (CAL-04), escribe `roofline_calibration.json` completo (CAL-05).
- `load_calibration()` rechaza si `plausibility_check_passed=False` (CAL-06/POST-15).
- `build_node_profile()`: solo lectura de `/proc/cpuinfo`, cachés (`coherency_line_size` — POST-10), NUMA; escribe `node_profile.json` (CAL-07/08). `node_id` estable (MET-03): usar un slug configurado en el manifest (`felix-sc3`), no el hostname de sesión.
- `build_calibration_references()`: ≥5 repeticiones del kernel de referencia → P95 de IPC/MPKI/MissRate + `cv_pct`, `calibration_references.json` (CAL-09/10/11).
- Tests: stdouts de STREAM/ERT/NPB como fixtures de texto; ningún test ejecuta binarios reales. Grep de `FP_ARITH` en el módulo debe dar vacío.

### F2.4 — `runner.py` extensión `--exec` 
- Pasar `--exec/--exec-args` desde `KernelEntry` (CAT-06: args vacíos → cadena vacía, no omitir); re-verificar C01/C02 antes de **cada** corrida (CAT-07).

### F2.5 — `postprocess.py` (POST-01…POST-16)
- `samples.csv` → `windows.csv` con **todas** las `REQUIRED_OUTPUT_COLUMNS` de la sección 10.4 de la Guía Maestra, incluidas las columnas de energía con `energy_valid=False` cuando no hay RAPL (nunca omitir columnas).
- Puntos críticos: primera muestra `first_sample_no_delta` (POST-01); wrap/deltas negativos (POST-02/05); `running_ratio` vs `running_ratio_min` → `pmu_degraded` (POST-03); tasas con `delta_t_ns` real (POST-04); warmup por tiempo de pared del catálogo, conservando las filas (POST-07); `bytes_moved_window = delta_LLC_misses × llc_line_size(node_profile)` y si es 0 → `operational_intensity=NaN` + `intensity_undefined` (POST-08/10); FLOPs por ventana prorrateados del total del stdout del binario proporcionalmente a `delta_instructions` (documentar el prorrateo en el docstring — es una aproximación declarada, no medición por PMU) (POST-09); `phase_label_train` **solo** por `operational_intensity` vs `i_ridge` (POST-11); features relativas siempre y sin recorte (POST-12/13/16); trazabilidad por fila (POST-14, MLT-01).
- Tests sobre `tests/orchestrator/fixtures/fake_samples.csv` ampliado con casos: wrap, running_ratio bajo, bytes 0, warmup.

### F2.6 — `validation.py` (VAL-01…VAL-08)
- Orden determinista I04 → C02/C03 → E06-E08 → resto (VAL-07); rechazos con `factor_id`, `accepted=False`, nunca borrar (VAL-06); D03 invalida campaña completa (VAL-05); rechazo por ventana no invalida corrida (VAL-08).

### F2.7 — `campaign.py` (CAM-01…CAM-07)
- Matriz kernels × niveles (en felix: solo REF) × repeticiones; `random.Random(seed).shuffle()` plano (CAM-01); par baseline+telemetry atómico (CAM-04); reanudación accepted→saltar / rechazada→reintentar (CAM-03); semilla y orden completo en metadata (CAM-02/MET-06); contabilidad hora-núcleo del piloto (CAM-05/OPS-01); timeouts por fase (CAM-06); cierre siempre restaura frecuencia y verifica (CAM-07 — en `unavailable` la verificación es que no había nada que restaurar).

### F2.8 — `metadata_schema.py`, `report.py`, `cli.py` (MET-01…MET-07)
- Merge con detección de colisiones, nunca `{**a, **b}` (MET-01); `governor_restored_verified` por lectura (MET-02); reporte con tabla por `factor_id` sumando 100% (MET-04) y advertencia CV (MET-05); trazabilidad completa (MET-07). `cli.py`: subcomandos `diagnose` (ya existe como módulo), `calibrate`, `run-campaign`, `postprocess`, `report`.

**Criterio de salida Fase 2:** suite completa de tests unitarios en verde en local (los 69 existentes + los nuevos); checklist de las secciones 12.5–12.11 marcable módulo a módulo.

---

## Fase 3 — Kernels reales: NPB, STREAM, ERT en felix + catálogo con ground truth

**Meta:** binarios compilados **en felix**, checksums en `kernels/catalog.yaml`, y las fuentes de verdad (FLOPs/BW por stdout) verificadas. Sin esto no hay etiquetado confiable.

### F3.1 — Compilación en felix (vía srun/sbatch, módulo `gnu14` ya cargado por defecto)
- **NPB 3.4.x OMP** (descargar de nas.nasa.gov en local y subir tarball por rsync; guardar URL y sha256 del tarball en el catálogo como procedencia): compilar `ep`, `mg`, `cg`, `is`, `ft`, `lu` — clases **S y W para smoke** y **A/B para dataset** (elegir clase final tras medir `expected_runtime_seconds` reales en F3.3; objetivo ≥ 60 s por corrida para ≥50 ventanas útiles a 1 ms tras warmup). `make CLASS=A ep` etc., `FC=gfortran CC=gcc`, flags por defecto del suite + `-fopenmp`. Sin `-march=native` prohibido: es aceptable (MLT-07, modelo por nodo).
- **STREAM** (mccalpin oficial, `stream.c`): compilar con `-O3 -fopenmp -DSTREAM_ARRAY_SIZE=<4× L3 total ≈ 200M elementos>` — el array debe superar con holgura los 96 MB de L3 agregada del nodo (usar los 24 MB del socket si la campaña es un solo socket; documentar la elección en el catálogo).
- **ERT** (Empirical Roofline Toolkit) — requiere config; si su despliegue en felix resulta pesado, alternativa aceptada por CAL-03: microbenchmark de FLOPs pico **de suite reconocida** que reporte GFLOPs por stdout. No escribir uno propio sin registrar la decisión en el `Registro_Cambios_Fuera_Plan_Original.md` como enmienda ARC nueva.
- Todo se compila y queda en `~/hyperion-kernels/bin/` en el clúster; los `sha256sum` de cada binario se copian al catálogo local y se hace rsync del catálogo actualizado (CAT-02, C02, ARC-23).

### F3.2 — Catálogo (`kernels/catalog.yaml`)
- Entradas `role=dataset` para npb_ep (hint compute_bound), npb_mg, npb_cg, npb_is (hint memory_bound), npb_ft, npb_lu (hint intermedio) con `size_variant`, `expected_runtime_seconds` (medido, no estimado — F3.3), `warmup_seconds`, `success_check: stdout_regex "VERIFICATION SUCCESSFUL"`, y el campo de extracción de FLOPs: regex sobre el stdout de NPB (`Mop/s total` × tiempo, o el campo `Mops` total que NPB imprime).
- Entradas `role=calibration`: stream (`reports_bandwidth_stdout: true`, regex sobre la línea `Triad:`), ert (`reports_flops_stdout: true`).
- CAT-01…CAT-08 aplican; los checksums son los de F3.1.

### F3.3 — Medición de tiempos reales
- Una pasada por kernel/clase en felix (srun 8 cores nomultithread) para fijar `expected_runtime_seconds` reales → timeouts RUN-03 y proyección de presupuesto OPS-01/I09.

### F3.4 — **Validación del ground truth de bytes (crítica, nueva)**
El eslabón más débil del etiquetado es `bytes_moved_window` (LLC misses × 64). Verificarlo empíricamente **antes** de confiar en `phase_label_train`:
- Correr STREAM bajo el launcher (modo `--exec`) en felix. STREAM mueve una cantidad de bytes conocida analíticamente (3 arrays × tamaño × iteraciones por kernel triad/copy/scale/add).
- Comparar bytes teóricos vs `Σ delta_cache_misses × 64` medidos (y también vs `LLC-load-misses + LLC-store-misses + LLC-prefetch-misses` si se decide ampliar el set de eventos — felix los expone).
- **Criterio:** acuerdo dentro de ±30%. Si el evento genérico `cache-misses` subestima gravemente (prefetchers), registrar enmienda ARC proponiendo el set LLC-* explícito y actualizar harness/postprocess en consecuencia. Documentar el resultado en `docs/retoma/` — este número acota la incertidumbre de `operational_intensity` y por tanto del ground truth.

**Estado (2026-07-31):** F3.1/F3.2/F3.3/F3.4 completados en felix — ver ARC-31 (decisión ERT), ARC-32 (compilación, selección de clase NPB, checksums y regex reales) y ARC-33 (validación de bytes) en `Registro_Cambios_Fuera_Plan_Original.md`. Los 8 binarios (6 NPB clase B + STREAM + ert_probe) están compilados en `~/hyperion-kernels/bin` en felix, con checksums reales en `orchestrator/schemas/kernels/catalog.yaml`. El harness C++ también quedó compilado en felix (`telemetry/build-felix`, cmake+gnu14, los 10 CTest pasan en el nodo real — adelanta parte de F4.1). F3.4 corrió STREAM bajo el launcher real: `Σ delta_cache_misses × 64` da -33.8% vs el byte count teórico de STREAM, apenas fuera del ±30% objetivo. Investigado y no atribuible a un defecto de medición (`perf stat` nativo y el desglose LLC-load/store-misses coinciden con el harness dentro de 3.5%); la explicación más plausible es el prefetcher de hardware de felix (Nehalem-EX, streaming agresivo) ocultando parte del tráfico real de DRAM al contador genérico de LLC-miss. Se documenta como sesgo sistemático conocido, no bloqueante — ver informe completo en `docs/retoma/Informe_Piloto_F3_2026-07-31.md`.

**Criterio de salida Fase 3:** catálogo completo con checksums reales de felix (☑); tiempos medidos (☑); informe de validación de bytes con desviación cuantificada (☑, desviación de -33.8% documentada y explicada, ligeramente fuera de ±30% — ver informe).

---

## Fase 4 — Despliegue y verificación extremo a extremo en felix

### F4.1 — Despliegue
- Auditar `~/hyperion` remoto (¿qué versión hay?); rsync del repo actualizado; crear entorno Conda con `environment-hpc.yml` (o venv con python3 del módulo `conda/machine_learning` si el create falla; registrar cuál).
- Compilar el harness C++ en felix (cmake + gnu14) **dentro de un srun**, y correr los CTest ahí: los tests unitarios del harness deben pasar en el nodo real.
- Decidir `output_dir`: verificar `/scratch` (¿escribible?, ¿quota?); si no, `$HOME` con proyección I09 contra los ~16 GB libres de la quota. Estimar: 720 corridas × ~60 s × 1 kHz × ~200 B/muestra ≈ 9 GB → **preferir /scratch**, con copia final de `windows.csv` + metadata a `$HOME`.

### F4.2 — Diagnóstico y preflight reales
- `python -m orchestrator.diagnostics --manifest orchestrator/schemas/campaign_sc3_audit.yaml --use-allowed-cpus` dentro del srun estándar → conservar `startup_diagnostic.json` nuevo (el perfil pudo cambiar desde julio; kernel del nodo fue actualizado en mayo).
- Escribir el manifest real `campaign_felix_ref.yaml`: `environment_tier: hpc_sc3`, niveles = solo `REF`, kernels del catálogo F3.2, `cores` resueltos contra el cpuset efectivo del job (p. ej. con 8 físicos `0-7`: `delegated_cpus: 0-5`, `collector_cpu: 6`, `consumer_cpu: 7` — un solo NUMA, E04), política SMT explícita (`nomultithread`, E05), `rapl.enabled: false`, semilla fija.
- Correr el preflight completo: todo verde o advertencias justificadas y documentadas.

**Estado (2026-08-01):** primer `run_campaign_preflight()` real corrido contra felix (`delegated_cpus=0-5`, entorno Conda `hyperion-hpc` ya existente en el clúster). Encontró y corrigió dos bloqueadores reales (ver ARC-36): `_parse_cpu_list()` no entendía el formato real de `freqdomain_cpus` (afectaba E10, un check de seguridad), y `catalog.yaml` le faltaba `estimated_memory_bytes` en las entradas de calibración (bloqueaba C05 siempre). Con ambos corregidos, el preflight queda en verde salvo **D05** (capacidad de PMC), pendiente porque `pmc_count` nunca se implementó en `environment.py` (gap ya conocido, ARC-20). Aún no se escribió el manifest real `campaign_felix_ref.yaml` (el catálogo de kernels va a cambiar — DGEMM y posibles ajustes — antes de fijarlo).

### F4.3 — Verificación de medición en el nodo (INT-T11 real)
- Launcher sin cgroup sobre `npb_ep.S` OpenMP vs `perf stat` externo del mismo binario: conteos < 5% de diferencia. **Este es el gate que certifica que la librería C++ mide bien en felix.**
- Inspección anti-error-silencioso: los deltas de `samples.csv` crecen durante toda la corrida (no plano + salto final).

### F4.4 — Campaña piloto mínima (INT-T01/T02/T06/T07/T09/T10 adaptados a REF)
- 2 kernels (npb_ep + npb_mg, clase pequeña) × REF × 3 repeticiones, vía `campaign.py` completo: calibración → matriz → validación → postproceso → reporte.
- Gates: ≥90% corridas aceptadas; ≥50 ventanas `ok` por repetición; overhead baseline-vs-telemetry estable (CV < 10%); `node_profile.json`, `roofline_calibration.json`, `calibration_references.json` completos; `windows.csv` con features relativas numéricas.

### F4.5 — **Gate de ground truth (INT-T08)** — el criterio que pidió el usuario
- En el piloto: `phase_label_train` de npb_ep mayormente `compute_bound` y de npb_mg mayormente `memory_bound` (umbral sugerido: ≥80% de las ventanas `ok` de cada uno en la clase esperada).
- Si no separa: **no entrenar nada**; investigar en orden: (1) I_ridge implausible (revisar D03 y stdout de STREAM/ERT), (2) bytes_moved sesgado (volver a F3.4), (3) prorrateo de FLOPs. Documentar el hallazgo como enmienda ARC.
- Con el gate en verde, lanzar la campaña piloto extendida (6 kernels × REF × 10 reps) y congelar ese dataset como el primer artefacto candidato para Fase 2.

**Criterio de salida Fase 4:** campaña piloto REF completa en felix, reproducible desde su metadata, con INT-T08 en verde. **Esto es "listo para medir".**

---

## Parte H — Acciones humanas (no delegables; el agente las reporta, no las ejecuta)

| # | Acción | Desbloquea | Estado |
|---|---|---|---|
| H1 | Solicitar a administración SC3 la delegación de escritura cpufreq (`scaling_setspeed`/governor `userspace`, ya confirmado disponible) sobre cores delegados en felix. **Incluir explícitamente**: confirmar que la asignación Slurm de la campaña siempre cubrirá el dominio de frecuencia completo (`freqdomain_cpus` = todo el socket, 8 cores físicos + 8 SMT), para que el control de frecuencia nunca se filtre a cores de otro usuario en el mismo socket | Niveles F0–F4 → dataset DVFS real, sin riesgo de interferencia entre usuarios | Pendiente — el diagnóstico F4.2 aporta la evidencia técnica para la solicitud; la ampliación 2026-07-31 (dominio por socket) ya es evidencia adicional lista para incluir |
| H2 | Preguntar a SC3 por medición de energía externa (PDU/IPMI del rack) — RAPL es imposible en felix por hardware | Features de energía/EDP | Pendiente |
| H3 | Prueba de caos de freqctl (INT-T03) en PC local bare-metal con root | Declarar freqctl terminado; requisito previo a usar F0–F4 cuando H1 se conceda | Pendiente (irrelevante mientras solo haya REF) |
| H4 | Decisiones del director: alcance multinodo (A/B/C), nomenclatura F0/REF (ARC-15), clase NPB definitiva | Alcance formal; matriz definitiva | Pendiente — llevar el reporte del piloto F4.4 como insumo |
| H5 | Confirmar GPU: `srun --gres=gpu:1` + `nvidia-smi` (las Titan X existen pero no se ven sin `--gres`) | Ruta GPU (G01–G03) | Rápida; opcional para esta fase |

---

## Orden de ejecución y paralelismo

```
F1 (C++ PID+inherit)  ──┐
                        ├──► F3 (kernels en felix; F3.4 necesita F1)
F2.1–F2.2 (runner, freqctl) ─┤
F2.3–F2.8 (resto orquestador)┘
                              ──► F4.1–F4.3 (despliegue + INT-T11)
                                    ──► F4.4–F4.5 (piloto + gate ground truth)
H1, H2, H4, H5 pueden iniciarse por el usuario desde ya, en paralelo con todo.
```

F1 y F2.1 pueden hacerse en sesiones intercaladas; F3.1–F3.3 no dependen de F2 y pueden adelantarse en cuanto F1 esté (para F3.4). La ruta crítica es **F1 → F2.5 (postprocess) → F4**.

## Definición global de "hecho"

Todo lo siguiente verdadero:
1. `ctest` (harness) y `pytest tests/orchestrator` en verde, local y en felix.
2. Checklist secciones 12.1–12.13 de la Guía Maestra marcado para los módulos construidos.
3. `windows.csv` de la campaña piloto en felix con INT-T08 en verde y validación de bytes (F3.4) documentada con desviación cuantificada.
4. Reporte de campaña con tabla de `factor_id` y trazabilidad completa por fila.
5. `Registro_Cambios_Fuera_Plan_Original.md` actualizado con toda decisión tomada fuera del plan (incluida la resolución de ERT y el resultado de F3.4).
