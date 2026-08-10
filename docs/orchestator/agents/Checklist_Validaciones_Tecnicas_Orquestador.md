Checklist de validaciones técnicas del orquestador de campañas — Fase 1 DVFS 

# **Checklist de Validaciones Técnicas del Orquestador** 

_Agente DVFS — Fase 1: Recolección de Telemetría Una regla por fila · marca ☑ cuando el módulo la satisface · agrupa por tipo de módulo_ 

**Convenciones de color:** cada franja de cabecera identifica el módulo responsable de las reglas debajo de ella. El ID de cada regla sigue el esquema PREFIJO-NN, donde el prefijo es el módulo y NN es el número de regla dentro de ese módulo. Los factor_id (E0x, I0x, C0x, D0x, M0x, G0x) corresponden a los códigos de verificación del Plan de Implementación, sección 5 (factores de confusión) y Guía Técnica, secciones 4 y 11. 

|**1. Manife**<br>**ID**|**st (manifest.py) — validación de la entrada declarativa de campaña**<br>**Regla de validación / invariante técnica que debe cumplir el módulo**|**✓**|
|---|---|---|
|**MAN-01**|Rechazar el manifest si environment_tier: hpc_sc3 y no existe cgroup_path<br>(obligatorio en ese tier).|☐|
|**MAN-02**|Rechazar si repetitions_per_combination < 3 (sin sentido estadístico).|☐|
|**MAN-03**|Calcular y loguear el tamaño total de la matriz (kernels × freq × repeticiones,<br>doble por baseline) antes de continuar.|☐|
|**MAN-04**|Rechazar si output_dir ya existe y overwrite: false (factor I07 — debe fallar,<br>nunca silenciar).|☐|
|**MAN-05**|Rechazar si seed está ausente o no es entero — no generar semilla aleatoria<br>en tiempo de ejecución (rompe reproducibilidad).|☐|
|**MAN-06**|Rechazar si cores.delegated_cpus, cores.collector_cpu y<br>cores.consumer_cpu se solapan entre sí.|☐|
|**MAN-07**|La sección calibration debe tener<br>1 kernel con<br>≥<br>reports_bandwidth_stdout: true y<br>1 con reports_fops_stdout: true; si<br>≥<br>falta alguno, no se puede calcular I_ridge.|☐|
|**MAN-08**|Ningún kernel_ref declarado en calibration puede aparecer en la sección<br>kernels (dataset), ni viceversa.|☐|
|**MAN-09**|Todo kernel_ref referenciado en calibration y en kernels debe existir en<br>catalog_path; validar antes de tocar el nodo.|☐|
|**MAN-10**|Validar que frequency_levels contiene exactamente un nivel con mode:<br>native_governor (REF) y los demás con mode: fixed y fraction en [0.0, 1.0].|☐|
|**MAN-11**|Validar que running_ratio_min está en (0.0, 1.0] y que interval_ns > 0.|☐|
|**2. Catálo**|**go de kernels (catalog.py) — integridad de los binarios externos**||
|**ID**|**Regla de validación / invariante técnica que debe cumplir el módulo**|**✓**|
|**CAT-01**|C01 — Verificar que exec_path existe en disco y que tiene permisos de<br>ejecución (os.access X_OK). Bloqueante.|X<br>☐|
|**CAT-02**|C02 — Verificar que sha256(exec_path) coincide con binary_checksum del<br>catálogo. Bloqueante. Detecta recompilaciones accidentales entre<br>sesiones/nodos.|X<br>☐|
|**CAT-03**|C03 — Verificar que success_check es un tipo reconocido (exit_code o<br>stdout_regex) y que el regex compila sin error antes de ejecutar nada.|X<br>☐|
|**CAT-04**|Verificar que todo kernel con role: dataset tiene phase_label_hint,<br>size_variant, expected_runtime_seconds y warmup_seconds declarados.|X<br>☐|
|**CAT-05**|Verificar que todo kernel con role: calibration tiene exactamente uno de<br>reports_bandwidth_stdout o reports_flops_stdout en true, no ambos (son<br>roles distintos).|X<br>☐|
|**CAT-06**|resolve_exec_command() no debe inferir argumentos de la suite — el campo|X<br>☐|



Página 1 de 8 

Checklist de validaciones técnicas del orquestador de campañas — Fase 1 DVFS 

||exec_args del catálogo es la fuente única. Si está vacío, pasar cadena vacía,<br>no omitir el argumento.||
|---|---|---|
|**CAT-07**|Repetir C01 y C02 (reducido) inmediatamente antes de cada corrida<br>individual, no solo al inicio de campaña — detecta si el binario cambió a<br>mitad de campaña.|☐|
|**CAT-08**|El ID de cada entrada del catálogo debe ser único dentro del archivo —<br>rechazar el catálogo si hay duplicados.|X<br>☐|
|**3. Entorn**|**o y capacidades (environment.py) — detección de solo lectura del no**|**do**|
|**ID**|**Regla de validación / invariante técnica que debe cumplir el módulo**|**✓**|
|**ENV-01**|detect_environment() es de SOLO LECTURA — no debe escribir nada en<br>sysfs ni en disco de sistema. Todo otro módulo que necesite saber qué<br>puede controlar debe preguntarle a environment.py, nunca repetir la<br>detección.|X<br>☐|
|**ENV-02**|freq_control_capable = False si scaling_driver no es intel_pstate, acpi-<br>cpufreq ni amd-pstate, o si scaling_available_frequencies contiene un único<br>valor.|X<br>☐|
|**ENV-03**|rapl_capable = False si /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj<br>no existe o no cambia entre dos lecturas con 100 ms de diferencia bajo<br>carga sintética mínima.|☐X|
|**ENV-04**|El manifest no puede forzar rapl.enabled: true si environment.py determina<br>rapl_capable: false — el orquestador debe sobrescribir el flag y dejarlo<br>registrado en la metadata, no fallar en silencio.|☐X|
|**ENV-05**|Si freq_control_capable == False, todas las corridas de esa campaña<br>quedan marcadas not_eligible_for_training_dataset: true en su metadata.|X<br>☐|
|**ENV-06**|Registrar la topología NUMA completa: nodos NUMA disponibles, cores por<br>nodo, y a qué nodo NUMA pertenecen los delegated_cpus. Necesario para<br>el check E04.|X<br>☐|
|**ENV-07**|Registrar los siblings SMT de cada core delegado (thread_siblings_list) y la<br>política elegida (un hilo por core físico vs. todos). La política debe quedar en<br>la metadata de campaña, no solo en logs.|X<br>☐|
|**ENV-08**|Leer y registrar el subconjunto real de eventos de perf soportados por esta<br>PMU — necesario para anticipar multiplexación excesiva (factor I02) y para<br>el node_profile.json (sección de calibración).|X<br>☐|
|**ENV-09**|Serializar un environment_report.json al inicio de campaña con todos los<br>campos de EnvironmentProfile — este artefacto permite reproducir el<br>contexto de una campaña incluso semanas después.|X<br>☐|
|**4. Preflig**|**ht (preflight.py) — verificaciones de solo lectura, bloqueantes o adve**|**rtencias**|
|**ID**|**Regla de validación / invariante técnica que debe cumplir el módulo**|**✓**|
|**PRE-E01**|E01 — Leer y registrar el estado de Turbo Boost / HWP al inicio de<br>campaña. Fijarlo como constante para toda la campaña — no permitir que<br>cambie entre corridas. Bloqueante si varía.|☐|
|**PRE-E02**|E02 (por corrida) — Leer temperatura de paquete si el sensor existe.<br>Rechazar la corrida si está fuera del rango normal declarado.|☐|
|**PRE-E04**|E04 — Verificar que todos los cores en delegated_cpus pertenecen a un<br>único nodo NUMA. Si están repartidos en varios nodos, la telemetría de<br>memoria es ambigua. Bloqueante.|☐|
|**PRE-E05**|E05 — Identificar siblings SMT de delegated_cpus; la política SMT debe<br>estar declarada explícitamente en el manifest, no asumida. Bloqueante si la<br>política no está definida.|☐|
|**PRE-E06**|E06 (por corrida) — Verificar que no hay procesos ajenos con afinidad a|☐|



Página 2 de 8 

Checklist de validaciones técnicas del orquestador de campañas — Fase 1 DVFS 

||delegated_cpus antes de cada corrida. Bloqueante.||
|---|---|---|
|**PRE-E07**|E07 (por corrida) — Leer scaling_governor antes y después de cada corrida.<br>Rechazar si el governor efectivo no coincide con el esperado (governor drift).|☐|
|**PRE-E08**|E08 (por corrida) — Leer load average normalizado. Rechazar si la carga<br>externa del nodo supera el umbral configurado en el manifest.|☐|
|**PRE-I05**|I05 — Leer max_energy_range_uj. Si no existe, marcar<br>rapl_wrap_correction: unavailable en la metadata de campaña (no es<br>bloqueante, pero condiciona el postproceso).|☐|
|**PRE-I07**|I07 — Verificar que output_dir y el run_id de la corrida actual no existen ya<br>en disco. Bloqueante. Salvo overwrite: true explícito, nunca pisar datos<br>previos.|☐|
|**PRE-C01**|C01 (campaña + por corrida) — Verificar existencia y permisos de ejecución<br>del binario del catálogo. Bloqueante.|☐|
|**PRE-C02**|C02 (campaña + por corrida) — Verificar checksum del binario contra el<br>catálogo. Bloqueante.|☐|
|**PRE-C03**|C03 — Verificar que el success_check de cada kernel está bien configurado<br>antes de ejecutar cualquier corrida. Bloqueante.|☐|
|**PRE-D01**|D01 — Verificar disponibilidad del toolchain (gfortran, gcc, make) si se va a<br>recompilar en este entorno. Bloqueante solo si se recompila.|☐|
|**PRE-D02**|D02 — Después de correr calibración, verificar que la salida de STREAM y<br>ERT es parseable (BW y FLOPs extraídos con éxito). Bloqueante para toda<br>la campaña.|☐|
|**PRE-D03**|D03 — Verificar que BW_pico y P_pico están dentro de ±40% de la ficha<br>técnica declarada del hardware (rango amplio para atrapar errores groseros).<br>Bloqueante para toda la campaña.|☐|
|**PRE-D04**|D04 — Verifcar que cv_pct de las referencias de calibración P95 es<br>≤<br>5% (umbral confgurable). Solo advertencia — no bloqueante, pero se<br>registra en el reporte.|☐|
|**PRE-G01**|G01 — Si gpu.enabled: true, verificar que no hay procesos CUDA activos en<br>la GPU asignada. Bloqueante si hay actividad ajena.|☐|
|**PRE-G02**|G02 — Si gpu.enabled: true, leer y registrar el estado de persistence mode.<br>No es bloqueante, pero debe ser constante durante toda la campaña.|☐|
|**PRE-G03**|G03 — Si gpu.enabled: true, leer y registrar la configuración MIG activa y<br>qué partición se usará. No es bloqueante, pero condiciona la interpretación<br>de NVML.|☐|
|**5. Contro**|**l de frecuencia (freqctl.py) — la pieza más sensible del sistema**||
|**ID**|**Regla de validación / invariante técnica que debe cumplir el módulo**<br>|**✓**|
|**FRQ-01**|snapshot_original_state() debe llamarse UNA SOLA VEZ al inicio de<br>campaña, antes de tocar cualquier core. El estado capturado es la referencia<br>de restauración de todo el sistema.|☐|
|**FRQ-02**|apply_frequency() debe VERIFICAR la frecuencia aplicada releyendo<br>scaling_cur_freq después de escribir scaling_setspeed. Si difiere más allá de<br>la tolerancia configurada, reportar CheckResult(passed=False,<br>factor_id='E01'), no asumir que se aplicó.|☐|
|**FRQ-03**|resolve_level_to_khz() debe guardar TANTO el valor solicitado (fracción ×<br>rango) COMO el valor discreto efectivamente aplicado. La metadata debe<br>contener ambos. Nunca solo uno.|☐|
|**FRQ-04**|restore_original_state() debe ser IDEMPOTENTE — puede llamarse más de<br>una vez sin lanzar excepción. Debe verificar por lectura que el<br>governor/frecuencia volvieron al estado original, no solo que el comando de|☐|



Página 3 de 8 

Checklist de validaciones técnicas del orquestador de campañas — Fase 1 DVFS 

||escritura no falló.||
|---|---|---|
|**FRQ-05**|install_emergency_handlers() debe registrar restore_original_state() en:<br>atexit.register(), signal.SIGINT y signal.SIGTERM. Cubre Ctrl-C, kill, y salida<br>normal del proceso.|☐|
|**FRQ-06**|Si environment.freq_control_capable == False, freqctl.py NO debe intentar<br>escribir en sysfs bajo ninguna circunstancia. Registrar frequency_control:<br>unavailable en metadata y marcar la corrida como<br>not_eligible_for_training_dataset: true.|☐|
|**FRQ-07**|La calibración Roofline (STREAM + ERT) debe correr a F0 (frecuencia<br>máxima). freqctl.py se invoca para fijar F0 antes de la calibración y restaura<br>al terminar, usando la misma rutina de restauración que para las corridas de<br>dataset.|☐|
|**FRQ-08**|Prueba de caos OBLIGATORIA antes de usar en hardware real: lanzar una<br>corrida y enviar SIGINT a mitad de ejecución. Verificar por lectura de sysfs<br>que el governor/frecuencia quedó exactamente igual al estado previo a la<br>campaña.|☐|
|**FRQ-09**|El governor userspace debe fijarse ÚNICAMENTE sobre los cores en<br>delegated_cpus, nunca a nivel global del nodo.|☐|
|**FRQ-10**|Registrar el valor efectivo de la frecuencia observada (scaling_cur_freq) en<br>cada ventana de muestreo — no solo la frecuencia nominal aplicada al inicio<br>de la corrida. Esto produce la columna freq_khz_observed en windows.csv.|☐|
|**6. Calibr**|**ación Roofline y perfil de nodo (calibration.py, node_profile.py)**||
|**ID**|**Regla de validación / invariante técnica que debe cumplir el módulo**<br>|**✓**|
|**CAL-01**|La calibración debe ejecutarse a F0 (frecuencia máxima). Ejecutarla a una<br>frecuencia reducida subestimaría P_pico y BW_pico, distorsionando I_ridge<br>para toda la campaña.|☐|
|**CAL-02**|BW_pico se extrae EXCLUSIVAMENTE del stdout del binario STREAM<br>(valor auto-reportado por la suite). No inferir de contadores de PMU de<br>hardware.|☐|
|**CAL-03**|P_pico se extrae EXCLUSIVAMENTE del stdout del binario ERT o del micro-<br>benchmark de FLOPs. No usar el evento FP_ARITH_INST_RETIRED ni<br>equivalentes — no son portables entre Intel/AMD ni entre generaciones.|☐|
|**CAL-04**|i_ridge_flops_per_byte = p_pico_flops_per_s / bw_pico_bytes_per_s. El<br>cálculo y la verificación de plausibilidad D03 deben ocurrir en la misma<br>función; si D03 falla, lanzar excepción bloqueante antes de generar la matriz<br>de dataset.|☐|
|**CAL-05**|roofline_calibration.json debe incluir: campaign_id, timestamp,<br>delegated_cpus, BW_pico, P_pico, I_ridge, stdout crudo de STREAM y ERT,<br>plausibility_check_passed. Todo lo necesario para auditar la calibración de<br>una campaña pasada.|☐|
|**CAL-06**|load_calibration() lo usa postprocess.py — debe validar que el JSON existe y<br>que plausibility_check_passed == true antes de usarlo para etiquetar<br>ventanas. Si es false, rechazar el post-procesamiento de toda la campaña.|☐|
|**CAL-07**|build_node_profile() es de SOLO LECTURA: lee /proc/cpuinfo,<br>/sys/devices/system/cpu/*/cache/index*/, /sys/devices/system/node/ y el<br>EnvironmentProfile ya calculado. No ejecuta nada nuevo sobre el hardware.|☐|
|**CAL-08**|node_profile.json debe incluir: node_id, hostname, cpu_model, sockets,<br>cores_total, threads_per_core, numa_nodes, cache_l1/l2/llc_kb,<br>cache_llc_shared, freq_min/max_khz, scaling_driver, perf_events_supported,<br>rapl_domains_available.|☐|
|**CAL-09**|build_calibration_references() debe correr<br>5 repeticiones del kernel<br>≥<br>de referencia y calcular P95 de IPC, IPS, MPKI y MissRate entre|☐|



Página 4 de 8 

Checklist de validaciones técnicas del orquestador de campañas — Fase 1 DVFS 

||repeticiones. El CV% entre esas repeticiones es el indicador de<br>estabilidad.||
|---|---|---|
|**CAL-10**|Si cv_pct > umbral (defecto 5.0, configurable), accepted = False. El check<br>D04 en preflight.py debe leer este campo y generar una advertencia en el<br>reporte — no es bloqueante, pero sí condiciona la interpretación de las<br>Propuestas A y B.|☐|
|**CAL-11**|Los tres artefactos (roofline_calibration.json, node_profile.json,<br>calibration_references.json) deben generarse en la misma fase de campaña,<br>antes de comenzar la matriz de dataset. No se puede recolectar datos de<br>entrenamiento sin calibración completa.|☐|
|**7. Ejecuc**|**ión de corrida individual (runner.py)**||
|**ID**|**Regla de validación / invariante técnica que debe cumplir el módulo**|**✓**|
|**RUN-01**|El comando del launcher se construye SIEMPRE desde el KernelEntry del<br>catálogo — nunca hardcodeando rutas, argumentos ni flags por suite. Si<br>entry.exec_args está vacío, pasar string vacío, no omitir el parámetro.|☐|
|**RUN-02**|run_id es DETERMINISTA: f'{campaign_id}__{kernel_ref}__{freq_level.id}<br>__rep{repetition_index:02d}'. Nunca usar timestamps ni UUIDs — la<br>determinismo es lo que permite la reanudación.|☐|
|**RUN-03**|subprocess.run() debe tener timeout = entry.expected_runtime_seconds<br>× SAFETY_MARGIN (<br>3×). Si el proceso supera el timeout, matarlo<br>≥<br>explícitamente (process.kill()) y registrar rechazo con factor_id de<br>timeout.|☐|
|**RUN-04**|Después de cada corrida, verificar que no quedan procesos hijos vivos (via<br>psutil o /proc/<pid>/status) antes de iniciar la siguiente combinación. Si<br>quedan, matarlos y registrar la anomalía.|☐|
|**RUN-05**|Aplicar success_check del catálogo contra el resultado: si es exit_code,<br>verificar returncode == 0; si es stdout_regex, buscar el patrón en la salida. Si<br>el check falla, marcar la corrida como rechazada con factor_id C03.|☐|
|**RUN-06**|La metadata final de cada corrida es la FUSIÓN de: metadata.json del<br>launcher + campos del orquestador (kernel_ref, binary_checksum,<br>roofline_calibration_ref, node_profile_ref, calibration_ref, environment_tier,<br>node_id, resultado del preflight reducido, frecuencia aplicada/observada).<br>Nunca solo uno de los dos.|☐|
|**RUN-07**|Registrar stdout y stderr completos de cada corrida en archivos separados<br>dentro de output_dir/<run_id>/. Son la única evidencia para depurar un<br>success_check fallido o un crash silencioso.|☐|
|**RUN-08**|Warmup se controla por tiempo de pared (warmup_seconds del catálogo),<br>aplicado en postprocess.py. Runner no tiene que hacer nada especial para<br>warmup — simplemente deja correr el binario completo.|☐|
|**8. Gener**|**ación y secuenciación de campaña (campaign.py)**||
|**ID**|**Regla de validación / invariante técnica que debe cumplir el módulo**|**✓**|
|**CAM-01**|M01 — Aleatorizar el orden de la matriz SIEMPRE con<br>random.Random(seed).shuffle(). Nunca usar random global sin semilla.<br>Nunca ejecutar en bloques por kernel o por frecuencia.|☐|
|**CAM-02**|La semilla y el orden resultante (lista de run_id en el orden ejecutado) deben<br>guardarse en la metadata de campaña, no solo en logs. Necesario para<br>reproducir o auditar el orden de cualquier campaña pasada.|☐|
|**CAM-03**|Reanudación: si output_dir/<run_id>/metadata.json existe y accepted ==<br>True, saltar esa combinación. Si accepted == False, reintentarla (una corrida<br>rechazada no es lo mismo que una corrida hecha).|☐|



Página 5 de 8 

|**CAM-04**|Checklist de validaciones técnicas del orquestador d<br>Baseline y telemetry son un par ATÓMICO por combinación — se ejecutan<br>consecutivamente y no se pueden separar en el orden aleatorizado. El<br>orquestador los trata como una unidad, no como dos corridas<br>independientes.|e campañas — Fase 1 DVFS<br>☐|
|---|---|---|
|**CAM-05**|Contabilizar el consumo real de hora-núcleo de la campaña piloto antes de<br>lanzar la campaña completa. Detener y alertar si se proyecta un excedente<br>del presupuesto disponible antes de que ocurra.|☐|
|**CAM-06**|Cada corrida debe tener timeout de fase (arranque, ready, ejecución, cierre).<br>Si se supera cualquier fase, matar el proceso, registrar el rechazo y<br>continuar con la siguiente combinación — nunca colgarse esperando<br>indefinidamente.|☐|
|**CAM-07**|Al cierre de campaña (normal o por interrupción), llamar siempre a<br>freqctl.restore_original_state() y confirmar la restauración por lectura antes<br>de declarar la campaña terminada.|☐|
|**9. Post-p**|**rocesamiento (postprocess.py) — de samples.csv a windows.csv**||
|**ID**|**Regla de validación / invariante técnica que debe cumplir el módulo**<br>**✓**||
|**POST-01**|La PRIMERA muestra de cada repetición recibe quality_status =<br>'first_sample_no_delta' y no se le calculan deltas. Nunca imputar un delta<br>artificial para la primera muestra.|☐|
|**POST-02**|Deltas negativos de contadores (instructions, cycles, cache_misses) sin<br>corrección de wrap conocida<br>quality_status = 'energy_invalid' o<br>→<br>equivalente. Nunca tratar un delta negativo como válido.|☐|
|**POST-03**|I02 — running_ratio = delta_running_ns / delta_enabled_ns por ventana. Si<br>running_ratio < running_ratio_min (defecto 0.90), quality_status =<br>'pmu_degraded'. La ventana no se usa para entrenar.|☐|
|**POST-04**|I03 — Usar el intervalo real medido (timestamp[t] - timestamp[t-1]) para<br>calcular tasas, NO el valor nominal de --interval-ns. Registrar ambos;<br>comparar para detectar jitter excesivo.|☐|
|**POST-05**|I05 — Corrección de wrap-around de RAPL: si max_energy_range_uj está<br>disponible, aplicarla. Si no está disponible y se detecta un delta negativo de<br>energía, marcar energy_valid = false — nunca usar ese delta para calcular<br>potencia.|☐|
|**POST-06**|I06 — Si el lector RAPL devuelve 0 ante error, debe propagarse una bandera<br>de invalidez explícita (no energy_uj = 0). Nunca interpretar 0 J como<br>consumo real sin verificar la bandera de validez.|☐|
|**POST-07**|M02 — Ventanas dentro del rango warmup_seconds del catálogo<br>→<br>quality_status = 'warmup_excluded'. Conservarlas en windows.csv con<br>esa marca; NO descartarlas silenciosamente.|☐|
|**POST-08**|compute_operational_intensity(): si bytes_moved_window == 0, retornar<br>float('nan'), NO dividir por cero. La ventana recibe quality_status =<br>'intensity_undefined' y NO entra al dataset de entrenamiento.|☐|
|**POST-09**|~~flops_window_estimate = run_flops_total × (delta_t_ns / run_duration_ns).~~<br>**SUPERADO (ARC-97/ARC-100, 2026-08-10):** los FLOPs por ventana se miden directamente por hardware (FP_ARITH_INST_RETIRED, `flops_measured_window`), validado en campaña real (~1.29M ventanas, 100% medido). El prorrateo descrito aquí fue eliminado del código (`postprocess.py` ya no tiene `run_flops_total` ni `flops_window_estimate`); queda documentado como antecedente en el registro de cambios (ARC-27.1), no como diseño vigente.|☑|
|**POST-10**|bytes_moved_window = delta_cache_misses × LLC_LINE_SIZE_BYTES.<br>LLC_LINE_SIZE_BYTES debe leerse de la topología real del nodo<br>(node_profile.json), no asumirse como 64 bytes.|☐|
|**POST-11**|phase_label_train: 'memory_bound' si operational_intensity <<br>i_ridge_fops_per_byte, 'compute_bound' si<br>i_ridge_fops_per_byte.<br>≥<br>NUNCA se copia de phase_label_hint ni se infere estadísticamente.|☐|
|**POST-12**|compute_relative_features(): ipc_relative, mpki_relative y miss_rate_relative|☐|



Página 6 de 8 

Checklist de validaciones técnicas del orquestador de campañas — Fase 1 DVFS 

||se calculan SIEMPRE en todas las filas válidas, independientemente de qué<br>alternativa multinodo se adopte. No son opcionales.||
|---|---|---|
|**POST-13**|Las features relativas NO se recortan a [0, 1] — pueden superar 1<br>legítimamente. Un ratio > 1 es información válida (la ventana supera el P95<br>de referencia), no un error a corregir.|☐|
|**POST-14**|node_id, node_profile_ref y calibration_ref deben propagarse a CADA FILA<br>de windows.csv — son indispensables para el group-split por nodo en<br>cualquier análisis posterior.|☐|
|**POST-15**|load_calibration() debe verificar plausibility_check_passed == true antes de<br>usar el I_ridge. Si es false, lanzar excepción — no etiquetar silenciosamente<br>ventanas con una calibración sospechosa.|☐|
|**POST-16**|windows.csv contiene TANTO features absolutas COMO features relativas.<br>Nunca generar una vista que solo tenga unas u otras — ambas deben estar<br>siempre presentes para que el dataset sea reutilizable.|☐|
|**10. Valid**|**ación y criterios de rechazo (validation.py) — toda corrida pasa por aq**|**uí**|
|**ID**|**Regla de validación / invariante técnica que debe cumplir el módulo**<br>|**✓**|
|**VAL-01**|I04 — samples_collected == 0 para algún backend activo es equivalente a<br>push_retries > 0: rechazo inmediato, sin posibilidad de reparación a nivel de<br>ventana. Registrar con factor_id I04.|☐|
|**VAL-02**|I07 — Si el run_id ya existía en disco al inicio de la corrida (no detectado en<br>preflight), rechazar de todas formas. El preflight no es la única defensa.|☐|
|**VAL-03**|C02 — Si el checksum del binario ejecutado (leído en runner.py al momento<br>de ejecutar) difiere del catálogo, la corrida se rechaza con factor_id C02,<br>aunque haya terminado sin error.|☐|
|**VAL-04**|C03 — Si success_check no se cumple (exit_code != 0 o patrón no<br>encontrado en stdout), rechazar con factor_id C03. La corrida 'terminó' pero<br>el kernel no completó su verificación interna.|☐|
|**VAL-05**|D03 — Si la calibración de la sesión está marcada como no plausible<br>(plausibility_check_passed == false), TODAS las corridas de esa sesión<br>quedan marcadas como rechazadas o no_eligible_for_training_dataset. No<br>es un rechazo por corrida sino por campaña completa.|☐|
|**VAL-06**|Las corridas RECHAZADAS nunca se borran — se conservan en<br>output_dir/<run_id>/ con una bandera accepted: false y rejection_factor_id<br>explícito. Son evidencia de auditoría.|☐|
|**VAL-07**|validate_run() aplica los criterios en orden determinista: primero I04 (vacío de<br>datos), luego C02/C03 (integridad del binario), luego E06-E08 (contención),<br>luego los demás. El primer criterio que falla es el factor_id reportado.|☐|
|**VAL-08**|El rechazo a nivel de VENTANA (I01, I02, I03, M02, energy_invalid,<br>intensity_undefined) no rechaza la corrida completa — la corrida puede ser<br>'accepted' aunque tenga ventanas con quality_status != 'ok'. El reporte debe<br>mostrar el % de ventanas válidas por corrida.|☐|
|**11. Meta**|**data y reporte de campaña (metadata_schema.py, report.py)**||
|**ID**|**Regla de validación / invariante técnica que debe cumplir el módulo**<br>|**✓**|
|**MET-01**|La metadata de cada corrida es la FUSIÓN de la del launcher<br>(samples_collected, push_retries, backend, etc.) y la del orquestador (ver<br>sección 7, RUN-06). Deben coexistir en un único metadata.json por corrida.|☐|
|**MET-02**|governor_restored_verified debe ser un booleano que refleja una LECTURA<br>posterior a la restauración, no solo 'el comando de restauración no falló'. Si<br>la lectura confirma el governor correcto, true; si no, false.|☐|
|**MET-03**|node_id es un identificador ESTABLE del nodo — no el hostname de la<br>sesión (que puede cambiar entre reboots). Puede ser el hostname canónico|☐|



Página 7 de 8 

Checklist de validaciones técnicas del orquestador de campañas — Fase 1 DVFS 

||del clúster, definido en el manifest o en environment.py, pero debe ser<br>consistente entre campañas del mismo nodo.|
|---|---|
|**MET-04**|El reporte final de campaña debe contener: tabla de corridas<br>aceptadas/rechazadas por factor_id con porcentaje, I_ridge de la sesión, %<br>de ventanas con quality_status = 'intensity_undefined', cv_pct de las<br>referencias de calibración.<br>☐|
|**MET-05**|Si cv_pct > umbral, el reporte lo señala como advertencia visible (no solo en<br>logs) — porque afecta la interpretabilidad de las Propuestas A y B, aunque<br>no invalide la C.<br>☐|
|**MET-06**|La seed y el orden completo de run_ids ejecutados (en el orden efectivo,<br>incluyendo los saltados por reanudación) deben quedar en la metadata de<br>campaña, no solo en el log de ejecución.<br>☐|
|**MET-07**|Toda corrida debe tener trazabilidad completa hasta cualquier fila de<br>windows.csv: run_id, kernel_ref, node_id, roofline_calibration_ref,<br>node_profile_ref, calibration_ref y binary_checksum deben aparecer en<br>ambos lugares.<br>☐|
|**12. Estra**|**tegia multinodo — invariantes de la capa 'sin arrepentimiento'**|
|**ID**|**Regla de validación / invariante técnica que debe cumplir el módulo**<br>**✓**|
|**MLT-01**|node_id debe estar presente en CADA corrida y en CADA FILA de<br>windows.csv. Sin este campo no es posible hacer group-split por nodo bajo<br>ninguna de las tres alternativas (A, B o C).<br>☐|
|**MLT-02**|node_profile.json debe generarse ANTES de la matriz de dataset, en la<br>misma fase de calibración. Generarlo retroactivamente sobre datos ya<br>recolectados no es posible si el hardware cambia.<br>☐|
|**MLT-03**|calibration_references.json (P95 de IPC/IPS/MPKI/MissRate + CV%) debe<br>generarse con<br>5 repeticiones del kernel de referencia. No calcular<br>≥<br>P95 con menos repeticiones — el valor no sería estadísticamente<br>signifcativo.<br>☐|
|**MLT-04**|Las tres features relativas (ipc_relative, mpki_relative, miss_rate_relative)<br>deben calcularse SIEMPRE, aunque la Propuesta B no se adopte.<br>Calcularlas retroactivamente requeriría tener el node_profile de cada corrida<br>vieja disponible y consistente.<br>☐|
|**MLT-05**|Los manifests deben ser parametrizables por nodo cambiando<br>ÚNICAMENTE environment_tier y cores. Cualquier campo que cambie entre<br>nodos (cores delegados, cgroup_path, etc.) debe estar en el manifest, no en<br>el código.<br>☐|
|**MLT-06**|El protocolo de campaña (manifest, catálogo, harness, orquestador) debe<br>estar versionado (commit hash) y ese hash debe quedar en la metadata de<br>TODA corrida. Dos campañas son comparables entre nodos SOLO si<br>corrieron exactamente el mismo protocolo versionado.<br>☐|
|**MLT-07**|Compilar NPB/STREAM/ERT con -march=native es aceptable si el modelo<br>es por nodo (Propuesta C). Si se exploran las Propuestas A o B, esta<br>decisión debe revisarse con el director — binarios compilados con flags<br>distintos no son directamente comparables entre nodos.<br>☐|
|**MLT-08**|NO comprometer tiempo de campaña ni presupuesto de hora-núcleo<br>ejecutando la matriz completa en un segundo o tercer nodo sin la decisión<br>formal del director. Toda la infraestructura anterior puede construirse en un<br>único nodo.<br>☐|



Página 8 de 8 

