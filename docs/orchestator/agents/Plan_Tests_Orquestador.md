Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

# **Plan de Tests del Orquestador de Campañas** 

_Agente DVFS — Fase 1: Recolección de Telemetría_ 

_Derivado del Checklist de Validaciones Técnicas · Cubre las 103 invariantes del plan_ 

## **Propósito** 

Este documento define los tests que verifican que cada módulo del orquestador cumple las invariantes técnicas del Checklist de Validaciones. Cada test referencia implícitamente las reglas del checklist a través de su módulo y número de secuencia: por ejemplo, CAT-T03 verifica la invariante CAT-01 del checklist. 

Los tests están ordenados en el mismo orden que el checklist (manifest → catálogo → entorno → preflight → freqctl → calibración → runner → campaign → postprocess → validación → metadata → integración). Los tests de integración (sección 12) cubren el flujo de punta a punta y deben ejecutarse únicamente en hardware baremetal real con las suites compiladas. 

## **Convenciones** 

- **Fixture: archivo, objeto o estado de disco/sistema preparado antes del test.** 

- **Mock: sustitución de una dependencia real (sysfs, subprocess, filesystem) por un doble de prueba controlado.** 

- **Spy: mock que también registra cómo fue invocado (qué argumentos, cuántas veces).** 

- **Los tests de las secciones 1–11 no requieren hardware real — usan mocks de sysfs y fixtures de disco. Deben pasar en cualquier entorno Linux (local, CI, cloud).** 

- **Los tests de la sección 12 requieren un PC bare-metal con root, las suites NPB/STREAM/ERT compiladas, y permiso para modificar cpufreq. Nunca correr en el SC3 hasta que el checklist de la sección 11 del Plan de Implementación esté completo.** 

**_Nota:_** _Un test que pasa no implica que la feature está completa — implica que la invariante específica que ese test verifica se cumple. Correr la suite completa antes de declarar un módulo listo._ 

### **1 · manifest.py — parsing y validación de la entrada declarativa de campaña** 

|**ID de test**|**Precondición /**<br>**Fixture**|**Qué se verifica (assert principal + cómo)**|**Resultado**<br>**esperado**|**✓**|
|---|---|---|---|---|
|MAN-T01|campaign.yaml<br>válido, todos los<br>campos<br>presentes.|Llamar manifest.load(); verificar que retorna<br>un objeto Manifest sin lanzar excepción.|Sin excepción;<br>todos los<br>campos del<br>dataclass<br>están<br>poblados.|☐|
|MAN-T02|campaign.yaml<br>sin cgroup_path,<br>environment_tier:<br>hpc_sc3.|Llamar manifest.load(); verificar que lanza<br>ManifestValidationError.|Excepción con<br>mensaje que<br>menciona<br>cgroup_path.|☐|
|MAN-T03|campaign.yaml<br>con<br>repetitions_per_c<br>ombination: 2.|Llamar manifest.load().|ManifestValidat<br>ionError,<br>mensaje<br>menciona<br>mínimo de 3.|☐|
|MAN-T04|output_dir que ya<br>existe en disco<br>(overwrite: false).|Crear el directorio antes; llamar<br>manifest.load().|ManifestValidat<br>ionError con<br>factor_id I07.|☐|
|MAN-T05|campaign.yaml|Llamar manifest.load().|ManifestValidat|☐|



Página 1 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

||sin campo seed.||ionError;<br>verificar que no<br>se generó<br>ninguna<br>semilla<br>aleatoria como<br>efecto<br>secundario.||
|---|---|---|---|---|
|MAN-T06|delegated_cpus:<br>'2-5',<br>collector_cpu: 3<br>(solape).|Llamar manifest.load().|ManifestValidat<br>ionError por<br>solapamiento<br>de cores.|☐|
|MAN-T07|Sección<br>calibration sin<br>ningún kernel con<br>reports_flops_std<br>out: true.|Llamar manifest.load().|ManifestValidat<br>ionError que<br>menciona que<br>I_ridge no es<br>calculable.|☐|
|MAN-T08|Un kernel_ref que<br>existe en kernels<br>(dataset) también<br>aparece en<br>calibration.|Llamar manifest.load().|ManifestValidat<br>ionError por<br>solapamiento<br>de roles.|☐|
|MAN-T09|Un kernel_ref en<br>kernels que no<br>existe en<br>catalog.yaml.|Llamar manifest.load() con catálogo real o<br>mock.|ManifestValidat<br>ionError antes<br>de tocar el<br>nodo.|☐|
|MAN-T10|frequency_levels<br>con una entrada<br>fraction: 1.5<br>(fuera de rango).|Llamar manifest.load().|ManifestValidat<br>ionError por<br>fracción<br>inválida.|☐|
|MAN-T11|Calcular tamaño<br>de matriz: 3<br>|Llamar manifest.compute_matrix_size();<br>verificar el entero retornado.|Retorna 60<br>(300 con<br>|☐|
|**2 · catalo**|kernels × 4<br>frecuencias × 5<br>reps.<br>**g.py — integridad**|**de los binarios externos**|baseline).||
|**ID de test**|**Precondición /**<br>**Fixture**|**Qué se verifica (assert principal + cómo)**|**Resultado**<br>**esperado**|**✓**|
|CAT-T01|catalog.yaml<br>válido con 3<br>entradas (2<br>dataset, 1<br>calibration).|Llamar catalog.load_catalog(); verificar<br>longitud y tipos.|Diccionario de<br>3 KernelEntry;<br>roles correctos.|☐|
|CAT-T02|Fixture: binario de<br>prueba sintético<br>existente con<br>checksum<br>correcto en el<br>catálogo.|Llamar catalog.verify_binary(entry); verificar<br>CheckResult.|passed=True,<br>ambos checks<br>C01 y C02 en<br>observed.|☐|
|CAT-T03|Fixture: ruta<br>exec_path apunta<br>a un archivo que<br>no existe.|Llamar catalog.verify_binary(entry).|passed=False,<br>factor_id='C01'<br>.|☐|
|CAT-T04|Fixture: binario<br>existe pero su<br>sha256 difiere del<br>catálogo.|Modificar 1 byte del binario o usar sha256<br>incorrecto en el catálogo.|passed=False,<br>factor_id='C02'<br>.|☐|
|CAT-T05|Fixture:|Llamar catalog.load_catalog() o|ManifestValidat|☐|



Página 2 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

||success_check<br>con type:<br>stdout_regex y<br>pattern: '[invalid<br>regex'.|verify_binary().|ionError/<br>CatalogValidati<br>onError<br>mencionando<br>C03 y regex<br>inválido.||
|---|---|---|---|---|
|CAT-T06|Kernel de dataset<br>sin campo<br>warmup_seconds<br>.|Llamar catalog.load_catalog().|CatalogValidati<br>onError; campo<br>obligatorio para<br>role: dataset.|☐|
|CAT-T07|Kernel de<br>calibración con<br>reports_bandwidt<br>h_stdout: true Y<br>reports_flops_std<br>out: true.|Llamar catalog.load_catalog().|CatalogValidati<br>onError; solo<br>uno de los dos<br>puede ser true.|☐|
|CAT-T08|catalog.yaml con<br>dos entradas de<br>id: 'npb_ep'.|Llamar catalog.load_catalog().|CatalogValidati<br>onError por ID<br>duplicado.|☐|
|CAT-T09|Llamar<br>resolve_exec_co<br>mmand() con<br>entry.exec_args:<br>''.|Invocar la función y revisar la lista de<br>argumentos.|El argumento --<br>exec-args<br>aparece con<br>valor '' (cadena<br>vacía), no se<br>omite.|☐|



### **3 · environment.py — detección de capacidades del nodo (solo lectura, con mocks de sysfs)** 

|**ID de test**|**Precondición /**<br>**Fixture**|**Qué se verifica (assert principal + cómo)**|**Resultado**<br>**esperado**|**✓**|
|---|---|---|---|---|
|ENV-T01|Mock de sysfs:<br>scaling_driver='int<br>el_pstate',<br>scaling_available<br>_frequencies='12<br>00000 2400000<br>3600000'.|Llamar environment.detect_environment('2-5');<br>verificar EnvironmentProfile.|freq_control_ca<br>pable=True,<br>available_frequ<br>encies_khz=[1<br>200000,24000<br>00,3600000].|☐|
|ENV-T02|Mock de sysfs:<br>scaling_driver='ac<br>pi-cpufreq',<br>scaling_available<br>_frequencies='24<br>00000' (un único<br>valor).|Llamar detect_environment().|freq_control_ca<br>pable=False.|☐|
|ENV-T03|Mock:<br>scaling_driver='hy<br>pervisor-virtual'<br>(driver<br>desconocido).|Llamar detect_environment().|freq_control_ca<br>pable=False.|☐|
|ENV-T04|Mock:<br>/sys/class/powerc<br>ap/intel-rapl/ no<br>existe.|Llamar detect_environment().|rapl_capable=<br>False.|☐|
|ENV-T05|Mock: energy_uj<br>devuelve el<br>mismo valor en<br>dos lecturas (no<br>cambia).|Llamar detect_environment() con la lógica de<br>doble lectura.|rapl_capable=<br>False.|☐|
|ENV-T06|Mock: energy_uj|Llamar detect_environment().|rapl_capable=|☐|



Página 3 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

||cambia entre<br>lecturas.||True.||
|---|---|---|---|---|
|ENV-T07|manifest.rapl.en<br>abled: true pero<br>detect_environ<br>ment()<br>→<br>rapl_capable=Fa<br>lse.|Llamar validate_environment_vs_manifest().|El flag<br>rapl.enabled se<br>fuerza a False<br>y se registra en<br>la metadata; no<br>se lanza<br>excepción.|☐|
|ENV-T08|Mock de<br>topología NUMA:<br>cpu0-3 en nodo<br>0, cpu4-7 en<br>nodo 1;<br>delegated_cpus='<br>2-5'.|Llamar detect_environment('2-5').|numa_nodes=<br>2; el conjunto<br>delegado está<br>en más de un<br>nodo NUMA<br>(campo en<br>EnvironmentPr<br>ofile).|☐|
|ENV-T09|Llamar<br>detect_environme<br>nt() dos veces<br>seguidas.|Verificar que no se modificó ningún archivo de<br>sysfs (detect no escribe).|Sin llamadas<br>de escritura<br>detectadas<br>(mock de<br>os.write/open(<br>mode='w')).|☐|
|ENV-T10<br>**4 · preflig**|Fixture válida<br>completa.<br>**ht.py — verificac**|Llamar detect_environment() y verificar que se<br>genera environment_report.json.<br>**iones de solo lectura (mocks de sysfs + f**|Archivo JSON<br>generado con<br>todos los<br>campos de<br>EnvironmentPr<br>ofile.<br>**ixtures de disco)**|☐|
|**ID de test**|**Precondición /**<br>**Fixture**|**Qué se verifica (assert principal + cómo)**|**Resultado**<br>**esperado**|**✓**|
|PRE-T01|delegated_cpus<br>en dos nodos<br>NUMA diferentes<br>(mock topología).|Llamar<br>preflight.check_numa(delegated_cpus).|CheckResult(p<br>assed=False,<br>factor_id='E04',<br>blocking=True)<br>.|☐|
|PRE-T02|SMT habilitado;<br>manifest sin<br>campo<br>smt_policy.|Llamar preflight.check_smt(env, manifest).|CheckResult(p<br>assed=False,<br>factor_id='E05',<br>mensaje pide<br>política<br>explícita).|☐|
|PRE-T03|cgroup.procs no<br>vacío (mock que<br>retorna un PID<br>ajeno).|Llamar<br>preflight.check_cgroup_clean(cgroup_path).|CheckResult(p<br>assed=False,<br>factor_id='E06'<br>).|☐|
|PRE-T04|scaling_governor<br>actual =<br>'schedutil';<br>governor<br>esperado =<br>'userspace'.|Llamar<br>preflight.check_governor(delegated_cpus,<br>expected='userspace').|CheckResult(p<br>assed=False,<br>factor_id='E07'<br>).|☐|
|PRE-T05|load average del<br>nodo = 0.9;<br>umbral<br>configurado =<br>0.8.|Llamar<br>preflight.check_external_load(threshold=0.8).|CheckResult(p<br>assed=False,<br>factor_id='E08'<br>).|☐|



Página 4 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

|PRE-T06|output_dir/<br><run_id>/ ya<br>existe en disco.|Crear el directorio; llamar<br>preflight.check_run_id_unique(output_dir,<br>run_id).|CheckResult(p<br>assed=False,<br>factor_id='I07').|☐|
|---|---|---|---|---|
|PRE-T07|max_energy_ran<br>ge_uj no existe<br>en sysfs (mock).|Llamar preflight.check_rapl_wrap(env).|CheckResult(p<br>assed=True,<br>blocking=False<br>,<br>observed={'rapl<br>_wrap_correcti<br>on':'unavailable<br>'}).|☐|
|PRE-T08|C01: exec_path<br>apunta a archivo<br>inexistente.|Llamar preflight.check_binary_exists(entry).|CheckResult(p<br>assed=False,<br>factor_id='C01'<br>).|☐|
|PRE-T09|C02: exec_path<br>existe pero<br>sha256<br>incorrecto.|Llamar<br>preflight.check_binary_checksum(entry).|CheckResult(p<br>assed=False,<br>factor_id='C02'<br>).|☐|
|PRE-T10|D02: stdout de<br>STREAM no<br>contiene el patrón<br>de BW esperado.|Mock de run_calibration() que retorna stdout<br>malformado; llamar<br>preflight.check_calibration_output().|CheckResult(p<br>assed=False,<br>factor_id='D02'<br>,<br>blocking=True)<br>.|☐|
|PRE-T11|D03: BW_pico<br>calculado = 0.5<br>GB/s; ficha<br>técnica declarada<br>= 40 GB/s.|Llamar<br>preflight.check_calibration_plausibility(bw=0.5<br>e9, p=200e9, spec_bw=40e9, spec_p=400e9).|CheckResult(p<br>assed=False,<br>factor_id='D03'<br>,<br>blocking=True)<br>.|☐|
|PRE-T12|D04: cv_pct =<br>8.5% (por encima<br>del umbral 5.0%).|Llamar<br>preflight.check_calibration_stability(cv_pct=8.5<br>).|CheckResult(p<br>assed=False,<br>blocking=False<br>): solo<br>advertencia, no<br>bloqueante.|☐|
|PRE-T13<br>**5 · freqc**|Preflight completo<br>con todas las<br>condiciones<br>correctas.<br>**tl.py — control de**|Llamar<br>preflight.run_campaign_preflight(manifest,<br>env, catalog).<br>**frecuencia y restauración de emergencia**|Lista de<br>CheckResult<br>todos con<br>passed=True;<br>ningún<br>factor_id de<br>error.<br>**(mocks de sysfs)**|☐|



|**ID de test**|**Precondición /**<br>**Fixture**|**Qué se verifica (assert principal + cómo)**|**Resultado**<br>**esperado**|**✓**|
|---|---|---|---|---|
|FRQ-T01|available_khz =<br>[1200000,<br>2400000,<br>3600000]; level<br>fraction=0.75.|Llamar freqctl.resolve_level_to_khz(level,<br>available_khz).|Retorna<br>{'requested_kh<br>z': 2400000,<br>'applied_khz':<br>2400000}<br>(valor discreto<br>más cercano).|☐|
|FRQ-T02|available_khz =<br>[1000000,<br>3000000];<br>fraction=0.60<br>→|Llamar resolve_level_to_khz().|applied_khz =<br>3000000 (más<br>cercano del<br>conjunto<br>discreto).|☐|



Página 5 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

||valor continuo =<br>2200000.||||
|---|---|---|---|---|
|FRQ-T03|Mock de sysfs:<br>escribir<br>scaling_governor;<br>releer devuelve el<br>valor escrito.|Llamar freqctl.apply_frequency([2,3,4,5], level,<br>available_khz) con mock.|Sin excepción;<br>AppliedFreque<br>ncy.verified=Tr<br>ue;<br>scaling_cur_fre<br>q dentro de<br>tolerancia.|☐|
|FRQ-T04|Mock:<br>scaling_cur_freq<br>después de<br>escribir devuelve<br>un valor muy<br>diferente al<br>solicitado (>5%<br>de diferencia).|Llamar apply_frequency() con mock de<br>discrepancia.|Retorna<br>CheckResult(p<br>assed=False,<br>factor_id='E01'<br>).|☐|
|FRQ-T05|Llamar<br>snapshot_original<br>_state([2,3]) con<br>governor='schedu<br>til' y<br>freq=2400000.|Verificar _original_state después de la<br>llamada.|_original_state[<br>2] =<br>('schedutil',<br>2400000);<br>_original_state[<br>3] =<br>('schedutil',<br>2400000).|☐|
|FRQ-T06|Tras snapshot y<br>apply, llamar<br>restore_original_s<br>tate() con mock<br>de escritura.|Verificar que se escribió el governor y la<br>frecuencia originales por core.|Se llamó<br>scaling_govern<br>or='schedutil' y<br>scaling_setspe<br>ed=2400000<br>para cada<br>core;<br>verified=True.|☐|
|FRQ-T07|Llamar<br>restore_original_s<br>tate() dos veces<br>seguidas<br>(idempotencia).|Segunda llamada tras restauración ya<br>realizada.|Sin excepción;<br>sin efectos<br>secundarios<br>inesperados.|☐|
|FRQ-T08|freq_control_capa<br>ble=False en<br>EnvironmentProfil<br>e.|Llamar apply_frequency() cuando<br>env.freq_control_capable=False.|Ninguna<br>escritura en<br>sysfs; retorna<br>AppliedFreque<br>ncy(control='un<br>available').|☐|
|FRQ-T09|Verificar que<br>install_emergenc<br>y_handlers()<br>registra los tres<br>manejadores.|Mock de atexit.register, signal.signal; llamar<br>install_emergency_handlers().|atexit.register<br>llamado 1 vez;<br>signal.SIGINT<br>y<br>signal.SIGTER<br>M registrados.|☐|
|FRQ-T10|El governor<br>userspace no se<br>aplica a cores<br>fuera de|Mock de sysfs: spy sobre qué rutas se<br>escriben; apply_frequency(['2','3'], ...).|Solo se<br>escriben /cpu2/<br>y /cpu3/; no<br>/cpu0/ /cpu1/|☐|
||<br>delegated_cpus.||,<br>ni otras.||
|**6 · calibr**|**ation.py + node_p**|**rofile.py — Roofline, perfil de hardware**|**y referencias P95**||
|**ID de test**<br>CAL-T01|**Precondición /**<br>**Fixture**<br>Fixture: stdout de|**Qué se verifica (assert principal + cómo)**<br>Llamar|**Resultado**<br>**esperado**<br>bw_pico_bytes|**✓**<br>☐|



Página 6 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

||STREAM con<br>línea 'Best Rate<br>MB/s: 42000.5'.|calibration._parse_stream_output(stdout).|_per_s =<br>42000.5 × 1e6<br>= 4.2005e10.||
|---|---|---|---|---|
|CAL-T02|Fixture: stdout de<br>ERT con línea<br>'Peak GFLOPS:<br>350.2'.|Llamar calibration._parse_ert_output(stdout).|p_pico_flops_p<br>er_s = 350.2 ×<br>1e9 =<br>3.502e11.|☐|
|CAL-T03|bw=4.2e10,<br>p=3.5e11.|Llamar calibration.compute_i_ridge(p, bw).|i_ridge_fops_<br>per_byte<br>≈<br>8.33.|☐|
|CAL-T04|bw=0.5e9<br>(absurda),<br>spec_bw=40e9,<br>tolerance=0.4.|Llamar<br>calibration.check_plausibility(bw=0.5e9,<br>spec_bw=40e9).|plausibility_che<br>ck_passed=Fal<br>se; D03 falla.|☐|
|CAL-T05|Calibración<br>correcta:<br>bw=38e9,<br>p=380e9,<br>i_ridge=10.0.|Llamar<br>calibration.run_calibration(mock_runner,<br>manifest, catalog).|roofine_calib<br>ration.json<br>generado con<br>plausibility_c<br>heck_passed=<br>True e<br>i_ridge<br>10.<br>≈|☐|
|CAL-T06|roofline_calibratio<br>n.json con<br>plausibility_check<br>_passed=False.|Llamar calibration.load_calibration(output_dir).|Excepción que<br>impide usar la<br>calibración en<br>postprocess.|☐|
|CAL-T07|Mock de<br>/proc/cpuinfo y<br>/sys/.../cache/<br>para un Intel con<br>8 cores, LLC 12<br>MB.|Llamar node_profile.build_node_profile(env,<br>delegated_cpus).|NodeProfile<br>con<br>cores_total=8,<br>cache_llc_kb=<br>12288,<br>scaling_driver<br>del mock.|☐|
|CAL-T08|5 corridas de<br>referencia con<br>IPC=[2.1, 2.0,<br>2.2, 2.1, 2.0].|Llamar<br>node_profile.build_calibration_references(runs<br>).|ipc_p95<br>2.2;<br>≈<br>cv_pct<br>calculado y <<br>5%;<br>accepted=Tru<br>e.|☐|
|CAL-T09|5 corridas con<br>IPC=[1.0, 3.0,<br>1.5, 2.8, 1.2] (alta<br>variabilidad).|Llamar build_calibration_references(runs).|cv_pct > 5%;<br>accepted=Fals<br>e (D04<br>advertencia).|☐|
|CAL-T10|build_node_profil<br>e() con mock de<br>sysfs|Verificar que no se modificó ningún archivo.|Sin llamadas<br>de escritura;<br>solo lectura|☐|
|**7 · runne**|.<br>**r.py — ejecución**|**individual de corrida con launcher C++ (**|<br>de /proc y /sys.<br>**mock de subproc**|**ess)**|
|**ID de test**|**Precondición /**<br>**Fixture**|**Qué se verifica (assert principal + cómo)**|**Resultado**<br>**esperado**|**✓**|
|RUN-T01|KernelEntry con<br>exec_path='/bin/n<br>pb_ep.x',<br>exec_args=''.|Llamar runner.build_command(combination,<br>manifest, entry); verificar la lista de strings.|Lista incluye '--<br>exec<br>/bin/npb_ep.x<br>--exec-args  --<br>run-id <run_id><br>...' sin<br>argumentos|☐|



Página 7 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

|RUN-T02|combination con<br>kernel_ref='npb_<br>mg',<br>freq_level='F2',<br>repetition_index=<br>3,<br>campaign_id='ca<br>mp1'.|Llamar runner.make_run_id(combination).|inventados.<br>Retorna<br>'camp1__npb_<br>mg__F2__rep0<br>3'.|☐|
|---|---|---|---|---|
|RUN-T03|Mock de<br>subprocess:<br>devuelve<br>returncode=0,<br>stdout='VERIFIC<br>ATION<br>SUCCESSFUL',<br>stderr=''.|Llamar runner.run_single() con<br>entry.success_check={type:stdout_regex,<br>pattern:'VERIFICATION SUCCESSFUL'}.|RunResult.acc<br>epted=True;<br>C03 passed.|☐|
|RUN-T04|Mock de<br>subprocess:<br>returncode=0,<br>stdout sin el<br>patrón de<br>verificación.|Llamar run_single() con success_check<br>stdout_regex.|RunResult.acc<br>epted=False,<br>factor_id='C03'<br>.|☐|
|RUN-T05|Mock de<br>subprocess:<br>proceso excede<br>el timeout<br>(subprocess.Time<br>outExpired).|Llamar run_single() con<br>expected_runtime_seconds=2 y mock que<br>tarda más.|RunResult.acc<br>epted=False,<br>factor_id='time<br>out'; proceso<br>hijo terminado<br>explícitamente.|☐|
|RUN-T06|Mock de psutil:<br>quedan 2<br>procesos hijos<br>vivos tras la<br>corrida.|Llamar runner._cleanup_children() o verificar<br>que run_single() lo llama.|Los hijos son<br>terminados;<br>RunResult<br>registra la<br>anomalía en<br>metadata.|☐|
|RUN-T07|RunResult con<br>metadata del<br>launcher y<br>campos del<br>orquestador.|Llamar<br>runner._merge_metadata(launcher_meta,<br>orchestrator_meta).|Dict resultante<br>contiene<br>AMBOS<br>conjuntos de<br>campos:<br>samples_collec<br>ted (launcher)<br>y node_id,<br>binary_checks<br>um<br>(orquestador).|☐|
|RUN-T08|Llamar<br>run_single()<br>cuando<br>env.freq_control_<br>capable=False.|Mock de environment; verificar que<br>freqctl.apply_frequency NO se invoca.|RunResult.met<br>adata.frequenc<br>y_control =<br>'unavailable';<br>not_eligible_for<br>_training_datas<br>et=True.|☐|
|RUN-T09|stdout y stderr del<br>proceso<br>mockeado con<br>contenido.|Llamar run_single(); verificar archivos en<br>output_dir/<run_id>/.|stdout.txt y<br>stderr.txt<br>creados con el<br>contenido del<br>proceso.|☐|
|**8 · camp**|**aign.py — genera**|**ción, aleatorización y secuenciación de l**|**a campaña**||
|**ID de test**|**Precondición /**|**Qué se verifica (assert principal + cómo)**|**Resultado**|**✓**|



Página 8 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

||**Fixture**||**esperado**||
|---|---|---|---|---|
|CAM-T01|Manifest con 2<br>kernels, 3<br>frequency_levels,<br>4 repeticiones.|Llamar campaign.build_matrix(manifest).|Lista de 24<br>Combination;<br>producto<br>cartesiano<br>correcto.|☐|
|CAM-T02|Misma lista de<br>combinaciones,<br>seed=12345.|Llamar campaign.randomize(matrix,<br>seed=12345) dos veces.|Ambas<br>llamadas<br>producen<br>exactamente el<br>mismo orden.|☐|
|CAM-T03|Misma lista con<br>seed=12345 y<br>seed=99999.|Llamar randomize con cada semilla.|Los órdenes<br>son distintos<br>(no<br>bloqueados<br>por<br>compilador/plat<br>aforma, pero<br>estadísticamen<br>te muy<br>probables).|☐|
|CAM-T04|output_dir/<br><run_id>/<br>metadata.json<br>existe con<br>accepted=True<br>para 3<br>combinaciones.|Llamar campaign.run_campaign() con mock<br>de runner; spy sobre qué run_ids se ejecutan.|Las 3<br>combinaciones<br>con<br>accepted=True<br>existente se<br>saltan; las<br>demás se<br>ejecutan.|☐|
|CAM-T05|output_dir/<br><run_id>/<br>metadata.json<br>existe con<br>accepted=False<br>para 1<br>combinación.|Llamar campaign.run_campaign() con mock.|La<br>combinación<br>rechazada SÍ<br>se reintenta<br>(un rechazo no<br>es lo mismo<br>que una<br>corrida hecha).|☐|
|CAM-T06|Inspeccionar el<br>orden de<br>ejecución de 6<br>combinaciones<br>con 2 kernels × 3<br>frecuencias.|Llamar run_campaign(); registrar el orden<br>efectivo.|El orden NO es<br>bloque-por-<br>kernel<br>(K1F1,K1F2,K1<br>F3,K2F1…),<br>está mezclado<br>por la<br>aleatorización.|☐|
|CAM-T07|Corrida que<br>excede el timeout<br>de fase.|Mock de runner que lanza TimeoutError;<br>llamar run_campaign().|La campaña<br>continúa con la<br>siguiente<br>combinación;<br>la corrida<br>queda<br>rechazada con<br>factor_id de<br>timeout.|☐|
|CAM-T08|Llamar<br>run_campaign();<br>verificar metadata<br>de campaña al<br>finalizar.|Inspeccionar el archivo de metadata de<br>campaña en output_dir.|Contiene seed,<br>lista completa<br>de run_ids en<br>el orden<br>efectivo<br>ejecutado<br>(incluyendo|☐|



Página 9 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

|CAM-T09|Comprobar que<br>cada par<br>baseline/telemetr<br>y se ejecuta<br>consecutivament<br>e.|Mock de runner; spy sobre el orden de<br>run_ids.|saltados).<br>Para cada<br>combinación,<br><run_id>__bas<br>eline y<br><run_id>__tele<br>metry<br>aparecen<br>consecutivos<br>en el orden de<br>ejecución.|☐|
|---|---|---|---|---|
|**9 · postpr**|**ocess.py — de s**|**amples.csv a windows.csv con features y**|**etiquetas (fixtur**|**es CSV)**|
|**ID de test**|**Precondición /**<br>**Fixture**|**Qué se verifica (assert principal + cómo)**|**Resultado**<br>**esperado**|**✓**|
|POST-T01|Fixture: primera<br>fila de una<br>repetición (sin fila<br>anterior).|Llamar postprocess.compute_windows() sobre<br>el fixture.|Fila 0 tiene<br>quality_status='<br>first_sample_n<br>o_delta';<br>columnas de<br>delta son<br>NaN/None.|☐|
|POST-T02|Fixture:<br>delta_cycles =<br>-100 (negativo,<br>sin corrección de<br>wrap).|Llamar compute_windows() con el fixture.|Esa ventana<br>tiene<br>quality_status='<br>energy_invalid'<br>(o similar); no<br>se propaga un<br>valor negativo.|☐|
|POST-T03|Fixture:<br>delta_running_n<br>s=800000,<br>delta_enabled_n<br>s=1000000<br>→<br>ratio=0.80 (bajo<br>umbral 0.90).|Llamar compute_windows().|quality_status='<br>pmu_degraded<br>'; ventana<br>marcada como<br>no apta.|☐|
|POST-T04|Fixture:<br>timestamps con<br>jitter: delta real =<br>1.8 ms vs<br>nominal = 1 ms.|Llamar compute_windows(); verificar cálculo<br>de tasas.|Las tasas usan<br>el delta_t_ns<br>real (1.8 ms),<br>no el nominal<br>de --interval-<br>ns.|☐|
|POST-T05|Fixture:<br>pkg_delta_uj<br>negativo sin<br>max_energy_ran<br>ge_uj en<br>run_metadata.|Llamar compute_windows().|energy_valid=F<br>alse en esa<br>ventana;<br>power_w no se<br>calcula.|☐|
|POST-T06|Fixture: primera<br>ventana de una<br>repetición dentro<br>de<br>warmup_seconds<br>=1.0 (t_start < 1<br>s).|Llamar compute_windows().|quality_status='<br>warmup_exclu<br>ded'; ventana<br>conservada en<br>windows.csv<br>pero marcada.|☐|
|POST-T07|Fixture:<br>delta_cache_miss<br>es=0<br>(bytes_moved_wi<br>ndow=0).|Llamar compute_windows().|quality_status='<br>intensity_undef<br>ined';<br>operational_int<br>ensity=NaN;|☐|



Página 10 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

||||phase_label_tr<br>ain no<br>asignado.||
|---|---|---|---|---|
|POST-T08|Fixture:<br>run_flops_total=1<br>e9,<br>run_duration_ns=<br>1e9,<br>delta_t_ns=1e7,<br>LLC_LINE=64,<br>delta_cache_miss<br>es=1000.|Llamar compute_operational_intensity().|fops_window<br>1e7;<br>≈<br>bytes_moved=<br>64000;<br>I<br>156.25<br>≈<br>FLOP/byte.|☐|
|POST-T09|I=5.0 <<br>I_ridge=10.0 en<br>la calibración de<br>la sesión.|Llamar compute_windows() con esa<br>calibración.|phase_label_tr<br>ain='memory_b<br>ound'.|☐|
|POST-T10|I=15.0 ><br>I_ridge=10.0.|Llamar compute_windows().|phase_label_tr<br>ain='compute_<br>bound'.|☐|
|POST-T11|phase_label_hint<br>='compute_bound<br>' pero I < I_ridge.|Llamar compute_windows() y verificar ambas<br>columnas.|phase_label_tr<br>ain='memory_b<br>ound';<br>phase_label_hi<br>nt='compute_b<br>ound' (se<br>conserva sin<br>modificar).|☐|
|POST-T12|CalibrationRefere<br>nces con<br>ipc_p95=2.0;<br>ventana con<br>ipc=2.5.|Llamar compute_relative_features().|ipc_relative=1.<br>25 (no<br>recortado a<br>1.0).|☐|
|POST-T13|Corrida con<br>node_id='node07'<br>, node_profile_ref<br>y calibration_ref.|Llamar compute_windows().|Cada fila de<br>windows.csv<br>tiene<br>node_id='node<br>07',<br>node_profile_r<br>ef y<br>calibration_ref<br>poblados.|☐|
|POST-T14|load_calibration()<br>con<br>roofline_calibratio<br>n.json donde<br>plausibility_check<br>_passed=False.|Llamar compute_windows() que invoca<br>load_calibration().|Excepción<br>antes de<br>procesar<br>ninguna<br>ventana; no se<br>genera<br>windows.csv<br>con calibración<br>inválida.|☐|
|POST-T15<br>**10 · valid**|Verificar que las<br>columnas<br>REQUIRED_OUT<br>PUT_COLUMNS<br>están presentes.<br>**ation.py — criteri**|Llamar compute_windows() y revisar<br>windows_df.columns.<br>**os de aceptación y rechazo por corrida**|Todas las<br>columnas del<br>conjunto<br>requerido<br>están<br>presentes; no<br>hay columnas<br>de más ni de<br>menos.|☐|



Página 11 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

|**ID de test**|**Precondición /**<br>**Fixture**|**Qué se verifica (assert principal + cómo)**|**Resultado**<br>**esperado**|**✓**|
|---|---|---|---|---|
|VAL-T01|RunResult con<br>metadata.push_r<br>etries=1.|Llamar validation.validate_run().|Verdict(accept<br>ed=False,<br>factor_id='I04').|☐|
|VAL-T02|RunResult con<br>metadata.sample<br>s_collected=0.|Llamar validate_run().|Verdict(accept<br>ed=False,<br>factor_id='I04')<br>(equivalente a<br>push_retries).|☐|
|VAL-T03|RunResult con<br>binary_checksum<br>en metadata<br>diferente al del<br>catálogo.|Llamar validate_run().|Verdict(accept<br>ed=False,<br>factor_id='C02'<br>).|☐|
|VAL-T04|RunResult con<br>success_check=F<br>alse en metadata.|Llamar validate_run().|Verdict(accept<br>ed=False,<br>factor_id='C03'<br>).|☐|
|VAL-T05|RooflineCalibratio<br>n con<br>plausibility_check<br>_passed=False.|Llamar<br>validation.validate_campaign_calibration().|Lanza<br>CampaignAbor<br>tError; todas<br>las corridas de<br>la sesión se<br>marcan<br>no_eligible_for<br>_training_datas<br>et.|☐|
|VAL-T06|RunResult válido<br>(sin ningún<br>problema).|Llamar validate_run().|Verdict(accept<br>ed=True,<br>factor_id=None<br>).|☐|
|VAL-T07|RunResult con<br>I04 Y C02<br>simultáneamente.|Llamar validate_run(); verificar cuál factor_id<br>se reporta.|factor_id='I04'<br>(I04 tiene<br>prioridad por<br>ser el primer<br>criterio en el<br>orden<br>determinista).|☐|
|VAL-T08|output_dir/<br><run_id>/<br>después de un<br>rechazo.|Llamar validate_run() con<br>Verdict(accepted=False) y verificar disco.|El directorio<br>output_dir/<run<br>_id>/ existe;<br>metadata.json<br>contiene<br>accepted=Fals<br>e y<br>rejection_factor<br>_id.|☐|
|**11 · meta**|**data_schema.py**|**+ report.py — trazabilidad y reporte de c**|**ampaña**||
|**ID de test**|**Precondición /**<br>**Fixture**|**Qué se verifica (assert principal + cómo)**|**Resultado**<br>**esperado**|**✓**|
|MET-T01|metadata del<br>launcher con<br>samples_collecte<br>d=500 y<br>metadata del<br>orquestador con<br>node_id='node07'<br>.|Llamar metadata.merge(launcher_meta,<br>orchestrator_meta).|Dict resultante<br>contiene<br>samples_collec<br>ted=500 Y<br>node_id='node<br>07'; sin<br>colisiones.|☐|



Página 12 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

|MET-T02|governor_restore<br>d_verified: la<br>lectura post-<br>restauración<br>devuelve el valor<br>original.|Llamar freqctl.restore_original_state() con<br>mock de lectura; verificar el campo.|governor_resto<br>red_verified=Tr<br>ue; no se<br>infiere del éxito<br>de la escritura.<br>☐|
|---|---|---|---|
|MET-T03|governor_restore<br>d_verified: la<br>lectura post-<br>restauración<br>devuelve un valor<br>distinto.|Mock de sysfs que no restaura correctamente.|governor_resto<br>red_verified=F<br>alse<br>(registrado, no<br>silenciado).<br>☐|
|MET-T04|Campaña con 30<br>corridas: 27<br>aceptadas, 2<br>rechazadas por<br>I04, 1 por C02.|Llamar<br>report.build_campaign_report(verdicts).|Tabla<br>resumen:<br>I04<br>2 (6.7%),<br>→<br>C02<br>1 (3.3%),<br>→<br>aceptadas<br>27<br>→<br>(90%). La<br>suma es 30.<br>☐|
|MET-T05|Una corrida con<br>200 ventanas: 20<br>quality_status='int<br>ensity_undefined'<br>.|Llamar<br>report.compute_window_stats(windows_df).|Reporte<br>muestra 10%<br>de ventanas<br>intensity_undef<br>ined para esa<br>corrida.<br>☐|
|MET-T06|calibration_refere<br>nces con<br>cv_pct=7.2<br>(>5%).|Llamar report.build_campaign_report().|Advertencia<br>visible en el<br>reporte<br>mencionando<br>cv_pct=7.2 y<br>las Propuestas<br>A/B afectadas.<br>☐|
|MET-T07|Una fila de<br>windows.csv.|Verificar que los campos run_id, kernel_ref,<br>node_id, roofline_calibration_ref,<br>node_profile_ref, calibration_ref y|Todos los 7<br>campos de<br>trazabilidad<br>☐|
|||binarychecksum están presentes.|presentes y no|
|**12 · Prue**|**bas de integració**|_<br>**n — campaña piloto real de punta a punt**|<br>nulos.<br>**a (hardware local)**|
|**ID de test**|**Precondición /**<br>**Fixture**|**Qué se verifica (assert principal + cómo)**|**Resultado**<br>**esperado**<br>**✓**|
|INT-T01|PC local bare-<br>metal; kernels<br>sintéticos<br>(gemm_naive,<br>stream_triad) en<br>modo --exec.|Correr campaign piloto: 1 kernel × 2<br>frecuencias × 3 reps. Verificar que el pipeline<br>completo termina sin excepción.|Directorio de<br>campaña<br>generado;<br>samples.csv,<br>metadata.json<br>y windows.csv<br>presentes para<br>cada corrida.<br>☐|
|INT-T02|Campaña piloto<br>local con 1 kernel<br>NPB real (npb_ep<br>clase S)<br>compilado y en el<br>catálogo.|Ejecutar la campaña; verificar windows.csv.|windows.csv<br>tiene<br>operational_i<br>ntensity > 0,<br>phase_label_tr<br>ain poblado y<br>50 ventanas<br>≥<br>con<br>quality_status<br>☐|



Página 13 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

||||='ok'.||
|---|---|---|---|---|
|INT-T03|Prueba de caos:<br>lanzar campaña<br>(freqctl activo) y<br>enviar SIGINT al<br>proceso<br>orquestador.|Leer scaling_governor de los cores delegados<br>después de la interrupción.|El governor y<br>la frecuencia<br>de CADA core<br>delegado<br>coincide<br>exactamente<br>con el estado<br>previo a la<br>campaña.|☐|
|INT-T04|Interrumpir<br>campaña a mitad<br>(corrida 4 de 12);<br>relanzar el mismo<br>comando.|Verificar qué corridas se ejecutan en el<br>segundo lanzamiento.|Las corridas 1–<br>3 (accepted) se<br>saltan; la<br>corrida que fue<br>interrumpida se<br>reintenta; las 5<br>–12 siguen.|☐|
|INT-T05|Campaña piloto<br>en cloud_own<br>con<br>RAPL/cpufreq no<br>disponibles (VM<br>estándar).|Ejecutar la misma campaña con<br>environment_tier: cloud_own.|Todas las<br>corridas tienen<br>frequency_cont<br>rol='unavailabl<br>e' y<br>not_eligible_for<br>_training_datas<br>et=True en<br>metadata; el<br>resto del<br>pipeline<br>funciona igual.|☐|
|INT-T06|Campaña piloto<br>completa local;<br>inspeccionar el<br>reporte de<br>campaña.|Verificar porcentaje de corridas aceptadas.|90% de las<br>≥<br>corridas del<br>piloto<br>aceptadas<br>(criterio del<br>Plan de<br>Implementaci<br>ón §7.1).|☐|
|INT-T07|Campaña piloto<br>local; medir<br>overhead real de<br>baseline vs.<br>telemetry.|Para cada combinación, comparar wall-time<br>de baseline y telemetry.|El overhead<br>(telemetry/base<br>line - 1) es<br>estable entre<br>las 3<br>repeticiones de<br>la misma<br>condición<br>(coeficiente de<br>variación <<br>10%).|☐|
|INT-T08|Campaña local<br>con 2 kernels:<br>npb_ep<br>(compute_bound<br>esperado) y<br>npb_mg<br>(memory_bound<br>esperado).|Verificar phase_label_train en windows.csv.|El grueso de<br>las ventanas<br>de npb_ep<br>tienen<br>phase_label_tr<br>ain='compute_<br>bound' y las de<br>npb_mg<br>'memory_boun<br>d', consistente<br>con<br>phase_label_hi<br>nt.|☐|



Página 14 de 15 

Plan de tests — Orquestador de Campañas de Telemetría (Fase 1 DVFS) 

|INT-T09|node_profile.json<br>y<br>calibration_refere<br>nces.json<br>generados en la<br>campaña piloto.|Verificar su contenido.|node_profile.js<br>on tiene todos<br>los campos de<br>NodeProfile;<br>calibration_refe<br>rences.json<br>tiene cv_pct<br>calculado y<br>accepted=True<br>o False con<br>valor explícito.|☐|
|---|---|---|---|---|
|INT-T10|windows.csv de<br>la campaña<br>piloto.|Verificar que ipc_relative, mpki_relative y<br>miss_rate_relative están presentes y no son<br>nulos para las filas con quality_status='ok'.|Las tres<br>columnas<br>relativas tienen<br>valores<br>numéricos<br>positivos;<br>ninguna es<br>NaN para filas<br>válidas.|☐|



Página 15 de 15 

