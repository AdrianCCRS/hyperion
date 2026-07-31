Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

# **GUÍA DE DESARROLLO ASISTIDO POR IA** 

### **Orquestador de Campañas de Telemetría — Fase 1 DVFS** 

_Módulo por módulo: descripción, responsabilidades, contexto, prompt sugerido y verificación_ 

###### Universidad Industrial de Santander 

Complementa: Plan de Implementación · Guía Técnica · Checklist de Validaciones · Plan de Tests 

Página 1 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

## **1. Cómo usar esta guía** 

Este documento traduce el Plan de Implementación y la Guía Técnica del Orquestador en instrucciones concretas para desarrollar cada módulo con ayuda de un asistente de IA (por ejemplo, Claude Code). No repite el código completo — eso ya está en la Guía Técnica — sino que da, por módulo, el contexto mínimo necesario, un prompt de arranque, y cómo verificar que lo generado es correcto antes de avanzar al siguiente módulo. 

#### **1.1 Principio rector: un módulo, una sesión, una verificación** 

Cada módulo se desarrolla en su propia sesión de trabajo con la IA, se verifica contra el Checklist de Validaciones y el Plan de Tests correspondientes, y solo entonces se avanza al siguiente. No se debe pedir a la IA que genere varios módulos a la vez: la superficie de error crece más rápido que la capacidad de revisarla, y los módulos más sensibles (freqctl.py, en particular) no admiten ese riesgo. 

#### **1.2 Ciclo de trabajo por módulo** 

- 1. Reunir el contexto: copiar en la conversación las secciones indicadas de la Guía Técnica para ese módulo (no el documento completo — la sección específica basta y evita que la IA se distraiga con información de otros módulos). 

- 2. Usar el prompt sugerido de esta guía como punto de partida, ajustándolo con cualquier detalle real del entorno (rutas exactas, nombres de CPU, etc.) que ya se conozca. 

- 3. Pedir a la IA que genere el módulo junto con sus tests unitarios (los IDs de test listados en la ficha del módulo), no el módulo solo — un módulo sin sus tests no está listo para revisión. 

- 4. Correr los tests generados. Si fallan, iterar con la IA mostrándole el error concreto, no re-explicando el módulo desde cero. 

- 5. Marcar en el Checklist de Validaciones las reglas que ese módulo satisface, una por una, releyendo el código — no asumir que pasan solo porque los tests pasan (los tests verifican lo que se pensó en probar, no lo que se pudo haber olvidado). 

- 6. Solo entonces avanzar al siguiente módulo, en el orden de la sección 2. 

#### **1.3 Qué documento de la serie usar para qué** 

- Plan de Implementación → para entender el porqué de una regla cuando la IA pregunta o cuando algo no tiene sentido a primera vista. 

- Guía Técnica del Orquestador → la referencia de código: firmas de funciones, dataclasses, formato de los artefactos (JSON, CSV). Es el contrato que la IA debe cumplir. 

- Checklist de Validaciones Técnicas → la lista de invariantes que el código debe satisfacer. Se marca módulo por módulo, regla por regla. 

- Plan de Tests → los tests concretos a implementar y correr por módulo. Cada test del plan debe existir como test real en el repositorio antes de dar el módulo por terminado. 

- Esta guía → el orden de desarrollo, el prompt de arranque, y los riesgos específicos de dejar que una IA programe cada pieza. 

#### **1.4 Nota sobre la herramienta** 

Para este tipo de desarrollo — múltiples módulos Python con dependencias entre sí, tests que deben correr localmente, y iteración rápida sobre errores reales — conviene una herramienta que pueda ejecutar comandos y ver resultados de tests directamente, no solo generar texto de código para copiar y pegar. 

## **2. Orden de desarrollo** 

Página 2 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) El orden sigue las dependencias reales entre módulos, no el orden de numeración de la Guía Técnica (esa numeración está organizada para lectura, no para construcción). Los módulos sin flecha de entrada se pueden desarrollar en paralelo si hay más de una persona trabajando. 

Orden de construcción (de arriba hacia abajo; == indica que se pueden hacer en paralelo) 

|1. manifest.py           ==  2. environment.py<br>\                    /<br>v                  v<br>3. prefight.py  <-------+<br>|<br>v<br>4. runner.py (modo --kernel sintetico, sin --exec todavia)<br>|<br>v<br>5. freqctl.py   (sensible: pair-programming + prueba de caos obligatoria)<br>|<br>v<br>6. catalog.py   (independiente de freqctl, pero va despues por prioridad<br>de riesgo: primero lo peligroso, luego lo mecanico)<br>|<br>v<br>7. calibration.py  ==  8. node_profle.py<br>|                    |<br>+--------------------+<br>v<br>9. runner.py (extension a modo --exec, usa catalog.py)<br>|<br>v<br>10. postprocess.py   (usa calibration.py y node_profle.py)<br>|<br>v|
|---|
|11. validation.py<br>|<br>v<br>12. campaign.py       (el integrador: orquesta 1-11 en un solo fujo)<br>|<br>v<br>13. metadata_schema.py + report.py<br>(el esquema se DEFINE junto con el modulo 1, pero el reporte<br>consolidado solo se completa al fnal, cuando existen todos|
|los factor_id que debe resumir)|



**_Riesgo típico de la IA aquí:_** _La IA, si no se le da este orden explícitamente, tiende a proponer construir campaign.py primero ("porque es el punto de entrada") o a generar los 13 módulos de un tirón en una sola respuesta larga. Ninguna de las dos cosas es deseable: campaign.py no se puede probar de verdad sin que los módulos que orquesta ya existan, y una respuesta larga generando todo a la vez es imposible de revisar con el mismo cuidado que un módulo a la vez._ 

**_Cómo verificarlo:_** _antes de aceptar un plan de trabajo que la IA proponga, comparar el orden contra este diagrama. Si difiere, señalarlo explícitamente en vez de dejar que la IA decida el orden por su cuenta._ 

## **3. Módulos, uno por uno** 

#### **<mark>3.1 · manifest.py — Parsing y validación del manifest de campaña</mark>** 

**Descripción** 

Convierte campaign.yaml en un objeto Manifest validado, o rechaza el archivo con un mensaje claro antes de que la campaña toque el nodo. 

Página 3 de 18 

||Guía de Desarrollo Asistidopor IA — Orquestador de Campañas (Fase 1 DVFS)|
|---|---|
|**Qué hace**|●<br>Parsea el YAML a dataclasses tipadas (Manifest, Combination,<br>FrequencyLevel).<br>●<br>Aplica las 11 reglas de validación de MAN-01 a MAN-11 (repeticiones<br>mínimas, output_dir/overwrite, cgroup_path obligatorio en hpc_sc3, roles<br>calibration/dataset sin solape, referencias cruzadas al catálogo, etc.).<br>●<br>Calcula y expone el tamaño total de la matriz antes de que el orquestador<br>ejecute nada.|
|**Depende de**|Ninguno. Es el primer módulo — se puede empezar aquí sin que exista nada más del<br>sistema.|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 2 (Catálogo de kernels y manifest de campaña)<br>completa, con el ejemplo de campaign.yaml.<br>●<br>Checklist de Validaciones, sección 1 (manifest.py).|
|**Checklist**<br>**relacionado**|MAN-01 a MAN-11 (11 reglas)|
|**Tests relacionados**|MAN-T01 a MAN-T11 (11 tests)|



##### **Prompt sugerido** 

Implementa orchestrator/manifest.py en Python 3.11+. 

<mark>Contexto: pego a continuación la sección 2 de la Guía Técnica del Orquestador, que defne el formato de campaign.yaml y las reglas de validación de manifest.py (sección 2.3).</mark> 

<mark>[pegar aquí el contenido de la sección 2 completa]</mark> 

<mark>Requisitos:</mark> 

<mark>- Dataclasses tipadas para Manifest, Combination, FrequencyLevel, Cores, Timeouts, tal como se describen en el ejemplo de campaign.yaml.</mark> 

<mark>- Una función manifest.load(path) -> Manifest que parsea y valida.</mark> 

<mark>- Cada regla de validación debe lanzar ManifestValidationError con un mensaje que incluya el campo específco que falló, no un mensaje genérico.</mark> 

<mark>- Una función manifest.compute_matrix_size(manifest) -> int.</mark> 

<mark>- No asumas valores por defecto para seed, cgroup_path ni overwrite si no están en el YAML — estos campos son obligatorios o deben fallar la validación explícitamente, no recibir un default silencioso.</mark> 

<mark>Junto con el módulo, genera tests/orchestrator/test_manifest.py cubriendo los casos MAN-T01 a MAN-T11 del Plan de Tests (los listo abajo):</mark> 

[pegar aquí la fila "1 · manifest.py" completa del Plan de Tests] 

**_Riesgo típico de la IA aquí:_** _la IA tiende a poner valores por defecto razonables (p. ej. seed=0, overwrite=False silencioso) para que el "camino feliz" funcione rápido, en vez de fallar explícitamente cuando un campo obligatorio falta. Eso rompe MAN-05 y MAN-04 sin que se note hasta mucho después._ 

**_Cómo verificarlo:_** _correr MAN-T04 y MAN-T05 primero — son los que detectan defaults silenciosos. Si pasan sin que el código realmente lance la excepción esperada (por ejemplo, porque el test también asume un default), el test está mal, no el código._ 

#### **<mark>3.2 · environment.py — Detección de capacidades reales del entorno (solo lectura)</mark>** 

|**Descripción**|Convierte "en qué máquina estoy" en un EnvironmentProfile: qué se puede controlar<br>de verdad (frecuencia, RAPL) y qué no, sin escribir nada.|
|---|---|
|**Qué hace**|●<br>Lee sysfs (cpufreq, powercap, topología NUMA y SMT) y arma un|



Página 4 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

||EnvironmentProfile.<br>●<br>Aplica las reglas duras: freq_control_capable y rapl_capable en False<br>cuando el driver o el hardware no lo permiten realmente (caso típico: VM<br>cloud sin passthrough de MSR).<br>●<br>Es la única fuente autorizada de esa detección — otros módulos no deben<br>repetirla por su cuenta.|
|---|---|
|**Depende de**|Ninguno. Se puede desarrollar en paralelo con manifest.py.|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 0.3 (advertencia RAPL/cpufreq en VM) y sección 3<br>(environment.py) completas.|
|**Checklist**<br>**relacionado**|ENV-01 a ENV-09 (9 reglas)|
|**Tests relacionados**|ENV-T01 a ENV-T10 (10 tests)|



##### **Prompt sugerido** 

Implementa orchestrator/environment.py en Python 3.11+. 

<mark>Contexto — pego la sección 0.3 (advertencia técnica sobre RAPL/cpufreq en VMs cloud) y la sección 3 (environment.py) de la Guía Técnica:</mark> 

<mark>[pegar aquí las secciones 0.3 y 3 completas]</mark> 

<mark>Requisitos:</mark> 

<mark>- detect_environment(delegated_cpus: str) -> EnvironmentProfle, de SOLO LECTURA: no debe escribir en ningún archivo de sysfs bajo ninguna circunstancia. Verifca esto explícitamente en el código con comentarios y en los tests con un mock que falle el test si se detecta cualquier operación de escritura.</mark> 

<mark>- freq_control_capable = False si scaling_driver no es intel_pstate, acpi-cpufreq o amd-pstate, o si solo hay una frecuencia disponible.</mark> 

<mark>- rapl_capable = False si energy_uj no existe o no cambia entre dos lecturas con 100 ms de diferencia.</mark> 

<mark>- Debe funcionar contra sysfs real Y contra un mock inyectable para tests (no hardcodees rutas de /sys directamente en la lógica sin una capa de acceso a flesystem que se pueda sustituir en tests).</mark> 

<mark>- Genera environment_report.json con todos los campos de EnvironmentProfle.</mark> 

<mark>Junto con el módulo, genera tests/orchestrator/test_environment.py cubriendo ENV-T01 a ENV-T10 (los listo abajo), todos con sysfs mockeado, sin tocar hardware real:</mark> 

[pegar aquí la fila "3 · environment.py" completa del Plan de Tests] 

**_Riesgo típico de la IA aquí:_** _es fácil que la IA hardcodee rutas de /sys directamente en la lógica de negocio, sin una capa de acceso a filesystem inyectable — lo cual hace que ENV-T01 a ENV-T09 (que dependen de mockear sysfs) sean imposibles de escribir limpiamente, y termine escribiendo tests que sí tocan el sistema de archivos real del entorno de CI._ 

**_Cómo verificarlo:_** _revisar que los tests generados NO usan rutas reales de /sys/... en ningún assert ni fixture — deben usar exclusivamente el mock inyectado. Si un test necesita permisos de root o falla en un contenedor sin sysfs real, algo está mal._ 

#### **<mark>3.3 · preflight.py — Verificaciones de solo lectura antes de campaña y por corrida</mark>** 

|**Descripción**|Aplica, en orden, todas las verificaciones E/I/C/D/G y decide qué es bloqueante y qué<br>es advertencia, sin modificar nada del sistema.|
|---|---|
|**Qué hace**|●<br>Preflight de campaña (una vez): E01, E04, E05, I05, I07, G01-G03, C01-C03,|



Página 5 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

||D01-D04.<br>●<br>Preflight reducido (por corrida): E02, E06, E07, E08, I07, C01-C02, C03.<br>●<br>Retorna una lista de CheckResult uniformes, con factor_id, passed, blocking<br>y observed.|
|---|---|
|**Depende de**|manifest.py (para saber qué validar) y environment.py (para saber qué es realmente<br>posible en este nodo).|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 4 (Preflight) completa, con las dos tablas (4.1<br>campaña, 4.2 reducido).<br>●<br>Checklist de Validaciones, sección 4 (preflight.py) — las 19 reglas PRE-*.|
|**Checklist**<br>**relacionado**|PRE-E01 a PRE-G03 (19 reglas, todas las categorías)|
|**Tests relacionados**|PRE-T01 a PRE-T13|



##### **Prompt sugerido** 

Implementa orchestrator/preflight.py en Python 3.11+. 

<mark>Contexto — pego la sección 4 completa de la Guía Técnica (las dos tablas de checks, de campaña y reducido) y la sección 4 del Checklist de Validaciones:</mark> 

<mark>[pegar aquí la sección 4 de la Guía Técnica y la sección 4 del Checklist]</mark> 

<mark>Requisitos: - CheckResult como dataclass: factor_id, name, passed, blocking, observed (dict), message. - run_campaign_prefight(manifest, env, catalog) -> list[CheckResult], ejecuta TODOS los checks de la tabla 4.1, sin cortar en el primero que falla — se necesita ver todos los resultados para diagnosticar. - run_reduced_prefight(...) -> list[CheckResult] para la tabla 4.2, invocado antes de cada corrida individual. - Los checks bloqueantes deben poder distinguirse programáticamente de las advertencias (D04 en particular es la única no bloqueante de toda la tabla de campaña — verifícalo). - No implementes la lógica de detección desde cero si ya existe en environment.py o catalog.py — prefight.py orquesta llamadas a esos módulos, no duplica su lógica.</mark> 

<mark>Junto con el módulo, genera tests/orchestrator/test_prefight.py cubriendo PRE-T01 a PRE-T13:</mark> 

[pegar aquí la fila "4 · preflight.py" completa del Plan de Tests] 

**_Riesgo típico de la IA aquí:_** _la IA suele implementar el preflight con "cortocircuito" (return al primer check que falla), lo cual es más simple de programar pero oculta el resto de los problemas — si hay 3 fallas simultáneas, el usuario solo se entera de la primera y repite el ciclo de prueba y error 3 veces._ 

**_Cómo verificarlo:_** _construir deliberadamente un escenario con 2+ fallas simultáneas (p. ej. NUMA mal y governor incorrecto a la vez) y confirmar que el preflight reporta AMBAS, no solo la primera._ 

#### **<mark>3.4 · runner.py (modo sintético) — Ejecución de una corrida con el launcher C++, modo --kernel</mark>** 

|**Descripción**|Invoca telemetry_kernel_launcher en modo sintético (--kernel gemm_naive, etc.),<br>captura su salida y fusiona la metadata. Es la base sobre la que luego se agrega el<br>modo --exec.|
|---|---|
|**Qué hace**|●<br>Construye el comando del launcher a partir de la combinación y el manifest.|



Página 6 de 18 

||Guía de Desarrollo Asistidopor IA — Orquestador de Campañas (Fase 1 DVFS)<br>●<br>Ejecuta con subprocess.run() y timeout, capturando stdout/stderr/exit_code.<br>●<br>Verifica que no queden procesos hijos vivos al terminar.<br>●<br>Fusiona la metadata del launcher con la del orquestador.|
|---|---|
|**Depende de**|manifest.py y preflight.py.|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 8 (Ejecución de una corrida individual) — en esta<br>primera pasada, solo la parte de construcción de comando adaptada al modo<br>--kernel en vez de --exec (ese modo se agrega en el módulo 3.9).|
|**Checklist**<br>**relacionado**|RUN-02, RUN-04, RUN-07 (parcial; el resto se cubre al extender a --exec en 3.9)|
|**Tests relacionados**|RUN-T02, RUN-T05, RUN-T06, RUN-T09|



##### **Prompt sugerido** 

Implementa orchestrator/runner.py en Python 3.11+, primera versión en <mark>modo --kernel sintético (sin catálogo todavía — eso viene después).</mark> 

<mark>Contexto — pego la sección 8 de la Guía Técnica (la lógica de run_single aplica igual, solo cambia cómo se arma el comando: por ahora usa --kernel {name} --size {size} en vez de --exec):</mark> 

<mark>[pegar aquí la sección 8 completa]</mark> 

<mark>Requisitos:</mark> 

<mark>- run_single(combination, env, manifest) -> RunResult, invocando telemetry_kernel_launcher --kernel <nombre> --size <n> --perf-cpus ... --output-dir ... --run-id ...</mark> 

<mark>- run_id determinista: f"{campaign_id}__{kernel_name}__{freq_level.id}__rep{n:02d}". - timeout explícito en subprocess.run(); si expira, matar el proceso y registrar el rechazo, no dejar el proceso colgado. - Verifcación de procesos hijos vivos tras cada corrida (psutil o /proc).</mark> 

<mark>- Guardar stdout.txt y stderr.txt completos en el directorio de la corrida.</mark> 

<mark>Genera tests/orchestrator/test_runner.py con mocks de subprocess cubriendo RUN-T02, RUN-T05, RUN-T06, RUN-T09 (los que no dependen todavía del catálogo — el resto de RUN-T* se agrega cuando se extienda a --exec):</mark> 

[pegar aquí las filas RUN-T02, RUN-T05, RUN-T06, RUN-T09 del Plan de Tests] 

**_Riesgo típico de la IA aquí:_** _la IA a veces omite matar el proceso explícitamente cuando expira el timeout (confía en que subprocess.TimeoutExpired ya lo hizo), pero eso no siempre es cierto según cómo se invoque subprocess — puede dejar un binario NPB corriendo en background consumiendo el core delegado._ 

**_Cómo verificarlo:_** _en el test de timeout, verificar explícitamente (con psutil o listando /proc) que el PID del proceso ya no existe después de que run_single() retorna, no solo que la función retornó._ 

#### **3.5 · freqctl.py — Control de frecuencia y restauración de emergencia — el módulo más sensible** 

|**Descripción**|Lee, discretiza, aplica y verifica frecuencias; y garantiza, pase lo que pase, que el<br>estado original del nodo se restaura.|
|---|---|
|**Qué hace**|●<br>Lee las frecuencias disponibles y discretiza cualquier fracción solicitada al<br>valor real más cercano.|
||●<br>Aplica governor userspace y frecuencia, verificando por relectura que se<br>aplicó de verdad.|
||●<br>Registra el estado original una sola vez por campaña y lo restaura de forma|



Página 7 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

||idempotente.<br>●<br>Instala manejadores de emergencia (atexit, SIGINT, SIGTERM) para que la<br>restauración ocurra incluso ante una interrupción forzada.|
|---|---|
|**Depende de**|environment.py (para saber si freq_control_capable es real).|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 7 (Control de frecuencia) completa, las 4<br>subsecciones.<br>●<br>Checklist de Validaciones, sección 5 (freqctl.py) completa — las 10 reglas<br>FRQ-*.|
|**Checklist**<br>**relacionado**|FRQ-01 a FRQ-10 (10 reglas — todas críticas)|
|**Tests relacionados**|FRQ-T01 a FRQ-T10, más la prueba de caos INT-T03 en hardware real|



##### **Prompt sugerido** 

Implementa orchestrator/freqctl.py en Python 3.11+. 

<mark>ADVERTENCIA para ti (IA): este es el módulo más sensible de todo el sistema. Un error aquí puede dejar un core de un servidor compartido en un estado de frecuencia incorrecto después de que termine el programa. Prioriza correctitud y verifcación por lectura sobre elegancia o brevedad del código.</mark> 

<mark>Contexto — pego la sección 7 completa de la Guía Técnica y la sección 5 completa del Checklist de Validaciones:</mark> 

<mark>[pegar aquí la sección 7 de la Guía Técnica y la sección 5 del Checklist]</mark> 

<mark>Requisitos no negociables:</mark> 

<mark>- apply_frequency() debe RELEER scaling_cur_freq después de escribir y comparar contra el valor aplicado, no asumir éxito porque la escritura no lanzó excepción.</mark> 

- <mark>snapshot_original_state() se llama exactamente UNA VEZ por campaña.</mark> 

<mark>- restore_original_state() debe ser IDEMPOTENTE (poder llamarse 2+ veces sin error) y debe verifcar por LECTURA que la restauración ocurrió, guardando ese resultado en governor_restored_verifed.</mark> 

<mark>- install_emergency_handlers() debe registrar la restauración en atexit.register(), signal.SIGINT y signal.SIGTERM, todos apuntando a la misma función de restauración.</mark> 

- <mark>Si env.freq_control_capable es False, esta clase/módulo NO debe escribir en sysfs bajo ninguna circunstancia — ni siquiera para "intentar".</mark> 

- <mark>El governor solo se toca en los cores de delegated_cpus, nunca fuera.</mark> 

<mark>Genera tests/orchestrator/test_freqctl.py cubriendo FRQ-T01 a FRQ-T10, TODOS con mocks de sysfs (ninguno debe tocar hardware real):</mark> 

<mark>[pegar aquí la fla "5 · freqctl.py" completa del Plan de Tests]</mark> 

<mark>Al terminar, dime explícitamente qué parte del código NO pudiste verifcar solo con mocks y que requiere la prueba de caos en hardware real (sección</mark> 3.5 de esta guía). 

**_Riesgo típico de la IA aquí:_** _es el módulo donde una IA con más confianza de la debida es más peligrosa: puede generar código que "se ve bien" — snapshot, apply, restore, handlers — pero que nunca fue ejecutado contra sysfs real, y los mocks pueden estar validando la forma del código, no su comportamiento real bajo interrupción._ 

Página 8 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) **_Cómo verificarlo:_** _NUNCA dar este módulo por terminado solo porque los tests con mocks pasan. Es el único módulo de toda la guía donde la prueba de caos en hardware real (INT-T03 del Plan de Tests: enviar SIGINT a mitad de una corrida y verificar por lectura de sysfs que todo volvió al estado previo) es un requisito, no un extra._ 

#### **<mark>3.6 · catalog.py — Integridad de los binarios externos (NPB, STREAM, ERT)</mark>** 

|**Descripción**|Convierte kernels/catalog.yaml en un diccionario de KernelEntry validados, y verifica<br>que cada binario referenciado existe y coincide con su checksum.|
|---|---|
|**Qué hace**|●<br>Parsea el catálogo declarativo a KernelEntry con todos sus campos.<br>●<br>verify_binary(): C01 (existe y es ejecutable) y C02 (checksum coincide).<br>●<br>resolve_exec_command(): traduce un KernelEntry a los argumentos --exec/--<br>exec-args del launcher.<br>●<br>Valida separación de roles (dataset vs. calibration) y campos obligatorios por<br>rol.|
|**Depende de**|manifest.py (para la referencia cruzada kernel_ref) — pero se puede desarrollar y<br>testear de forma aislada con fixtures propias.|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 2.1 (catálogo declarativo, con el ejemplo completo de<br>kernels/catalog.yaml) y sección 6 (catalog.py).<br>●<br>Checklist, sección 2 (catalog.py) — las 8 reglas CAT-*.|
|**Checklist**<br>**relacionado**|CAT-01 a CAT-08 (8 reglas)|
|**Tests relacionados**|CAT-T01 a CAT-T09|



##### **Prompt sugerido** 

Implementa orchestrator/catalog.py en Python 3.11+. 

<mark>Contexto — pego la sección 2.1 (ejemplo de kernels/catalog.yaml) y la sección 6 (catalog.py) de la Guía Técnica, y la sección 2 del Checklist:</mark> 

<mark>[pegar aquí la sección 2.1 y 6 de la Guía Técnica, y la sección 2 del Checklist]</mark> 

<mark>Requisitos:</mark> 

<mark>- KernelEntry como dataclass con todos los campos del ejemplo (id, suite, role, exec_path, binary_checksum, phase_label_hint, size_variant, expected_runtime_seconds, warmup_seconds, success_check, reports_bandwidth_stdout, reports_fops_stdout).</mark> 

- <mark>load_catalog(path) -> dict[str, KernelEntry], rechazando IDs duplicados.</mark> 

<mark>- verify_binary(entry) -> CheckResult con factor_id C01 o C02 según cuál falle. Usa hashlib.sha256 sobre el contenido real del archivo, no sobre su ruta ni su tamaño.</mark> 

- <mark>Un kernel con role=calibration debe tener exactamente UNO de reports_bandwidth_stdout / reports_fops_stdout en true, nunca ambos. - Un kernel con role=dataset debe tener phase_label_hint, size_variant, expected_runtime_seconds y warmup_seconds obligatoriamente.</mark> 

- <mark>resolve_exec_command(entry) -> list[str], sin inventar argumentos que no estén en exec_args.</mark> 

<mark>Genera tests/orchestrator/test_catalog.py cubriendo CAT-T01 a CAT-T09, usando binarios de prueba SINTÉTICOS (un script bash trivial que haga echo y exit 0), nunca NPB real:</mark> 

[pegar aquí la fila "2 · catalog.py" completa del Plan de Tests] 

Página 9 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

**_Riesgo típico de la IA aquí:_** _la IA puede implementar la verificación de checksum comparando el tamaño del archivo o su fecha de modificación en vez de un hash criptográfico real — funciona en el caso feliz pero no detecta una recompilación con el mismo tamaño de binario._ 

**_Cómo verificarlo:_** _en CAT-T04, modificar deliberadamente 1 solo byte del binario de prueba sin cambiar su tamaño, y confirmar que verify_binary() sigue detectando la discrepancia._ 

#### **<mark>3.7 · calibration.py — Calibración Roofline (P_pico, BW_pico, I_ridge)</mark>** 

|**Descripción**|Ejecuta STREAM y ERT a frecuencia máxima, extrae sus métricas del stdout, y<br>calcula el ridge point del nodo que se usará para etiquetar el entrenamiento.|
|---|---|
|**Qué hace**|●<br>Ejecuta los kernels de calibración vía runner.run_single() con<br>role=calibration.<br>●<br>Parsea el stdout de cada suite con un regex específico para extraer BW_pico<br>y P_pico.<br>●<br>Calcula I_ridge = P_pico / BW_pico y verifica su plausibilidad frente a la ficha<br>técnica declarada (D03).<br>●<br>Serializa roofline_calibration.json.|
|**Depende de**|catalog.py (kernels de calibración), runner.py (para ejecutarlos), freqctl.py (para fijar<br>F0).|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 6.1 (Calibración Roofline) completa.<br>●<br>Checklist, sección 6, reglas CAL-01 a CAL-06 (las de node_profile.py son un<br>módulo aparte, ficha 3.8).|
|**Checklist**<br>**relacionado**|CAL-01 a CAL-06 (6 reglas)|
|**Tests relacionados**|CAL-T01 a CAL-T06|



##### **Prompt sugerido** 

Implementa la parte de calibración Roofline de orchestrator/calibration.py <mark>en Python 3.11+ (node_profle.py y las referencias P95 son un módulo hermano aparte, no lo incluyas aquí todavía).</mark> 

<mark>Contexto — pego la sección 6.1 de la Guía Técnica y las reglas CAL-01 a CAL-06 del Checklist:</mark> 

<mark>[pegar aquí la sección 6.1 de la Guía Técnica y CAL-01 a CAL-06 del Checklist]</mark> 

<mark>Requisitos:</mark> 

<mark>- run_calibration(manifest, catalog) -> RoofineCalibration: localiza en manifest.calibration los kernels con reports_bandwidth_stdout=True y reports_fops_stdout=True, los ejecuta con runner.run_single() a F0 (usa freqctl para fjar la frecuencia máxima antes), y parsea su stdout.</mark> 

<mark>- El parseo de BW_pico y P_pico debe ser exclusivamente del stdout del binario (documenta con un comentario por qué: portabilidad de contadores de FLOPs entre Intel/AMD). No uses ningún evento de perf para esto.</mark> 

<mark>- compute_i_ridge(p_pico, bw_pico) -> foat, función pura, fácil de testear sin mocks de subprocess.</mark> 

<mark>- check_plausibility(bw, p, spec_bw, spec_p, tolerance=0.4) -> bool. - load_calibration(output_dir) -> RoofineCalibration, debe lanzar excepción si plausibility_check_passed es False — no debe ser posible usar una calibración marcada como no plausible sin que el llamador lo note explícitamente.</mark> 

<mark>Genera tests/orchestrator/test_calibration.py cubriendo CAL-T01 a CAL-T06, con fxtures de stdout SINTÉTICO de STREAM/ERT (strings hardcodeados en el</mark> 

Página 10 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

<mark>test), sin ejecutar binarios reales:</mark> 

[pegar aquí las filas CAL-T01 a CAL-T06 del Plan de Tests] 

**_Riesgo típico de la IA aquí:_** _la IA puede intentar "ser útil" agregando un método alternativo de cálculo de FLOPs vía contadores de PMU "por si el stdout no está disponible" — esto reintroduce exactamente el problema de portabilidad que este diseño evita a propósito. Rechazar esa adición si aparece._ 

**_Cómo verificarlo:_** _leer el código generado buscando cualquier referencia a eventos de perf tipo FP_ARITH o similar dentro de calibration.py — no debería haber ninguna._ 

#### **3.8 · node_profile.py — Perfil de hardware y referencias de calibración P95 (estrategia multinodo)** 

|**Descripción**|Módulo hermano de calibration.py: arma el perfil de hardware del nodo y calcula las<br>referencias de estabilidad P95, necesarios para las Propuestas A y B de la estrategia<br>multinodo.|
|---|---|
|**Qué hace**|●<br>build_node_profile(): reorganiza información de solo lectura ya disponible<br>(topología, caché, NUMA) en un NodeProfile con node_id estable.<br>●<br>build_calibration_references(): corre<br>5 repeticiones de un kernel de<br>≥<br>referencia y calcula P95 + coefciente de variación de<br>IPC/IPS/MPKI/MissRate.<br>●<br>Serializa node_profile.json y calibration_references.json.|
|**Depende de**|environment.py (para la topología ya detectada), runner.py (para las repeticiones del<br>kernel de referencia).|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 6.2 (node_profile.py) completa, y sección 0.4 (por qué<br>existe esta capa multinodo).<br>●<br>Checklist, sección 6, reglas CAL-07 a CAL-11.|
|**Checklist**<br>**relacionado**|CAL-07 a CAL-11 (5 reglas)|
|**Tests relacionados**|CAL-T07 a CAL-T10|



##### **Prompt sugerido** 

Implementa orchestrator/node_profile.py en Python 3.11+. 

<mark>Contexto — pego la sección 0.4 (por qué existe la capa multinodo "sin arrepentimiento") y la sección 6.2 de la Guía Técnica, y las reglas CAL-07 a CAL-11 del Checklist:</mark> 

<mark>[pegar aquí las secciones 0.4 y 6.2 de la Guía Técnica, y CAL-07 a CAL-11]</mark> 

<mark>Requisitos:</mark> 

<mark>- build_node_profle(env, delegated_cpus) -> NodeProfle: SOLO LECTURA de</mark> 

<mark>/proc/cpuinfo, /sys/devices/system/cpu/*/cache/index*/,</mark> 

<mark>/sys/devices/system/node/ y del EnvironmentProfle ya calculado. No ejecutes nada nuevo sobre el hardware.</mark> 

<mark>- build_calibration_references(calibration_runs) -> CalibrationReferences: requiere al menos 5 repeticiones; calcula P95 de IPC, IPS, MPKI y MissRate entre esas repeticiones, y el coefciente de variación (CV%). accepted = (cv_pct <= 5.0), con el umbral como parámetro confgurable, no hardcodeado sin poder cambiarlo.</mark> 

<mark>- Este módulo es independiente de calibration.py (Roofine) en cuanto a propósito, pero corre en la misma fase de campaña — no dupliques la lógica de ejecutar kernels de referencia si ya existe en calibration.py.</mark> 

Página 11 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

<mark>Genera tests/orchestrator/test_node_profle.py cubriendo CAL-T07 a CAL-T10:</mark> 

[pegar aquí las filas CAL-T07 a CAL-T10 del Plan de Tests] 

**_Riesgo típico de la IA aquí:_** _con menos de 5 repeticiones el cálculo de P95 no es estadísticamente significativo, pero una IA puede "optimizar" el número de repeticiones a la baja para que los tests corran más rápido en CI, sin señalar que eso cambia el significado del resultado._ 

**_Cómo verificarlo:_** _confirmar que el código rechaza (o al menos advierte fuertemente) si se le pide construir referencias con menos de 5 repeticiones — no debe ser un límite que se pueda bajar accidentalmente vía un parámetro por defecto bajo._ 

#### **<mark>3.9 · runner.py (extensión --exec) — Ejecución de kernels reales del catálogo</mark>** 

|**Descripción**|Extiende runner.py (módulo 3.4) para construir el comando en modo --exec a partir de<br>un KernelEntry del catálogo, en vez del modo --kernel sintético.|
|---|---|
|**Qué hace**|●<br>Resuelve el comando desde catalog[combination.kernel_ref] en vez de un<br>nombre de kernel hardcodeado.<br>●<br>Aplica success_check (C03) contra el resultado real de la ejecución.<br>●<br>Ajusta el timeout usando expected_runtime_seconds del catálogo con<br>margen de seguridad.|
|**Depende de**|runner.py (3.4, ya existente), catalog.py (3.6).|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 8 completa (ahora sí, con el modo --exec).<br>●<br>El propio runner.py generado en 3.4, para que la IA extienda en vez de<br>reescribir.|
|**Checklist**<br>**relacionado**|RUN-01, RUN-03, RUN-05, RUN-06, RUN-08 (completa lo pendiente de 3.4)|
|**Tests relacionados**|RUN-T01, RUN-T03, RUN-T04, RUN-T07, RUN-T08|



##### **Prompt sugerido** 

Extiende orchestrator/runner.py (pego el código actual del módulo, generado <mark>en la sesión anterior en modo --kernel sintético) para soportar también el modo --exec usando el catálogo.</mark> 

<mark>Contexto:</mark> 

<mark>[pegar aquí el código actual de runner.py]</mark> 

<mark>[pegar aquí la sección 8 completa de la Guía Técnica]</mark> 

<mark>Requisitos:</mark> 

<mark>- run_single() debe aceptar ahora también el parámetro catalog y resolver entry = catalog[combination.kernel_ref] para construir el comando --exec entry.exec_path --exec-args entry.exec_args (o vacío si no hay args), en vez de --kernel/--size.</mark> 

<mark>- El modo --kernel sintético existente NO debe romperse — sigue siendo el modo usado para pruebas del propio orquestador (sección 13 de la Guía Técnica). Ambos modos deben coexistir en el módulo.</mark> 

<mark>- timeout = entry.expected_runtime_seconds * SAFETY_MARGIN (usa 3x como valor por defecto, pero que sea un parámetro, no una constante fja sin posibilidad de ajuste).</mark> 

- <mark>Aplicar entry.success_check contra el resultado (exit_code o stdout_regex) y marcar C03 si falla. - Fusionar en la metadata: checksum del binario ejecutado, kernel_ref, roofine_calibration_ref, node_profle_ref, calibration_ref.</mark> 

<mark>Actualiza tests/orchestrator/test_runner.py agregando RUN-T01, RUN-T03, RUN-T04, RUN-T07, RUN-T08 (los que dependen del catálogo):</mark> 

Página 12 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

[pegar aquí las filas correspondientes del Plan de Tests] 

**_Riesgo típico de la IA aquí:_** _al pedir una "extensión" de un módulo existente, algunas IA reescriben el archivo completo en vez de modificarlo incrementalmente, lo cual puede perder silenciosamente comportamiento del modo --kernel sintético que ya funcionaba y estaba testeado._ 

**_Cómo verificarlo:_** _correr TODOS los tests de runner.py (los de 3.4 y los de 3.9) después de esta sesión, no solo los nuevos — un test viejo que empieza a fallar es la señal de que algo se perdió en la reescritura._ 

|**3.10 · postproces**|**s.py — De samples.csv a windows.csv: features y etiqueta de entrenamiento**|
|---|---|
|**Descripción**|El módulo con más lógica de negocio del sistema: calcula deltas, tasas, intensidad<br>operacional, la etiqueta de entrenamiento por Roofline, y las features relativas para la<br>estrategia multinodo.|
|**Qué hace**|●<br>Deltaiza contadores acumulados entre muestras consecutivas, marcando la<br>primera fila de cada repetición.<br>●<br>Calcula running_ratio, corrige wrap-around de RAPL, marca ventanas de<br>warmup por tiempo de pared.<br>●<br>Calcula operational_intensity y deriva phase_label_train comparando contra<br>I_ridge — nunca copiando phase_label_hint.<br>●<br>Calcula las tres features relativas (ipc_relative, mpki_relative,<br>miss_rate_relative) siempre, sin recortarlas a [0,1].|
|**Depende de**|calibration.py y node_profile.py (para I_ridge y las referencias P95).|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 10 completa (postprocess.py), incluyendo el listado<br>REQUIRED_OUTPUT_COLUMNS.<br>●<br>Checklist, sección 9 (postprocess.py) completa — las 16 reglas POST-*, la<br>sección más larga del checklist.|
|**Checklist**<br>**relacionado**|POST-01 a POST-16 (16 reglas — la sección más extensa)|
|**Tests relacionados**|POST-T01 a POST-T15|



##### **Prompt sugerido** 

Implementa orchestrator/postprocess.py en Python 3.11+. Este es el módulo <mark>con más lógica de negocio de todo el sistema — tómate tiempo para revisar cada regla del Checklist antes de escribir código, no solo al fnal.</mark> 

<mark>Contexto — pego la sección 10 completa de la Guía Técnica y la sección 9 completa del Checklist de Validaciones (16 reglas):</mark> 

<mark>[pegar aquí la sección 10 de la Guía Técnica y la sección 9 del Checklist]</mark> 

<mark>Requisitos no negociables:</mark> 

- <mark>La primera muestra de cada repetición: quality_status =</mark> 

- <mark>'frst_sample_no_delta', columnas de delta como None/NaN, nunca 0 imputado.</mark> 

- <mark>Ningún delta negativo de contadores de hardware se propaga como válido sin marcar la ventana correspondiente.</mark> 

- <mark>compute_operational_intensity(): si bytes_moved_window == 0, retorna foat('nan'), NUNCA levanta ZeroDivisionError ni retorna 0.</mark> 

<mark>- phase_label_train se deriva EXCLUSIVAMENTE comparando operational_intensity contra i_ridge_fops_per_byte de la calibración cargada. No debe existir ningún camino de código que copie o infera esta columna de otra forma. - Las tres features relativas se calculan para TODA fla con quality_status aceptable, sin excepción, y sin recortarse a [0,1].</mark> 

Página 13 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

<mark>- windows.csv debe contener exactamente las columnas de</mark> 

<mark>REQUIRED_OUTPUT_COLUMNS — ni de más ni de menos.</mark> 

<mark>- load_calibration() debe rechazar (excepción) una calibración con plausibility_check_passed=False antes de procesar cualquier ventana.</mark> 

<mark>Genera tests/orchestrator/test_postprocess.py cubriendo POST-T01 a POST-T15, con fxtures/fake_samples.csv sintético (constrúyelo tú con los casos específcos que cada test necesita, documentando qué fla prueba qué):</mark> 

[pegar aquí la fila "9 · postprocess.py" completa del Plan de Tests] 

**_Riesgo típico de la IA aquí:_** _es el módulo con más cálculos numéricos del sistema, y el más propenso a que la IA use operadores de división sin verificar el denominador, o a que "simplifique" el manejo de NaN devolviendo 0 en su lugar porque es más simple de encadenar en pandas._ 

**_Cómo verificarlo:_** _buscar en el código generado cada división (/) y confirmar que hay una verificación explícita del denominador antes, o un manejo de excepción/NaN documentado. Grep de 'except ZeroDivisionError' o similar como señal de que se está silenciando el problema en vez de prevenirlo._ 

#### **<mark>3.11 · validation.py — Criterios de aceptación y rechazo por corrida</mark>** 

|**Descripción**|El punto único de decisión sobre si una corrida entra o no al dataset. Aplica, en orden,<br>todos los factor_id de rechazo.|
|---|---|
|**Qué hace**|●<br>validate_run(): aplica I01-I07, E06-E08, M02, C02, C03, D03 en un orden<br>determinista.<br>●<br>Nunca borra una corrida rechazada — la deja marcada en disco con su<br>factor_id.<br>●<br>Distingue rechazo a nivel de ventana (no invalida toda la corrida) de rechazo<br>a nivel de corrida completa.|
|**Depende de**|runner.py (para el RunResult a validar), catalog.py (checksums), postprocess.py<br>conceptualmente (para los criterios a nivel de ventana, aunque la implementación de<br>esos vive en postprocess.py).|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 11 (Validación y criterios de rechazo) completa, con la<br>tabla íntegra de 15 factor_id.<br>●<br>Checklist, sección 10 (validation.py) — las 8 reglas VAL-*.|
|**Checklist**<br>**relacionado**|VAL-01 a VAL-08 (8 reglas)|
|**Tests relacionados**|VAL-T01 a VAL-T08|



##### **Prompt sugerido** 

Implementa orchestrator/validation.py en Python 3.11+. 

<mark>Contexto — pego la sección 11 completa de la Guía Técnica (tabla completa de 15 factor_id de rechazo) y la sección 10 del Checklist:</mark> 

<mark>[pegar aquí la sección 11 de la Guía Técnica y la sección 10 del Checklist]</mark> 

<mark>Requisitos:</mark> 

<mark>- Verdict como dataclass: accepted (bool), factor_id (str|None), message. - validate_run(run_result, manifest) -> Verdict, aplicando los criterios en ESTE orden: I04 (o samples_collected==0) primero, luego C02/C03, luego E06-E08, luego el resto. El primer criterio que falla es el factor_id reportado — documenta este orden explícitamente en un comentario o constante, no lo dejes implícito en la secuencia de ifs.</mark> 

Página 14 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

<mark>- validate_campaign_calibration(calibration) -> lanza CampaignAbortError si plausibility_check_passed es False — esto es un rechazo de CAMPAÑA completa (D03), no de una corrida individual.</mark> 

<mark>- Una corrida rechazada se serializa en disco con accepted=False y rejection_factor_id — el directorio de la corrida y su metadata NUNCA se eliminan.</mark> 

<mark>Genera tests/orchestrator/test_validation.py cubriendo VAL-T01 a VAL-T08, incluyendo el caso de dos factores fallando simultáneamente (VAL-T07):</mark> 

[pegar aquí la fila "10 · validation.py" completa del Plan de Tests] 

**_Riesgo típico de la IA aquí:_** _si no se le da el orden determinista explícitamente, la IA puede implementar los checks en el orden en que se le ocurrió escribirlos, lo cual hace que el factor_id reportado ante fallas simultáneas sea impredecible entre ejecuciones o entre reescrituras futuras del módulo._ 

**_Cómo verificarlo:_** _VAL-T07 es el test clave: construir un RunResult con I04 y C02 fallando a la vez, correr validate_run() varias veces, y confirmar que SIEMPRE reporta el mismo factor_id (I04, según el orden documentado)._ 

#### **<mark>3.12 · campaign.py — El integrador: genera, aleatoriza y secuencia la campaña completa</mark>** 

|**Descripción**|El único módulo que orquesta a todos los demás en un fujo completo: prefight<br>calibración<br>matriz<br>por cada combinación (prefight reducido, frecuencia,<br>→<br>→<br>→<br>ejecución, validación)<br>restauración<br>post-procesamiento<br>reporte.<br>→<br>→<br>→|
|---|---|
|**Qué hace**|●<br>build_matrix(): producto cartesiano de kernels de dataset × niveles de<br>frecuencia × repeticiones.<br>●<br>randomize(): aleatoriza con semilla, nunca en bloques por kernel o<br>frecuencia (M01).<br>●<br>run_campaign(): el bucle principal, con reanudación (saltar run_id ya<br>aceptados) y manejo de baseline/telemetry como par atómico.|
|**Depende de**|TODOS los módulos anteriores (3.1 a 3.11). Es deliberadamente el último en<br>desarrollarse, salvo metadata_schema.py/report.py que se cierran en paralelo.|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 9 (campaign.py) completa.<br>●<br>Checklist, sección 8 (campaign.py) — las 7 reglas CAM-*.<br>●<br>El código YA GENERADO de manifest.py, environment.py, preflight.py,<br>freqctl.py, catalog.py, calibration.py, node_profile.py, runner.py,<br>postprocess.py y validation.py — campaign.py los importa y orquesta, no<br>reimplementa nada de su lógica.|
|**Checklist**<br>**relacionado**|CAM-01 a CAM-07 (7 reglas)|
|**Tests relacionados**|CAM-T01 a CAM-T09, más la campaña piloto real (INT-T01 a INT-T10)|



##### **Prompt sugerido** 

Implementa orchestrator/campaign.py en Python 3.11+. Este módulo ORQUESTA <mark>los módulos ya construidos — no debe reimplementar ninguna lógica que ya exista en manifest.py, prefight.py, freqctl.py, runner.py, validation.py, calibration.py, node_profle.py ni postprocess.py.</mark> 

<mark>Contexto — pego las frmas públicas de todos los módulos ya implementados (no el código completo, para no saturar el contexto) y la sección 9 de la Guía Técnica:</mark> 

<mark>[pegar aquí las frmas de funciones públicas de cada módulo ya construido] [pegar aquí la sección 9 completa de la Guía Técnica]</mark> 

Página 15 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

<mark>Requisitos:</mark> 

<mark>- build_matrix(manifest) -> list[Combination]: producto cartesiano de kernels con role=dataset × frequency_levels × range(repetitions).</mark> 

<mark>- randomize(matrix, seed) -> list[Combination]: random.Random(seed).shufe, nunca random global. Guarda la semilla y el orden resultante. - run_campaign(manifest, catalog) -> CampaignReport: el fujo completo descrito en la sección 9, invocando cada módulo en su punto correcto, con reanudación basada en metadata.json existente con accepted=True.</mark> 

<mark>- Cada combinación de dataset se ejecuta junto a su baseline como PAR ATÓMICO (no se separan en el orden aleatorizado). - Al fnal (normal o por interrupción), SIEMPRE llamar freqctl.restore_original_state() antes de retornar.</mark> 

<mark>Genera tests/orchestrator/test_campaign.py cubriendo CAM-T01 a CAM-T09, usando mocks de runner.py y freqctl.py (no ejecutes binarios reales en estos tests):</mark> 

<mark>[pegar aquí la fla "8 · campaign.py" completa del Plan de Tests]</mark> 

<mark>Al terminar, dime si detectaste alguna inconsistencia entre las frmas de los módulos que te di y lo que esta sección de la Guía Técnica asume —</mark> puede indicar que algún módulo previo necesita un ajuste menor. 

**_Riesgo típico de la IA aquí:_** _al ser el módulo integrador con el contexto más grande, es donde más fácilmente la IA "rellena" una función de un módulo anterior que percibe incompleta, en vez de señalar la inconsistencia y detenerse — silenciosamente duplicando o divergiendo de la lógica ya construida y testeada._ 

**_Cómo verificarlo:_** _pedir explícitamente, como se sugiere al final del prompt, que la IA señale inconsistencias en vez de resolverlas por su cuenta. Revisar el diff de campaign.py buscando cualquier lógica que debería vivir en otro módulo (p. ej. un cálculo de checksum reimplementado en vez de llamar a catalog.verify_binary)._ 

#### **<mark>3.13 · metadata_schema.py + report.py — Esquema de trazabilidad y reporte consolidado</mark>** 

|**Descripción**|El contrato de datos que atraviesa todo el sistema (se define temprano, junto con<br>manifest.py) y el reporte final que resume una campaña completa (se cierra al final,<br>cuando todos los factor_id ya existen).|
|---|---|
|**Qué hace**|●<br>metadata_schema.py: dataclasses/JSON Schema de la metadata por corrida<br>y por campaña — el campo común que todos los demás módulos escriben y<br>leen.<br>●<br>report.py: agrega los Verdict de toda la campaña en una tabla de<br>aceptadas/rechazadas por factor_id, más I_ridge, % de ventanas<br>intensity_undefined y cv_pct de calibración.|
|**Depende de**|El esquema (metadata_schema.py) se define en paralelo con manifest.py, al principio.<br>El reporte (report.py) depende de que existan validation.py y campaign.py.|
|**Contexto a darle a**<br>**la IA**|●<br>Guía Técnica, sección 12 completa (metadata y reporte), con el JSON de<br>ejemplo íntegro.<br>●<br>Checklist, sección 11 (metadata_schema.py + report.py) — las 7 reglas<br>MET-*.|
|**Checklist**<br>**relacionado**|MET-01 a MET-07 (7 reglas)|
|**Tests relacionados**|MET-T01 a MET-T07|



##### **Prompt sugerido** 

Parte A (desarrollar EN PARALELO con manifest.py, al inicio): <mark>Implementa orchestrator/metadata_schema.py en Python 3.11+ con las</mark> 

Página 16 de 18 

Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

<mark>dataclasses del esquema de metadata por corrida, tal como el JSON de ejemplo de la sección 12.1 de la Guía Técnica.</mark> 

<mark>[pegar aquí la sección 12.1 de la Guía Técnica]</mark> 

<mark>Requisitos: una función merge(launcher_meta: dict, orchestrator_meta: dict)</mark> 

<mark>-> dict que fusiona ambos orígenes sin que ningún campo de uno pise silenciosamente al otro si hay colisión de nombres (debe lanzar error si hay una clave con valores distintos en ambos diccionarios, no quedarse con el último).</mark> 

<mark>---</mark> 

<mark>Parte B (desarrollar AL FINAL, cuando validation.py y campaign.py ya existen):</mark> 

<mark>Implementa orchestrator/report.py en Python 3.11+.</mark> 

<mark>Contexto — pego la sección 12.2 de la Guía Técnica y la frma pública de validation.py y campaign.py:</mark> 

<mark>[pegar aquí la sección 12.2 de la Guía Técnica y las frmas relevantes]</mark> 

<mark>Requisitos: build_campaign_report(verdicts, calibration, node_refs) debe producir una tabla con conteo y porcentaje de corridas por factor_id (que sume exactamente el 100% del total), más una fla de i_ridge_fops_per_byte de la sesión, % de ventanas intensity_undefned, y cv_pct de calibración con advertencia visible si supera el umbral.</mark> 

<mark>Genera tests/orchestrator/test_metadata_schema.py (parte A, MET-T01 a MET-T03) y agrega a test_report.py (parte B, MET-T04 a MET-T07):</mark> 

[pegar aquí la fila "11 · metadata_schema.py + report.py" completa del Plan de Tests] 

**_Riesgo típico de la IA aquí:_** _en la fusión de metadata (merge), es común que la IA implemente un simple {**dict1, **dict2} de Python, que resuelve colisiones silenciosamente quedándose con el último valor — exactamente lo que MET-01 pide evitar._ 

**_Cómo verificarlo:_** _MET-T01 debe incluir deliberadamente una clave presente en ambos diccionarios de entrada con valores distintos, y confirmar que merge() lanza una excepción en vez de elegir uno silenciosamente._ 

## **4. Buenas prácticas generales al revisar código generado por IA en este proyecto** 

Estas prácticas aplican a los 13 módulos, más allá del riesgo específico de cada uno listado arriba. 

#### **4.1 Antes de aceptar cualquier módulo como terminado** 

- Los tests existen como archivos reales en tests/orchestrator/ y corren en verde — no basta con que la IA describa qué testearía. 

- Cada regla del Checklist de Validaciones para ese módulo se revisó leyendo el código, no solo corriendo los tests (un test puede estar mal escrito y pasar igual). 

- Ningún path de sysfs, ninguna ruta de binario, ningún nombre de evento de perf está hardcodeado sin pasar por una capa de configuración o de acceso inyectable — si algo cambia entre nodos, no debería exigir editar la lógica de negocio. 

- El módulo no reimplementa lógica que ya vive en otro módulo (violación de la separación de responsabilidades de la sección 1 de la Guía Técnica). 

#### **4.2 Señales de alerta al leer una respuesta de la IA** 

Página 17 de 18 

   - Guía de Desarrollo Asistido por IA — Orquestador de Campañas (Fase 1 DVFS) 

- Código que "simplifica" un caso límite en vez de manejarlo explícitamente (silenciar excepciones, usar valores por defecto donde el Checklist exige fallar). 

- Comentarios que dicen "asumiendo que..." sobre algo que el Plan de Implementación o la Guía Técnica ya definieron explícitamente — es señal de que la IA no leyó o no priorizó ese contexto. 

- Funciones que hacen más de lo que se pidió (p. ej. freqctl.py agregando lógica de reintentos no especificada) — puede introducir comportamiento no auditado en el módulo más sensible del sistema. 

- Tests que solo cubren el camino feliz cuando el Plan de Tests especificaba explícitamente un caso de falla — comparar el conteo de tests generados contra el conteo de IDs de test listados en la ficha del módulo. 

#### **4.3 Cuando la IA y el Checklist no coinciden** 

Si al revisar un módulo terminado alguna regla del Checklist no queda clara cómo verificarla en el código generado, ese es el momento de pedirle a la IA que señale explícitamente dónde se cumple esa regla — no de asumir que se cumple porque "el código se ve razonable". Un módulo no se marca como terminado en el Checklist hasta que cada regla tiene una ubicación identificable en el código o en un test. 

#### **4.4 Prueba de caos y campaña piloto: no delegables a la IA** 

La prueba de caos de freqctl.py (sección 3.5) y la campaña piloto de integración (sección 3.12, tests INT-*) requieren hardware real, permisos de root, y presencia humana durante la ejecución. Ninguna sesión de IA reemplaza esto — son el paso final de verificación antes de considerar el orquestador listo para operar, incluso en el entorno local de desarrollo, y mucho más antes de solicitar acceso al clúster SC3. 

Página 18 de 18 

