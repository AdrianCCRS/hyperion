# Especificación de validación para `orchestrator/catalog.py`

**Proyecto:** Hyperion — Fase 1  
**Versión de esta especificación:** 1.0  
**Base documental:** Guía Técnica Maestra del Orquestador v3.1, Plan Maestro Consolidado Fase 1 v3.1 y cronograma técnico de 12 semanas.  
**Objetivo:** permitir que un agente implemente `catalog.py` sin depender de documentos anteriores ni de frases como “se conservan las reglas de la versión previa”.

---

## 1. Propósito de `catalog.py`

`catalog.py` administra el contrato declarativo entre el orquestador y los binarios de workloads o calibración.

Debe encargarse de:

1. cargar `kernels/catalog.yaml`;
2. validar su estructura y semántica;
3. construir objetos tipados e inmutables;
4. validar la coherencia interna de cada entrada;
5. comprobar la existencia, ejecutabilidad e identidad del binario;
6. comprobar la coherencia entre el catálogo y el build manifest;
7. validar la configuración de `success_check`;
8. resolver el comando de ejecución como `argv`, sin shell;
9. exponer búsquedas deterministas por `kernel_ref`;
10. producir errores y resultados estructurados con IDs estables.

`catalog.py` **no debe**:

- ejecutar campañas;
- cambiar frecuencia;
- crear cgroups;
- interpretar telemetría;
- calcular features;
- decidir si una corrida es aceptada;
- copiar `phase_label_hint` como etiqueta de entrenamiento;
- ejecutar `system()`, `shell=True` o `/bin/sh -c`;
- ignorar campos desconocidos silenciosamente.

---

## 2. Relación con otros módulos

| Módulo | Responsabilidad |
|---|---|
| `manifest.py` | Valida el manifest de campaña y cruza sus `kernel_ref` con el catálogo. |
| `catalog.py` | Valida las entradas del catálogo, los binarios, build manifests, argumentos y success checks. |
| `preflight.py` | Ejecuta C01–C04 usando las funciones públicas de `catalog.py`. |
| `runner.py` | Ejecuta el `ExecSpec` resuelto y evalúa los success checks sobre el resultado real. |
| `calibration.py` | Usa las entradas de calibración y parsea BW/FLOP/s reportados. |
| `validation.py` | Decide `accepted/rejected`; consume resultados de ejecución y preflight. |
| `report.py` | Consolida hashes, build refs, checks y advertencias. |

### 2.1 Mapeo con IDs canónicos

| ID canónico | Responsable principal | Regla |
|---|---|---|
| M04 | `manifest.py` + `catalog.py` | El catálogo existe, es legible y cumple el schema. |
| M05 | `manifest.py` | Cada `kernel_ref` existe exactamente una vez. |
| M07 | `manifest.py` | No se mezclan roles de calibración y dataset. |
| M19 | `manifest.py` + `catalog.py` | `success_check` usa tipos soportados y parámetros válidos. |
| C01 | `catalog.py`/`preflight.py` | El binario existe, es archivo regular y ejecutable. |
| C02 | `catalog.py`/`preflight.py` | El checksum efectivo coincide con catálogo/build manifest. |
| C03 | `catalog.py` | `success_check` es válido y coherente. |
| C04 | `catalog.py` | `role`, `argv`, `environment`, tamaño, runtime y ROI son coherentes. |

C01 y C02 deben ejecutarse:

- una vez para todas las entradas referenciadas antes de la campaña;
- nuevamente para la entrada concreta inmediatamente antes de cada corrida.

Así se detecta una recompilación o sustitución del binario a mitad de campaña.

---

## 3. Formato canónico del catálogo

```yaml
catalog_version: "1.0"

kernels:
  - id: stream_official
    suite: STREAM
    role: calibration
    exec_path: bin/stream_c.exe
    argv: []
    environment: {}
    build_manifest_ref: builds/local/stream.json
    binary_checksum: "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    expected_runtime_seconds: 8
    warmup_seconds: 0
    reports_bandwidth_stdout: true
    reports_flops_stdout: false
    success_check:
      - type: exit_code
        expected: 0
      - type: stdout_regex
        pattern: "(?m)^Triad:"
    roi_mode: process_lifetime

  - id: ert_probe
    suite: ERT
    role: calibration
    exec_path: bin/ert_probe.x
    argv: []
    environment: {}
    build_manifest_ref: builds/local/ert.json
    binary_checksum: "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    expected_runtime_seconds: 15
    warmup_seconds: 0
    reports_bandwidth_stdout: false
    reports_flops_stdout: true
    success_check:
      - type: exit_code
        expected: 0
    roi_mode: process_lifetime

  - id: npb_ep_omp_s
    suite: NPB-OMP
    role: dataset
    exec_path: bin/ep.S.x
    argv: []
    environment:
      OMP_NUM_THREADS: "4"
      OMP_PROC_BIND: "close"
      OMP_PLACES: "cores"
      OMP_DYNAMIC: "false"
    build_manifest_ref: builds/local/npb-omp.json
    binary_checksum: "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    size_variant: S
    expected_runtime_seconds: 4
    warmup_seconds: 1.0
    reports_bandwidth_stdout: false
    reports_flops_stdout: false
    success_check:
      - type: exit_code
        expected: 0
      - type: stdout_regex
        pattern: "VERIFICATION SUCCESSFUL"
    roi_mode: instrumented
    phase_label_hint: compute_bound
```

---

## 4. Modelo de datos recomendado

La etiqueta narrativa `synthetic/dev` usada en algunos documentos debe normalizarse en código como `synthetic`.

```python
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping


class KernelRole(StrEnum):
    DATASET = "dataset"
    CALIBRATION = "calibration"
    SYNTHETIC = "synthetic"


class RoiMode(StrEnum):
    INSTRUMENTED = "instrumented"
    PROCESS_LIFETIME = "process_lifetime"
    SYNTHETIC_READY_GO = "synthetic_ready_go"


class SuccessCheckType(StrEnum):
    EXIT_CODE = "exit_code"
    STDOUT_REGEX = "stdout_regex"
    ARTIFACT_EXISTS = "artifact_exists"


@dataclass(frozen=True)
class SuccessCheck:
    type: SuccessCheckType
    expected: int | None = None
    pattern: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class KernelEntry:
    id: str
    suite: str
    role: KernelRole
    exec_path: Path
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    build_manifest_ref: Path | None
    binary_checksum: str
    size_variant: str | None
    expected_runtime_seconds: float
    warmup_seconds: float
    success_checks: tuple[SuccessCheck, ...]
    roi_mode: RoiMode
    phase_label_hint: str | None
    reports_bandwidth_stdout: bool
    reports_flops_stdout: bool


@dataclass(frozen=True)
class Catalog:
    catalog_version: str
    source_path: Path
    entries: Mapping[str, KernelEntry]


@dataclass(frozen=True)
class ExecSpec:
    executable: Path
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    cwd: Path | None = None


@dataclass(frozen=True)
class CheckResult:
    factor_id: str
    status: str          # pass | fail | warning | skipped
    blocking: bool
    entry_id: str | None
    message: str
    details: Mapping[str, object]
```

---

# 5. Reglas completas que debe validar `catalog.py`

## 5.1 Estructura global del archivo

### CAT001 — Archivo legible

- La ruta del catálogo debe existir.
- Debe ser un archivo regular y legible.
- Un directorio, socket, FIFO o dispositivo no es válido.
- Si falla: error bloqueante asociado a M04.

### CAT002 — YAML válido y seguro

- El YAML debe parsearse con un loader seguro.
- El documento debe producir un único objeto raíz.
- El objeto raíz debe ser un mapping.
- YAML vacío o raíz escalar/lista: rechazo.

### CAT003 — `catalog_version` obligatorio

- Debe existir.
- Debe ser string.
- Debe pertenecer al conjunto de versiones soportadas.
- Para esta implementación inicial: `{"1.0"}`.
- Una versión futura no debe aceptarse suponiendo compatibilidad.

### CAT004 — `kernels` obligatorio

- Debe existir.
- Debe ser una lista.
- No puede estar vacía para campañas que dependan del catálogo.
- Cada elemento debe ser un mapping.

### CAT005 — Campos desconocidos

Política estricta:

- en nivel raíz solo se permiten `catalog_version` y `kernels`;
- en cada entrada solo se permiten los campos declarados en esta especificación;
- en `success_check` solo se permiten los campos del tipo correspondiente;
- no ignorar campos desconocidos;
- el error debe indicar la ruta completa, por ejemplo:

```text
kernels[2].sucess_check: campo desconocido; ¿quiso decir success_check?
```

### CAT006 — IDs únicos

- Cada `id` debe aparecer exactamente una vez.
- No se permite “último valor gana”.
- Debe reportar todos los índices donde aparece el duplicado.

### CAT007 — ID estable y seguro

`id`:

- debe ser string no vacío;
- debe cumplir `^[a-z][a-z0-9_]{2,63}$`;
- no debe contener espacios, `/`, `\\`, `..`, caracteres de control ni separadores de shell;
- debe ser estable y apto para metadata, rutas y `kernel_ref`.

Ejemplos válidos:

```text
stream_official
ert_probe
npb_ep_omp_s
synthetic_gemm
```

---

## 5.2 Suite y rol

### CAT008 — `suite` obligatoria y registrada

- Debe ser string no vacío.
- Suites iniciales registradas:

```text
STREAM
ERT
NPB-SER
NPB-OMP
synthetic
```

- Se pueden agregar suites mediante un registro explícito; no aceptar cualquier string por defecto.
- `NPB-SER` y `NPB-OMP` son condiciones distintas y no deben normalizarse como una sola suite.

### CAT009 — `role` obligatorio

Valores canónicos:

```text
dataset
calibration
synthetic
```

No aceptar `synthetic/dev`, `train`, `benchmark` o `probe` salvo migración explícita.

### CAT010 — Coherencia suite/rol

| Suite | Roles permitidos |
|---|---|
| STREAM | `calibration` |
| ERT | `calibration` |
| NPB-SER | `dataset` |
| NPB-OMP | `dataset` |
| synthetic | `synthetic` |

Un cambio futuro debe hacerse en el registro de suites, no dispersando condicionales.

### CAT011 — Entradas sintéticas no elegibles para dataset formal

`catalog.py` debe marcar `role=synthetic` de forma inequívoca. La elegibilidad final la decide `manifest.py`, pero el catálogo no debe representar un kernel sintético como `dataset`.

---

## 5.3 Ruta y seguridad del ejecutable

### CAT012 — `exec_path` obligatorio

- Debe ser string no vacío.
- No debe contener byte NUL.
- No se expande mediante shell.
- Si es relativo, se resuelve respecto al directorio del `catalog.yaml`, no respecto al directorio actual del proceso.
- Debe conservarse en metadata el valor declarado y la ruta absoluta resuelta.

### CAT013 — C01: existencia y tipo

La ruta resuelta debe:

- existir;
- apuntar a un archivo regular;
- no ser un directorio;
- poder abrirse para lectura.

Resultado:

```text
factor_id = C01
blocking = true
```

### CAT014 — C01: ejecutabilidad

- En Linux debe pasar `os.access(path, os.X_OK)`.
- Si no es ejecutable: C01 falla.
- No intentar `chmod` automáticamente.
- No intentar ejecutar mediante `bash <archivo>` para evadir el permiso.

### CAT015 — Política de symlinks

- Se permite un symlink solo si su destino resuelto es un archivo regular y ejecutable.
- El hash se calcula sobre el contenido del destino efectivo.
- Se registran tanto la ruta declarada como `realpath`.
- Un symlink roto falla C01.

### CAT016 — Verificación repetible

`verify_binary()` debe ser libre de efectos secundarios:

- no modifica permisos;
- no recompila;
- no ejecuta el binario;
- solo inspecciona filesystem y hash.

---

## 5.4 Checksum e identidad del build

### CAT017 — `binary_checksum` obligatorio

Formato exacto:

```text
sha256:<64 caracteres hexadecimales en minúscula>
```

Regex:

```regex
^sha256:[0-9a-f]{64}$
```

No aceptar MD5, SHA-1, hash sin prefijo, placeholders, mayúsculas no normalizadas o string vacío.

### CAT018 — C02: hash efectivo

- Calcular SHA-256 leyendo el archivo por bloques.
- El hash calculado debe coincidir con `binary_checksum`.
- Registrar checksum esperado, checksum efectivo, tamaño, mtime informativo y ruta efectiva.

Si falla:

```text
factor_id = C02
blocking = true
```

### CAT019 — `build_manifest_ref` requerido para binarios reales

Obligatorio para:

```text
STREAM
ERT
NPB-SER
NPB-OMP
```

Puede omitirse para `synthetic` si el ejecutable se genera como fixture de tests y queda documentado.

### CAT020 — Build manifest existente y parseable

`build_manifest_ref`:

- se resuelve respecto al catálogo;
- debe existir y ser legible;
- debe ser JSON válido;
- debe contener, como mínimo:

```text
source_url_or_commit
compiler
compiler_version
flags
libraries
architecture
build_timestamp
binary_checksum
```

También se recomienda registrar:

```text
suite
hostname
node_id
operating_system
language_runtime
make_variables
openmp_configuration
linking
artifacts
```

### CAT021 — Consistencia de checksum con build manifest

Se deben cumplir simultáneamente:

```text
hash efectivo == catalog.binary_checksum
hash efectivo == build_manifest.binary_checksum
```

Si catálogo y build manifest difieren aunque el archivo coincida con uno de ellos, C02 falla.

### CAT022 — Build por nodo

No se exige que dos nodos generen el mismo hash.

Se exige que:

- cada nodo/sesión use el build manifest correcto;
- el hash del catálogo corresponda al binario del nodo actual;
- no se mezcle telemetría de builds distintos como si fueran la misma condición sin metadata diferenciada.

---

## 5.5 `argv` y resolución del comando

### CAT023 — `argv` obligatorio

- Debe existir.
- Debe ser una lista, aunque esté vacía.
- Cada elemento debe ser string.
- No se aceptan números, booleanos, mappings ni listas anidadas.
- Cada string debe estar libre de byte NUL.

### CAT024 — Prohibición de comandos de shell

No aceptar una cadena única como:

```yaml
argv: "--class S && rm -rf /tmp/x"
```

El formato válido es:

```yaml
argv: ["--class", "S"]
```

`resolve_exec_spec()` debe devolver executable, argv y entorno por separado. No debe concatenar una línea de comandos, usar `shlex.split`, `shell=True`, `/bin/sh -c` o expansión de shell.

### CAT025 — Argumentos deterministas

- Con la misma entrada se produce exactamente el mismo `argv`.
- No ordenar argumentos.
- No eliminar duplicados.
- No expandir globbing, `~`, variables o sustituciones de comando.

---

## 5.6 Variables de entorno

### CAT026 — `environment` tipado

- Opcional; si falta, se normaliza a `{}`.
- Debe ser mapping `str -> str`.
- No convertir números o booleanos silenciosamente a string.
- Claves válidas: `^[A-Za-z_][A-Za-z0-9_]*$`.
- Clave o valor con byte NUL: rechazo.

### CAT027 — Entorno explícito y auditable

`catalog.py` devuelve únicamente las variables declaradas. `runner.py` decide cómo fusionarlas con un entorno base mínimo y registrado.

No debe permitir reemplazar silenciosamente variables reservadas, por ejemplo:

```text
HYPERION_RUN_ID
HYPERION_CAMPAIGN_ID
HYPERION_ROI_FD
```

### CAT028 — NPB-OMP

Para `suite=NPB-OMP`, declarar explícitamente:

```text
OMP_NUM_THREADS
OMP_PROC_BIND
OMP_PLACES
OMP_DYNAMIC
```

Validaciones mínimas:

- `OMP_NUM_THREADS`: entero positivo representado como string;
- `OMP_DYNAMIC`: `true` o `false`;
- `OMP_PROC_BIND`: no vacío;
- `OMP_PLACES`: no vacío.

No completar valores por defecto silenciosamente.

### CAT029 — NPB-SER

No debe presentarse como equivalente a NPB-OMP. Si declara variables OpenMP, debe producir error estricto o advertencia bloqueante para training/final.

---

## 5.7 Tamaño, runtime y warmup

### CAT030 — `size_variant`

Obligatorio para `role=dataset`.

Para NPB, conjunto inicial:

```text
S
W
A
B
C
```

Para suites no NPB puede ser otro identificador discreto validado por su adaptador.

### CAT031 — `expected_runtime_seconds`

- Obligatorio para toda entrada ejecutable real.
- Número finito mayor que cero.
- No aceptar booleanos, `NaN`, infinito ni cero.
- Alimenta el dry-run y la validación de `timeouts_seconds.run`.

### CAT032 — `warmup_seconds`

- Si falta, se normaliza a `0.0`.
- Debe ser finito y mayor o igual que cero.
- Debe ser menor que `expected_runtime_seconds`.
- En binarios externos con `roi_mode=process_lifetime`, debe declararse si se excluye startup por tiempo.
- No interpretar warmup como número de iteraciones internas.

---

## 5.8 `success_check`

### CAT033 — Lista obligatoria

- Debe ser una lista no vacía.
- La v3.1 canónica usa lista; no aceptar mapping único silenciosamente.
- Compatibilidad antigua, si existe, debe emitir deprecación y no aplicarse a training/final sin migración.

### CAT034 — Tipos soportados

```text
exit_code
stdout_regex
artifact_exists
```

No aceptar tipos no definidos como `contains`, `script`, `python` o `stderr_regex`.

### CAT035 — `exit_code`

```yaml
- type: exit_code
  expected: 0
```

- `expected` obligatorio;
- entero real, no booleano;
- campos permitidos: `type`, `expected`;
- rango recomendado `0..255`.

### CAT036 — `stdout_regex`

```yaml
- type: stdout_regex
  pattern: "VERIFICATION SUCCESSFUL"
```

- `pattern` obligatorio, string no vacío;
- debe compilar con `re.compile`;
- campos permitidos: `type`, `pattern`;
- no evaluar el patrón durante la carga.

### CAT037 — `artifact_exists`

```yaml
- type: artifact_exists
  path: "results/verification.txt"
```

- `path` obligatorio y no vacío;
- relativo al output de la corrida;
- no absoluto;
- no puede escapar mediante `..`;
- la existencia se evalúa después de ejecutar.

### CAT038 — Semántica AND

Todos los checks deben cumplirse. No implementar OR implícito.

### CAT039 — Checks duplicados o contradictorios

Rechazar dos exit codes distintos o duplicados exactos. Se permite más de un `stdout_regex` si todos deben cumplirse.

### CAT040 — Reglas específicas por suite

#### STREAM

Debe incluir:

- `exit_code`;
- al menos un `stdout_regex` que pruebe salida parseable, por ejemplo `Triad:`;
- `reports_bandwidth_stdout: true`.

#### ERT

Debe incluir:

- `exit_code`;
- `reports_flops_stdout: true`;
- adaptador/parser registrado para FLOP/s.

#### NPB-SER / NPB-OMP

Debe incluir:

- `exit_code: 0`;
- `stdout_regex` para `VERIFICATION SUCCESSFUL` cuando la suite lo soporte.

No basta con código cero si existe verificación interna.

#### synthetic

Debe incluir al menos `exit_code`; puede añadir regex de protocolo.

### CAT041 — Separación configuración/evaluación

`catalog.py` valida la configuración. `runner.py` evalúa el resultado real. Una regex válida pero ausente en stdout es fallo de corrida, no error de parseo del catálogo.

---

## 5.9 ROI

### CAT042 — `roi_mode` obligatorio

```text
instrumented
process_lifetime
synthetic_ready_go
```

### CAT043 — Coherencia por rol

| Rol | Modos permitidos |
|---|---|
| `calibration` | `process_lifetime`, `instrumented` |
| `dataset` | `instrumented`, `process_lifetime` |
| `synthetic` | `synthetic_ready_go`, `instrumented` |

La aceptación de `process_lifetime` en una campaña final se decide con M17.

### CAT044 — `synthetic_ready_go`

Solo si el ejecutable implementa el protocolo sintético `ready/go`. No usar para binarios externos caja negra.

### CAT045 — `instrumented`

Debe representar soporte real de marcadores ROI/phase. `catalog.py` debe exigir un adaptador o capability registrada; no asumirlo por nombre.

### CAT046 — `process_lifetime`

Significa ROI igual a vida del proceso, quizá con exclusión temporal de startup. No prueba transiciones internas.

---

## 5.10 Hints y salidas reportadas

### CAT047 — `phase_label_hint`

- Opcional.
- Solo permitido para `role=dataset`.
- Es auditoría/prior de literatura.
- Nunca es `phase_label_train` ni feature.

Valores recomendados:

```text
compute_bound
memory_bound
mixed
unknown
```

### CAT048 — Flags `reports_*`

`reports_bandwidth_stdout` y `reports_flops_stdout` deben ser booleanos reales.

### CAT049 — Coherencia de reportes

- STREAM: bandwidth true.
- ERT: flops true.
- Un flag declarado exige parser registrado.
- Significa que el binario informa la magnitud por stdout.
- No significa que perf la mida automáticamente.
- No depender de un contador PMU universal de FLOPs.

### CAT050 — Calibración

Una entrada de calibración debe aportar al menos una capacidad conocida: bandwidth, flops o referencias. La obligación de que la campaña tenga una fuente de cada tipo pertenece a M06.

---

## 5.11 Coherencia transversal

### CAT051 — Identidad experimental distinta

No colapsar:

- NPB-SER y NPB-OMP;
- clases diferentes;
- builds diferentes;
- argumentos distintos;
- entornos OpenMP distintos.

### CAT052 — Duplicados semánticos

Si dos entradas tienen exactamente suite, role, realpath, argv, environment, size, hash y ROI, pero IDs distintos, emitir error o advertencia estricta.

### CAT053 — Inmutabilidad

Entradas y mappings deben ser de solo lectura después de validar.

### CAT054 — Orden determinista

Preservar orden declarado para reportes, pero resolver por ID sin depender del orden. Serialización canónica determinista.

### CAT055 — Digest del catálogo

Calcular opcionalmente:

```text
catalog_digest = sha256(serialización canónica del catálogo validado)
```

Debe incluir campos semánticos y excluir timestamps del filesystem.

---

# 6. API pública requerida

```python
def load_catalog(catalog_path: str | Path, *, strict: bool = True) -> Catalog:
    """Carga, valida, normaliza rutas y construye Catalog sin ejecutar binarios."""


def validate_catalog(catalog: Catalog) -> list[CheckResult]:
    """Ejecuta reglas estáticas y devuelve todos los errores posibles."""


def get_entry(catalog: Catalog, kernel_ref: str) -> KernelEntry:
    """Resuelve exactamente una entrada o produce error compatible con M05."""


def verify_binary(entry: KernelEntry) -> list[CheckResult]:
    """Ejecuta C01 y C02 sin modificar ni ejecutar el binario."""


def verify_build_manifest(entry: KernelEntry) -> list[CheckResult]:
    """Valida schema y coherencia del build manifest."""


def validate_success_checks(entry: KernelEntry) -> list[CheckResult]:
    """Ejecuta C03 estático."""


def validate_entry_coherence(entry: KernelEntry) -> list[CheckResult]:
    """Ejecuta C04."""


def resolve_exec_spec(entry: KernelEntry) -> ExecSpec:
    """Devuelve executable, argv y entorno por separado; nunca una cadena shell."""


def compute_catalog_digest(catalog: Catalog) -> str:
    """SHA-256 de representación canónica determinista."""
```

---

# 7. Modelo de errores

Cada error debe contener:

```text
code
factor_id
catalog_path
entry_id
field_path
declared_value
message
suggestion
blocking
```

Ejemplo:

```json
{
  "code": "CAT036_INVALID_REGEX",
  "factor_id": "C03",
  "catalog_path": "kernels/catalog.yaml",
  "entry_id": "npb_mg_omp_s",
  "field_path": "success_check[1].pattern",
  "declared_value": "(VERIFICATION",
  "message": "La expresión regular no compila: missing ), unterminated subpattern.",
  "suggestion": "Corrija el patrón antes de ejecutar la campaña.",
  "blocking": true
}
```

No mostrar stack traces como mensaje principal.

---

# 8. Orden de validación recomendado

```text
1. filesystem del catalog.yaml
2. parseo YAML seguro
3. schema raíz
4. catalog_version
5. estructura de kernels
6. IDs y duplicados
7. parseo tipado de cada entrada
8. C04: coherencia semántica
9. C03: success checks
10. build_manifest_ref y schema
11. C01: binario
12. C02: checksum
13. digest canónico
```

Si C01 falla, C02 debe quedar `skipped` por dependencia, no producir un error confuso adicional.

---

# 9. Integración con `manifest.py`

Reglas cruzadas:

1. todo `kernel_ref` existe;
2. cada referencia resuelve una sola entrada;
3. `calibration` referencia solo `role=calibration`;
4. `kernels` referencia solo `role=dataset`, salvo smoke explícito;
5. Roofline incluye una fuente de BW y una de cómputo;
6. timeout run no es menor que runtime esperado sin justificación;
7. ROI del manifest es compatible con `roi_mode`;
8. synthetic y calibration no generan dataset final;
9. hints no se promueven a etiqueta.

---

# 10. Integración con `runner.py`

`runner.py` recibe un `ExecSpec`, no una cadena.

```python
subprocess.Popen(
    exec_spec.argv,
    executable=str(exec_spec.executable),
    env=merged_environment,
    shell=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=False,
)
```

Debe:

- capturar stdout/stderr sin deadlocks;
- propagar exit status y señal;
- no reinterpretar argv;
- evaluar todos los checks;
- registrar hash efectivo inmediatamente antes de ejecutar;
- rechazar si el hash cambió desde preflight.

---

# 11. Pruebas mínimas obligatorias

## 11.1 Válidos

1. STREAM válido.
2. ERT válido.
3. NPB-SER EP S válido.
4. NPB-OMP MG S con entorno completo.
5. Sintético con ready/go.
6. Varias regex AND.
7. Rutas relativas resueltas desde el catálogo.

## 11.2 Estructura

8. YAML vacío.
9. YAML inválido.
10. raíz no mapping.
11. versión ausente/no soportada.
12. kernels ausente/no lista.
13. entrada no mapping.
14. campo desconocido.
15. ID duplicado/inseguro.

## 11.3 Binarios/builds

16. binario faltante.
17. ruta directorio.
18. no ejecutable.
19. symlink roto.
20. checksum mal formado/incorrecto.
21. binario modificado después del preflight.
22. build manifest faltante/inválido.
23. hash de build distinto.
24. build de otro nodo accidental.

## 11.4 argv/environment

25. argv string.
26. elemento no string.
27. byte NUL.
28. operadores shell no se ejecutan.
29. environment inválido.
30. nombre de variable inválido.
31. NPB-OMP sin variables obligatorias.
32. NPB-SER con OMP contradictorio.

## 11.5 Roles/tamaños

33. suite/role desconocido.
34. STREAM como dataset.
35. NPB como calibration.
36. dataset sin size.
37. clase inválida.
38. runtime cero/negativo/NaN.
39. warmup negativo o >= runtime.
40. hint en calibration.
41. prueba que impide usar hint como target.

## 11.6 Success checks

42. lista vacía.
43. tipo desconocido.
44. exit sin expected.
45. expected booleano.
46. regex ausente/vacía/inválida.
47. artifact absoluto o con `..`.
48. exit codes contradictorios.
49. NPB sin verificación textual.
50. STREAM sin salida parseable.
51. un check falla y el resultado AND falla.

## 11.7 ROI/reportes

52. roi desconocido.
53. synthetic_ready_go en NPB.
54. report flag no booleano.
55. STREAM sin bandwidth.
56. ERT sin flops.
57. flag sin parser.
58. calibration sin capacidad.
59. process_lifetime incompatible con final.

## 11.8 Propiedades

60. resolve nunca usa shell.
61. resultado determinista.
62. objetos inmutables.
63. digest estable.
64. digest cambia con campo semántico.
65. C01/C02 repetibles por corrida.
66. se reportan varios errores independientes.

---

# 12. Criterios de aceptación

La tarea termina cuando:

1. existe `schemas/kernel_catalog.schema.json`;
2. se implementan las APIs públicas;
3. CAT001–CAT055 tienen cobertura;
4. C01–C04 producen `CheckResult`;
5. binario faltante aborta antes de ejecutar;
6. mutación entre preflight y run se detecta;
7. regex inválida falla en validate;
8. NPB exit 0 sin verificación falla en runtime;
9. STREAM/ERT solo calibración;
10. NPB-SER y NPB-OMP diferenciados;
11. hint nunca llega al target;
12. no hay shell, `system()` ni `/bin/sh -c`;
13. argv y entorno son deterministas;
14. build manifest, hash y catálogo quedan enlazados en metadata;
15. pytest cubre válidos, inválidos y mutación;
16. errores muestran código, campo y sugerencia;
17. `plan` resuelve binarios sin ejecutarlos;
18. preflight completo y reducido reutilizan C01/C02.

---

# 13. Decisiones explícitas

1. Rol sintético canónico: `synthetic`.
2. Success checks: AND.
3. Tipos iniciales: exit code, stdout regex y artifact exists.
4. Rutas relativas: respecto al catálogo.
5. Shell: prohibido.
6. Checksum: SHA-256 con prefijo.
7. Builds multinodo: hashes distintos permitidos si están documentados por nodo.
8. Hints: no son etiquetas.
9. Warmup externo: segundos de pared.
10. Process lifetime no prueba transiciones.
11. NPB: exit 0 no basta si hay verificación interna.
12. STREAM/ERT reportan por stdout; perf no los sustituye.
13. C01/C02 se repiten por corrida.
14. Campos desconocidos nunca se ignoran.

---

# 14. Prompt breve para el agente programador

```yaml
task_id: HYP-CATALOG-01
role: senior_python_systems_engineer
objective: >
  Implementar orchestrator/catalog.py y schemas/kernel_catalog.schema.json
  siguiendo íntegramente docs/catalog_py_reglas_validacion.md.
requirements:
  - Cargar YAML con loader seguro.
  - Validar CAT001-CAT055.
  - Implementar C01-C04 con CheckResult estructurado.
  - Usar dataclasses inmutables.
  - Resolver rutas relativas desde catalog.yaml.
  - Verificar SHA-256 contra catálogo y build manifest.
  - Resolver ExecSpec como argv estructurado, nunca shell.
  - Validar success_check como lista AND.
  - Diferenciar NPB-SER y NPB-OMP.
  - No usar phase_label_hint como target.
  - Revalidar binario/hash antes de cada corrida.
deliverables:
  - orchestrator/catalog.py
  - schemas/kernel_catalog.schema.json
  - orchestrator/tests/test_catalog.py
  - examples/kernels/catalog.valid.yaml
  - examples/kernels/catalog.invalid.yaml
tests:
  - Ejecutar todos los casos de la sección 11.
  - Incluir fake executables para exit code, stdout y artefactos.
  - Incluir test que modifica el binario después del preflight.
acceptance:
  - pytest completo en verde.
  - validate y plan no ejecutan binarios.
  - Ningún uso de shell=True, system() o /bin/sh -c.
  - Todos los errores incluyen code, factor_id, entry_id, field_path y suggestion.
```
