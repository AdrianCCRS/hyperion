"""Modelos analíticos de trabajo CUDA por lanzamiento.

Este registro es la nueva fuente de verdad para etiquetar ventanas GPU. Los
valores históricos de ``operational_intensity_flops_per_byte`` medidos con
``ncu`` no son un fallback: si un kernel de dataset no tiene modelo registrado,
la preparación de la campaña final debe fallar de forma explícita.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class AnalyticWork:
    flops: float
    bytes_moved: float
    model_id: str
    assumptions: str

    @property
    def operational_intensity(self) -> float:
        return self.flops / self.bytes_moved


ModelFn = Callable[[str, int, Mapping[str, int | float | str]], AnalyticWork]
OperationModelFn = Callable[[Sequence[tuple[str, str, int]], int, Mapping[str, int | float | str]], AnalyticWork]


@dataclass(frozen=True)
class OperationSpec:
    """Regla para una operación CUDA que activa varios eventos Activity."""

    events_per_operation: int
    work: OperationModelFn
    allow_unmatched_prefix: bool = False

    def __post_init__(self) -> None:
        if self.events_per_operation <= 0:
            raise ValueError("events_per_operation debe ser positivo")


def gaussian_work(
    kernel_name: str,
    launch_index: int,
    parameters: Mapping[str, int | float | str],
) -> AnalyticWork:
    """Trabajo de Fan1/Fan2 en el paso ``t`` de eliminación gaussiana.

    Rodinia lanza exactamente ``Fan1, Fan2`` por cada ``t=0..N-2``. Con
    ``n = N-1-t``, Fan1 procesa ``n`` filas y Fan2 una región
    ``n x (n+1)``. Los bytes son el tráfico algorítmico mínimo FP32: cada
    arreglo independiente se lee/escribe una vez por lanzamiento, suponiendo
    reutilización ideal del pivote, la fila pivote y el multiplicador de cada
    fila. Por eso es AOI analítica, no tráfico DRAM observado.
    """
    try:
        size = int(parameters["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("rodinia_gaussian requiere parameters['size'] entero") from exc
    if size < 2:
        raise ValueError("rodinia_gaussian requiere size >= 2")
    if launch_index < 0:
        raise ValueError("launch_index debe ser no negativo")

    step = launch_index // 2
    if step >= size - 1:
        raise ValueError(
            f"launch_index={launch_index} excede los {2 * (size - 1)} lanzamientos esperados"
        )
    n = size - 1 - step
    fp32_bytes = 4
    if "Fan1" in kernel_name:
        # m[row,t] = a[row,t] / a[t,t]: una división por fila.
        flops = float(n)
        # a[row,t] + m[row,t] por fila, más el pivote a[t,t] compartido.
        bytes_moved = float(fp32_bytes * (2 * n + 1))
        model_id = "rodinia_gaussian/fan1/aoi-v1"
    elif "Fan2" in kernel_name:
        # A -= m*A_pivot (2 FLOP por celda) y, para y=0, b -= m*b_pivot
        # (2 FLOP adicionales por fila).
        flops = float(2 * n * (n + 1) + 2 * n)
        # A objetivo read+write; m por fila; fila pivote por columna;
        # b objetivo read+write por fila; b pivote una vez. m ya fue contado.
        bytes_moved = float(
            fp32_bytes * (2 * n * (n + 1) + n + (n + 1) + 2 * n + 1)
        )
        model_id = "rodinia_gaussian/fan2/aoi-v1"
    else:
        raise ValueError(f"kernel CUDA inesperado para rodinia_gaussian: {kernel_name!r}")
    return AnalyticWork(
        flops=flops,
        bytes_moved=bytes_moved,
        model_id=model_id,
        assumptions="FP32; bytes algorítmicos mínimos con reutilización ideal intra-lanzamiento",
    )


def _positive_int(parameters: Mapping[str, int | float | str], name: str, *, minimum: int = 1) -> int:
    try:
        value = int(parameters[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"modelo analítico requiere parameters[{name!r}] entero") from exc
    if value < minimum:
        raise ValueError(f"parameters[{name!r}] debe ser >= {minimum}")
    return value


def stencil_work(kernel_name: str, launch_index: int, parameters: Mapping[str, int | float | str]) -> AnalyticWork:
    """Jacobi 2D FP64 de ``stencil_gpu_dispatch.cu`` por lanzamiento."""
    if "jacobi_kernel" not in kernel_name:
        raise ValueError(f"kernel CUDA inesperado para stencil: {kernel_name!r}")
    n = _positive_int(parameters, "size", minimum=3)
    cells = (n - 2) ** 2
    return AnalyticWork(
        flops=float(5 * cells),
        bytes_moved=float(6 * 8 * cells),
        model_id="dual_stencil/fp64-5point-aoi-v1",
        assumptions="FP64; cuatro vecinos leídos y una salida escrita por celda interior",
    )


def axpy_work(kernel_name: str, launch_index: int, parameters: Mapping[str, int | float | str]) -> AnalyticWork:
    """AXPY FP64 de Dual-AXPY, un lanzamiento CUDA por iteración.

    La traza de validación del ELF directo identifica el kernel como
    ``axpy_kernel_val``.  ``y = a*x + y`` ejecuta una multiplicación y una
    suma por elemento; el denominador cuenta ``x`` y ``y`` leídos, y ``y``
    escrito, una vez por operación lógica.
    """
    if "axpy_kernel_val" not in kernel_name:
        raise ValueError(f"kernel CUDA inesperado para Dual-AXPY: {kernel_name!r}")
    n = _positive_int(parameters, "size")
    return AnalyticWork(
        flops=float(2 * n),
        bytes_moved=float(3 * n * 8),
        model_id="dual_axpy/fp64-aoi-v1",
        assumptions="FP64; x e y leídos y y escrito una vez por AXPY",
    )


def spmv_operation_work(
    events: Sequence[tuple[str, str, int]],
    operation_index: int,
    parameters: Mapping[str, int | float | str],
) -> AnalyticWork:
    """Un despacho completo CSR de Dual-SpMV, incluida E/S host--device.

    El código fuente transfiere ``row_ptr``, ``col_idx``, ``values`` y ``x``;
    cuSPARSE despacha dos kernels (partición y CSR-MV), y finalmente se baja
    ``y``. CUPTI debe mostrar esos siete eventos en ese orden. La fórmula se
    asigna una sola vez al grupo, nunca una vez a cada kernel interno.
    """
    n = _positive_int(parameters, "size")
    nnz = 7 * n
    expected = (
        4 * (n + 1),  # row_ptr H2D
        4 * nnz,      # col_idx H2D
        8 * nnz,      # values H2D
        8 * n,        # x H2D
        0, 0,          # kernels internos cuSPARSE
        8 * n,        # y D2H
    )
    if len(events) != len(expected):
        raise ValueError(f"Dual-SpMV requiere {len(expected)} eventos por operación, recibió {len(events)}")
    kinds = tuple(kind for kind, _, _ in events)
    if kinds != ("memcpy", "memcpy", "memcpy", "memcpy", "kernel", "kernel", "memcpy"):
        raise ValueError(f"secuencia CUPTI inesperada para Dual-SpMV: {kinds!r}")
    names = (events[4][1], events[5][1])
    # Nombres reales confirmados en paccaA100 (job 7588, 2026-09-23):
    # cusparse::binary_search_partition_kernel<...> y
    # cusparse::load_balancing_kernel<...> -- los nombres originales
    # (csr_partition_kernel/csrmv_v3_kernel) eran de otra version de
    # cuSPARSE y nunca coincidieron con el binario real de este nodo.
    if "binary_search_partition_kernel" not in names[0] or "load_balancing_kernel" not in names[1]:
        raise ValueError(f"kernels internos inesperados para Dual-SpMV: {names!r}")
    observed_transfers = tuple(bytes_moved for _, _, bytes_moved in events)
    if observed_transfers != expected:
        raise ValueError(
            "bytes CUPTI inesperados para Dual-SpMV; "
            f"observado={observed_transfers!r}, esperado={expected!r}"
        )
    return AnalyticWork(
        flops=float(2 * nnz),
        bytes_moved=float(sum(expected)),
        model_id="dual_spmv/fp64-csr7-dispatch-aoi-v1",
        assumptions="FP64 CSR con 7 NNZ/fila; cuatro H2D, cuSPARSE y una D2H por despacho",
    )


def cholesky_operation_work(
    events: Sequence[tuple[str, str, int]],
    operation_index: int,
    parameters: Mapping[str, int | float | str],
) -> AnalyticWork:
    """Una factorización ``cusolverDnDpotrf`` completa de Dual-Cholesky."""
    n = _positive_int(parameters, "size")
    matrix_bytes = 8 * n * n
    expected = (matrix_bytes, 0, 0, 4, matrix_bytes)
    if len(events) != len(expected):
        raise ValueError(f"Dual-Cholesky requiere {len(expected)} eventos por operación, recibió {len(events)}")
    kinds = tuple(kind for kind, _, _ in events)
    if kinds != ("memcpy", "kernel", "kernel", "memcpy", "memcpy"):
        raise ValueError(f"secuencia CUPTI inesperada para Dual-Cholesky: {kinds!r}")
    names = (events[1][1], events[2][1])
    if "set_info" not in names[0] or "getrf_wo_pivot" not in names[1]:
        raise ValueError(f"kernels internos inesperados para Dual-Cholesky: {names!r}")
    observed_transfers = tuple(bytes_moved for _, _, bytes_moved in events)
    if observed_transfers != expected:
        raise ValueError(
            "bytes CUPTI inesperados para Dual-Cholesky; "
            f"observado={observed_transfers!r}, esperado={expected!r}"
        )
    return AnalyticWork(
        flops=float(n ** 3 / 3),
        bytes_moved=float(sum(expected)),
        model_id="dual_cholesky/fp64-potrf-dispatch-aoi-v1",
        assumptions="FP64 potrf; H2D de A, estado de cuSOLVER, D2H de info y A factorizada",
    )


def fft_operation_work(
    events: Sequence[tuple[str, str, int]],
    operation_index: int,
    parameters: Mapping[str, int | float | str],
) -> AnalyticWork:
    """Una FFT Z2Z 2D completa de Dual-FFT, con sus transferencias."""
    n = _positive_int(parameters, "size")
    data_bytes = 16 * n * n  # cufftDoubleComplex
    if len(events) != 5:
        raise ValueError(f"Dual-FFT requiere 5 eventos por operación, recibió {len(events)}")
    kinds = tuple(kind for kind, _, _ in events)
    if kinds != ("memcpy", "kernel", "kernel", "kernel", "memcpy"):
        raise ValueError(f"secuencia CUPTI inesperada para Dual-FFT: {kinds!r}")
    names = tuple(name for _, name, _ in events[1:4])
    if not ("regular_fft" in names[0] and "regular_fft" in names[1] and "vector_fft" in names[2]):
        raise ValueError(f"kernels internos inesperados para Dual-FFT: {names!r}")
    observed_transfers = tuple(bytes_moved for _, _, bytes_moved in events)
    expected = (data_bytes, 0, 0, 0, data_bytes)
    if observed_transfers != expected:
        raise ValueError(
            "bytes CUPTI inesperados para Dual-FFT; "
            f"observado={observed_transfers!r}, esperado={expected!r}"
        )
    elements = n * n
    return AnalyticWork(
        flops=float(5 * elements * math.log2(elements)),
        bytes_moved=float(sum(expected)),
        model_id="dual_fft/fp64-z2z2d-dispatch-aoi-v1",
        assumptions="FFT compleja FP64 2D; H2D y D2H de N² cufftDoubleComplex",
    )


def kmeans_operation_work(
    events: Sequence[tuple[str, str, int]],
    operation_index: int,
    parameters: Mapping[str, int | float | str],
) -> AnalyticWork:
    """Una iteración de asignación de Rodinia k-means CUDA.

    ``kmeansPoint`` recorre todos los centros y dimensiones; cada distancia
    aplica resta, producto y acumulación. La transposición inicial de datos
    (`invert_mapping`) ocurre antes del bucle y se descarta mediante el
    prefijo no emparejado, nunca se etiqueta como una iteración k-means.
    """
    n_points = _positive_int(parameters, "n_points")
    n_features = _positive_int(parameters, "n_features")
    n_clusters = _positive_int(parameters, "n_clusters")
    membership_bytes = 4 * n_points
    cluster_bytes = 4 * n_clusters * n_features
    expected = (membership_bytes, cluster_bytes, 0, membership_bytes)
    if len(events) != len(expected):
        raise ValueError(f"Rodinia k-means requiere {len(expected)} eventos por iteración, recibió {len(events)}")
    kinds = tuple(kind for kind, _, _ in events)
    if kinds != ("memcpy", "memcpy", "kernel", "memcpy"):
        raise ValueError(f"secuencia CUPTI inesperada para Rodinia k-means: {kinds!r}")
    if "kmeansPoint" not in events[2][1]:
        raise ValueError(f"kernel inesperado para Rodinia k-means: {events[2][1]!r}")
    observed_transfers = tuple(bytes_moved for _, _, bytes_moved in events)
    if observed_transfers != expected:
        raise ValueError(
            "bytes CUPTI inesperados para Rodinia k-means; "
            f"observado={observed_transfers!r}, esperado={expected!r}"
        )
    return AnalyticWork(
        flops=float(3 * n_points * n_features * n_clusters),
        bytes_moved=float(4 * (n_points * n_features + n_clusters * n_features + 2 * n_points)),
        model_id="rodinia_kmeans/fp32-assignment-aoi-v1",
        assumptions="FP32; resta+producto+suma por punto-centro-dimensión; datos y centros únicos, membresía leída/escrita",
    )


def lud_work(kernel_name: str, launch_index: int, parameters: Mapping[str, int | float | str]) -> AnalyticWork:
    """Trabajo por lanzamiento del LUD bloqueado FP32 de Rodinia.

    ``lud_cuda`` despacha, para cada bloque diagonal de 16x16, un kernel
    diagonal, ``q`` perímetros y ``q²`` bloques internos; al final sólo queda
    la diagonal. La posición se recupera del índice CUPTI: el binario validado
    El cargador asigna un ordinal sólo entre kernels y por tanto el primer
    ``lud_diagonal`` es el índice 0, independientemente de copias CUDA de
    setup. FLOPs y bytes se cuentan directamente de
    ``lud_kernel.cu``: accesos globales de cada kernel, no bytes DRAM.
    """
    n = _positive_int(parameters, "size", minimum=16)
    block = 16
    if n % block:
        raise ValueError("rodinia_lud requiere size múltiplo de 16")
    relative = launch_index
    blocks = n // block
    if relative < 0:
        raise ValueError(f"índice CUPTI inesperado antes de LUD: {launch_index}")
    step, phase = divmod(relative, 3)
    is_last_diagonal = step == blocks - 1 and phase == 0
    if step >= blocks or (step == blocks - 1 and phase != 0):
        raise ValueError(f"índice CUPTI excede la secuencia LUD N={n}: {launch_index}")
    q = blocks - step - 1
    lower_upper_flops = sum((block - 1 - i) * (4 * i + 3) for i in range(block - 1))
    if "lud_diagonal" in kernel_name:
        if phase != 0:
            raise ValueError(f"fase CUPTI/nombre LUD inconsistentes: {launch_index}, {kernel_name!r}")
        return AnalyticWork(
            flops=float(lower_upper_flops),
            bytes_moved=float(4 * (block * block + block * (block - 1))),
            model_id="rodinia_lud/fp32-block16-diagonal-aoi-v1",
            assumptions="Rodinia LUD BLOCK_SIZE=16; cargas/escrituras globales explícitas de lud_diagonal",
        )
    if "lud_perimeter" in kernel_name:
        if phase != 1 or q <= 0:
            raise ValueError(f"fase CUPTI/nombre LUD inconsistentes: {launch_index}, {kernel_name!r}")
        # Por bloque: 16 columnas de fila y 16 de columna; la segunda suma
        # además contiene una división por pivote para cada elemento.
        flops_per_block = 16 * (2 * sum(range(1, block))) + 16 * (2 * sum(range(block)) + block)
        global_elements = 3 * block * block + (block - 1) * block + block * block
        return AnalyticWork(
            flops=float(q * flops_per_block),
            bytes_moved=float(4 * q * global_elements),
            model_id="rodinia_lud/fp32-block16-perimeter-aoi-v1",
            assumptions="Rodinia LUD BLOCK_SIZE=16; q bloques perímetro y accesos globales explícitos",
        )
    if "lud_internal" in kernel_name:
        if phase != 2 or q <= 0:
            raise ValueError(f"fase CUPTI/nombre LUD inconsistentes: {launch_index}, {kernel_name!r}")
        # 16 FMA (32 FLOPs) y una resta por celda; dos cargas y una escritura.
        return AnalyticWork(
            flops=float(q * q * block * block * (2 * block + 1)),
            bytes_moved=float(4 * q * q * 3 * block * block),
            model_id="rodinia_lud/fp32-block16-internal-aoi-v1",
            assumptions="Rodinia LUD BLOCK_SIZE=16; 16 FMA y resta por celda, dos cargas y una escritura globales",
        )
    raise ValueError(f"kernel CUDA inesperado para Rodinia LUD: {kernel_name!r}")


def minibude_work(kernel_name: str, launch_index: int, parameters: Mapping[str, int | float | str]) -> AnalyticWork:
    """Una evaluación ``fasten_main`` del modelo CUDA de miniBUDE.

    El propio benchmark reporta 40 FLOPs por interacción proteína--ligando--
    pose. El denominador es la huella algorítmica única de una evaluación:
    seis transformaciones y una energía por pose, más moléculas y forcefield.
    Es deliberadamente AOI (reutilización de los datos comunes), no tráfico
    HBM observado. ``Atom`` y ``FFParams`` son estructuras packed de 16 B.
    """
    # El ejecutable bm1 también despacha al inicio ``version(int*)``
    # (mangled como ``_Z7versionPi``). Es una escritura de un único entero
    # para informar la versión del benchmark; no forma parte del cálculo de
    # docking. Se conserva como trabajo nulo con sus 4 B explícitos en vez de
    # descartarlo, para que la cobertura de la traza siga siendo 1:1.
    if kernel_name == "_Z7versionPi" or kernel_name.startswith("version("):
        return AnalyticWork(
            flops=0.0,
            bytes_moved=4.0,
            model_id="minibude/cuda-version-int-store-v1",
            assumptions="Kernel auxiliar version(int*): una escritura global de int; sin FLOPs de docking",
        )
    if "fasten_main" not in kernel_name:
        raise ValueError(f"kernel CUDA inesperado para miniBUDE: {kernel_name!r}")
    poses = _positive_int(parameters, "poses")
    proteins = _positive_int(parameters, "proteins")
    ligands = _positive_int(parameters, "ligands")
    forcefields = _positive_int(parameters, "forcefields")
    flops = 40 * poses * proteins * ligands
    bytes_moved = 4 * (6 * poses + poses) + 16 * (proteins + ligands + forcefields)
    return AnalyticWork(
        flops=float(flops),
        bytes_moved=float(bytes_moved),
        model_id="minibude/cuda-fasten-aoi-v1",
        assumptions="40 FLOPs por interacción declarados por miniBUDE; entradas únicas por evaluación y reutilización ideal",
    )


def heartwall_work(kernel_name: str, launch_index: int, parameters: Mapping[str, int | float | str]) -> AnalyticWork:
    """Cota analítica conservadora de un frame de Heartwall CUDA.

    El kernel monolítico procesa 51 puntos. Su término dominante, visible en
    el código, es la convolución directa de un template ``T×T`` sobre la
    ventana ``S×S``: por punto ejecuta ``T²·S²`` multiplicaciones y el mismo
    número de acumulaciones. Se cuentan además una vez los tres arreglos
    persistentes de esa etapa (template, ventana y convolución) y el frame
    H2D. Esto es una cota inferior de FLOPs y una huella algorítmica, no una
    reconstrucción de tráfico HBM; basta para una etiqueta Roofline robusta
    porque queda muy por encima del ridge FP32 incluso antes de las demás
    etapas del rastreador.
    """
    if "kernel" not in kernel_name.lower():
        raise ValueError(f"kernel CUDA inesperado para Rodinia Heartwall: {kernel_name!r}")
    points = _positive_int(parameters, "points")
    template_side = _positive_int(parameters, "template_side")
    search_side = _positive_int(parameters, "search_side")
    frame_bytes = _positive_int(parameters, "frame_bytes")
    template_elements = template_side * template_side
    search_elements = search_side * search_side
    convolution_elements = (template_side + search_side - 1) ** 2
    return AnalyticWork(
        flops=float(2 * points * template_elements * search_elements),
        bytes_moved=float(frame_bytes + 4 * points * (template_elements + search_elements + convolution_elements)),
        model_id="rodinia_heartwall/fp32-dominant-convolution-aoi-lowerbound-v1",
        assumptions="Cota inferior: sólo MACs de convolución directa; frame H2D y huella única template/ventana/salida por punto",
    )


def cutlass_dgemm_work(kernel_name: str, launch_index: int, parameters: Mapping[str, int | float | str]) -> AnalyticWork:
    """GEMM SIMT FP64 cuadrado de CUTLASS, un lanzamiento por iteración."""
    if "cutlass" not in kernel_name.lower():
        raise ValueError(f"kernel CUDA inesperado para CUTLASS DGEMM: {kernel_name!r}")
    n = _positive_int(parameters, "size")
    return AnalyticWork(
        flops=float(2 * n ** 3),
        bytes_moved=float(3 * n ** 2 * 8),
        model_id="cutlass_simt_dgemm/fp64-aoi-v1",
        assumptions="FP64; A, B y C contados una vez por GEMM (reutilización ideal intra-lanzamiento)",
    )


def cutlass_conv2d_work(kernel_name: str, launch_index: int, parameters: Mapping[str, int | float | str]) -> AnalyticWork:
    """Conv2D fprop FP32 NHWC/KRSC de ``cutlass_simt_conv2d_bench.cu``."""
    if "cutlass" not in kernel_name.lower():
        raise ValueError(f"kernel CUDA inesperado para CUTLASS Conv2D: {kernel_name!r}")
    n = _positive_int(parameters, "n")
    h = _positive_int(parameters, "h")
    w = _positive_int(parameters, "w")
    c = _positive_int(parameters, "c")
    k = _positive_int(parameters, "k")
    r = _positive_int(parameters, "r")
    s = _positive_int(parameters, "s")
    p = _positive_int(parameters, "p")
    q = _positive_int(parameters, "q")
    return AnalyticWork(
        flops=float(2 * n * p * q * k * r * s * c),
        bytes_moved=float(4 * (n * h * w * c + k * r * s * c + n * p * q * k)),
        model_id="cutlass_simt_conv2d/fp32-aoi-v1",
        assumptions="FP32; activación, filtro y salida contados una vez por fprop (reutilización ideal intra-lanzamiento)",
    )


def _raja_real_bytes(elements: int) -> float:
    """Bytes de ``Real_type`` del binario RAJAPerf-CUDA (FP64 en pacca)."""
    return float(8 * elements)


def rajaperf_stream_triad_work(
    kernel_name: str, launch_index: int, parameters: Mapping[str, int | float | str],
) -> AnalyticWork:
    """Una invocación CUDA de ``Stream_TRIAD`` de RAJAPerf.

    RAJAPerf declara explícitamente 2 FLOPs y tres accesos ``Real_type`` por
    elemento.  El tamaño ya resuelto (no ``--sizefact``) se declara en el
    manifiesto para que el cálculo no dependa de heurísticas de la salida.
    """
    if "triad" not in kernel_name.lower():
        raise ValueError(f"kernel CUDA inesperado para RAJAPerf Stream_TRIAD: {kernel_name!r}")
    n = _positive_int(parameters, "elements")
    return AnalyticWork(
        flops=float(2 * n),
        bytes_moved=_raja_real_bytes(3 * n),
        model_id="rajaperf/stream-triad-fp64-aoi-v1",
        assumptions="RAJAPerf v2025.12.1: 2 FLOPs, dos lecturas y una escritura Real_type por kernel",
    )


def rajaperf_jacobi_2d_work(
    kernel_name: str, launch_index: int, parameters: Mapping[str, int | float | str],
) -> AnalyticWork:
    """Cada uno de los dos kernels de una repetición Polybench_JACOBI_2D."""
    if "jacobi" not in kernel_name.lower():
        raise ValueError(f"kernel CUDA inesperado para RAJAPerf JACOBI_2D: {kernel_name!r}")
    n = _positive_int(parameters, "side", minimum=3)
    interior = (n - 2) ** 2
    # RAJAPerf declara por repetición dos stencils: 10 FLOPs, dos lecturas
    # de N²-4 y dos escrituras del interior.  Cada evento es media repetición.
    return AnalyticWork(
        flops=float(5 * interior),
        bytes_moved=_raja_real_bytes((n * n - 4) + interior),
        model_id="rajaperf/polybench-jacobi2d-fp64-aoi-v1",
        assumptions="RAJAPerf v2025.12.1: mitad de las cuentas por repetición, una pasada stencil por kernel",
    )


def rajaperf_heat_3d_work(
    kernel_name: str, launch_index: int, parameters: Mapping[str, int | float | str],
) -> AnalyticWork:
    """Cada una de las dos pasadas de Polybench_HEAT_3D."""
    if "heat" not in kernel_name.lower():
        raise ValueError(f"kernel CUDA inesperado para RAJAPerf HEAT_3D: {kernel_name!r}")
    n = _positive_int(parameters, "side", minimum=3)
    interior = (n - 2) ** 3
    # Exactamente la mitad de setFLOPs/Bytes{Read,Written}PerRep del suite.
    read_elements = n ** 3 - 12 * (n - 2) - 8
    return AnalyticWork(
        flops=float(15 * interior),
        bytes_moved=_raja_real_bytes(read_elements + interior),
        model_id="rajaperf/polybench-heat3d-fp64-aoi-v1",
        assumptions="RAJAPerf v2025.12.1: una de dos pasadas 7-point por repetición",
    )


def rajaperf_gemm_work(
    kernel_name: str, launch_index: int, parameters: Mapping[str, int | float | str],
) -> AnalyticWork:
    """Una invocación Polybench_GEMM/Base_CUDA de RAJAPerf."""
    if "gemm" not in kernel_name.lower():
        raise ValueError(f"kernel CUDA inesperado para RAJAPerf GEMM: {kernel_name!r}")
    ni = _positive_int(parameters, "ni")
    nj = _positive_int(parameters, "nj")
    nk = _positive_int(parameters, "nk")
    return AnalyticWork(
        flops=float((1 + 3 * nk) * ni * nj),
        bytes_moved=_raja_real_bytes(ni * nk + nj * nk + ni * nj),
        model_id="rajaperf/polybench-gemm-fp64-aoi-v1",
        assumptions="RAJAPerf v2025.12.1: FLOPs/bytes declarados por Polybench_GEMM para una repetición",
    )


_MODELS: dict[str, ModelFn] = {
    "rodinia_gaussian": gaussian_work,
    "dual_axpy_gpu_N10000000": axpy_work,
    "dual_axpy_gpu_N1280000000": axpy_work,
    "dual_stencil_gpu_N1024": stencil_work,
    "dual_stencil_gpu_N36864": stencil_work,
    "gpu_cutlass_simt_dgemm_n4096": cutlass_dgemm_work,
    "gpu_cutlass_simt_conv2d": cutlass_conv2d_work,
    "gpu_rajaperf_stream_triad": rajaperf_stream_triad_work,
    "gpu_rajaperf_jacobi_2d": rajaperf_jacobi_2d_work,
    "gpu_rajaperf_heat_3d": rajaperf_heat_3d_work,
    "gpu_rajaperf_gemm": rajaperf_gemm_work,
    "rodinia_lud": lud_work,
    "minibude_cuda_bm1": minibude_work,
    "rodinia_heartwall": heartwall_work,
}

_OPERATION_MODELS: dict[str, OperationSpec] = {
    "dual_spmv_gpu_N1000000": OperationSpec(7, spmv_operation_work),
    "dual_spmv_gpu_N200000000": OperationSpec(7, spmv_operation_work),
    "dual_cholesky_gpu_N2048": OperationSpec(5, cholesky_operation_work),
    "dual_fft_gpu_N4096": OperationSpec(5, fft_operation_work),
    "rodinia_kmeans_gpu_N350000_D34_K200": OperationSpec(
        4, kmeans_operation_work, allow_unmatched_prefix=True,
    ),
}


def registered_kernel_refs() -> tuple[str, ...]:
    return tuple(sorted(set(_MODELS) | set(_OPERATION_MODELS)))


def operation_spec_for_kernel(kernel_ref: str) -> OperationSpec | None:
    return _OPERATION_MODELS.get(kernel_ref)


def work_for_operation(
    kernel_ref: str,
    events: Sequence[tuple[str, str, int]],
    operation_index: int,
    parameters: Mapping[str, int | float | str],
) -> AnalyticWork:
    try:
        spec = _OPERATION_MODELS[kernel_ref]
    except KeyError as exc:
        raise KeyError(f"kernel GPU sin modelo analítico por operación: {kernel_ref}") from exc
    return spec.work(events, operation_index, parameters)


def work_for_launch(
    kernel_ref: str,
    kernel_name: str,
    launch_index: int,
    parameters: Mapping[str, int | float | str],
) -> AnalyticWork:
    try:
        model = _MODELS[kernel_ref]
    except KeyError as exc:
        raise KeyError(
            f"kernel GPU sin modelo analítico registrado: {kernel_ref}; "
            "no se permite fallback silencioso a la OI histórica de ncu"
        ) from exc
    return model(kernel_name, launch_index, parameters)
