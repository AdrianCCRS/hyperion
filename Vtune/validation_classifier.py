"""Metodologia de clasificacion de VALIDACION (no es el etiquetado oficial
de Hyperion) a partir del Top-Down completo de Microarchitecture
Exploration. Ver docs/vtune/vtune_cross_validation.md secciones C y D para
la discusion completa de por que este veredicto es independiente del
mecanismo principal y por que sigue siendo util pese a compartir PMU.

Diferencia central frente a pipelinevtune/classifier.py (D3-v3): ese
clasificador tuvo que calibrar contra anclas STREAM/DGEMM porque
hpc-performance en este nodo NO exponia un 'Core Bound' aislado (el TMAM de
4 categorias completo vive en Microarchitecture Exploration, que no estaba
disponible entonces). Con uarch-exploration funcional, Back-End Bound se
descompone nativamente en Memory Bound y Core Bound DENTRO DEL MISMO
REPORTE de cada kernel -- ya no hace falta pedir prestada la posicion de
un kernel externo para saber si el numero es 'alto' o 'bajo'. STREAM/DGEMM
se conservan (ver expected_behavior mas abajo) pero como referencia
declarada de literatura para comparar el veredicto observado, no como
insumo matematico del veredicto.

Principio explicito, pedido por el usuario: nunca un umbral tipo
'Memory Bound > 50% => memory_bound' sin justificar por que ese punto de
corte tiene sentido. Cada umbral de este modulo esta declarado como
constante nombrada, con su razon en el comentario -- no un numero magico
suelto en medio de una expresion booleana.
"""
from __future__ import annotations

from dataclasses import dataclass

VALIDATION_CLASSES = ("compute_bound", "memory_bound", "ambiguous", "invalid")

# --- Umbrales declarados (no derivados de los datos, documentados aqui) ---

# Por debajo de este piso, Back-End Bound (Memory+Core) es demasiado
# pequeno en terminos absolutos para que discutir su composicion interna
# tenga sentido -- el kernel simplemente no esta stalled por back-end,
# punto. Referencia: EP real capturado en este proyecto con Back-End Bound
# bajo y Retiring dominante (ver pipelinevtune/docs/analisis_vtune.md).
BACKEND_NEGLIGIBLE_PP = 15.0

# Retiring por encima de este piso, combinado con Back-End Bound bajo
# (regla anterior), se interpreta como "pipeline eficiente sin cuello de
# botella de memoria ni de nucleo relevante" -- no es lo mismo que "muy
# intensivo en FLOPs", es la ausencia de evidencia de lo contrario. Se deja
# explicito en la justificacion, nunca se presenta como sinonimo exacto.
RETIRING_EFFICIENT_PP = 70.0

# Margen (en puntos porcentuales de Pipeline Slots, NO en razon) para
# decidir entre Memory Bound y Core Bound cuando Back-End Bound si es
# relevante. Puntos porcentuales, no ratio, porque ambas metricas ya viven
# en la misma escala (fraccion de slots) -- comparar su diferencia absoluta
# es mas defendible que un cociente cuando los valores pueden ser pequenos.
MEMORY_VS_CORE_MARGIN_PP = 5.0

# Si Front-End Bound o Bad Speculation superan a Back-End Bound Y superan
# este piso absoluto, el cuello de botella real no es memoria ni computo en
# el sentido del proyecto -- es front-end (fetch/decode) o especulacion mal
# resuelta (branch mispredict/machine clears). Ver notas de kernel LU/IS en
# pipelinevtune/context/03_kernels_notas.md para casos ya anticipados de
# este patron (sincronizacion, ordenamiento por enteros).
NON_MAPPING_DOMINANT_PP = 30.0

# Coherencia interna: las 4 categorias de Nivel 1 deberian sumar ~100% de
# Pipeline Slots por construccion del TMAM. Una desviacion mayor a esto es
# señal de multiplexacion degradada o metricas parciales, no un resultado
# valido para clasificar con confianza alta.
TOPDOWN_SUM_TOLERANCE_PP = 10.0

# Hints de literatura para expected_behavior -- MISMA fuente que
# phase_label_hint en docs/retoma/Guia_Maestra_Fase1_DVFS.md seccion 6.
# Declarados como hint, nunca como verdad asumida: la columna
# vtune_validation_class de este pipeline es la que decide, comparada
# contra esto, no al reves.
EXPECTED_BEHAVIOR_HINTS = {
    "stream": "memory_bound",
    "dgemm": "compute_bound",
    "ep": "compute_bound",
    "mg": "memory_bound",
    "cg": "memory_bound",
    "is": "memory_bound",
    "ft": "intermedio",
    "lu": "intermedio",
    "bt": "intermedio",
    "sp": "intermedio",
}


@dataclass
class ValidationVerdict:
    vtune_validation_class: str
    validation_confidence: str  # "alta" | "media" | "baja"
    validation_reason: str
    quality_status: str  # "ok" | "topdown_no_suma_100" | "metricas_insuficientes"


def _pct(v: float | None) -> str:
    return f"{v:.1f}%" if v is not None else "NA"


def classify(parsed: dict) -> ValidationVerdict:
    """parsed: dict de uarch_parser.parse_uarch_summary_text() de UNA
    corrida (kernel, clase, repeticion). No recibe anclas ni referencias
    externas -- toda la decision sale de este unico reporte, ver docstring
    del modulo."""
    retiring = parsed.get("retiring_pct")
    frontend = parsed.get("frontend_bound_pct")
    bad_spec = parsed.get("bad_speculation_pct")
    backend = parsed.get("backend_bound_pct")
    memory = parsed.get("memory_bound_pct")
    core = parsed.get("core_bound_pct")

    if None in (retiring, frontend, bad_spec, backend):
        return ValidationVerdict(
            "invalid", "NA",
            "Faltan una o mas de las 4 categorias de Nivel 1 (Retiring/Front-End "
            "Bound/Bad Speculation/Back-End Bound) en el reporte -- no se puede "
            "aplicar la metodologia sin el Top-Down completo.",
            "metricas_insuficientes",
        )

    total = retiring + frontend + bad_spec + backend
    quality = "ok"
    if abs(total - 100.0) > TOPDOWN_SUM_TOLERANCE_PP:
        quality = "topdown_no_suma_100"

    # Paso 1: ¿domina una categoria que no mapea al eje computo/memoria?
    non_mapping_max = max(frontend, bad_spec)
    non_mapping_label = "Front-End Bound" if frontend >= bad_spec else "Bad Speculation"
    if non_mapping_max > backend and non_mapping_max > NON_MAPPING_DOMINANT_PP:
        return ValidationVerdict(
            "ambiguous", "media",
            f"{non_mapping_label}={_pct(non_mapping_max)} domina sobre Back-End "
            f"Bound={_pct(backend)} y supera el piso de {NON_MAPPING_DOMINANT_PP:.0f}pp. "
            "Este cuello de botella (fetch/decode o especulacion mal resuelta) no "
            "se mapea al eje compute_bound/memory_bound del proyecto -- ver "
            "docs/vtune/vtune_cross_validation.md seccion D. "
            f"[Retiring={_pct(retiring)}, Front-End Bound={_pct(frontend)}, "
            f"Bad Speculation={_pct(bad_spec)}, Back-End Bound={_pct(backend)}]",
            quality,
        )

    # Paso 2: Back-End Bound irrelevante en magnitud absoluta + Retiring alto
    # -> pipeline eficiente, sin evidencia de cuello de botella de memoria
    # ni de nucleo.
    if backend < BACKEND_NEGLIGIBLE_PP and retiring > RETIRING_EFFICIENT_PP:
        return ValidationVerdict(
            "compute_bound", "media",
            f"Retiring={_pct(retiring)} (> {RETIRING_EFFICIENT_PP:.0f}pp) y "
            f"Back-End Bound={_pct(backend)} (< {BACKEND_NEGLIGIBLE_PP:.0f}pp): "
            "pipeline mayormente retirando trabajo util, sin cuello de botella "
            "de memoria ni de nucleo detectable. No implica intensidad "
            "aritmetica alta por si solo -- solo ausencia de stalls relevantes.",
            quality,
        )

    # Paso 3: comparar Memory Bound vs Core Bound directamente -- el nucleo
    # de la metodologia, ya no requiere anclas externas.
    if memory is None or core is None:
        return ValidationVerdict(
            "invalid", "NA",
            f"Back-End Bound={_pct(backend)} es relevante, pero Memory Bound "
            f"({_pct(memory)}) o Core Bound ({_pct(core)}) no estan disponibles "
            "en el reporte -- no se puede decidir la composicion interna.",
            "metricas_insuficientes",
        )

    diff = memory - core
    detalle_mem = _memory_sublevel_detail(parsed)
    detalle_core = _core_sublevel_detail(parsed)
    base = (
        f"Back-End Bound={_pct(backend)}, Memory Bound={_pct(memory)}, "
        f"Core Bound={_pct(core)} (margen declarado={MEMORY_VS_CORE_MARGIN_PP:.0f}pp). "
        f"Retiring={_pct(retiring)}."
    )

    if diff > MEMORY_VS_CORE_MARGIN_PP:
        confianza = "alta" if diff > 2 * MEMORY_VS_CORE_MARGIN_PP else "media"
        return ValidationVerdict(
            "memory_bound", confianza,
            f"Memory Bound supera a Core Bound por {diff:.1f}pp. {base}{detalle_mem}",
            quality,
        )
    if -diff > MEMORY_VS_CORE_MARGIN_PP:
        confianza = "alta" if -diff > 2 * MEMORY_VS_CORE_MARGIN_PP else "media"
        return ValidationVerdict(
            "compute_bound", confianza,
            f"Core Bound supera a Memory Bound por {-diff:.1f}pp. {base}{detalle_core}",
            quality,
        )
    return ValidationVerdict(
        "ambiguous", "baja",
        f"Memory Bound y Core Bound difieren en {diff:.1f}pp, dentro del margen "
        f"declarado de {MEMORY_VS_CORE_MARGIN_PP:.0f}pp -- no hay un componente "
        f"dominante claro dentro de Back-End Bound. {base}",
        quality,
    )


def _memory_sublevel_detail(parsed: dict) -> str:
    levels = [
        ("L1 Bound", parsed.get("l1_bound_pct")),
        ("L2 Bound", parsed.get("l2_bound_pct")),
        ("L3 Bound", parsed.get("l3_bound_pct")),
        ("DRAM Bound", parsed.get("dram_bound_pct")),
        ("Store Bound", parsed.get("store_bound_pct")),
    ]
    present = [(name, v) for name, v in levels if v is not None]
    if not present:
        return ""
    dominant = max(present, key=lambda kv: kv[1])
    return f" Dentro de Memory Bound, el nivel dominante es {dominant[0]}={_pct(dominant[1])}."


def _core_sublevel_detail(parsed: dict) -> str:
    levels = [
        ("Divider", parsed.get("divider_pct")),
        ("Ports Utilization", parsed.get("ports_utilization_pct")),
    ]
    present = [(name, v) for name, v in levels if v is not None]
    if not present:
        return ""
    dominant = max(present, key=lambda kv: kv[1])
    return f" Dentro de Core Bound, el nivel dominante es {dominant[0]}={_pct(dominant[1])}."


def expected_behavior_for(kernel: str) -> str:
    return EXPECTED_BEHAVIOR_HINTS.get(kernel.lower(), "desconocido")


def agrees_with_hint(vtune_validation_class: str, expected_behavior: str) -> str:
    """'si' / 'no' / 'na' -- columna de apoyo, no de decision. 'intermedio'
    en expected_behavior nunca se marca 'no' contra ambiguous/compute/memory
    -- es un hint deliberadamente abierto (ver notas de MG/FT/LU en
    pipelinevtune/context/03_kernels_notas.md, mezcla de fases dentro de
    una misma corrida)."""
    if vtune_validation_class == "invalid" or expected_behavior == "desconocido":
        return "na"
    if expected_behavior == "intermedio":
        return "na"
    return "si" if vtune_validation_class == expected_behavior else "no"
