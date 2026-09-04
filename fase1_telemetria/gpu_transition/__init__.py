"""F1-GPU-002 -- agregación de las corridas del probe de transición de reloj GPU.

El probe (`common/telemetry/experiments/gpu_clock_transition_probe.cpp`) mide UNA
transición dirigida `(origen -> destino, réplica)` por ejecución y escribe un
`gpu_clock_transition_summary.json`. `aggregate_transition_matrix` junta varios
de esos JSON y deriva `T_transicion_gpu_ns_conservative` como el MÁXIMO sobre
pares y réplicas realmente medidos (nunca un promedio), el valor que alimenta
`--t-transicion-gpu-ns` de `fase3_daemon/policy/derive_policy_table.py` y el
piso de `min_dwell_ns` del daemon.

Uso:
    python3 -m fase1_telemetria.gpu_transition.aggregate_transition_matrix --help

Import:
    from fase1_telemetria.gpu_transition.aggregate_transition_matrix import (
        aggregate_summaries, conservative_transition_ns, load_summaries,
    )
"""
