# Revisión para Claude: candidatos GPU compute-bound

Fecha: 2026-09-14. Revisión local y de fuentes originales; no se ejecutaron cargas en pacca ni se modificaron campañas. Los OI 2.81/2.50/1.95 se toman del contexto proporcionado, no de CSV revalidados. No fue posible acceder al chat «Screen CPU analysis»: Orca devolvió `runtime_unavailable`, también después de intentar abrir la aplicación.

## Recomendación

Hacer una búsqueda final acotada, centrada en GEMM optimizado sin Tensor Cores y una familia adicional. No gastar otra campaña en los seis Polybench restantes ni recortar memory-bound como sustituto de diversidad algorítmica.

## 1. Hay una contradicción documental que afecta la decisión

`Seguimiento_Cambios_Plan_Director.md`, F1-GPU-009/010, atribuye el OI casi cero de cuBLAS a una incapacidad de contar SASS de kernels propietarios y extiende esa sospecha a CUTLASS. Sin embargo, `fase1_telemetria/catalog/catalog.yaml`, comentarios ARC-76 y ARC-110 alrededor de las líneas 400 y 470, documenta:

- Kernel `cutlass_80_tensorop_d884gemm`.
- Contador DMMA positivo y DFMA cero.
- OI previo 68 derivado de FLOPs analíticos y bytes medidos, no del conteo escalar.

Es evidencia histórica local, pendiente de cotejar con los perfiles originales, pero ya contradice la explicación «cuBLAS propietario invisible». `fase1_telemetria/ncu_convergence.py`, función `build_kernel_report`, excluye expresamente Tensor Cores porque su contrato solo convierte ADD/MUL/FMA escalares a FLOPs. La exclusión es correcta bajo ese contrato; no demuestra que GEMM sea memory-bound.

Primera prueba recomendada: configurar el benchmark existente con `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)` y comprobar retorno, corrección numérica, nombre del kernel, DFMA y ausencia de actividad Tensor. La documentación de [cuBLAS CUDA 12.0, Tensor Core Usage](https://docs.nvidia.com/cuda/archive/12.0.0/cublas/index.html#tensor-core-usage) indica que los modos pedantic deshabilitan Tensor Cores. Confirmarlo en el binario y versión reales; no asumir que el cambio de API produjo la ruta deseada. Es una configuración del proveedor, no escribir un GEMM propio. Registrar un nuevo checksum y variante.

Alternativa: benchmark oficial CUTLASS con implementación SIMT. Su [profiler](https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/profiler.md) permite `--op_class=simt` y selección de kernels concretos. CUTLASS no implica Tensor Cores. Verificar disponibilidad en una versión compatible con CUDA instalado. Separar los lanzamientos de referencia/verificación del kernel objetivo: la validación puede invocar otra biblioteca y contaminar el perfil agregado.

No reinstalar el OI histórico 68 como verdad actual. Si se decide admitir DMMA, hace falta un método explícito de FLOPs Tensor y techo Tensor compatible por frecuencia; no sumar sus FLOPs al techo escalar actual.

## 2. 2MM/3MM: resultado válido, explicación causal todavía incompleta

El [código original de 3MM](https://raw.githubusercontent.com/LLNL/RAJAPerf/develop/src/polybench/POLYBENCH_3MM-Cuda.cpp) ejecuta E=A·B, F=C·D, G=E·F mediante tres kernels separados. Esto no equivale simplemente a una cadena A·B·C·D. La versión upstream revisada debe contrastarse con el checkout instalado.

Separar kernels no obliga a que toda lectura global siguiente venga de DRAM: puede servirse desde L2. Tampoco agregar productos hace caer necesariamente la intensidad. Para etapas medidas sobre el mismo dominio:

`OI_total = sum(F_i) / sum(B_i) = sum(B_i * OI_i) / sum(B_i)`.

Si las etapas tuvieran el mismo OI, el conjunto conservaría ese OI. Las dimensiones, accesos, reutilización y estado de caché determinan la diferencia real. Los intermedios agregan tráfico, pero por sí solos no prueban la secuencia 2.81 → 2.50 → 1.95. `roofline_label_eligible` valida condiciones del método, no esa explicación causal.

Además, el screening local usa `--launch-count` sobre todos los procesos y suma sus métricas: revisar warmups, inicializaciones y ciclos completos de 2/3 etapas. Los conteos 5/20/50 no son ciclos completos de 3MM. Separar FLOPs/bytes por nombre de kernel antes de interpretar la caída.

## 3. Lista corta de candidatos

| Prioridad | Candidato | Motivo y límite |
|---|---|---|
| 1 | DGEMM cuBLAS pedantic o GEMM CUTLASS SIMT | Recuperar una familia optimizada excluida por ruta Tensor; medir, sin prometer etiqueta. |
| 2 | RAJAPerf `Apps_LTIMES` | Contracción de transporte con dimensiones configurables; inspeccionar las opciones y variante CUDA realmente compiladas antes del piloto. No asumir independencia de GEMM solo por nombre. |
| 3 | miniBUDE CUDA | Benchmark de docking de terceros, candidato a otra aplicación intensiva en cómputo si se amplía el conjunto de suites. Auditar relación con la familia de interacciones moleculares para el split. |
| Opcional | CUTLASS convolución SIMT | Operación distinta de GEMM explícito, aunque comparte maquinaria de cómputo; agrupar conservadoramente para no sobreestimar generalización. |

[LTIMES upstream v2025.12.0](https://raw.githubusercontent.com/LLNL/RAJAPerf/v2025.12.0/src/apps/LTIMES.cpp) expone dimensiones d/g/m y un conteo de 2·z·g·m·d FLOPs. Eso justifica inspeccionarlo; no garantiza intensidad DRAM elevada. [miniBUDE](https://github.com/UoB-HPC/miniBUDE) tiene implementación CUDA, y sus autores discuten esta carga en el contexto de [workloads compute-bound](https://uob-hpc.github.io/assets/ISC-Bristol-Hans-Meueur-award-2021.pdf). La etiqueta en pacca sigue pendiente de medición. El profiler oficial de CUTLASS enlazado arriba también documenta convolución sobre CUDA cores.

Dentro de RAJAPerf existe [Basic_MAT_MAT_SHARED](https://raw.githubusercontent.com/LLNL/RAJAPerf/develop/src/basic/MAT_MAT_SHARED-Cuda.cpp), con tiling compartido. No lo priorizaría como ancla profunda: con tile T, FP64 y sin reutilización adicional entre bloques, el modelo simple da aproximadamente T/8 FLOP/byte; T=16/32 da 2/4, coherente con el problema ya observado. Caché puede elevarlo, pero hay que medirlo.

Tampoco priorizaría MASS3DPA/DIFFUSION3DPA solo por ser elementos finitos: las versiones consultadas fijan orden bajo y aumentar elementos no equivale a aumentar orden. Ver [MASS3DPA.hpp](https://raw.githubusercontent.com/LLNL/RAJAPerf/v2025.12.0/src/apps/MASS3DPA.hpp) y [DIFFUSION3DPA.hpp](https://raw.githubusercontent.com/LLNL/RAJAPerf/v2025.12.0/src/apps/DIFFUSION3DPA.hpp).

## 4. Piloto y balance

1. En el nodo de login o local: inventariar versiones, opciones y fuentes, registrar checksum y preparar validación. No compilar ni ejecutar pruebas en la asignación mientras mida una campaña.
2. Cuando no haya medición activa: prueba corta de corrección y ruta aritmética; perfil limitado de lanzamientos representativos y ciclos completos.
3. Solo si resulta prometedor: convergencia completa, tamaño fuera de L2 para el régimen DRAM buscado y duración sostenida mediante repeticiones. El tiempo objetivo corresponde al segmento GPU medido, no a inicialización de CPU. No exigir ~50% de VRAM a todos los algoritmos como condición universal.
4. Evaluar `OI/ridge(precisión, frecuencia)` con los techos medidos de la sesión. Una OI alta indica posición a la derecha del ridge; contrastar rendimiento y respuesta a frecuencia para descartar dominio de latencia, sincronización o funciones especiales.
5. No reducir familias memory-bound solo para igualar filas. Ajustar pesos o muestrear dentro de cada fold de entrenamiento, conservar evaluación por familias y reportar métricas por clase/familia. Los tamaños o variantes del mismo algoritmo deben permanecer juntos.
6. Si lavamd sigue siendo la única familia profunda, declarar esa limitación de cobertura. Una segunda variante de lavamd no permite validar generalización a otra familia profunda. Si además fuera la única familia con cualquier ejemplo compute-bound, el fold que la retira dejaría el entrenamiento sin esa clase.

La conclusión defendible hoy es «los kernels y configuraciones medidos no cubren la zona», no «no existe otra clase de algoritmo ni candidato de terceros». El mínimo de 5–6 familias por clase del plan tampoco se satisface equilibrando números de ventanas: cualquier reducción de esa meta debe registrarse como cambio de alcance explícito.
