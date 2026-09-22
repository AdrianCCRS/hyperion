# Umbral de $\alpha$ para que DVFS de CPU pague: insumo para Fases 3 y 4

Fecha: 2026-09-22. Estado: cerrado como análisis, NO incorporado al capítulo de
resultados a propósito (decisión del autor: el libro ya explica el cierre por el
piso de potencia, y una segunda parametrización de $\alpha$ solo confundiría al
lector). Se guarda aquí para que no se pierda, porque es el criterio que la
Fase 3 (daemon) y la Fase 4 (evaluación) necesitan si alguna vez se corre en
otro hardware.

## Qué es

El capítulo de resultados usa $\alpha$ como la pendiente negativa de $\log T$
frente a $\log f$ (elasticidad del tiempo respecto del reloj). Este documento
usa una parametrización distinta, tipo Amdahl, más cómoda para despejar un
umbral cerrado:

$$T(f)/T_{REF} = \alpha + (1-\alpha)\cdot \frac{3.2}{f}$$

donde $\alpha$ es la **fracción del tiempo insensible al reloj** (0 = el tiempo
escala por completo con la frecuencia, 1 = el tiempo no cambia al bajar el
reloj). Es aproximadamente el complemento de la $\alpha$ del libro. **No mezclar
las dos en el mismo texto.**

## El umbral

Con la potencia medida en paccaA100 sobre la campaña final (28 kernels
memory-bound, mediana entre kernels):

$$P(f)/P_{REF} = 0.803 + 0.058\,f \quad (f \text{ en GHz})$$

Es decir, el **81% de la potencia de paquete a 3.2 GHz no escala con la
frecuencia**. Bajar el reloj 4x (3.2 a 0.8 GHz) ahorra solo **14.6%** de
potencia mientras el tiempo sube 1.83x. El techo absoluto de ahorro, con
$f \to 0$, es 19.7%.

Como el producto energía--retardo es $EDP = P \cdot T^2$, el castigo del tiempo
entra al cuadrado. Imponiendo $EDP(f) < EDP_{REF}$ y despejando, con
$r = 3.2/f$ y $p = P(f)/P_{REF}$:

$$\alpha > \frac{r - \sqrt{1/p}}{r - 1}$$

| nivel | GHz | $P/P_{REF}$ | $\alpha$ mínimo para ganar algo | $\alpha$ mínimo para ganar 1% |
|---|---|---|---|---|
| F1 | 2.9 | 0.971 | 0.857 | 0.906 |
| F2 | 2.6 | 0.949 | 0.885 | 0.907 |
| F3 | 2.3 | 0.928 | 0.903 | 0.916 |
| F4 | 2.0 | 0.913 | 0.922 | 0.931 |
| F8 | 0.8 | 0.854 | 0.973 | 0.974 |

## Contra el catálogo real

Ajustando $\alpha$ por familia sobre los diez niveles medidos, las 19 familias
memory-bound van de 0.159 (`npb_lu`) a 0.881 (`rajaperf_stream`), con media
0.601. Contra los umbrales de arriba:

- Familias con $\alpha \ge 0.906$ (ganar 1% en F1): **0 de 19**
- Familias con $\alpha \ge 0.885$ (ganar algo en F2): **0 de 19**
- Familias con $\alpha \ge 0.857$ (ganar algo en F1): 3 de 19, y las tres apenas
  rozan el umbral (`rajaperf_stream` 0.881, `ptrchase` 0.876,
  `rajaperf_lcals` 0.873)

**La región objetivo está vacía en este procesador.** No fue un problema de
tamaño de muestra, ni del clasificador, ni del umbral elegido: ninguna carga del
catálogo, ni siquiera STREAM puro, tiene $\alpha$ suficiente. Esto explica hacia
atrás el `no_actuar` de la tabla por clase y las dos pruebas externas selladas
refutadas (V1 y V1b, ver `docs/libro/datos/cpu_calidad_30fam/politica/`).

## Para qué sirve en Fases 3 y 4

1. **Criterio de portabilidad.** En otro nodo, la pregunta "¿vale la pena actuar
   frecuencia de CPU aquí?" se responde con dos mediciones baratas: ajustar
   $P(f)$ para obtener la fracción no escalable, y ajustar $\alpha$ de unas
   pocas cargas. Si el $\alpha$ máximo del catálogo queda por debajo del umbral
   de la tabla, no hace falta montar la campaña completa.
2. **Cota superior para el objetivo 4.** Ninguna política, por perfecta que
   sea su clasificación, puede superar el 19.7% en esta plataforma, y en la
   práctica el techo real es mucho menor.
3. **Lo que tendría que cambiar para que fuera viable:** hardware con una
   fracción no escalable de potencia bastante menor que 81%. No un catálogo más
   grande, no un umbral mejor, no otro criterio de decisión.

## Corrección al plan

`Plan_Detallado_Realineacion_Hyperion.md` §3.4 cita, de una campaña anterior a
la reorganización, que "reducción de reloj 4x baja la potencia solo 28%", y pide
explícitamente re-verificarlo con la campaña reorganizada. Hecho: el valor real
medido es **14.6%**, no 28%. El plan queda corregido por medición, y el
argumento del plan se sostiene con más fuerza de la que anticipaba.
