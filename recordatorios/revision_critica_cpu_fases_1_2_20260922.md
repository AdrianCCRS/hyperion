# Revisión crítica de CPU (Fases 1 y 2): resultados, discusión y metodología

Fecha: 2026-09-22. Alcance: `docs/libro/secciones/02_metodologia.tex` (partes CPU),
`03_resultados.tex` §Resultados de CPU, `04_discusion.tex`, `05_conclusiones.tex`.
Todas las cifras se verificaron contra `docs/libro/datos/cpu_calidad_30fam/`.

**Estado (2026-09-22, aplicado por Sonnet 5):** A1-A6, B1-B6, C1 (parcial, ver nota),
C2-C6, D1 (versión ligera), D2, D3, F, E1, E3-E5, E7 aplicados y compilados (151
páginas, sin referencias/citas sin resolver). Detalle de lo hecho y lo pendiente:

- **A1-A6**: corregidos todos los números (cobertura 526\,279/783\,476; caption
  21\,108; F8 46.4\%; Conclusiones alineadas con Resultados; kNN 0.684 en toda
  parte; "árbol de profundidad seis" eliminado, ahora "seis tipos de candidato").
- **B1-B6**: umbral 0.749 (selección) vs. 0.738 (evaluación externa) ahora
  nombrados por separado; coberturas etiquetadas (por celda / por familia /
  agrupada); discrepancia del tope=1000 explicada; nota al pie sobre IC
  bootstrap vs. $p$ de Wilcoxon; IC de Optuna en regresión logística reportado
  completo; tabla de plataforma corregida (dominio por núcleo físico, no lógico).
- **C1 (calibración en línea)**: NO se subió a subsección propia (decisión
  pendiente del autor, ver `feedback-libro-solo-cambios-trascendentales`: no
  hay método/tabla/figura que la respalde en el capítulo 2). Se dejó la cifra
  donde estaba; si se quiere destacar, hace falta que el autor decida y aporte
  el método antes de escribir más.
- **C2-C6**: hecho (columna exact.-bal.-celda en la tabla de frecuencia;
  explicación del "rebote" de F8; redacciones corregidas; tabla de comparación
  de modelos reducida a 6 filas para no duplicar la tabla de variantes de entrada).
- **D1**: no se movió la subsección completa (arriesgaba romper el flujo de
  tablas); en su lugar se añadió un párrafo breve al inicio de §Clasificador de
  fase en CPU que define LOFO y exactitud-por-celda antes de la primera cifra.
- **D2/D3**: los cinco párrafos de narrativa de depuración del instrumento en
  Discusión se colapsaron en uno solo (principio general + 3 hechos vigentes,
  sin contradecir Resultados); el capítulo ahora entra a CPU inmediatamente
  después de ese párrafo, antes que GPU.
- **E1**: insertada `fig_cpu_correlacion_entradas_20260918.png` (recaptioned:
  es la matriz de Pearson de las 12 candidatas, no de las 6 finales).
- **E3**: nueva figura de perfiles gemelos (`fig_cpu_perfiles_gemelos_20260922`),
  con el par gap_pr/phasic y el par más cercano real (mat_mat_shared/pi_reduce).
  Ojo: la sospecha original de contradicción sobre mat_mat_shared **era
  incorrecta** — esa familia sí es mayoritariamente compute-bound
  (`memory_share=0.001`); lo confusable es su celda minoritaria memory (0.1%
  de sus intervalos) frente a la celda típica compute de pi_reduce. Corregido
  en el texto.
- **E4**: forest plot de la tabla de política (`fig_cpu_politica_forest_20260922`).
- **E5**: exactitud-por-celda + cobertura vs. frecuencia, doble eje
  (`fig_cpu_exactitud_cobertura_frecuencia_20260922`).
- **E6**: no se tocó (bajo valor, requeriría editar la figura de calibración
  existente para marcar los puntos 0.85/0.90).
- **E2 (Roofline scatter de intervalos reales)**: **hecha**
  (`fig_cpu_roofline_scatter_20260922`). El dataset de entrenamiento
  (`source30.csv`/`new6_eligible.csv`, job 7507) no trae FLOPs/bytes por
  intervalo, porque el contrato de columnas del clasificador los descarta antes
  de esa etapa. Se reconstruyeron desde `training_cpu_intervals.csv`, que sí
  vive dentro de cada directorio de corrida de la campaña
  `pacca_cpu_final_20260913` en pacca (1295 corridas concatenadas vía SSH,
  sin re-ejecutar nada, solo lectura de CSVs ya producidos), más
  `new6_eligible.csv` para las 6 familias añadidas en el recálculo de
  vector-6. El total (1\,309\,755 filas, 526\,279 compute / 783\,476 memory)
  coincide exactamente con las cifras ya verificadas del capítulo, confirmando
  que es el dataset correcto. Los techos Roofline usados son los calibrados
  reales de la campaña (`roofline_calibration_REF.json`/`_F8.json`: BW 58.28/44.34
  GB/s, P 506.39/129.52 GFLOP/s, ridge 8.69/2.92 FLOP/byte). Artefactos
  derivados (submuestreo estratificado 4000/nivel/clase + constantes de
  calibración, ~1 MB) guardados en
  `docs/libro/datos/cpu_calidad_30fam/roofline_scatter/`, siguiendo la
  convención del resto del directorio (artefactos pequeños, no dumps crudos).
- **E7**: párrafo de asimetría del margen del ridge (0.318 vs. 0.766) añadido
  en §Dónde está el límite del resultado.

Script nuevo: `docs/libro/scripts/generar_figuras_revision_20260922.py`
(reproduce las tres figuras nuevas desde los CSV/JSON ya existentes).

Este documento es la lista de acciones para la sesión que reescriba el texto.
Cada ítem indica archivo, ubicación, qué está mal y con qué evidencia.

---

## A. Errores numéricos verificados (corregir sí o sí)

### A1. La tabla de cobertura no cuadra con el conjunto ni consigo misma
`03_resultados.tex:26-27`, `tab:cpu-cobertura-conjunto` declara
513 508 intervalos *compute_bound* (39.6 %) y 783 473 *memory_bound* (60.4 %).
Suman **1 296 981**, no los 1 309 755 elegibles de la fila anterior de la misma tabla.

Dato real: `inventory.json` → `memory_share_global = 0.59819`;
`final_model.json` → soporte por clase compute 526 279 / memory 783 476.
Suman exactamente 1 309 755 y coinciden con `tab:cpu-metricas-clase` (`03:366-367`).

**Acción:** corregir la tabla a 526 279 (40.2 %) y 783 476 (59.8 %).

### A2. Conteo de la muestra inconsistente entre caption y tabla
`03:46` (caption de `fig:cpu-composicion-clase-kernel`) dice 21 105 intervalos
*memory_bound*; `tab:cpu-flujo-modelo` (`03:123`) dice 21 108.
21 350 + 21 108 = 42 458, que es el total declarado en todo el capítulo.

**Acción:** corregir el caption a 21 108.

### A3. Fracción *compute_bound* en F8
`03:53` dice que sube "hasta 45.1 % en F8".
`inventory.json` → `memory_share_by_freq["F8"] = 0.5355` → compute = **46.4 %**.
(REF 0.71 → 29.0 % sí es correcto.)

**Acción:** 46.4 %.

### A4. Conclusiones arrastra cuatro cifras obsoletas
`05_conclusiones.tex`, párrafo de Fase 2:

| En Conclusiones | En Resultados / datos | Fuente |
|---|---|---|
| piso = "83 % de la potencia a 3.2 GHz" | 84 % (`03:551`) | ajuste `P0 + c·f^k` del propio libro |
| "α ≲ 0.22" | "mediana 0.14, en ningún kernel supera 0.33" (`03:551`) | el 0.22 viene del análisis de agosto, ya superado |
| STREAM "ganancias de 4 a 12 %" | "entre 4 y 11 %" (`03:518`) | — |
| oráculo "2.5 % en promedio" | 2.2 % media, 12.5 % máx, 22/47 por debajo del 1 % | `politica/best_level_by_kernel.csv` (verificado: media 0.0217, máx 0.1248) |

**Acción:** unificar contra Resultados. Además, Conclusiones presenta las ganancias
STREAM como si fueran un hallazgo aprovechable, cuando `03:518` dice explícitamente
que no son monótonas, que son del orden de la dispersión y que "se reportan como
observación puntual y no como evidencia de una región aprovechable". La conclusión
contradice la advertencia de su propio capítulo de resultados: hay que reescribirla
para que herede la advertencia, no para que la borre.

### A5. kNN: 0.683 vs 0.684
`04_discusion.tex` dice 0.683; `03:479` dice 0.684.
`knn_ceiling.csv` → `pmu_only = 0.6843`. **Acción:** 0.684 en ambos.

### A6. El árbol de profundidad 6 nunca existió en CPU
`02_metodologia.tex:680` (párrafo de §2.4.6) habla de "los siete candidatos" y nombra
"árbol de profundidad seis" entre los modelos ajustables que perdieron contra XGBoost.

`matrix.csv` contiene exactamente estos modelos: `majority`, `stump_base`,
`logistic_base`, `logistic_inter`, `rf_base`, `et_base`, `xgb_base`, `xgb_inter`
(más ablaciones `*_nofreq`, `freq_only`, `mono`, `shallow`, `class_only`).
**No hay ningún árbol de profundidad 6.** Ese modelo existe en
`fase2_clasificador/training/model_specs.py::tunable_specs()`, pero esa función solo
la consume el pipeline GPU (`gpu_historical_nested_optuna.py`); el de CPU usa
`build_models()`. `tab:cpu-comparacion-modelos` tampoco lo lista.

**Acción:** reescribir ese párrafo. Son **seis** tipos de modelo candidatos, y los dos
modelos ajustables que perdieron y no se re-optimizaron son Random Forest y Extra Trees.
El resto del razonamiento (costo para los que ya perdieron; exclusión permanente por
diseño de la regla mayoritaria y del árbol de profundidad uno) se mantiene tal cual.

---

## B. Incoherencias internas: la misma cantidad con dos valores

### B1. Exactitud por celda sobre lo decidido a umbral 0.85: 0.749 o 0.738
- `tab:cpu-umbral-rejilla` (`03:432`): 0.749 → `threshold_final_grid.csv` = 0.7487
- `tab:cpu-selectiva` (`03:453`) y texto (`03:419`): 0.738 → `final_model.json.selective` = 0.7384

Ambos valores son correctos pero corresponden a cálculos distintos (rejilla de selección
vs modelo final evaluado). El libro los presenta como la misma magnitud, con el mismo
nombre, a cinco líneas de distancia.

**Acción:** nombrar cada uno de forma distinta y explicar en una frase qué los separa.

### B2. Tres coberturas usadas indistintamente
0.722 (agrupada sobre intervalos), 0.751 (media por familia), 0.714 (media por celda).
El párrafo `03:417` pasa de "baja la cobertura de 0.714 a 0.649" a "6.5 puntos más de
intervalos que el agente no resuelve": salta de cobertura por celda a una afirmación
sobre intervalos sin avisar.

**Acción:** etiquetar cada cifra con su unidad cada vez que aparece.

### B3. El tope 1000 da 0.723 en una tabla y 0.728 en todo el resto del capítulo
`tab:cpu-tope-muestreo` (`03:146`) → `cap_sensitivity.csv` = 0.7227 para cap=1000,
que es exactamente la configuración final cuyo resultado se reporta como 0.728
(`final_model.json` = 0.7282) en las otras once apariciones.

**Acción:** explicar la diferencia (distinto número de semillas / distinta evaluación)
o armonizar. Tal como está, el lector atento concluye que una de las dos está mal.

### B4. IC bootstrap que excluye el cero junto a un p no significativo
`tab:cpu-politica-ic`, clase memory: F1 = −5.4 con IC [−6.8; −3.4] y p = 0.120;
F2 = −13.4 con IC [−18.5; −11.2] y p = 0.073.

No es un error (el bootstrap es sobre la ganancia agregada y Wilcoxon es pareado por
kernel: no miden lo mismo), pero presentados en columnas contiguas sin comentario es
justo el tipo de detalle que un jurado marca como contradicción.

**Acción:** una frase al pie de la tabla explicando que ambas columnas responden
preguntas distintas.

### B5. La búsqueda Optuna en regresión logística sí es significativa
`03:238` agrupa la logística en "no mejora de forma distinguible" y reporta solo "+0.002".
`optuna_paired.json` → `logistic.ci95 = [0.00036, 0.00415]`, que **excluye el cero**
(14 familias mejoran, 8 empeoran en XGBoost; 5 vs 2 en logística).

**Acción:** reportar el IC y decir lo que realmente ocurre: el efecto es
estadísticamente distinguible de cero y a la vez irrelevante en magnitud. Es más
honesto y más fuerte que omitirlo.

### B6. El dominio de frecuencia declarado contradice el resultado que lo mide
`tab:plataforma` (`02:43`) declara "Dominio de frecuencia: **Por núcleo lógico**", y
`02:51` dice que se evitan los hilos hermanos SMT.

Pero `03:67` demuestra lo contrario: restringiendo solo los procesadores lógicos
delegados la razón de rendimiento fue 0.995 (sin efecto alguno), y solo al restringir
**también** sus hermanos SMT subió a 3.78. Si el dominio fuera realmente por núcleo
lógico, lo primero habría bastado.

**Acción:** corregir la tabla de plataforma (el dominio efectivo de actuación abarca
ambos hilos del núcleo físico) y decirlo en Metodología, que es donde el lector lo
necesita para entender el protocolo, no solo en Resultados como hallazgo suelto.

---

## C. Información suelta o sin respaldo metodológico

### C1. La calibración en línea está enterrada en una oración subordinada
`03:509` (§Síntesis, bloque de "trabajo futuro"): "la exactitud balanceada sobre bloques
con cambios de fase reales pasa de 0.654 a 0.847 etiquetando el 2.6 % de los intervalos".

Las cifras son correctas (`adaptation_phases.csv`: `mixed_blocks/base` = 0.6541,
`mixed_blocks/adapt_periodic` = 0.8471; `adaptation_phases_meta.json`:
`periodic_rows_frac = 0.0264`). Pero **no hay método en el capítulo 2, ni tabla, ni
figura, ni definición de "bloque con cambios de fase"**. Es, con diferencia, el
resultado con más potencial de toda la Fase 2 (+0.19 de exactitud por 2.6 % de
etiquetado) y aparece como cláusula subordinada.

**Acción:** o sube a subsección propia con su método en `02_metodologia.tex`, su tabla
y su figura, o baja a una frase de trabajo futuro **sin cifras**. Lo que no se sostiene
es reportar un número tan grande sin nada que lo respalde.

### C2. La tabla por frecuencia abandona la métrica principal sin avisar
`tab:cpu-resultado-frecuencia` (`03:399-408`) reporta **exactitud global**
(`final_model_by_frequency.csv`), en un capítulo que insiste en que la exactitud global
"no debe leerse como la capacidad del modelo" (`03:345`).

Los valores en la métrica principal existen: `by_frequency_level.csv` →
`cell_balanced_acc` REF 0.823, F0 0.777, F1 0.768, F2 0.753, F3 0.730, F4 0.751,
F5 0.702, F6 0.692, F7 0.657, F8 0.659.

**Acción:** añadir esa columna (o sustituir). De paso resuelve C3.

### C3. El rebote de F8 no se explica
`03:388` dice que el resultado "decrece, en general, al bajar el reloj", pero la tabla
sube de 0.710 (F7) a 0.757 (F8) y nadie lo comenta. Con la métrica por celda el rebote
prácticamente desaparece (0.657 → 0.659), lo que confirma que es un artefacto de la
métrica global y de la composición de clases de ese nivel.

**Acción:** una frase. Si se adopta C2, la frase se escribe sola.

### C4. Cabos sueltos del catálogo
`03:39` dice que tres kernels de Rodinia (kmeans, srad, particlefilter) quedaron fuera.
Pero `tab:catalogo` (`02:326-327`) lista como candidatos de ampliación en CPU a
**kmeans, SRAD, NW y particlefilter**, más **EP de NAS** ("ancla compute-bound pura en CPU").
NW y EP no vuelven a aparecer en ninguna parte del documento.

**Acción:** cerrar el cabo. Qué pasó con NW y con EP: se midieron, se descartaron, o
nunca se llegaron a correr. Si EP nunca se midió, es relevante decirlo, porque el
capítulo argumenta repetidamente que falta diversidad *compute-bound*.

### C5. Redacción defectuosa en pasajes concretos
- `03:8`: "procede de una campaña **la cual** cubrió" → anacoluto.
- `03:39`: "por solo agregar **mas** redundancia **memory bound**" → falta tilde,
  la clase va en cursiva y con guion como en el resto del documento, y el registro
  es coloquial. Reescribir entera.
- `03:355`: "Las dos clases se reconocen con sensibilidad **parecida**" para 0.806
  frente a 0.714 (nueve puntos). No es parecida. Y la frase que sigue ("la diferencia
  cambia de signo según el promedio") es correcta pero está redactada de forma que
  cuesta seguirla; merece dos oraciones.
- `03:206`: "La regresión logística queda por debajo: con seis variables está 0.048
  por debajo (IC95 −0.092 a **0.000**)". Ese IC toca el cero; la afirmación fuerte solo
  se sostiene con doce variables. Matizar.
- `03:345`: "La exactitud balanceada clásica (0.760) y la F1 macro agrupada (0.760)
  coinciden entre sí" — es una coincidencia numérica de este conjunto, no una identidad.
  Decirlo, o el lector busca la relación que no existe.
- `tab:cpu-protocolos` usa "±" sin declarar que es la desviación entre las cinco
  semillas (`protocols.json` → `sd`).
- Captions alternan "Fuente: Elaboración propia." y "Nota. Elaboración propia."
  Unificar en todo el capítulo.

### C6. Solapamiento entre la comparación de modelos y la de vectores de entrada
`tab:cpu-comparacion-modelos` incluye filas "Regresión logística (12 var.)" y
"XGBoost (12 var.)", que es exactamente lo que ya compara `tab:cpu-variantes-entrada`
una subsección antes. El párrafo `03:206` mezcla ambas discusiones.

**Acción:** decidir dónde vive cada comparación y no repetirla. Sugerencia: la tabla de
modelos se queda solo con los seis modelos a seis variables, y las variantes de vector
se quedan íntegramente en su propia subsección.

---

## D. Estructura y orden de lectura

### D1. El capítulo se lee hacia atrás
`03:103` anuncia el orden "muestra → variables → modelo → métricas → resultado". Pero
las tres primeras subsecciones ya reportan números en la exactitud balanceada por celda
y bajo el protocolo LOFO, **que solo se definen en la cuarta**. `03:206` lo admite
explícitamente al remitir hacia adelante a `sec:resultados-cpu-metricas`.

**Acción:** mover "Métricas de evaluación" y el párrafo "Esquema de evaluación" al
inicio de §Clasificador de fase en CPU, antes de la muestra. Es el cambio que más
mejora la legibilidad del capítulo entero y no toca ninguna cifra.

### D2. La Discusión contiene narrativa de depuración del instrumento, y además se
contradice con Resultados
`04_discusion.tex` dedica **cinco párrafos seguidos** a la historia de defectos del
instrumento (un permiso comunicado como concedido que no coincidía con el sistema; un
chequeo de carga externa que falla a escala; el presupuesto de contadores que deja de
sostenerse por el vigilante NMI; un microbenchmark de calibración "sistemáticamente
incorrecto por casi un orden de magnitud"; la corrección de ancho vectorial aplicada
selectivamente). Eso contradice la regla editorial ya establecida para este libro
(solo metodología final y resultado final, sin narrativa de bugs). Pero además hay
tres problemas verificables:

1. Referencia a `sec:resultados-cpu-dvfs` para "la corrección del chequeo de carga
   externa": esa sección no contiene nada sobre carga externa. La referencia compila
   pero apunta a otro contenido.
2. Afirma que la calibración resultó "sistemáticamente incorrecta por casi un orden de
   magnitud", mientras `03:97` (§Validación de los techos Roofline) reporta que la
   calibración **coincide** con Intel Advisor dentro del 12–15 %. El capítulo 4
   contradice al capítulo 3.
3. Describe la refutación del control de frecuencia atribuyéndola al turbo global,
   cuando `03:67` ya reporta otra causa distinta y vigente (los hermanos SMT).

**Acción:** recortar esos cinco párrafos a **uno** de principio metodológico
("ningún mecanismo se dio por funcional sin verificarlo contra el hardware"), sin
historia, sin cifras de campañas superadas y sin referencias a secciones que ya no
cuentan ese episodio. El párrafo equivalente de Conclusiones ya dice eso bien y puede
servir de molde.

### D3. La discusión de CPU empieza en el párrafo siete
Para un lector que evalúa Fases 1–2 de CPU, el capítulo 4 abre con seis párrafos de
instrumento y de GPU antes de llegar al resultado de CPU.

**Acción:** reordenar, CPU primero.

---

## E. Figuras: qué falta y qué sobra

### E1. Hay una figura generada y nunca insertada
`figuras/fig_cpu_correlacion_entradas_20260918.png` **no está citada en ningún `.tex`**
(verificado sobre `main.tex` y todas las secciones). Mientras tanto, `03:181-183`
describe en prosa densa las correlaciones (0.74 Spearman entre IPC y stall, 0.72 entre
IPC e IPS) y los VIF (5.5 y 7.3).

**Acción:** insertarla en §Variables de entrada y aligerar la prosa.

### E2. No existe ninguna figura Roofline del conjunto CPU real
El libro tiene el Roofline conceptual (marco), el ridge por frecuencia y el histograma
de error por margen, pero **nunca muestra los intervalos medidos en el plano
intensidad-operacional / rendimiento contra el ridge**. Es la figura que ancla la
definición de la etiqueta de toda la tesis y es la primera que un jurado espera ver en
un trabajo basado en Roofline.

**Acción:** generarla (nube de intervalos por clase + techos + ridge, para REF y F8,
que además hace visible el desplazamiento del ridge con la frecuencia).

### E3. Los perfiles gemelos merecen figura, y el par elegido es discutible
`03:479` describe el argumento central del límite del resultado solo con números en
prosa (MPKI 0.82 vs 0.94, IPC 1.11 vs 0.97, tasa de fallos 0.76 vs 0.72).

Dos cosas de `twin_cells.csv`:
- El par más cercano **no es** el que destaca el libro. `gap_pr`(memory) ↔
  `phasic`(compute) están a distancia 0.357. `rajaperf_basic_mat_mat_shared`(memory) ↔
  `rajaperf_basic_pi_reduce`(compute) están a **0.18**, la mitad, y son dos kernels de
  la misma suite, lo que hace el argumento más incómodo y más convincente.
- Ojo: `03:35` lista `basic_mat_mat_shared` entre las "cargas de cómputo denso" que
  "concentran sus intervalos en *compute_bound*", pero `twin_cells.csv` la registra con
  etiqueta memory. **Verificar cuál es la composición real de esa familia antes de
  escribir nada**, porque una de las dos afirmaciones está mal.

**Acción:** figura de coordenadas paralelas (o radar) con las seis variables de los
pares confusables, y revisar qué par se presenta como caso principal.

### E4. La tabla de política pide un forest plot
`tab:cpu-politica-ic` son 54 números en 9 filas. Un gráfico de ganancia con IC por nivel
y clase (dos paneles, línea en cero) comunica el "no actuar" de un vistazo. La tabla
puede quedarse como anexo.

### E5. La tabla por frecuencia pide un gráfico de dos ejes
Exactitud (por celda, ver C2) y cobertura frente al nivel de frecuencia, en un mismo
panel. Hace visible la caída conjunta de ambas y el comportamiento de F8.

### E6. Posible redundancia tabla/figura en el umbral
El panel derecho de `fig:cpu-umbral-calibracion` ya es la curva exactitud-cobertura, y
`tab:cpu-umbral-rejilla` la repite en números.

**Acción:** conservar ambas solo si se marcan sobre la curva el punto congelado (0.85)
y el que proponía la regla (0.90); si no, eliminar la tabla.

### E7. Hay una asimetría medida y no explotada
`error_by_ridge_margin.csv`: justo **por encima** del ridge, en (0, 0.5], la exactitud
es **0.318**; justo **por debajo**, en (−0.5, 0], es **0.766**. El error cerca de la
frontera no es simétrico: el modelo empuja sistemáticamente hacia *memory-bound*.

Eso explica de forma concreta el déficit de sensibilidad *compute-bound* (0.714 frente
a 0.806) que `03:355` atribuye de forma vaga a "perfiles gemelos", y se ve en la figura
que ya existe (`fig:cpu-margen-ridge`).

**Acción:** un párrafo nuevo en §Dónde está el límite del resultado. Es de las
observaciones más valiosas que están en los datos y no en el texto.

---

## F. Detalle metodológico menor

- `02:687` describe la curva de aprendizaje "de 3 a 23 familias"
  (`learning_curve.csv` confirma k = 3…23), pero el modelo final se ajusta con 29
  familias en cada pliegue. Decir por qué se detiene en 23, porque el lector lo pregunta.
- `02:687` dice "un clasificador de $k$ vecinos más cercanos" sin fijar k; Resultados
  dice 25 (`knn_ceiling.csv` → `knn_k25`). Declarar k en Metodología.
- El dato "0.037 por duplicación del catálogo" es correcto:
  `learning_curve_fit.json` → `slope_per_log_family = 0.0539`; ×ln2 = 0.0374. Sin cambio.
- Los rechazos por causa (`tab:cpu-rechazos`, 5.0 % + 3.0 % = 8.1 %) tienen un
  redondeo inconsistente: 5.0 + 3.0 = 8.0, no 8.1. Usar un decimal más o declarar
  el redondeo.

---

## Orden sugerido de ejecución

1. **A1–A6** (errores numéricos): mecánicos, sin decisiones. Empezar por aquí.
2. **B6 y D2**: las dos contradicciones entre capítulos. Son las que más daño hacen
   en una defensa.
3. **D1**: reordenar métricas antes de muestra. No toca cifras.
4. **B1–B5, C2, C3, C5, F**: precisión de redacción y etiquetado.
5. **C1 y C4**: decidir qué se hace con la calibración en línea y con NW/EP.
   Requieren una decisión del autor, no solo redacción.
6. **E1** (insertar figura ya generada), luego **E7** (párrafo con datos ya existentes),
   luego **E4/E5** (figuras nuevas baratas), y por último **E2/E3** (figuras nuevas que
   requieren volver a los datos crudos).

Recompilar con `pdflatex` al cerrar cada bloque y verificar que no aparezcan
referencias ni citas sin resolver.
