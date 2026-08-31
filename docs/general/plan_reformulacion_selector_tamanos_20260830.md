# Plan de reformulación del selector CPU/GPU orientado a tamaños no vistos

**Fecha:** 2026-08-30  
**Estado:** plan de trabajo activo; enmendado después de R1/R2
**Alcance documental:** Fases 2–4 del selector; incluye una reformulación
propuesta de objetivos específicos, pero no modifica por sí solo el libro ni
los objetivos formales del trabajo de grado  
**Condición de alcance:** el objetivo general permanece inalterado

El plan distingue expresamente tres niveles de compromiso:

1. **núcleo obligatorio:** lo necesario para implementar y evaluar un agente
   DVFS real y cumplir el objetivo general;
2. **trabajo confirmatorio condicionado por evidencia:** campañas adicionales
   que solo se ejecutan si el análisis de los datos existentes las justifica;
3. **extensiones deseables:** capacidades que se intentarán si el tiempo lo
   permite, sin convertir su ausencia en incumplimiento del núcleo del trabajo.

## 1. Decisión metodológica propuesta

El selector se evaluará como un sistema de **dominio cerrado respecto al tipo
de operación** y de **generalización respecto al tamaño**.

Esto significa que:

- las seis operaciones del catálogo dual son conocidas por el sistema:
  GEMM, FFT, AXPY, stencil, Cholesky y SpMV;
- el modelo puede utilizar la identidad de la operación y sus descriptores
  analíticos;
- la capacidad que se pondrá a prueba es seleccionar correctamente para
  tamaños de entrada no utilizados durante el entrenamiento;
- no se afirmará que el modelo generaliza a familias de operaciones nunca
  observadas;
- `leave-one-operation-out` podrá conservarse como análisis de robustez
  exploratorio, pero no será la prueba principal de la contribución.

Esta delimitación es coherente con el tamaño efectivo disponible: el conjunto
actual contiene 68 configuraciones distintas distribuidas entre solo seis
operaciones. Sostener generalización abierta a operaciones nuevas requeriría
un catálogo mucho más diverso y no debe inferirse a partir de este conjunto.

## 2. Problema que se quiere resolver

La formulación vigente intenta elegir directamente una de 40 acciones
`dispositivo × frecuencias` mediante una clasificación binaria por candidato.
Cada grupo tiene un positivo y 39 negativos. Aunque la implementación permite
ordenar las acciones, esta representación presenta cuatro limitaciones:

1. aumenta el número de filas sin aumentar el número de configuraciones
   independientes;
2. trata igual a una acción casi óptima y a una acción extremadamente mala;
3. mezcla en una sola etiqueta la decisión robusta de dispositivo y la
   decisión más ruidosa de frecuencia;
4. no representa el horizonte de reutilización necesario para amortizar la
   inicialización de GPU.

La reformulación separará tres preguntas:

1. **Inicialización y amortización:** ¿conviene pagar el arranque de GPU si la
   operación se repetirá cierto número de veces?
2. **Selección de dispositivo:** dado el estado actual, ¿conviene CPU o GPU?
3. **Selección de frecuencia:** una vez elegido el dispositivo, ¿existe una
   frecuencia cuya ventaja sea distinguible del ruido y superior al costo de
   actuación?

## 3. Hipótesis de investigación operativas

### H1 — Generalización por tamaño

Para una operación conocida, los descriptores analíticos y, cuando esté
disponible, la telemetría de una primera ejecución permiten predecir la
relación de costo CPU/GPU para tamaños no observados durante el entrenamiento.

### H2 — Valor de la telemetría

El modelo con sondeo debe superar al modelo que usa únicamente descriptores
estáticos. Si no lo supera, la telemetría no aporta valor predictivo suficiente
para justificar su costo en esta plataforma.

### H3 — Separación entre dispositivo y frecuencia

La mayor parte de la reducción potencial de EDP proviene de elegir el
dispositivo correcto. La frecuencia solo debe actuarse cuando su beneficio
estimado sea robusto frente a la incertidumbre experimental y al overhead de
la actuación. Esto no equivale a declarar que todo el eje DVFS carece de
señal: la estrategia sin sondeo presenta un margen mediano de frecuencia CPU
de 11,31 %, mientras que la elección fina en la estrategia con sondeo presenta
un margen mediano de 1,03 %. La pregunta pendiente no es solo si existe
variación, sino si un modelo supera una política constante como REF o F0
después de incorporar incertidumbre y overhead.

### H4 — Necesidad de amortización

La conveniencia de GPU para una primera ejecución depende del número esperado
de reutilizaciones. Una comparación de un único despacho frío no es suficiente
para formular la política de una operación repetida.

## 4. Alcance y exclusiones

### Incluido

- Generalización a tamaños no vistos de las seis operaciones conocidas.
- Estados `none_ready`, `cpu_ready` y `gpu_ready`.
- Costos `cold` y `warm` medidos, incluidas las transferencias presentes en
  los kernels actuales.
- Comparación CPU/GPU a frecuencia REF.
- DVFS condicionado por evidencia y con posibilidad de abstención.
- Validación offline y posterior validación extremo a extremo.

### No afirmado por este plan

- Generalización a una séptima operación desconocida.
- Generalización automática a otra plataforma de hardware.
- Optimización de aplicaciones con residencia de datos arbitraria sin modelar
  primero ese estado.
- Ganancia neta antes de medir inferencia, actuación y transferencias entre
  operaciones.
- Cambio de los objetivos formales del trabajo, de la plataforma experimental
  o del marco institucional sin confirmación explícita del autor y, cuando
  corresponda, del director.

## 5. Unidad experimental y tamaño efectivo

La unidad independiente será `config_id = operación × tamaño`, no la fila
candidata ni la ventana de telemetría.

El conjunto actual tiene:

- 68 `config_id`;
- 40 acciones medidas por configuración;
- 3 repeticiones por acción;
- 68 contextos para la estrategia sin recursos preparados;
- 136 contextos para la estrategia con sondeo, pero solo 68 configuraciones
  físicas subyacentes.

Todas las particiones deben mantener juntos:

- las 40 acciones de un `config_id`;
- sus repeticiones;
- sus regiones `cold` y `warm`;
- sus dos estados derivados de sondeo.

## 6. Arquitectura propuesta del selector

### 6.1 Capa de estado

El agente mantendrá explícitamente, como mínimo:

- dispositivo inicializado;
- dispositivo que produjo el sondeo de la operación;
- disponibilidad de telemetría previa de esa operación;
- tamaño de la operación;
- número esperado o estimado de invocaciones restantes;
- cuando se implemente la evaluación de secuencias, residencia de los datos.

El estado de inicialización debe ser global al proceso/dispositivo. La
telemetría de sondeo sí puede conservarse por operación y tamaño.

### 6.2 Política para `none_ready`

La decisión inicial no se formulará únicamente con el EDP de un despacho
frío. Para cada dispositivo y horizonte `K` se calculará:

```text
E_total(d, K) = E_cold(d) + (K - 1) * E_warm(d)
T_total(d, K) = T_cold(d) + (K - 1) * T_warm(d)
EDP_total(d, K) = E_total(d, K) * T_total(d, K)
```

El punto de amortización es el menor `K` para el cual el EDP total de GPU es
menor que el de CPU. Si no existe dentro del horizonte analizado, la política
permanece en CPU.

La primera versión será una tabla empírica por operación y tamaño. Después se
evaluará si un modelo puede predecir el punto de amortización para tamaños no
vistos.

En el alcance obligatorio, `K` se tratará como una entrada conocida,
suministrada por la aplicación o por el escenario experimental. Estimar `K`
en línea constituiría un segundo problema de predicción y se conserva como
extensión. Los resultados principales se reportarán como función de `K`.

`K_break_even` no se presentará como un entero exacto sin incertidumbre. Los
costos `cold` y `warm` se estiman a partir de repeticiones experimentales y la
región fría contiene acciones de baja resolución temporal: 5,9 % de las
acciones de la estrategia sin sondeo están marcadas de baja resolución y
excluirlas modifica 2 de 68 ganadores. Esa incertidumbre debe propagarse para
obtener un intervalo o análisis de sensibilidad de `K_break_even`.

### 6.3 Política para `cpu_ready`

> **Supersedida para `K>1` por la enmienda 2026-08-30-A.** La regla fija de
> esta subsección describe únicamente `K=1`. Para horizontes mayores se aplica
> `argmin_d EDP_total(d,K|estado)`: 22 de las 68 configuraciones migran a GPU
> desde `cpu_ready` dentro del horizonte explorado.

En los datos actuales CPU resulta óptima en los 68 grupos `cpu_ready`. Por
tanto, esta rama comenzará como una regla explícita: permanecer en CPU.

No se entrenará un modelo para reproducir una clase constante. La regla se
revisará únicamente si los tamaños confirmatorios nuevos producen cruces de
dispositivo en este estado.

### 6.4 Política para `gpu_ready`

Esta es la tarea de aprendizaje principal del conjunto actual. Hay 68 grupos:

- 56 donde conviene permanecer en GPU;
- 12 donde conviene cambiar a CPU.

Se construirá una fila por `config_id` y se predecirá la razón de costos a
frecuencia REF:

```text
y = log(EDP_GPU_REF / EDP_CPU_REF)
```

- `y < 0`: permanecer en GPU;
- `y > 0`: cambiar a CPU;
- valores cercanos a cero: abstenerse y aplicar la política segura definida
  para el estado.

Se priorizará regresión porque conserva la magnitud del error. Los modelos
iniciales serán deliberadamente simples:

- regresión Ridge;
- Elastic Net;
- regresión Huber;
- árbol de regresión pequeño;
- Random Forest regressor como comparación no lineal.

También se construirá una variante de clasificación binaria para contrastar,
pero no se asumirá que es superior.

> **Nota 2026-08-30 (post-R2), ver §6.4-bis más abajo.** El resultado de R2
> con esta formulación directa (regresor sobre `y` completo, evaluado sobre
> la rejilla de `K` de la enmienda 2026-08-30-A del protocolo congelado) fue
> mixto: gana a la mejor baseline en 2 de 48 rebanadas de
> `(régimen, estado, K)` -- justo en la zona de cruce de dispositivo (`K=3`,
> `cpu_ready`/`none_ready`) -- y pierde de forma inestable en valores de `K`
> cercanos (hasta -72,4 % en `none_ready` `K=10`). El diagnóstico y la
> reformulación estructurada que responde a esto quedan en §6.4-bis; esta
> sección se conserva sin editar como registro de lo que efectivamente se
> ejecutó primero.

### 6.4-bis Reformulación estructurada de la política de horizonte (nota 2026-08-30, post-R2)

**Diagnóstico.** `y = log(EDP_GPU_REF / EDP_CPU_REF)` no es una cantidad
primitiva: es una función cerrada de ocho costos medibles por separado (`E` y
`T`, fríos y calientes, por dispositivo), compuestos según la fórmula de
`EDP_total(d, K)` de §12.1 del protocolo congelado. Pedirle a un regresor que
aprenda `y` directamente, para toda la rejilla de `K` a la vez, con 68
`config_id`, es pedirle que redescubra esa composición sin dársela -- de ahí
la inestabilidad entre valores de `K` cercanos observada en R2.

Verificado sobre los datos exploratorios, antes de tocar ningún dato
confirmatorio:

- el costo **caliente** (`E_warm`, `T_warm`) sigue una ley de potencias casi
  perfecta en `log(costo) ~ log(N)` por operación: R² entre 0,974 y 0,998 en
  las cuatro combinaciones dispositivo×magnitud, en las seis operaciones;
- el costo **frío** no correlaciona con el tamaño de la misma forma (R² entre
  0,000 y 0,918, errático) porque está dominado por un término de arranque
  aproximadamente constante: el arranque de GPU en tiempo tiene mediana
  0,618 s con CV 0,09 a través de las seis operaciones y trece tamaños; el de
  CPU es un orden de magnitud menor y sí depende de la operación;
- la telemetría de una única ejecución de sondeo coincide con el costo
  **frío** medido del dispositivo que la produjo (error relativo mediano
  4,88 %, dentro del piso de ruido de la región fría, 5,76 %) -- no con el
  costo caliente. Esto le da al sondeo un rol estructural preciso: mide
  directamente uno de los ocho costos primitivos, no una aproximación de la
  razón final.

**Reformulación propuesta.** Reemplazar el objetivo de aprendizaje único por
tres capas:

1. **Predicción**: cuatro primitivas de costo **calientes** (`E_warm`,
   `T_warm` × CPU, GPU), como función de `(operación, tamaño)` -- tarea de
   regresión simple, ya validada con R² > 0,97 en la mayoría de las
   operaciones.
2. **Calibración**: arranque por dispositivo. GPU como constante (con su
   incertidumbre); CPU por operación si la dependencia se sostiene bajo
   validación cruzada.
3. **Composición**: costo frío = caliente + arranque; `EDP_total(d, K)` y
   `K_break_even` se derivan con la fórmula ya congelada en §12.1 del
   protocolo, no se aprenden.

La decisión de dispositivo y el `K` de cambio salen de la capa 3, no de un
cuarto modelo. El sondeo, cuando está disponible, sustituye directamente la
primitiva fría predicha por la medida real del dispositivo que sondeó
(mejora esperable acotada por el 4,88 % de error ya verificado, no una
mejora libre).

**Qué modelos entran en cada capa.** No se introduce una familia nueva fuera
de las ya congeladas en §6.4: Ridge es el candidato natural para la capa 1
(un ajuste lineal en `log(costo)` contra `log(N)` es exactamente una ley de
potencias); árbol de regresión chico y Random Forest siguen disponibles como
contraste no lineal en la misma capa; Elastic Net y Huber quedan como
contraste de regularización/robustez. La capa 2 (arranque) puede no
necesitar ajuste de hiperparámetros -- los datos exploratorios sugieren una
constante, verificable con una media y su intervalo. **XGBoost no se agrega**:
ya se excluyó deliberadamente por el tamaño de muestra (n=68); el patrón de
sobreajuste local ya observado en Random Forest (gana en `K=3`, pierde hasta
-72 % en `K` cercanos) es una razón adicional para no incorporar un modelo
todavía más flexible antes de resolver la inestabilidad con una reformulación
más simple, no más compleja.

**Extensión condicionada a evidencia, no obligatoria todavía**: una cuarta
capa de corrección residual -- un modelo pequeño que aprenda el error entre
la ley de potencias ideal y el costo medido, usando telemetría de sondeo
como entrada -- queda registrada como extensión (mismo nivel que §19,
punto 5) a evaluar solo si la composición de tres capas dejara un residual
sistemático y aprendible después de aplicarse. No se implementa por
anticipado.

**Gobernanza.** Esta reformulación amplía §6.4 sin contradecir el núcleo
obligatorio de §14 (R2 sigue siendo obligatoria; cambia su implementación,
no su objetivo). Como toca la formulación del target que el protocolo
congelado fija en su §1, requiere su propia enmienda fechada en
`protocolo_congelado_confirmatorio_20260830.md` antes de tocar cualquier
dato confirmatorio -- verificado al redactar esta nota: los jobs 6763/6764
seguían sin producir datos. Ver la enmienda 2026-08-30-B en ese documento.

> **Corrección 2026-08-30-C.** Los conteos iniciales `2/48` y `4/48`
> seleccionaban la mejor familia y una fila de pliegue después de observar
> cada test. Quedan supersedidos por la evaluación de políticas únicas y
> pliegues pareados documentada en
> `resultados_selector_r2_corregidos_20260830.md`: la comparación de tres vías
> supera la baseline en 2/24 rebanadas de interpolación agregada, 0/24 en
> `extrapolation_top1` y 2/24 en `extrapolation_top2`. La formulación
> estructurada reduce la inestabilidad del target directo, pero no la elimina.

### 6.5 Política de frecuencia

La frecuencia no se tratará inicialmente como una clase exacta obligatoria.
Para cada grupo se distinguirán:

- frecuencia claramente separada;
- conjunto de frecuencias estadísticamente equivalentes;
- frecuencia indefinida por resolución insuficiente.

La actuación DVFS se habilitará solo si:

1. la ventaja supera la incertidumbre combinada de las mediciones;
2. la ganancia esperada supera el costo de actuación;
3. el modelo supera una política REF en validación externa.

Cuando estas condiciones no se cumplan, se utilizará REF. Esto es una
abstención deliberada, no un error del selector.

La capa DVFS forma parte del núcleo obligatorio porque el objetivo general
exige diseñar, implementar y evaluar ajuste dinámico de frecuencia. Un
resultado negativo es admisible —por ejemplo, demostrar que REF o F0 no son
superados después del overhead—, pero la capa no puede omitirse sin ser
implementada y evaluada. La selección de dispositivo complementa a DVFS; no lo
reemplaza como objeto del trabajo.

La implementación candidata reutilizará la separación estructurada de R2 sin
presuponer una forma funcional antes de contrastarla:

1. predecir tiempo y energía o potencia por nivel de frecuencia;
2. calibrar los términos que no escalan con frecuencia;
3. componer EDP por nivel y devolver un conjunto de frecuencias equivalentes,
   no un `argmin` puntual cuando las diferencias estén bajo incertidumbre.

Las formas tipo Amdahl para tiempo y potencia estática más un término
dependiente de frecuencia se tratarán como hipótesis candidatas que deben
compararse contra los datos y contra REF, no como leyes ya demostradas en esta
plataforma. No se incorporan aquí afirmaciones bibliográficas nuevas.

## 7. Características de entrada

### 7.1 Modelo estático

- identidad de la operación;
- tamaño y `log10(tamaño)`;
- FLOPs analíticos;
- bytes lógicos analíticos;
- intensidad aritmética analítica;
- estado del recurso;
- horizonte esperado `K` cuando aplique.

### 7.2 Modelo con sondeo

A las anteriores se añadirán características obtenidas de una única ejecución
real:

- tiempo y energía por despacho;
- potencia media;
- métricas CPU o GPU disponibles;
- indicadores de baja resolución y ausencia;
- dispositivo que produjo el sondeo.

La ausencia estructural de métricas CPU en sondeos GPU y viceversa se tratará
explícitamente. Se evaluará si es más claro mantener dos esquemas de
preprocesamiento por dispositivo que imputar todas las columnas en un único
vector.

### 7.3 Prevención de fuga

No podrán usarse como entrada:

- EDP de las acciones candidatas;
- acción ganadora;
- margen contra el segundo lugar;
- identificadores de corrida o repetición;
- estadísticas construidas con repeticiones que no existirían en despliegue;
- información de tamaños reservados para evaluación confirmatoria.

## 8. Partición y protocolo de validación

### 8.1 Validación principal: tamaños no vistos

Se utilizarán particiones agrupadas por `config_id` y estratificadas dentro de
cada operación.

Se reportarán dos regímenes:

1. **Interpolación:** tamaños internos retenidos mientras existen tamaños
   menores y mayores de la misma operación en entrenamiento.
2. **Extrapolación:** entrenamiento únicamente con tamaños menores y prueba en
   el extremo superior no observado.

La extrapolación es la prueba más importante para justificar la campaña de
tamaños grandes.

### 8.2 Evaluación confirmatoria con tamaños nuevos

Antes de medir los tamaños suplementarios se congelarán:

- formulación del target;
- características;
- familias de modelos;
- procedimiento de selección;
- baselines;
- métricas;
- regla de abstención.

Los tamaños nuevos no se utilizarán para ajustar hiperparámetros ni umbrales.
Serán un conjunto confirmatorio externo. Después de publicar su resultado se
podrá entrenar un modelo final con todos los datos, manteniendo separado el
resultado confirmatorio original.

### 8.3 Análisis secundario

`Leave-one-operation-out` se conservará únicamente para medir transferencia
fuera del dominio principal. Un resultado negativo allí no invalida la
generalización a tamaños, y uno positivo no bastará para afirmar
generalización abierta con solo seis operaciones.

## 9. Baselines obligatorias

Ningún modelo se considerará útil sin compararlo contra:

1. siempre CPU REF;
2. siempre GPU REF;
3. permanecer en el dispositivo preparado usando REF;
4. mejor acción constante estimada solo con entrenamiento;
5. umbral simple por tamaño;
6. umbral simple por intensidad aritmética;
7. tabla de cruce por operación construida solo con tamaños de entrenamiento;
8. oráculo con conocimiento posterior.

Para la frecuencia se añadirán:

- siempre REF;
- mejor frecuencia constante por dispositivo estimada en entrenamiento;
- no actuar si la mejora esperada no supera el costo de conmutación.

La selección final tendrá una regla bloqueante: si el modelo no supera la
mejor baseline pertinente en evaluación externa, se conserva la baseline y se
reporta que ML no aportó valor adicional.

## 10. Métricas y reporte de sanidad

### 10.1 Data card obligatorio

Antes de entrenar se generará un informe con:

- número de `config_id` y filas derivadas;
- tamaños disponibles por operación;
- distribución de dispositivo ganador por operación, tamaño y estado;
- distribución de acciones ganadoras;
- proporción de óptimos separados e inciertos;
- márgenes CPU/GPU y márgenes entre frecuencias;
- coeficiente de variación por acción;
- faltantes por característica y causa;
- cobertura de telemetría;
- balance de cada fold de entrenamiento y prueba.

### 10.2 Métricas del selector de dispositivo

- matriz de confusión;
- balanced accuracy;
- MCC;
- precision y recall de la decisión de migrar;
- error de la razón logarítmica de EDP;
- regret medio, mediano, p95 y máximo;
- porcentaje del ahorro del oráculo capturado;
- resultado desglosado por operación y régimen de tamaños;
- cobertura y calidad de la abstención.

### 10.3 Métricas operativas

- latencia p50, p95 y p99 de inferencia;
- tiempo de actuación de frecuencia;
- costo energético de inferencia y actuación;
- número mínimo de despachos para amortizar el control;
- EDP bruto y EDP neto después de overhead.

### 10.4 Dos agregaciones de EDP

Se reportarán por separado:

- suma o promedio del EDP por despacho, útil para comparar decisiones
  individuales;
- EDP de la secuencia completa:
  `(suma de energía) × (suma de tiempo)`.

No se presentará la suma de EDP individuales como si fuera automáticamente el
EDP de una aplicación completa.

## 11. Uso de los datos existentes antes de nuevas campañas

Antes de solicitar más horas de cómputo se deben producir cinco artefactos con
las 8.160 corridas ya aceptadas:

1. mapa `K_break_even` CPU/GPU por operación y tamaño;
2. dataset compacto de razón CPU/GPU a REF;
3. evaluación de baselines en particiones por tamaño;
4. comparación de modelo estático contra modelo con sondeo;
5. cuantificación del beneficio máximo adicional de DVFS después de elegir el
   dispositivo.

Estos resultados decidirán si la campaña suplementaria está justificada y qué
tamaños aportan información nueva.

Esta es la actividad inmediata y precede a cualquier lanzamiento `big`. En
particular, el mapa de amortización puede cambiar tanto los tamaños necesarios
como la necesidad misma de una campaña suplementaria. No se comprometerán
horas adicionales para completar una matriz antes de producir estos cinco
artefactos.

## 12. Campaña suplementaria de tamaños grandes

### 12.1 Propósito

La campaña no tendrá como objetivo “balancear clases” artificialmente. Su
propósito será observar y validar cruces de amortización y dispositivo en el
extremo superior de operaciones con escalamiento computacional relevante.

Las candidatas iniciales son:

- GEMM;
- FFT;
- Cholesky.

Los tamaños propuestos deben someterse primero a pruebas individuales de
memoria, duración y corrección. Ningún tamaño se incorporará solo porque
produzca la clase deseada.

### 12.2 Contrato de campaña

- manifiestos `big-only` nuevos;
- `campaign_id` y directorios de salida nuevos;
- preservación intacta de las campañas base;
- prueba REF del mayor tamaño por operación y dispositivo;
- estimación de duración a partir de esas pruebas;
- sesiones reanudables compatibles con el límite de Slurm;
- integración posterior de base + suplemento sin hardcodear 68
  configuraciones;
- caracterización GPU específica cuando se vayan a utilizar etiquetas
  Roofline para los tamaños nuevos.

La adquisición se realizará por etapas:

1. CPU REF y GPU REF para los tamaños aprobados, con tres repeticiones y el
   contrato `cold`/`warm` vigente;
2. evaluación confirmatoria del dispositivo y de `K_break_even`;
3. únicamente si existe headroom DVFS justificable, barrido reducido de
   frecuencias sobre las configuraciones informativas;
4. niveles adicionales solo alrededor de un mínimo o cruce observado.

No se ejecutará por defecto el producto cartesiano completo de ocho niveles
CPU por cuatro niveles de anfitrión y ocho niveles GPU. Completar todas las
celdas de la formulación anterior no es por sí solo una justificación
científica.

### 12.3 Uso estadístico

Los tamaños nuevos se reservarán primero como prueba confirmatoria. No se
emplearán para decidir retrospectivamente qué modelo o qué umbral reportar.

## 13. Validación extremo a extremo

La implementación y evaluación de un agente mínimo es obligatoria. Después de
la validación offline se construirá un controlador en espacio de usuario que,
como mínimo:

- reciba operación, tamaño, `K` y estado de inicialización;
- consulte o reciba la telemetría disponible;
- ejecute la inferencia del modelo o la baseline seleccionada;
- elija dispositivo y una política DVFS, incluida la abstención;
- aplique realmente la frecuencia mediante los mecanismos ya validados;
- registre costo de inferencia, actuación, energía y tiempo;
- restaure el estado del hardware ante finalización o señal.

La evaluación mínima puede usar operaciones o una secuencia controlada y no
requiere resolver la residencia arbitraria de datos de una aplicación HPC
general. Su flujo es:

```text
estado actual
  -> operación y tamaño
  -> decisión CPU/GPU
  -> decisión de frecuencia o abstención
  -> costo de inferencia
  -> costo de actuación
  -> costo de transferencia
  -> actualización de estado y residencia
```

Se compararán como mínimo:

- ejecución íntegra en CPU REF;
- ejecución íntegra en GPU REF;
- política estática de mejor dispositivo;
- política de permanecer en el dispositivo preparado;
- selector propuesto;
- oráculo offline.

La contribución se medirá sobre EDP neto de la secuencia, no solo sobre la
exactitud de la acción.

Un evaluador general de secuencias, con dependencias entre operaciones,
residencia explícita de datos y planificación global, se conserva como
extensión deseable. Esa ampliación no debe bloquear el controlador mínimo.

## 14. Fases de ejecución y nivel de compromiso

### Fase R1 — Reanálisis sin nuevas mediciones — obligatoria e inmediata

- construir el mapa de amortización;
- construir el target de razón CPU/GPU;
- generar el data card;
- implementar baselines;
- definir las particiones por tamaño;
- medir el headroom real de dispositivo y frecuencia.

**Salida:** diagnóstico documentado de la señal disponible, del margen máximo
frente a las baselines y de si se justifica medir tamaños adicionales. Este
diagnóstico **no puede cancelar ni omitir R2**: aunque la mejor baseline quede
tan cerca del oráculo que anticipe un resultado negativo para ML, se deben
entrenar y evaluar los modelos predeclarados. Lo condicionado por el resultado
no es la evaluación del modelo, sino su adopción dentro del agente final.

### Fase R2 — Prototipo de modelos — obligatoria

- entrenar regresores simples;
- comparar estático frente a estático + sondeo;
- calibrar abstención únicamente dentro del entrenamiento;
- evaluar interpolación y extrapolación existentes;
- medir latencia y tamaño del modelo.

**Salida:** modelo candidato o conclusión negativa frente a baselines.

### Fase R3 — Agente DVFS mínimo — obligatoria

- implementar la máquina de estados mínima;
- integrar el modelo o la política simple ganadora;
- integrar abstención y fallback;
- aplicar DVFS real en CPU y GPU;
- conservar restauración y trazabilidad;
- medir latencia de inferencia y actuación.

**Salida:** agente ejecutable; no solamente un modelo serializado.

### Fase R4 — Evaluación mínima extremo a extremo — obligatoria

- ejecutar operaciones o secuencias controladas;
- comparar gobernadores nativos, políticas constantes, agente y oráculo;
- incorporar inferencia y actuación al EDP neto;
- distinguir beneficio de dispositivo y beneficio de frecuencia;
- reportar también el resultado negativo si ninguna política aprendida supera
  las baselines.

**Salida:** respuesta experimental al objetivo general y al overhead del
control.

### Fase C1 — Preparación confirmatoria — condicionada por R1–R2

- congelar protocolo y criterios;
- crear campañas suplementarias separadas;
- extender el constructor para múltiples campañas;
- ejecutar pruebas máximas individuales;
- revisar presupuesto de Slurm.

**Salida:** campaña confirmatoria reproducible y segura.

### Fase C2 — Tamaños nuevos — condicionada por C1

- recolectar únicamente configuraciones suplementarias aprobadas;
- integrar sin modificar los datos base;
- ejecutar una sola evaluación confirmatoria congelada;
- documentar aciertos, fallos y cambio de dominio.

**Salida:** evidencia externa de generalización por tamaño.

### Fase E1 — Secuencias y residencia de datos — extensión deseable

- representar residencia y dependencias de datos entre operaciones;
- contabilizar transferencias inducidas por cada transición;
- evaluar secuencias más cercanas a una aplicación científica completa;
- comparar una política local contra planificación con conocimiento de la
  secuencia.

**Salida:** evaluación ampliada del selector sobre secuencias con estado de
datos.

### Fase E2 — Estimación en línea de `K` — extensión deseable

- estudiar si el horizonte puede inferirse o actualizarse en ejecución;
- comparar contra `K` suministrado por la aplicación;
- propagar el error de estimación a la decisión de amortización.

**Salida:** eliminación opcional del requisito de conocer `K` de antemano.

### Fase E3 — Operaciones no vistas — extensión exploratoria

- ampliar el catálogo con nuevas familias de operación;
- reemplazar o complementar la identidad one-hot con descriptores
  transferibles;
- reevaluar `leave-one-operation-out` con diversidad suficiente.

**Salida:** evidencia exploratoria de generalización fuera del dominio
cerrado; no se reclamará con las seis operaciones actuales.

## 15. Criterios de interpretación final

### Resultado positivo completo

El modelo de dispositivo supera las baselines en tamaños nuevos y conserva
beneficio después del overhead; la capa DVFS también aporta una mejora robusta.

### Resultado positivo parcial

El modelo selecciona bien CPU/GPU, pero DVFS no supera REF o su ganancia no
amortiza la actuación. Este resultado solo se considera suficiente después de
implementar y evaluar la capa DVFS: la contribución sería un agente que
selecciona dispositivo y se abstiene justificadamente de cambiar frecuencia
cuando no hay evidencia de ganancia neta. No sería suficiente omitir la capa
DVFS por anticipado.

### Resultado de política simple

Una tabla de cruces o un umbral simple iguala a los modelos. Se adopta la
política simple y se reporta que la complejidad ML no está justificada.
Esta conclusión solo puede declararse después de completar R2 y la evaluación
confirmatoria aplicable con el protocolo congelado; el diagnóstico preliminar
de R1, por sí solo, no autoriza abandonar el entrenamiento ni la evaluación de
los modelos.

### Resultado negativo

Ningún método supera permanecer en el dispositivo actual o una acción
constante después de overhead. Se reporta el límite experimental y no se
despliega un modelo sin utilidad demostrada.

Los cuatro resultados son científicamente válidos. El plan no presupone que
ML ni DVFS deban ganar.

## 16. Decisiones que requieren confirmación

Antes de convertir esta propuesta en contrato normativo deben confirmarse:

1. que la afirmación principal será generalización a tamaños no vistos dentro
   de seis operaciones conocidas;
2. que el horizonte esperado `K` se tratará como entrada suministrada por la
   aplicación en el núcleo obligatorio, dejando su estimación en línea como
   extensión;
3. qué información de residencia de datos puede conocer el agente;
4. si el resultado positivo parcial —agente implementado, DVFS evaluado y
   abstención justificada— satisface el alcance esperado por el director;
5. cómo reconciliar esta formulación con la redacción vigente del segundo
   objetivo específico sin modificarlo unilateralmente;
6. qué tamaños suplementarios son seguros después de las pruebas máximas.

## 17. Criterio de finalización

Esta reformulación estará cerrada cuando exista evidencia reproducible para
responder, por separado, las siguientes preguntas:

1. ¿El selector generaliza a tamaños no vistos de una operación conocida?
2. ¿La telemetría mejora la decisión frente a descriptores estáticos?
3. ¿La selección de dispositivo supera las políticas simples?
4. ¿DVFS añade beneficio después de incertidumbre y actuación?
5. ¿La política completa reduce el EDP neto de una secuencia frente a CPU,
   GPU y permanencia en el dispositivo actual?
6. ¿El agente en espacio de usuario aplica y restaura DVFS correctamente con
   un overhead medido?

Hasta entonces, un modelo entrenado y serializado será un artefacto de
experimentación, no evidencia suficiente de utilidad del agente.

## 18. Alineación con los objetivos del trabajo de grado

### 18.1 Objetivo general — inalterado

El objetivo general se conserva sin modificación. En consecuencia, el núcleo
obligatorio debe contener caracterización, diseño, implementación y evaluación
de un agente en espacio de usuario basado en modelos ligeros, con actuación
DVFS real sobre CPU/GPU y medición de energía, rendimiento y overhead.

La selección de dispositivo y el análisis de amortización amplían la lógica de
decisión en un sistema heterogéneo, pero no sustituyen la obligación de
implementar y evaluar DVFS. Que DVFS no supere al gobernador nativo puede ser
un resultado válido; no evaluarlo no lo sería.

### 18.2 Reformulación propuesta de los objetivos específicos

Las siguientes redacciones son borradores para discusión. Quedan registradas
en este plan, pero **no se incorporarán al libro ni reemplazarán los objetivos
vigentes hasta recibir confirmación explícita del autor y la validación
académica que corresponda**.

#### Objetivo específico 1 — caracterización

> Caracterizar el comportamiento computacional, temporal y energético de
> cargas científicas representativas bajo distintos estados de frecuencia y
> estados de inicialización de CPU y GPU, recolectando telemetría mediante
> contadores de rendimiento e interfaces estándar de potencia, y distinguiendo
> los costos de primera ejecución y ejecuciones posteriores.

Mantiene la semántica del objetivo original y hace explícito el contrato
`cold`/`warm` que resultó determinante experimentalmente.

#### Objetivo específico 2 — modelo

> Entrenar y validar modelos clásicos y ligeros de Aprendizaje Automático que,
> a partir de descriptores de la operación y telemetría obtenida en tiempo de
> ejecución, estimen su respuesta energía–rendimiento y seleccionen una
> política de dispositivo y frecuencia para fases computacionales conocidas
> con tamaños no observados, manteniendo una baja latencia de inferencia.

Los kernels continúan representando fases computacionales completas. Cambia
el target operativo: en lugar de limitarse a clasificar cada ventana como
`compute_bound` o `memory_bound`, el modelo estima la decisión que alimenta al
controlador. La caracterización Roofline se conserva como explicación y
auditoría, no necesariamente como etiqueta de entrenamiento.

#### Objetivo específico 3 — agente

> Desarrollar un servicio de control en espacio de usuario que mantenga el
> estado de los recursos, procese la telemetría disponible, ejecute la
> inferencia del modelo y aplique políticas de DVFS sobre CPU y GPU, incluyendo
> una política explícita de abstención cuando la ganancia esperada no compense
> la incertidumbre o el costo de actuación.

La selección de dispositivo puede formar parte de la política del servicio,
pero no elimina su responsabilidad de aplicar y restaurar frecuencias.

#### Objetivo específico 4 — evaluación

> Evaluar empíricamente el agente mediante el Producto Energía–Retardo,
> comparándolo con los gobernadores nativos, políticas estáticas y un oráculo
> offline, e incorporando el costo de inferencia y actuación para determinar
> si las decisiones del modelo producen una mejora energética neta sin una
> penalización severa de rendimiento.

Este objetivo exige al menos una evaluación controlada del agente actuando
sobre hardware; una comparación exclusivamente offline entre modelos no es
suficiente.

### 18.3 Juicio de continuidad semántica

La reformulación permanece dentro de la semántica del proyecto siempre que se
mantengan conjuntamente:

- agente real en espacio de usuario;
- modelos ligeros guiados por telemetría;
- plataforma heterogénea CPU–GPU;
- actuación DVFS efectiva;
- comparación contra gobernadores nativos;
- EDP y rendimiento global incluyendo overhead.

El proyecto se desviaría del objetivo general si terminara únicamente como un
selector offline CPU/GPU a frecuencia REF o como un estudio de curvas de
amortización sin implementar y evaluar el agente.

## 19. Registro de extensiones deseables

Si el cronograma lo permite, se intentarán en este orden:

1. campaña confirmatoria reducida de tamaños no vistos, comenzando por REF y
   ampliando frecuencias solo donde exista headroom;
2. secuencias con residencia y dependencias de datos explícitas;
3. estimación en línea del horizonte `K`;
4. incorporación de operaciones adicionales y evaluación de transferencia a
   operaciones no vistas;
5. política adaptativa de repeticiones para configuraciones con márgenes
   estrechos;
6. planificación global de una secuencia frente a decisiones locales por
   operación.

Estas extensiones deben registrarse como trabajo condicionado por tiempo y
evidencia. No pueden desplazar el núcleo obligatorio ni justificar campañas
costosas antes de completar R1.
