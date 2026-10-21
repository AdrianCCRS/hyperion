# Guion de la sustentación (diapositivas 1 a 34)

Duración total: **30 minutos**. Ritmo de referencia: unas 130 palabras por minuto, más lento en
las cifras. Los textos están escritos para decirse en voz alta, no para leerse palabra por palabra.

## Presupuesto de tiempo

| Bloque | Diapositivas | Tiempo | Acumulado |
|---|---|---|---|
| Apertura | 1–2 | 0:35 | 0:35 |
| Contexto y problema | 3–7 | 2:40 | 3:15 |
| Estado del arte | 8–9 | 1:20 | 4:35 |
| Pregunta y objetivos | 10–13 | 1:50 | 6:25 |
| Fundamentos | 14–16 | 2:00 | 8:25 |
| Metodología | 17–25 | 7:25 | 15:50 |
| Resultados: Fases 1 a 3 | 26–34 | 7:50 | **23:40** |
| *Pendiente: Fase 4, conclusiones y cierre* | 35–43 | *≈ 6:20* | *30:00* |

Las diapositivas de índice que abren cada sección (3, 5, 8, 10, 14, 17, 26) se pasan con una frase de
transición, sin detenerse (unos 5 s cada una).

Convenciones: **[acción]** es lo que se hace o señala en pantalla; *[si hay tiempo]* es una frase
que se puede omitir si van atrasados.

---

## 1 · Portada · 0:20 (0:20)

Buenos días. Somos Yeison Adrián Cáceres y Ricardo Pérez, y presentamos nuestro trabajo de grado,
dirigido por el profesor Gilberto Díaz: un agente en espacio de usuario que ajusta la frecuencia de
la CPU y la GPU con modelos ligeros de aprendizaje automático.

## 2 · Contenido · 0:15 (0:35)

El recorrido es este: contexto y problema, estado del arte, pregunta y objetivos, fundamentos,
la metodología en cuatro fases, los resultados y las conclusiones.

## 3 · Sección: Contexto · 0:05 (0:40)

Empecemos por el contexto.

## 4 · La energía limita cuánto puede escalar un nodo CPU–GPU · 0:50 (1:30)

En computación de alto rendimiento los nodos combinan CPU y GPU, y los dos dispositivos comparten
la carga científica. Cada uno consume según su frecuencia y su actividad, y esa energía pesa en el
costo de operación y en la sostenibilidad.

La herramienta clásica para controlarla es DVFS, el escalado dinámico de voltaje y frecuencia.
**[señalar la tarjeta inferior]** Pero bajar la frecuencia no siempre conviene: el beneficio
depende de lo que la aplicación esté haciendo en cada fase.

## 5 · Sección: Problema · 0:05 (1:35)

¿Por qué depende de la fase?

## 6 · Una misma aplicación alterna fases · 0:50 (2:25)

**[señalar la franja]** Una aplicación científica no es homogénea: alterna fases.

**[tarjeta izquierda]** En una fase *compute-bound* el límite es el cómputo: si sube la
frecuencia, el tiempo baja. **[tarjeta derecha]** En una fase *memory-bound* el procesador pasa
esperando datos de memoria, y subir la frecuencia casi no cambia el tiempo.

Esa segunda fase es la oportunidad: ahí se puede bajar la frecuencia y ahorrar energía sin
perder rendimiento.

## 7 · Los gobernadores nativos no consideran qué limita la fase · 0:50 (3:15)

¿Qué hace hoy Linux? **[recorrer el flujo]** En nuestra plataforma el controlador es
`intel_pstate`, que ofrece dos gobernadores, *performance* y *powersave*. Deciden con la
utilización o con una política fija, y de ahí sale la frecuencia.

**[señalar la caja punteada]** Lo que no entra en la decisión es qué está limitando la fase, si
el cómputo o la memoria. Ese es el vacío que atacamos.

## 8 · Sección: Estado del arte · 0:05 (3:20)

Veamos qué ha hecho la literatura.

## 9 · La literatura resuelve el problema por partes · 1:15 (4:35)

**[recorrer la línea de tiempo de izquierda a derecha]**

Calore y colaboradores, en 2017, aplicaron DVFS por región en CPU y GPU, pero la decisión queda
atada a funciones instrumentadas en el código.

Guerreiro y Ali predicen cómo responde una GPU a DVFS, pero exigen caracterizar la aplicación
antes y solo cubren la GPU.

MCBound, en 2024, clasifica si una carga es *compute* o *memory-bound*, pero no actúa sobre la
frecuencia.

Simsek y DRLCap sí ajustan la frecuencia de la GPU en ejecución, aunque con instrumentación y solo
en GPU.

**[última columna]** Este trabajo junta las piezas: clasifica la fase y actúa en CPU y GPU,
durante la ejecución y sin modificar la aplicación.

## 10 · Sección: Pregunta y objetivos · 0:05 (4:40)

De ahí sale nuestra pregunta.

## 11 · Pregunta de investigación · 0:40 (5:20)

¿En qué medida un agente en espacio de usuario, guiado por modelos ligeros, permite ajustar la
frecuencia en CPU y GPU para reducir el consumo, sin que el costo de la inferencia degrade el
rendimiento, en comparación con los gobernadores nativos de Linux?

Tiene dos condiciones: que el agente ahorre energía, y que el propio agente no cueste más de lo
que ahorra.

## 12 · Objetivo general · 0:25 (5:45)

El objetivo general es caracterizar, diseñar, implementar y evaluar ese agente, buscando
eficiencia energética sin una penalización severa del rendimiento.

## 13 · Cuatro objetivos específicos, cuatro fases · 0:40 (6:25)

Se divide en cuatro objetivos, y cada uno es una fase del trabajo. **[señalar 1 a 4]**

Uno, caracterizar las cargas y su consumo en distintas frecuencias, con Perf, RAPL y NVML. Dos,
entrenar clasificadores ligeros de fase con baja latencia. Tres, construir el *daemon* que infiere
la fase y aplica DVFS. Y cuatro, evaluar con el producto energía–retardo si el ahorro compensa la
sobrecarga, frente a los gobernadores nativos.

## 14 · Sección: Fundamentos · 0:05 (6:30)

Dos conceptos sostienen el trabajo.

## 15 · Roofline: la intensidad operacional define la etiqueta · 1:05 (7:35)

El primero es el modelo Roofline, que nos da la etiqueta.

**[señalar la gráfica]** El rendimiento de un kernel está limitado por el menor de dos techos: el
de cómputo, *P pico*, o el de memoria, que es la intensidad operacional por el ancho de banda.
**[fórmula de *I*]** La intensidad operacional es cuántas operaciones de punto flotante se hacen
por cada byte traído de DRAM.

**[ridge]** Donde los techos se cruzan está el *ridge*. Por debajo, la fase es *memory-bound*; por
encima, *compute-bound*.

Los techos no los tomamos de la ficha técnica: los calibramos en el propio nodo, para cada nivel
de frecuencia y cada precisión.

## 16 · Bajar la frecuencia reduce la potencia, no siempre la energía · 0:50 (8:25)

El segundo concepto explica por qué bajar la frecuencia no siempre ahorra.

**[fila de potencia]** La potencia tiene un piso, *P cero*, que no depende del reloj, y una parte
dinámica que sí. **[barras]** Bajar la frecuencia solo recorta la parte dinámica.

**[energía]** La energía es potencia por tiempo: si el tiempo crece más de lo que baja la
potencia, la energía sube. **[EDP]** Por eso usamos el EDP, energía por tiempo, que premia ahorrar
energía pero penaliza alargar la ejecución.

## 17 · Sección: Metodología · 0:05 (8:30)

Pasemos a cómo lo hicimos.

## 18 · Cuatro fases encadenadas · 0:40 (9:10)

**[recorrer las fases]** La Fase 1 produce un conjunto de datos etiquetado con Roofline. La Fase 2,
los modelos, el umbral y la tabla de política. La Fase 3, el agente, como dos procesos en C++. Y la
Fase 4 lo evalúa con el EDP CPU+GPU frente a la base nativa.

**[pie]** Todo se midió en un nodo exclusivo, con un Xeon Gold 5315Y, seis núcleos delegados, y una
A100 PCIe de 40 GB.

## 19 · Fase 1: cada carga se mide en diez niveles de frecuencia · 0:50 (10:00)

Cada carga se mide en diez estados de frecuencia: el nativo y nueve niveles fijos. **[CPU]** En CPU
van de 3.2 a 0.8 GHz, con el turbo desactivado; **[GPU]** en GPU, de 1410 a 210 MHz. Cada
combinación se repite tres veces.

El catálogo tiene 30 familias algorítmicas en CPU y 16 en GPU. La telemetría viene de Perf y RAPL en
CPU, y de NVML en GPU.

## 20 · La etiqueta compara la intensidad medida con el ridge calibrado · 0:45 (10:45)

**[fila CPU]** En CPU la intensidad se calcula por intervalo, de unos 10 milisegundos, con FLOPs y
bytes de DRAM medidos por hardware. El *ridge* baja de 8.69 a 2.92 FLOP por byte al bajar el reloj.

**[fila GPU]** En GPU se calcula por kernel, fuera de línea, con un *ridge* de 14.97 en precisión
simple y 6.90 en doble.

## 21 · Fase 2: un clasificador ligero aproxima la etiqueta · 0:55 (11:40)

La etiqueta exige contar FLOPs y bytes. El agente no tiene esos contadores disponibles en
operación, y en GPU obtenerlos exige perfilar, lo que alarga mucho la ejecución. Por eso un
clasificador ligero aproxima la etiqueta con señales baratas.

**[variables]** En CPU usa seis variables de Perf, entre ellas IPC, MPKI, los ciclos detenidos por
memoria y la frecuencia. En GPU, tres señales de NVML: utilización de GPU, utilización de memoria y
su desviación.

**[prohibidas]** La intensidad, los FLOPs, los bytes y el *ridge* no pueden ser entradas, porque son
la etiqueta misma. **[salida]** El modelo responde *compute*, *memory* o *revisar*, cuando no está
seguro. Comparamos varios modelos, desde una regla mayoritaria hasta XGBoost.

## 22 · La validación retira familias completas · 1:25 (13:05)

**[pliegues]** Validamos dejando una familia fuera: en cada pliegue se retira una familia
algorítmica completa, se entrena con las demás y se evalúa sobre la retirada. Así medimos lo que
importa, cómo le va al modelo con un programa que nunca vio.

La métrica principal es la **exactitud balanceada por celda**. Una celda es una combinación de
familia y clase, por ejemplo «gaussian, *memory*». Para cada celda se calcula qué fracción acierta
el modelo, y la métrica es el promedio de todas las celdas.

¿Por qué esta y no otra? Por lo que necesitamos medir: cómo le va al modelo con un programa que
nunca vio, y no con las filas más abundantes. La exactitud común cuenta filas, así que mandan las
familias con más intervalos o corridas: un modelo que acierta las grandes y falla las pequeñas se
vería bien. La exactitud balanceada estándar, que promedia solo las dos clases, tampoco basta,
porque sigue dominada por las familias grandes. Con el promedio por celda, cada familia pesa lo
mismo, tenga cien filas o tres, y sirve aunque la familia tenga una sola clase. *[si hay tiempo]* La exactitud común se reporta igual, como
complemento, pero no se usa para elegir.

**[fórmula]** La decisión es selectiva: si la confianza *q* supera el umbral *tau*, se emite la
clase; si no, el modelo responde *revisar*.

**[tarjeta de política]** La política se deriva aparte del clasificador: para cada clase, el nivel
de frecuencia con mejor EDP frente a la base.

## 23 · Fase 3: dos procesos C++ cierran el lazo · 1:05 (14:10)

El agente son dos procesos en C++ que corren junto a la aplicación, sin modificarla.

**[fila CPU]** El de CPU lee Perf cada milisegundo, infiere con ONNX, decide con umbral 0.85 y una
histéresis de 50 ventanas, y escribe la frecuencia de los núcleos por *sysfs*.

**[fila GPU]** El de GPU muestrea NVML cada 50 milisegundos, decide sobre una ventana de 3 segundos
con umbral 0.90, y fija el reloj con `nvidia-smi`.

**[tarjetas inferiores]** Hay dos reglas de protección. Mientras la GPU está activa, la CPU no baja
de 3.2 GHz, para no frenar al acelerador. Y si el agente se interrumpe, devuelve el hardware a su
estado inicial.

## 24 · Fase 4: brazos comparables · 0:55 (15:05)

**[recorrer las filas]** Comparamos cuatro brazos. La base corre sin agente. La sombra ejecuta el
agente completo, muestrea y decide, pero no escribe ninguna frecuencia: así medimos cuánto cuesta
observar. El activo hace todo. Y F1 fijo no usa agente: fija el reloj de la GPU en F1 durante toda
la aplicación, para saber si clasificar aporta algo frente a una regla fija.

**[fórmula]** La métrica es el EDP CPU+GPU: la energía de los dos paquetes de CPU más la de la GPU,
por el tiempo. Los brazos se ejecutan en bloques aleatorizados.

## 25 · Seis escenarios · 0:45 (15:50)

**[grupo superior]** Cuatro escenarios usan aplicaciones armadas con el catálogo: la matriz
inicial, con las compuestas A y B; E-A y E-B, con fases de memoria largas de familias vistas e
inéditas; y E-C, que añade a E-A una fase de cómputo sensible al reloj.

**[grupo inferior]** Y dos aplicaciones externas que no armamos nosotros: LAMMPS, con cuatro
entradas, y CloverLeaf, una aplicación continua.

## 26 · Sección: Resultados · 0:05 (15:55)

Vamos a los resultados, en el mismo orden de las cuatro fases.

## 27 · Fase 1: el conjunto de datos cubre ambos regímenes · 1:10 (17:05)

**[transición]** Empezamos por lo que dejó la Fase 1: los datos.

**[tarjeta CPU]** El clasificador de CPU se entrenó con 42 458 intervalos de 30 familias, con un
tope de mil por familia y clase, y quedó prácticamente parejo entre las dos clases.

**[tarjeta GPU]** El de GPU, con 483 corridas de 16 familias, con más *memory* que *compute*.

**[pie]** Los dos ponderan cada celda de familia y clase, y vale la pena explicar cómo, porque
los datos están muy desparejos: hay familias con cientos de intervalos o corridas y otras con
unas pocas. Cada observación recibe un peso inversamente proporcional al tamaño de su celda,
normalizado a media uno. Así cada celda suma el mismo peso total, y una celda de tres corridas
cuenta tanto como una de decenas. No usamos el balanceo por clase global de la librería, porque
en GPU la mayoría de las familias tiene una sola clase y las grandes seguirían dominando
dentro de su clase. En CPU, además, se limita cada celda a mil intervalos; en GPU no hay tope, se
usan las 483 corridas completas. No remuestreamos ni generamos datos sintéticos. Y la misma
lógica gobierna la métrica: la exactitud promedia el acierto de cada celda, no el de cada fila.
*[si hay tiempo]* Como referencia, en CPU se midieron 1470 corridas y 1.31 millones de intervalos
elegibles.

## 28 · Fase 2: en CPU los modelos empatan; se eligió el más rápido · 0:50 (17:55)

**[transición]** Con esos datos, la Fase 2: ¿un modelo ligero aprende la etiqueta?

**[gráfica]** En CPU comparamos los modelos con el mismo protocolo y la misma métrica. Los de capacidad suficiente,
*extra trees*, *random forest* y XGBoost, tienen una exactitud prácticamente igual:
con 30 familias no podemos decir que uno sea mejor. Empatan.

**[latencia]** Elegimos XGBoost por latencia: clasifica un intervalo en 243 microsegundos, frente a
5.3 milisegundos de *random forest* y *extra trees*, unas 22 veces menos.

**[hiperparámetros]** Probamos además una búsqueda automática de los hiperparámetros de XGBoost, el
modelo que íbamos a desplegar. Mejoró 0.019, pero
es una diferencia pequeña que con estos datos no se distingue de que no haya mejorado nada. Por eso dejamos la configuración fija, que es la más simple
de reproducir.

## 29 · XGBoost: el modelo final de CPU · 1:45 (19:40)

**[configuración]** Este es el modelo final: XGBoost de cien árboles y profundidad seis, con las
seis variables de Perf y ponderado por celda. **[métricas]** La sensibilidad es, de las fases que
de verdad son de una clase, cuántas reconoce el modelo: reconoce 0.806 de las *memory* y 0.714 de
las *compute*. La exactitud común, 0.769, mira todos los intervalos juntos, y es más alta que la
métrica principal porque pesan más las familias grandes. F1 macro, AUC y Matthews confirman lo
mismo desde otros ángulos; la decisión se toma con la exactitud por celda.

**[0.728]** Con la familia retirada alcanza 0.728 de exactitud balanceada por celda. Es la cifra
honesta: si mezcláramos los intervalos al azar, el modelo reconocería algoritmos ya vistos y
subiría a 0.874.

**[0.738]** Con el umbral de 0.85 decide el 72.2 % de los intervalos y llega a 0.738.
*[si hay tiempo] La abstención no filtra los errores que el modelo comete con confianza alta en
familias que no conoce.*

**[techo de la exactitud]** ¿Por qué no más? Porque el límite está en la información que traen las seis señales,
no en el modelo. Hay cuatro pruebas de eso:

1. Un clasificador sin ajuste, de vecinos más cercanos, llega a 0.684: a menos de 0.045 del
   modelo final. Cambiar de modelo no ayuda.
2. Existen perfiles gemelos entre clases opuestas: una fase *memory* de `gap_pr` y una fase
   *compute* de `phasic` tienen contadores casi iguales (MPKI de 0.82 frente a 0.94). Ningún modelo
   que reciba solo esas señales puede separarlas.
3. Más familias sí ayudan: con 3 familias el resultado es 0.615 y con 23 sube a 0.718. Lo que
   gobierna es la diversidad del catálogo.
4. Y si le damos la intensidad operacional como entrada, llega a 0.999. Pero esa variable es la que
   define la etiqueta, y el agente no puede medirla en operación.

*[si hay poco tiempo, decir solo la 1 y la 4]* Los detalles están en el anexo A5.

## 30 · En GPU los modelos también empatan · 0:50 (20:30)

**[gráfica]** En GPU repetimos la comparación con el mismo protocolo y la misma métrica. Pasa lo
mismo que en CPU: *random forest*, *extra trees* y la regresión logística quedan prácticamente
igual, alrededor de 0.80. XGBoost queda por debajo.

**[latencia]** Aquí la latencia no desempata: *random forest* y *extra trees* tardan lo mismo,
unos 4.7 y 4.8 milisegundos por predicción. Para el agente de GPU eso no pesa: fijar el reloj ya
toma unos 50 milisegundos y decide sobre ventanas de 3 segundos.

**[hiperparámetros]** La búsqueda automática de hiperparámetros empeoró cuatro de los cinco
modelos ajustables, así que también dejamos la configuración fija.

## 31 · En GPU se eligió el modelo que no depende del reloj · 1:05 (21:35)

**[configuración]** El modelo final es un *random forest* con tres señales de NVML: utilización de
GPU, utilización de memoria y la variación de esta última. No usa el reloj ni la potencia, porque
el agente cambia el reloj, y un modelo que lo usara terminaría respondiendo a su propia acción.

**[métricas]** Se leen igual que en CPU. Reconoce 0.929 de las fases *memory* y 0.825 de las
*compute*. La exactitud común, 0.892, es más alta que la principal porque en GPU hay más corridas
*memory* y familias de tamaño desigual. F1 macro, AUC y Matthews coinciden.

**[0.819]** Con la familia retirada alcanza 0.819 de exactitud balanceada por celda.

**[0.900]** Con el umbral de 0.90 decide en promedio el 65.7 % de cada celda y llega a 0.900 sobre
lo que decide. *[si hay tiempo]* No se equivoca en ninguna fase *memory*, que es la clase en la
que el agente actúa.

## 32–33 · Fase 3: el piso de potencia decide la política · 1:10 (22:45)

**[transición]** Con el modelo dentro del agente, ¿qué decide en cada clase?

**[32, curva]** La respuesta la da el piso de potencia. En CPU, el 84 % de la potencia al reloj más
alto es piso: no baja aunque bajemos la frecuencia de los núcleos. Solo queda un 16 % para ganar.
En GPU el piso es el 55.8 %, así que hay mucho más margen.

**[33, fila CPU]** Por eso en CPU la política es no actuar en ninguna clase. En *compute* la mejor
opción gana solo 0.6 %, bajo el mínimo de 1 %; en *memory*, bajar a F1 empeora 5.4 %. Ni un
oráculo que eligiera la mejor frecuencia para cada kernel pasaría de 2.2 %.

**[fila GPU]** En GPU *compute*, bajar a F1 gana 2.1 % en promedio, pero en unas familias gana y
en otras pierde, así que tampoco se actúa.

**[celda verde]** La única celda accionable es GPU *memory*: fijar 1260 MHz mejora el EDP de la
GPU 8.9 %, en 7 de 8 familias. Esa cifra se calculó con las mismas familias; para un programa
nuevo esperamos 6.5 %, aunque con tan pocas familias no podemos descartar que sea cero.

*[si hay tiempo]* La clase marca la oportunidad, no la garantiza: `rodinia_gaussian` es *memory* y
pierde 27.3 % con F1.

## 34 · Inferir cuesta microsegundos; el proceso de CPU sí cuesta · 0:55 (23:40)

**[transición]** ¿Y cuánto cuesta tener al agente funcionando?

**[barras]** Inferir es casi gratis: el modelo de CPU responde en 16 microsegundos y el de GPU en
97. *[señalar la barra de 3 s]* Eso es nada frente a los 3 segundos que toma decidir una fase.
Lo más lento es actuar: cambiar el reloj de la GPU tarda unos 0.35 segundos.

**[Sin costo]** El agente de GPU, solo observando, deja el EDP igual que la base.

**[6–14 W]** El de CPU no: gasta entre 6 y 14 vatios de más sin cambiar ninguna frecuencia. Y no
es por la inferencia: el proceso lee 9 contadores en 6 núcleos cada milisegundo y decide unas
mil veces por segundo.

**[+4–6 %]** Por eso, en la matriz inicial, el agente de CPU activo empeoró el EDP CPU+GPU entre
4 y 6 %, con p = 0.0079. La conclusión es que el costo está en observar, no en actuar.

---

## Orden de Resultados por fases

| Fase | Diapositivas | Transición sugerida |
|---|---|---|
| Fase 1 | 27 datos | «Empezamos por lo que dejó la Fase 1: los datos.» |
| Fase 2 | 28–29 CPU · 30–31 GPU | «Con esos datos, ¿un modelo ligero aprende la etiqueta?» |
| Fase 3 | 32 piso de potencia · 33 política · 34 latencias y costo | «Con el modelo dentro del agente, ¿qué decide en cada clase?» · en la 34: «¿Y cuánto cuesta ejecutarlo?» |
| Fase 4 | 35 panorama de los seis escenarios · 36 fases de memoria largas (E-A y CloverLeaf) · 37 ¿clasificar o fijar F1? | «Con el agente funcionando, ¿ahorra energía frente a la base?» · en la 37: «¿Y hace falta clasificar, o basta con fijar F1?» |

## Nota para la diapositiva 36 (fases de memoria largas), a incorporar al escribir su guion

La diapositiva muestra solo el EDP CPU+GPU y la duración; esto se dice:
- Qué son los escenarios: E-A es una aplicación compuesta con familias vistas y fases de memoria de
  46 a 60 s; CloverLeaf es una aplicación externa continua (CUDA Fortran).
- De dónde viene el ahorro: la energía de la GPU baja 7.2 % (E-A) y 6.4 % (CloverLeaf). El EDP
  baja menos porque la CPU es el 64 % de la energía medida y el agente de GPU no la toca.
- 5 de 5 bloques es el mejor resultado posible con 5 bloques (p = 0.0625): dirección consistente,
  no confirmación estadística. No decir «significativo».
- El brazo sombra queda en paridad (0.997): el ahorro viene de actuar, no de observar.
- CloverLeaf tiene una sola decisión *memory* sostenida: se comportó como F1 fijo y no puntúa
  clasificación.

## Si preguntan: «¿por qué esa métrica y no otra?»

Respuesta corta: porque el agente se enfrentará a programas nuevos, y la métrica debe medir eso
sin que las familias grandes la inflen.

- **Exactitud común:** cuenta filas. Con 483 corridas de tamaño desigual y 1.31 millones de
  intervalos en CPU, las familias abundantes dominan. Es la que más sube (0.769 y 0.892) y por eso
  no sirve para elegir.
- **Exactitud balanceada estándar** (promedio de las dos clases): corrige la prevalencia de clase,
  pero no el volumen por familia, y el promedio sigue mandado por las familias grandes.
- **F1 macro:** se calcula sobre todas las filas juntas, con el mismo problema; se reporta como
  complemento.
- **AUC-ROC y Matthews:** miden la calidad de las puntuaciones y la correlación global, también
  agrupadas; no responden a «cómo le va con una familia nueva». Se reportan como complemento.
- **Por qué sirve la de celdas:** cada familia pesa igual, está bien definida cuando una familia
  tiene una sola clase, y se acopla al protocolo (familia retirada) y al peso del entrenamiento.
- **Límite que conviene admitir:** hay pocas familias (30 y 16), por eso los intervalos son
  anchos y varios modelos empatan. Y la cifra depende del protocolo: 0.728 con la familia
  retirada contra 0.874 mezclando al azar.

*Continúa en la diapositiva 35 (Fase 4, panorama de los escenarios). Quedan unos 6 minutos y 20
segundos para las diapositivas 35 a 43.*

---

## Preguntas probables (GPU, diapositivas 30 y 31; respaldo en el anexo A8)

**¿Por qué *random forest* y no *extra trees*, si empatan?**
Empatan en exactitud balanceada por celda (0.807 y 0.806) y en latencia, pero *random forest* es
más confiable en todo lo demás, en la misma ronda y con las mismas variables: reconoce más fases
de ambas clases (0.809 y 0.891 frente a 0.749 y 0.855), varía menos entre semillas (0.003 frente
a 0.013) y falla mucho menos en la familia más difícil, `dual_cholesky` (0.51 frente a 0.15).
*No decir que* extra trees *se auditó sin reloj: solo se auditó* random forest.

**¿Por qué no la regresión logística, si es la más rápida (0.4 ms)?**
Porque depende del reloj que el agente cambia. Al fijar el reloj en 1410 MHz cambia de clase el
26.7 % de sus predicciones. Con las mismas tres señales sin reloj ni potencia baja a 0.624, mientras
*random forest* llega a 0.819. Y 0.4 ms frente a 5 ms no importa con ventanas de 3 segundos.

## Preguntas probables (CPU, diapositivas 28 y 29)

**¿Por qué no se probó la búsqueda de hiperparámetros en los otros modelos?**
La búsqueda se evaluó en XGBoost, el modelo que se iba a desplegar, y en la regresión logística,
el otro candidato competitivo. En ninguno mejoró de forma distinguible de cero. Los demás modelos
ya empataban en exactitud y XGBoost se eligió por latencia, no por exactitud:
para que otro ganara, tendría que mejorar mucho más que +0.019, y a XGBoost no le sirvió. Además,
el límite no es el modelo (kNN sin ajuste: 0.684; perfiles gemelos). En GPU sí se probó en los cinco
modelos ajustables y empeoró cuatro. *No decir que se probó en todos en CPU.*

**¿Qué rangos se usaron?**
XGBoost: 50 a 300 árboles (de 25 en 25), profundidad de 2 a 10, tasa de aprendizaje de 0.01 a 0.5
(escala logarítmica) y peso mínimo por hoja de 1 a 10. Regresión logística: constante de
regularización C de 0.001 a 100 (escala logarítmica). Quince pruebas por pliegue para XGBoost y
siete para la regresión logística, con muestreador TPE de semilla fija. La búsqueda es anidada: se
hace dentro de cada pliegue sin ver la familia retirada, que se evalúa una sola vez. *Los rangos
salen del código, no del libro.*

**¿Quince pruebas no son pocas?**
El problema no era el presupuesto. El puntaje interno casi no variaba (0.734 a 0.761), mientras que
la exactitud sobre la familia retirada iba de 0.487 a 1.000: el criterio interno no distingue
configuraciones buenas de malas para una familia nueva. Más pruebas no lo habrían resuelto.

**¿Por qué no se adoptó la mejora de 0.019 del ajuste de hiperparámetros?**
Porque con estos datos no se distingue de que no hubiera mejora. De las 30 familias, 14 mejoraron y 8 empeoraron.
