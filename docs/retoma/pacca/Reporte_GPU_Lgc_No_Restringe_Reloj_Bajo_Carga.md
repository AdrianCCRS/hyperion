# Reporte para el admin: `nvidia-smi -lgc` no restringe el reloj real de GPU bajo carga (paccaA100)

**Estado:** borrador, no enviado. Registrado en `docs/orchestator/agents/Registro_Cambios_Fuera_Plan_Original.md` como ARC-113.

**Contexto:** tras la confirmación del permiso P4 (bloqueo de reloj de GPU vía sudo delegado sobre `nvidia-smi`), el smoke test de la campaña GPU DVFS expuso que el candado de reloj no tiene efecto real sobre el hardware bajo carga de cómputo activa. Se agotaron todas las verificaciones posibles del lado del proyecto (3 rondas) antes de escribir esto.

---

**Asunto:** nvidia-smi -lgc no restringe el reloj real de GPU bajo carga (paccaA100)

Hola,

Confirmo que la delegación de sudo para `nvidia-smi -lgc/-rgc` funciona correctamente (sin pedir contraseña, comando exitoso). Sin embargo, encontramos que el candado de reloj no tiene ningún efecto medible sobre el reloj real de la GPU una vez que hay una carga de cómputo activa, y agotamos todas las verificaciones posibles de nuestro lado antes de escribirles.

Lo que probamos, todo dentro de una asignación exclusiva de Slurm (`--exclusive --gres=gpu:1`):

1. Con un kernel CUDA propio corriendo (sin cuBLAS, un bucle FMA simple) y utilización de GPU confirmada al 100% de forma sostenida (varios segundos), ejecutamos:
   ```
   sudo nvidia-smi -i 0 -lgc <objetivo>,<objetivo>
   ```

2. Probamos objetivos de 210, 400, 555, 900 y 1200 MHz (todos dentro del rango de relojes soportados que reporta `nvidia-smi -q -d SUPPORTED_CLOCKS`), tanto por encima como por debajo del reloj observado, con tres sintaxis distintas (candado a un punto, rango min/max, valor único sin coma).

3. En todos los casos, el comando reporta éxito (`"GPU clocks set to ... All done"`, código de salida 0), pero `nvidia-smi --query-gpu=clocks.sm` sigue reportando 765 MHz sin importar el objetivo pedido -- el mismo valor en todas las combinaciones.

4. Repetimos la prueba con dos binarios completamente distintos (un microbenchmark propio y una eliminación gaussiana de Rodinia) para descartar que fuera un problema específico de un kernel -- mismo resultado en ambos.

5. Probamos aplicar el candado ANTES de lanzar el kernel (no solo durante) -- mismo resultado: 765 MHz tanto en reposo como bajo carga sostenida.

6. Descartamos causas de nuestro lado:
   - Versión del módulo del kernel vs. la librería userspace: coinciden exactamente (`595.45.04` en ambos).
   - GPU Operation Mode: `N/A` (no aplica en esta generación).
   - MIG: deshabilitado (`Current` y `Pending` ambos `Disabled`).
   - Ningún daemon de gestión/monitoreo de GPU corriendo (revisamos procesos activos) más allá de `nvidia-persistenced`, que ya sabíamos que corre.
   - Ningún otro proceso usando la GPU en el momento de la prueba.
   - Sin throttling térmico ni de potencia activo (`nvidia-smi -q -d CLOCK/PERFORMANCE`), Auto Boost en `N/A`.

7. Intentamos aislar si el problema es específico de la herramienta `nvidia-smi` o de la librería NVML subyacente, escribiendo un programa en C que llama directamente a `nvmlDeviceSetGpuLockedClocks()` -- pero la delegación de sudo está (correctamente) restringida al binario `nvidia-smi` específicamente, así que no pudimos ejecutar esa prueba sin permisos más amplios que los que ya nos dieron. Si quieren que probemos esto para acotar más el diagnóstico, dejamos el programa listo para correrlo con su ayuda.

8. Dato adicional relevante: 765 MHz es exactamente el reloj de "boost" no gestionado que ya habíamos medido meses atrás, antes de tener ningún permiso de frecuencia (sin ningún candado aplicado). Es decir, con el candado supuestamente activo hoy, la GPU se comporta idéntico a como se comportaba sin ningún candado en absoluto -- el mecanismo no parece estar restringiendo nada en la práctica.

9. Hallazgo más concluyente: con la sintaxis exacta que nos indicaron (`sudo nvidia-smi -i $CUDA_VISIBLE_DEVICES -lgc 1005`), bajo 100% de utilización sostenida, revisamos `nvidia-smi -q -d PERFORMANCE` para ver las "Clocks Event Reasons" -- el campo que el propio NVIDIA documenta como la causa oficial de cualquier restricción de reloj. Con la GPU en esas condiciones (27°C, 76W de 250W disponibles, muy lejos de cualquier límite térmico o de potencia), TODAS las razones aparecen como "Not Active": Idle, Applications Clocks Setting, SW Power Cap, HW Slowdown, Sync Boost, SW Thermal Slowdown, Display Clock Setting. Es decir, el propio driver no reporta ninguna causa reconocida para que el reloj se mantenga en 765 MHz en vez de subir al valor solicitado -- no es un límite térmico, de potencia, ni de ninguna de las categorías que NVIDIA expone como diagnóstico.

10. Descartamos también que el mecanismo de lanzamiento del trabajo fuera la causa. Todas las pruebas anteriores usaban `srun -p GPU ... bash -c '...'` (un solo comando por invocación); repetimos la prueba reservando el nodo con una asignación persistente (`salloc --no-shell`) y accediendo por `ssh` directo al nodo, sin pasar por el mecanismo de lanzamiento de `srun` en absoluto -- mismo resultado exacto. Repetimos una vez más con una terminal pseudo-interactiva real adjunta (`ssh -tt` + `script`), para descartar que la ausencia de una terminal interactiva en las pruebas anteriores influyera -- mismo resultado otra vez. Slurm, como orquestador de la asignación, queda descartado como causa.

Driver Version: 595.45.04, CUDA Version: 13.2 -- notamos que es una versión relativamente reciente para un A100 en producción; no sabemos si hay algún cambio de comportamiento de `-lgc` en esta rama del driver que debamos tener en cuenta, o si hay algún mecanismo adicional (por ejemplo, a nivel de `nv-hostengine`/DCGM u otra política del clúster que no veamos desde nuestra sesión) que esté sobreescribiendo el candado.

¿Podrían confirmar si el candado de reloj (`-lgc`) debería estar restringiendo el reloj real bajo carga en este nodo, y si hay algo adicional que debamos configurar de nuestro lado?

Quedamos atentos.

Saludos
