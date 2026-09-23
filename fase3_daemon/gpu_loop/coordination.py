"""Señal de coordinación CPU-GPU real (Bloque C, ítem C4, Plan_Fase3_Daemon.md
§0.1). El plan original asumía una variable atómica compartida entre ambos
loops porque asumía un solo proceso -- eso ya no se sostiene: el loop de
GPU es `run_daemon.py` (Python) y el loop de CPU es `cpu_loop_main`
(C++), dos procesos separados (ver el docstring de
`fase3_daemon/cpu_loop/include/cpu_loop_consumer.hpp`, que ya dejaba esto
documentado como decisión de alcance pendiente).

Mecanismo elegido: un archivo de estado de un solo byte, escrito
ATÓMICAMENTE (`write` a un temporal + `os.replace`, nunca sobrescritura in
situ) por el loop de GPU en cada transición de fase, y leído por el loop
de CPU en cada tick (`fase3_daemon/cpu_loop/include/gpu_active_reader.hpp`,
el espejo en C++ de este módulo). `os.replace`/`rename` es atómico a nivel
de sistema de archivos en Linux (POSIX): un lector concurrente nunca ve un
archivo a medio escribir, solo el contenido viejo completo o el nuevo
completo -- evita la alternativa (escribir in situ) que sí podría dejar al
lector con una lectura parcial ("1" a medias, p.ej.) en una carrera real.

Por qué un archivo y no un socket/pipe: ambos procesos se lanzan por
separado (Slurm los somete como dos jobs, o un mismo job con dos pasos),
sin garantía de orden de arranque ni de que ambos compartan una tubería
heredada -- un archivo en una ruta conocida no requiere que ninguno de los
dos espere al otro para existir (el lector trata "no existe todavía" como
"GPU inactiva", el estado seguro por defecto, igual que antes de que esta
señal existiera)."""
from __future__ import annotations

import os
from pathlib import Path

_ACTIVE_BYTE = b"1"
_INACTIVE_BYTE = b"0"


class GpuActiveSignalWriter:
    """Escribe el estado activo/inactivo de la GPU en `path`, atómicamente.
    Sin estado propio salvo la ruta -- no cachea el último valor escrito ni
    evita escrituras redundantes (escribir "0" cuando ya era "0" es barato
    y mantiene el archivo con timestamp fresco, útil para depurar si el
    loop de GPU sigue vivo)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._tmp_path = path.with_suffix(path.suffix + ".tmp")
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, active: bool) -> None:
        self._tmp_path.write_bytes(_ACTIVE_BYTE if active else _INACTIVE_BYTE)
        os.replace(self._tmp_path, self._path)  # atómico en el mismo filesystem (POSIX rename())

    @property
    def path(self) -> Path:
        return self._path
