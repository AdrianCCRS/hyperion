"""Paridad del actuador C++ con common/hpc/freqctl.py (Bloque C8).

Corre la misma secuencia de niveles con `cpu_freq_actuator_probe` (C++) y con
`freqctl.apply_frequency` (Python, estrategia bounded_range) sobre dos arboles
sysfs simulados identicos, y compara el estado de cada CPU (min/max) tras cada
paso y tras restaurar. Se omite si el binario de la sonda no esta construido.

Uso: python3 test_freqctl_parity.py <ruta a cpu_freq_actuator_probe>
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from common.hpc import freqctl  # noqa: E402

LO, HI = 800000, 3600000
CPUS = [0, 1]
SIBLINGS = {0: [0, 6], 6: [0, 6], 1: [1]}
SEQ = [3200000, 2900000, 3200000, 2300000]


def build_tree(root: Path) -> None:
    for cpu in (0, 1, 6):
        d = root / f"cpu{cpu}"
        (d / "cpufreq").mkdir(parents=True)
        (d / "topology").mkdir()
        (d / "topology" / "thread_siblings_list").write_text(",".join(map(str, SIBLINGS[cpu])) + "\n")
        for name, val in (("scaling_min_freq", LO), ("scaling_max_freq", HI), ("scaling_governor", "powersave"),
                          ("cpuinfo_min_freq", LO), ("cpuinfo_max_freq", HI)):
            (d / "cpufreq" / name).write_text(f"{val}\n")


def state(root: Path, cpus, label: str) -> list[str]:
    out = []
    for cpu in sorted(cpus):
        d = root / f"cpu{cpu}" / "cpufreq"
        out.append(f"{label} cpu{cpu} min={(d / 'scaling_min_freq').read_text().strip()} "
                   f"max={(d / 'scaling_max_freq').read_text().strip()}")
    return out


def run_python(root: Path) -> list[str]:
    env = SimpleNamespace(
        frequency_write_capable=True,
        frequency_control_strategy=freqctl.STRATEGY_BOUNDED,
        available_frequencies_khz=[LO, HI],
        smt_siblings={c: SIBLINGS[c] for c in CPUS},
        frequency_control_paths={
            c: {a: str(root / f"cpu{c}" / "cpufreq" / a) for a in ("scaling_governor", "scaling_min_freq", "scaling_max_freq")}
            for c in (0, 1, 6)
        },
    )
    original = freqctl.snapshot_original_state(CPUS, env)
    lines: list[str] = []
    for i, khz in enumerate(SEQ, 1):
        level = SimpleNamespace(id=f"k{khz}", mode="fixed", fraction=(khz - LO) / (HI - LO))
        freqctl.apply_frequency(CPUS, level, env, original=original)
        lines += state(root, original.cpus, f"paso{i}")
    assert freqctl.restore_original_state(original, env)
    return lines + state(root, original.cpus, "restaurado")


def main() -> int:
    probe = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if probe is None or not probe.exists():
        print("SKIP: sonda no construida")
        return 0
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        py_root, cc_root = Path(a), Path(b)
        build_tree(py_root)
        build_tree(cc_root)
        py_lines = run_python(py_root)
        proc = subprocess.run(
            [str(probe), "--root", str(cc_root), "--cpus", "0,1", "--khz-seq", ",".join(map(str, SEQ))],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"FALLO: la sonda salio con {proc.returncode}\n{proc.stderr}")
            return 1
        cc_lines = proc.stdout.strip().splitlines()
    if py_lines != cc_lines:
        print("FALLO: el estado difiere entre freqctl.py y el actuador C++")
        for p, c in zip(py_lines, cc_lines):
            print(("  ok   " if p == c else "  DIFF ") + f"py={p!r} cc={c!r}")
        return 1
    print(f"paridad OK: {len(py_lines)} lineas identicas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
