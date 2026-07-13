# Diagnóstico de arranque en SC3

Este diagnóstico carga y valida el manifest y el catálogo, y recopila el perfil
del nodo en modo solo lectura. No ejecuta kernels, no modifica sysfs y no requiere
permisos administrativos.

Desde la raíz del repositorio, crea el entorno Conda:

```bash
conda env create -f environment-hpc.yml
conda activate hyperion-hpc
```

Solicita una asignación interactiva. Ajusta la cuenta y la partición a las
autorizadas para tu usuario:

```bash
srun --partition=amd --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=2G --time=00:10:00 --hint=nomultithread --pty bash
```

Dentro de la asignación, desde la raíz del repositorio, ejecuta:

```bash
python -m orchestrator.diagnostics \
  --manifest orchestrator/schemas/campaign_sc3_audit.yaml \
  --output-dir artifacts/sc3-startup \
  --use-allowed-cpus
```

El comando imprime la ruta de `startup_diagnostic.json`. El informe incluye el
contexto Slurm y cgroup, los CPUs efectivos de la asignación, la topología NUMA,
SMT, cpufreq, RAPL, GPU y eventos perf visibles. La plantilla es de auditoría;
no debe reutilizarse como manifest de una campaña de medición.
