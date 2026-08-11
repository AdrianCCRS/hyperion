#!/bin/bash
echo "=== 1. version driver vs kernel module ==="
nvidia-smi --version
echo "---"
cat /proc/driver/nvidia/version 2>/dev/null || echo "no /proc/driver/nvidia/version"
echo ""
echo "=== 2. GPU Operation Mode ==="
nvidia-smi -i 0 -q | grep -A5 "GPU Operation Mode"
echo ""
echo "=== 3. MIG status ==="
nvidia-smi -i 0 -q | grep -A10 "MIG Mode"
echo ""
echo "=== 4. persistence mode ==="
nvidia-smi -i 0 -q | grep -i persist
echo ""
echo "=== 5. procesos de gestion/monitoreo GPU corriendo ==="
ps aux | grep -iE "dcgm|nv-hostengine|nvidia-smi|persistenced" | grep -v grep
echo ""
echo "=== 6. dmesg reciente relacionado a nvidia ==="
sudo -n dmesg 2>/dev/null | grep -i nvidia | tail -30 || echo "sin permiso dmesg o vacio"
echo ""
echo "=== 7. otros procesos usando la GPU ahora mismo ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo ""
echo "=== 8. clock policy / GOM completo ==="
nvidia-smi -i 0 -q -d CLOCK | head -50
