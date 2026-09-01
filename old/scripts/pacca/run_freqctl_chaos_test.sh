#!/bin/bash
# FRQ-05/CAM-07: prueba de caos real de la restauración de frecuencia CPU.
# Debe lanzarse desde el nodo pacca (login Slurm), no desde paccaA100 directo.
set -e -o pipefail

export PYTHONPATH="/home/latorresn/hyperion:${PYTHONPATH:-}"
export CHAOS_SIGNAL="${CHAOS_SIGNAL:-TERM}"
case "$CHAOS_SIGNAL" in
  TERM) export CHAOS_EXPECTED_RC=143 ;;
  INT)  export CHAOS_EXPECTED_RC=130 ;;
  *)
    echo "CHAOS_SIGNAL debe ser TERM o INT" >&2
    exit 64
    ;;
esac

srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --gres=gpu:1 --exclusive \
  --time=00:10:00 bash -c '
set -e -o pipefail

module load gnu12/12.4.0 openblas/0.3.21 2>&1 || true
export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib:${LD_LIBRARY_PATH:-}"
source ~/hyperion-venv/bin/activate
cd ~/hyperion-kernels

evidence_dir="$HOME/yacacerest/freqctl_chaos_$(date +%Y%m%dT%H%M%S)_job${SLURM_JOB_ID}"
mkdir -p "$evidence_dir"

snapshot_sysfs() {
  destination="$1"
  : > "$destination"
  for cpu in 0 1 2 3 4 5; do
    root="/sys/devices/system/cpu/cpu${cpu}/cpufreq"
    governor="$(<"$root/scaling_governor")"
    minimum="$(<"$root/scaling_min_freq")"
    maximum="$(<"$root/scaling_max_freq")"
    printf "%s\t%s\t%s\t%s\n" "$cpu" "$governor" "$minimum" "$maximum" >> "$destination"
  done
}

snapshot_sysfs "$evidence_dir/before.tsv"
{
  date --iso-8601=seconds
  hostname
  printf "slurm_job_id=%s\n" "$SLURM_JOB_ID"
  printf "chaos_signal=%s\nexpected_rc=%s\n" "$CHAOS_SIGNAL" "$CHAOS_EXPECTED_RC"
  sha256sum \
    ~/hyperion/orchestrator/freqctl.py \
    ~/hyperion/orchestrator/campaign.py \
    ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_freqctl_chaos.yaml
} > "$evidence_dir/context.txt"

python3 -m orchestrator.cli run-campaign \
  --manifest ~/hyperion/orchestrator/schemas/campaigns/campaign_pacca_freqctl_chaos.yaml \
  --node-id pacca-a100 \
  --hostname "$(hostname)" \
  --reference-kernel-ref npb_mg \
  --campaign-timeout-seconds 480 \
  > "$evidence_dir/campaign.stdout" \
  2> "$evidence_dir/campaign.stderr" &
campaign_pid=$!
printf "campaign_pid=%s\n" "$campaign_pid" >> "$evidence_dir/context.txt"

applied=0
for _ in $(seq 1 600); do
  snapshot_sysfs "$evidence_dir/during.tmp.tsv"
  if awk '\''
    BEGIN { pinned=1; changed=0 }
    NR == FNR { before_gov[$1]=$2; before_min[$1]=$3; before_max[$1]=$4; next }
    $3 != $4 { pinned=0 }
    $2 != before_gov[$1] || $3 != before_min[$1] || $4 != before_max[$1] { changed=1 }
    END { exit !(pinned == 1 && changed == 1) }
  '\'' "$evidence_dir/before.tsv" "$evidence_dir/during.tmp.tsv"; then
    mv "$evidence_dir/during.tmp.tsv" "$evidence_dir/during.tsv"
    applied=1
    break
  fi
  if ! kill -0 "$campaign_pid" 2>/dev/null; then
    break
  fi
  sleep 0.1
done

if [[ "$applied" -ne 1 ]]; then
  set +e
  kill -"$CHAOS_SIGNAL" "$campaign_pid" 2>/dev/null
  wait "$campaign_pid"
  campaign_rc=$?
  set -e
  snapshot_sysfs "$evidence_dir/after.tsv"
  printf "campaign_rc=%s\nresult=APPLY_NOT_OBSERVED\n" "$campaign_rc" >> "$evidence_dir/context.txt"
  echo "FRQ-05 FAIL: no se observó el nivel fixed; evidencia: $evidence_dir" >&2
  exit 1
fi

kill -"$CHAOS_SIGNAL" "$campaign_pid"
set +e
wait "$campaign_pid"
campaign_rc=$?
set -e

sleep 0.5
snapshot_sysfs "$evidence_dir/after.tsv"
printf "campaign_rc=%s\n" "$campaign_rc" >> "$evidence_dir/context.txt"

if [[ "$campaign_rc" -ne "$CHAOS_EXPECTED_RC" ]]; then
  printf "result=UNEXPECTED_SIGNAL_STATUS\n" >> "$evidence_dir/context.txt"
  echo "FRQ-05 FAIL: SIG${CHAOS_SIGNAL} produjo rc=$campaign_rc, se esperaba $CHAOS_EXPECTED_RC; evidencia: $evidence_dir" >&2
  exit 1
fi

if ! cmp -s "$evidence_dir/before.tsv" "$evidence_dir/after.tsv"; then
  printf "result=RESTORE_MISMATCH\n" >> "$evidence_dir/context.txt"
  diff -u "$evidence_dir/before.tsv" "$evidence_dir/after.tsv" \
    > "$evidence_dir/restore.diff" || true
  echo "FRQ-05 FAIL: sysfs no volvió al snapshot; evidencia: $evidence_dir" >&2
  exit 1
fi

printf "result=PASS\n" >> "$evidence_dir/context.txt"
echo "FRQ-05 PASS: nivel fixed observado, SIG${CHAOS_SIGNAL}=$campaign_rc y snapshot restaurado"
echo "EVIDENCE_DIR=$evidence_dir"
'
