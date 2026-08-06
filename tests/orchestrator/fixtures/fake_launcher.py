#!/usr/bin/env python3
"""Fake telemetry_kernel_launcher for runner.py tests. No hardware, no perf.

Accepts the same CLI shape runner.build_command() produces and writes the
same three artifacts the real launcher writes (samples.csv, metadata.json,
stdout), so runner.py's parsing/merging logic is exercised without needing
the real C++ binary or PMU permissions.

Behavior is selected through FAKE_LAUNCHER_BEHAVIOR:
  ok   (default) - writes fixtures, exits 0, prints VERIFICATION SUCCESSFUL.
  fail            - exits 1 without VERIFICATION SUCCESSFUL.
  hang            - spawns a grandchild and sleeps well past any test timeout,
                     to exercise RUN-03/RUN-04 process-group cleanup.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec")
    parser.add_argument("--exec-args", default="")
    parser.add_argument("--perf-cpus")
    parser.add_argument("--pin-workload-cpus")
    parser.add_argument("--collector-cpu")
    parser.add_argument("--consumer-cpu")
    parser.add_argument("--interval-ns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cgroup-path")
    parser.add_argument("--no-perf", action="store_true")
    args = parser.parse_args()

    run_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    behavior = os.environ.get("FAKE_LAUNCHER_BEHAVIOR", "ok")

    if behavior == "hang":
        # A real grandchild, so tests can verify the whole process group
        # (not just this pid) is gone after runner.py's RUN-04 cleanup.
        subprocess.Popen(["sleep", "300"])
        time.sleep(300)
        return 0

    if behavior == "fail":
        sys.stderr.write("simulated failure\n")
        return 1

    with open(os.path.join(run_dir, "samples.csv"), "w", encoding="utf-8") as samples_file:
        samples_file.write("run_id,repetition,kernel,label,timestamp_ns,tag\n")

    metadata = {
        "run_id": args.run_id,
        "kernel": args.exec,
        "perf_attach_mode": "pid_inherit",
        "measured_pids": [os.getpid()],
        "samples_collected": 0,
        "push_retries": 0,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file)

    print("VERIFICATION SUCCESSFUL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
