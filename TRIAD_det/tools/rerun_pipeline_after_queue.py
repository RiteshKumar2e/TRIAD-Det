"""Wait for the master queue (tools/run_everything.py) to exit, then re-run the base-pipeline stages
that failed with CUDA out-of-memory on 2026-11-05 14:34 (predict, evaluate, figures, compare, export).
Runs alone on the GPU, with expandable CUDA segments. Logged to runs/run_everything.log.

    python tools/rerun_pipeline_after_queue.py --wait-pid 17700
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_everything as q  # noqa: E402


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    q.say(f"rerun_pipeline_after_queue: waiting for PID {args.wait_pid}")
    while alive(args.wait_pid):
        time.sleep(120)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    q.run(["run_all.py", "--stages", "predict,evaluate,figures,compare,export"],
          name="pipeline predict/evaluate/figures/compare/export (re-run after OOM)")
    q.refresh()
    q.say("rerun_pipeline_after_queue finished")


if __name__ == "__main__":
    main()
