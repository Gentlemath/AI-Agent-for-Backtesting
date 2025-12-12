from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence

from backtester.tasks import allowed_tasks

LOG_DIR = Path("logs")

def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _log_path() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"performance_compare_{_timestamp()}.log"

def _run_command(cmd: Sequence[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def _write_log(log_file: Path, header: str, cmd: Sequence[str], rc: int, stdout: str, stderr: str) -> None:
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"\n========== {header} ==========\n")
        fh.write(f"Command: {' '.join(cmd)}\n")
        fh.write(f"Exit code: {rc}\n")
        if stdout.strip():
            fh.write("\n[stdout]\n")
            fh.write(stdout.strip() + "\n")
        if stderr.strip():
            fh.write("\n[stderr]\n")
            fh.write(stderr.strip() + "\n")
        fh.write("================================\n")

def _print_progress(task: str, label: str, iteration: int | None, rc: int) -> None:
    iter_text = f" #{iteration}" if iteration is not None else ""
    status = "OK" if rc == 0 else f"FAIL({rc})"
    print(f"[{task}] {label}{iter_text}: {status}")

def main(task: str | None):
    tasks = [task] if task else list(allowed_tasks())
    log_file = _log_path()
    print(f"Logging detailed output to {log_file}")

    for t in tasks:
        print(f"\n=== Task: {t} ===")

        # # Agentic multi-attempt run (single invocation)
        # cmd_agentic = ["python", "scripts/run_adaptive_agent.py", "--task", t]
        # rc, stdout, stderr = _run_command(cmd_agentic)
        # _write_log(log_file, f"{t} - adaptive agent", cmd_agentic, rc, stdout, stderr)
        # _print_progress(t, "agentic", None, rc)

        # # Single-shot runs (5 times)
        # for i in range(1, 6):
        #     cmd_single = ["python", "scripts/run_single_shot_agent.py", "--task", t, "--rounds", str(i)]
        #     rc, stdout, stderr = _run_command(cmd_single)
        #     _write_log(log_file, f"{t} - single_shot run {i}", cmd_single, rc, stdout, stderr)
        #     _print_progress(t, "single_shot", i, rc)

        # Pure LLM runs (5 times)
        for i in range(1, 6):
            cmd_pure = ["python", "scripts/run_pure_llm.py", "--task", t, "--rounds", str(i)]
            rc, stdout, stderr = _run_command(cmd_pure)
            _write_log(log_file, f"{t} - pure_llm run {i}", cmd_pure, rc, stdout, stderr)
            _print_progress(t, "pure_llm", i, rc)

    print("\nAll runs complete. See log for details.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare adaptive agent vs single-shot vs pure LLM backtests.")
    parser.add_argument("--task", help="Optional single task to run. Defaults to all tasks.", default=None)
    args = parser.parse_args()
    main(args.task)
