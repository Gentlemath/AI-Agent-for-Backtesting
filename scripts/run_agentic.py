from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from backtester.orchestrator import Orchestrator
from backtester.tasks import allowed_tasks

def _read_user_prompt() -> str:
    """Interactively collect a prompt from stdin."""
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print(
        "Enter a JSON strategy specification or plain-text instructions "
        "(Ctrl-D / Ctrl-Z to submit, blank line to finish):"
    )
    lines: list[str] = []
    try:
        while True:
            line = input()
            if not line and lines:
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines).strip()

def _resolve_prompt(task: str | None, prompt: str | None, prompt_file: str | None) -> str:
    if prompt_file:
        return Path(prompt_file).read_text()
    if prompt:
        return prompt
    if task:
        return json.dumps({"task": task})
    user_text = _read_user_prompt()
    if user_text:
        return user_text
    return json.dumps({"task": allowed_tasks()[0]})

def main(task: str | None, prompt: str | None, prompt_file: str | None):
    payload = _resolve_prompt(task, prompt, prompt_file)
    orch = Orchestrator(
        data_path="data/etf",
        kb_root="src/backtester/kb",
        workdir=".adaptive_workdir",
    )
    state = orch.execute(payload)
    print("=== Agentic Backtest ===")
    print(f"Task: {state.spec.task}")
    print(f"Verdict: {state.verdict}")
    print(f"Metrics: {state.metrics}")
    print(f"Attempts: {state.attempts}")
    print(f"Tools: {state.tool_refs}")
    print(f"Artifacts: {state.code_artifacts[-1]}")
    print(f"Report: {state.figures[-1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agentic pipeline on a user prompt.")
    parser.add_argument("--task", help="Task name to seed the prompt.")
    parser.add_argument("--prompt", help="Raw JSON prompt string.")
    parser.add_argument("--prompt-file", help="Path to a file containing the prompt JSON.")
    args = parser.parse_args()
    main(args.task, args.prompt, args.prompt_file)
