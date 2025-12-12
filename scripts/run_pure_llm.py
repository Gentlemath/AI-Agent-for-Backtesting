from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

from backtester.agents.spec_guard import SpecGuardAgent
from backtester.llm import ArgonneLLM, LLMGenerationError
from backtester.schemas import StrategySpec

WORKDIR = Path(".pure_llm")

def _read_user_prompt() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Enter your strategy specification (Ctrl-D/Ctrl-Z to submit, blank line to finish):")
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
        text = Path(prompt_file).read_text().strip()
    elif prompt:
        text = prompt.strip()
    elif task:
        text = json.dumps({"task": task})
    else:
        text = _read_user_prompt()
    text = text.strip()
    if not text:
        raise SystemExit("No prompt supplied.")
    return text

def _build_system_prompt() -> str:
    return dedent(
        """
        You generate a stand-alone Python script that will be executed inside our repo.
        Requirements:
        - Import pandas as pd and numpy as np (if needed).
        - Import DataLoader from `backtester.utils.data_loader` and call `ensure_symbols(universe, start_date, end_date)` to fetch a price panel.
        - Do not rely on other internal helpers (no backtester.kb.* etc).
        - Define `SPEC` as provided, implement `run_strategy(prices: pd.DataFrame, spec: dict) -> dict`.
        - Implement `main()` that instantiates `DataLoader(disk_dir="data")`, calls `ensure_symbols` with the SPEC window/universe, and runs the strategy.
        - Respect `spec["frequency"]`: for "weekly" resample to weekly closes/returns; for "daily" stay on daily data.
        - Smooth positions over the holding period (rolling mean) before normalizing.
        - Normalize weights via custom code (unit gross leverage scaled by max_leverage) and compute turnover using those weights.
        - You cannot use today's data to make today's trading decisions. Always shift signals/weights by one period to avoid lookahead bias.
        - Implement the strategy logic as described in SPEC["signal"] and SPEC["rules"].
        - Compute metrics: annualized return, annualized vol, Sharpe, max drawdown, turnover, hit rate, profit factor.
        - At the end of `main()`, print one line per metric plus a JSON-like summary so we can capture the output.
        - Include `if __name__ == "__main__": main()`.
        Output only the Python code (no explanation, no Markdown).
        """
    ).strip()

def _build_user_prompt(spec: StrategySpec, raw_prompt: str) -> str:
    spec_json = json.dumps(spec.model_dump(mode="json"), indent=2)
    return dedent(
        f"""
        StrategySpec JSON:
        {spec_json}

        Additional user instructions:
        {raw_prompt}

        Use only pandas/numpy/DataLoader plus standard library. Print metrics to stdout.
        The method in DataLoader to fetch prices is `ensure_symbols(universe, start_date, end_date)`.
        universe is a list of ticker strings; start_date and end_date are "YYYY-MM-DD" strings.
        """
    ).strip()

def _extract_code(raw: str) -> str:
    if raw is None:
        raise LLMGenerationError("LLM response did not include any text content.")
    matches = re.findall(r"```(?:python)?\n(.*?)```", raw, flags=re.DOTALL)
    return matches[-1] if matches else raw

def _generate_script(spec: StrategySpec, raw_prompt: str, rounds:int) -> Path:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    llm = ArgonneLLM()
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(spec, raw_prompt)
    response = llm.call_reasoning_api(user=user_prompt, system=system_prompt)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMGenerationError(f"Malformed Argonne response: {response}") from exc
    code = _extract_code(content).strip()
    if not code:
        raise LLMGenerationError("LLM returned empty script.")
    path = WORKDIR / f"pure_llm_{spec.name}_attempt{rounds}.py"
    path.write_text(code + "\n")
    return path

def _run_script(path: Path) -> tuple[int, str, str]:
    env = os.environ.copy()
    src = str((Path(__file__).resolve().parent.parent / "src").resolve())
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env and env["PYTHONPATH"] else "")
    proc = subprocess.run(
        ["python", str(path)],
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr

def main(task: str | None, prompt: str | None, prompt_file: str | None, rounds: int) -> None:
    prompt_text = _resolve_prompt(task, prompt, prompt_file)
    guard = SpecGuardAgent()
    spec = guard.validate_and_struct(prompt_text)
    script_path = _generate_script(spec, prompt_text, rounds)
    print(f"[pure-llm] Generated script at {script_path}")
    rc, stdout, stderr = _run_script(script_path)
    if rc == 0:
        print("[pure-llm] Execution succeeded. Output:")
        print(stdout.strip() or "(no stdout)")
        if stderr.strip():
            print("\n[pure-llm] stderr:")
            print(stderr.strip())
    else:
        print(f"[pure-llm] Execution failed with exit code {rc}.")
        if stdout.strip():
            print("\n[stdout]")
            print(stdout.strip())
        if stderr.strip():
            print("\n[stderr]")
            print(stderr.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and execute a pure LLM-coded backtest.")
    parser.add_argument("--task", help="Task name to seed the prompt.")
    parser.add_argument("--prompt", help="Inline prompt string.")
    parser.add_argument("--prompt-file", help="File containing the prompt JSON/text.")
    parser.add_argument("--rounds", type=int, default=1, help="Number of attempts (default: 1).")
    args = parser.parse_args()
    main(args.task, args.prompt, args.prompt_file, args.rounds)
