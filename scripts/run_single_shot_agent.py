from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from backtester.agents.spec_guard import SpecGuardAgent
from backtester.agents.retriever import RetrieverAgent
from backtester.agents.coder import CoderAgent
from backtester.agents.code_verifier import CodeVerifierAgent
from backtester.agents.runner import RunnerAgent
from backtester.agents.test_result_verifier import BTVerifierAgent
from backtester.agents.reporter import ReporterAgent
from backtester.utils.data_loader import DataLoader
from backtester.schemas import StrategySpec, BacktestResult

DEFAULT_TOOLS = ["returns", "sharpe", "drawdown", "normalize_weights", "compute_turnover"]
KB_ROOT = "src/backtester/kb"
WORKDIR = ".sigle_shot_workdir"

def _read_user_prompt() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Enter your task specification (Ctrl-D/Ctrl-Z to submit, blank line to finish):")
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
        raise SystemExit("No prompt supplied to LLM baseline.")
    return text

def _log(msg: str) -> None:
    print(msg)

def _single_shot_backtest(prompt_text: str, rounds:int) -> tuple[str, StrategySpec | None, BacktestResult | None, str, list[str], str | None, str]:
    guard = SpecGuardAgent()
    retriever = RetrieverAgent(KB_ROOT)
    coder = CoderAgent(WORKDIR)
    codever = CodeVerifierAgent()
    loader = DataLoader()
    runner = RunnerAgent(data_loader=loader)
    verifier = BTVerifierAgent()
    reporter = ReporterAgent(out_dir="reports/sigle_shot")

    logs: list[str] = []
    spec: StrategySpec | None = None
    artifact_path = ""
    try:
        _log("Validating specification via guard...")
        spec = guard.validate_and_struct(prompt_text)
        _log(f"Spec resolved: {spec.name} ({spec.task}) {spec.start_date}→{spec.end_date} on {len(spec.universe)} symbols")
        tools = spec.tools or DEFAULT_TOOLS
        _log(f"Requested tools: {tools}")
        specs = retriever.fetch(tools)
        _log(f"Fetched {len(specs)} tool modules.")
        path, _code = coder.write_module(spec, specs, attempt=rounds)
        artifact_path = path
        logs.append(f"coder generated strategy at {path}")
        _log(f"Coder emitted module at {path}")
        codever.verify(path)
        logs.append("code verification passed")
        _log("Code verification passed.")
        _log("Running backtest...")
        result = runner.run(path, spec)
        _log("Backtest completed.")
        logs.append("runner executed successfully")
        ok, failures = verifier.evaluate(result)
        result.issues = failures
        _log(f"Verifier outcome: {'pass' if ok else 'fail'}; issues={failures}")
        if ok:
            report_path = reporter.write_summary(spec.name, result)
            _log(f"Summary report written to {report_path}")
            return "pass", spec, result, report_path, logs, None, artifact_path
        reason = f"verifier failed checks: {failures}"
        logs.append(reason)
        failure_path = reporter.write_failure(spec.name, reason, logs)
        _log(f"Failure report written to {failure_path}")
        return "fail", spec, result, failure_path, logs, reason, artifact_path
    except Exception as exc:
        reason = str(exc)
        spec_name = spec.name if isinstance(spec, StrategySpec) else "unknown"
        failure_path = ReporterAgent().write_failure(spec_name, reason, logs)
        return "fail", spec, None, failure_path, logs, reason, artifact_path

def _print_result(verdict: str, spec: StrategySpec | None, result: BacktestResult | None, report_path: str, artifact_path: str, reason: str | None):
    print("=== LLM Single-Shot Backtest ===")
    if spec:
        print(f"Task: {spec.task}")
    print(f"Verdict: {verdict}")
    if verdict == "pass" and result:
        metrics = {
            "ann_return": result.ann_return,
            "ann_vol": result.ann_vol,
            "sharpe": result.sharpe,
            "max_dd": result.max_dd,
            "turnover": result.turnover,
        }
        print("Metrics:")
        for key, value in metrics.items():
            if key in {"ann_return", "ann_vol", "max_dd"}:
                print(f"  - {key}: {value:.3%}")
            elif key == "turnover":
                print(f"  - turnover: {value:.3f}")
            else:
                print(f"  - {key}: {value:.3f}")
    elif reason:
        print(f"Reason: {reason}")
    if artifact_path:
        print(f"Artifact: {artifact_path}")
    print(f"Report: {report_path}")

def main(task: str | None, prompt: str | None, prompt_file: str | None, rounds: int):
    prompt_text = _resolve_prompt(task, prompt, prompt_file)
    verdict, spec, result, report_path, _logs, reason, artifact = _single_shot_backtest(prompt_text, rounds)
    _print_result(verdict, spec, result, report_path, artifact, reason)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the single-shot LLM baseline.")
    parser.add_argument("--task", help="Task name to seed the prompt.")
    parser.add_argument("--prompt", help="Inline prompt string.")
    parser.add_argument("--prompt-file", help="File containing the prompt (same JSON used by agentic run).")
    parser.add_argument("--rounds", type=int, default=1, help="Number of attempts (default: 1).")   
    args = parser.parse_args()
    main(args.task, args.prompt, args.prompt_file, args.rounds)
