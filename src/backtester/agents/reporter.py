from __future__ import annotations
from pathlib import Path
from textwrap import dedent

from ..schemas import BacktestResult

class ReporterAgent:
    def __init__(self, out_dir: str = "reports"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_summary(self, spec_name: str, res: BacktestResult) -> str:
        p = self.out_dir / f"summary_{spec_name}.md"
        issues = ", ".join(res.issues) if res.issues else "none"
        diag_lines = "\n".join(
            f"- {k}: {v:.4f}"
            for k, v in sorted(res.diagnostics.items())
            if isinstance(v, (int, float))
        )
        content = dedent(
            f"""
            # Result: {spec_name}

            | Metric | Value |
            | ------ | ----- |
            | Annual Return | {res.ann_return:.3%} |
            | Annual Vol | {res.ann_vol:.3%} |
            | Sharpe | {res.sharpe:.3f} |
            | Max Drawdown | {res.max_dd:.2%} |
            | Turnover | {res.turnover:.3f} |
            | Trades | {res.trades} |

            Diagnostics:
            {diag_lines or '-'}

            Issues detected: {issues}
            """
        ).strip() + "\n"
        p.write_text(content)
        return str(p)

    def write_failure(self, spec_name: str, reason: str, logs: list[str]) -> str:
        p = self.out_dir / f"failure_{spec_name}.md"
        log_block = "\n".join(f"- {line}" for line in logs) or "- no logs recorded"
        content = dedent(
            f"""
            # Failure Report: {spec_name}

            **Reason:** {reason or "unspecified"}

            ## Recent Logs
            {log_block}
            """
        ).strip() + "\n"
        p.write_text(content)
        return str(p)
