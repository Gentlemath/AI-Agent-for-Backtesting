from __future__ import annotations

from typing import Iterable, List

from ..schemas import StrategySpec
from .retriever import ToolSpec
from .coder import CoderAgent

class CodeFixerAgent:
    """Suggests repairs and re-triggers code generation via the LLM coder."""

    def __init__(self, coder: CoderAgent):
        self.coder = coder

    def _build_hint(
        self,
        spec: StrategySpec,
        error: str | None = None,
        failed_checks: List[str] | None = None,
        attempt: int = 1,
    ) -> str:
        pieces = [f"Repair attempt {attempt} for task {spec.task}"]
        normalized_error = (error or "").strip()
        if normalized_error:
            pieces.append(f"runtime: {normalized_error}")
            lower_err = normalized_error.lower()
            if "lookback" in lower_err or "top_k" in lower_err or "holding_period" in lower_err:
                pieces.append("access params via `params = spec['params']`; read lookback/top_k/holding_period from `params`.")
        if failed_checks:
            pieces.append("verifier: " + ", ".join(failed_checks))
        pieces.append(
            "ensure pandas index alignment, diagnostics scalars (turnover, sharpe, dd), "
            "avoid DataFrame masks as column selectors; smooth positions across holding_period, "
            "normalize via normalize_weights before computing turnover/returns; leverage <= 1."
        )
        return " | ".join(pieces)

    def repair(
        self,
        spec: StrategySpec,
        tools: Iterable[ToolSpec],
        attempt: int,
        error: str | None = None,
        failed_checks: List[str] | None = None,
    ) -> tuple[str, str]:
        hint = self._build_hint(spec, error=error, failed_checks=failed_checks, attempt=attempt)
        path, code = self.coder.write_module(spec, tools, attempt=attempt, hint=hint)
        return path, hint, code
