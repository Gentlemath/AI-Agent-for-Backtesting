from __future__ import annotations
from pathlib import Path
from typing import Iterable
import json
from textwrap import dedent     


from ..schemas import StrategySpec
from ..llm import ArgonneLLM, LLMGenerationError
from .retriever import ToolSpec

class CoderAgent:
    def __init__(self, out_dir: str, llm: ArgonneLLM | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm or ArgonneLLM()

    def write_module(
        self,
        spec: StrategySpec,
        tools: Iterable[ToolSpec],
        attempt: int = 1,
        hint: str | None = None,
    ) -> str:
        file_name = f"strategy_{spec.name}_attempt{attempt}.py"
        path = self.out_dir / file_name
        try:
            content = self.synthesize_strategy(spec, list(tools), attempt=attempt, hint=hint)
        except Exception as exc:
            raise RuntimeError(f"LLM generation failed: {exc}")
        path.write_text(content)
        return str(path), content
    
    def synthesize_strategy(
        self,
        spec: StrategySpec,
        tools: Iterable[ToolSpec],
        attempt: int = 1,
        hint: str | None = None,
    ) -> str:
        system_prompt, user_prompt = self._build_prompts(spec, tools, attempt, hint)

        # 🔹 Call the LLM via our wrapper
        data = self.llm.call_reasoning_api(
            user=user_prompt,
            system=system_prompt,
        )

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMGenerationError(f"Malformed Argonne response: {data}") from exc

        code = self._extract_code(content)
        if not code.strip():
            raise LLMGenerationError("Argonne LLM returned empty body.")

        return dedent(code).strip() + "\n"

    def _build_prompts(
        self,
        spec: "StrategySpec",
        tools: Iterable["ToolSpec"],
        attempt: int,
        hint: str | None,
    ) -> tuple[str, str]:
        tool_lines = []
        for tool in tools:
            desc = f" – {tool.description}" if tool.description else ""
            tool_lines.append(
                f"- {tool.name}: `{tool.module}.{tool.symbol}`{desc} (path={tool.path})"
            )
        tool_block = "\n".join(tool_lines) if tool_lines else "- returns: none supplied"
        spec_json = json.dumps(spec.model_dump(mode="json"), indent=2)
        hint_text = hint or "Initial synthesis. Produce fully runnable module."
        system_prompt = dedent(
            """
            You are the coding agent in a quantitative backtesting pipeline.
            Understand your task carefully!!!
            Output a complete Python module defining `run_strategy(prices: pd.DataFrame, spec: dict)`.
            Respect the supplied tools and the StrategySpec JSON. Please explore it properly in your code. Do not invent extra dependencies.
            The structure of StrategySpec is as follows (fields may be omitted if not needed):
            - name: str
            - task: str
            - description: str = ""
            - universe: List[str]
            - frequency: Literal["daily", "weekly"]
            - signal: str  # high-level description or DSL
            - rules: Dict[str, str]  # {"entry": "...", "exit": "..."}
            - tools: List[str] = Field(default_factory=list)
            - required_metrics: List[str] = Field(default_factory=list)
            - params: Dict[str, Any] = Field(default_factory=dict)  # custom parameters for the strategy
            - costs_bps: float = 1.0
            - start_date: date
            - end_date: date
            - seed: int = 42
            - max_leverage: float = 1.0
            Diagnostics must be numeric scalars except for the `returns` Series.
            Always align pandas objects by both index and columns; do not pass 2-D masks where a list of column labels is expected.
            Use vectorized pandas ops (no explicit Python loops over dates) and prefer `.where`, `.rolling`, `.rank`, etc.
            Prefer using `normalize_weights` and `compute_turnover` from `backtester.kb.strategies` to stay consistent with the toolkit.
            Ensure turnover is a scalar float (e.g., average absolute position change) and enforce leverage/cost constraints from the spec.
            Smooth positions over the requested holding period (e.g., via rolling means) before normalization so exposures decay instead of flipping instantly.
            Honor spec.frequency: if spec.frequency == "weekly", resample prices to weekly closes, compute weekly returns/signals, and annualize with 52 periods; if "daily", stay on daily data (252 periods). Never mix frequencies.
            You must only return Python code; no commentary or markdown.
            """
        ).strip()
        user_prompt = dedent(
            f"""
            Attempt: {attempt}
            Repair hint: {hint_text}

            StrategySpec (JSON):
            {spec_json}

            Tools you may call:
            {tool_block}

            Requirements:
            - Import pandas as pd.
            - Use the provided tools and `backtester.kb.strategies.STRATEGY_REGISTRY` when appropriate.
            - Return diagnostics dict containing at least `\"returns\"` (pd.Series) and `\"turnover\"` (float).
            - Handle NaNs, align indexes, and respect leverage <= spec.max_leverage.
            - Never index columns with DataFrame-valued masks; use boolean DataFrames via `.where` or multiply by masks.
            - Collapse portfolio statistics to floats before storing them in diagnostics.
            - When constructing positions, normalize via `normalize_weights` and compute turnover via `compute_turnover` from `backtester.kb.strategies`.
            - Smooth raw signals across the holding period specified in the StrategySpec before calling `normalize_weights`, so positions unwind gradually.
            - Measure turnover using the normalized weights (the same array used for returns).
            - Respect `spec["frequency"]`: use weekly resampling/annualization for weekly strategies, daily cadence for daily ones.
            - For cross-sectional rankings, rank in descending order (higher signal is better) and build top-k masks via `(ranks <= top_k)`; avoid `.apply(nlargest, ...)` patterns that drop column alignment.
            """
        ).strip()
        return system_prompt, user_prompt

    @staticmethod
    def _extract_code(raw: str) -> str:
        # Look for ```python\n ... ``` blocks
        import re
        matches = re.findall(r"```(?:python)?\n(.*?)```", raw, flags=re.DOTALL)
        if matches:
            return matches[-1]
        return raw
