from __future__ import annotations
from copy import deepcopy
import json
from datetime import date as dt_date
from typing import Any, Dict

from ..schemas import StrategySpec
from ..tasks import TASK_LIBRARY, allowed_tasks, allowed_universe, DATA_BOUNDS

class SpecGuardAgent:
    """Turns user intent into a validated StrategySpec."""

    REQUIRED_KEYS = ["task", "universe", "frequency", "start_date", "end_date"]

    def __init__(self, task_library: Dict[str, Dict[str, Any]] | None = None):
        self._library = task_library or TASK_LIBRARY

    def validate_and_struct(self, prompt: str | Dict[str, Any]) -> StrategySpec:
        payload = self._parse_prompt(prompt)
        task_name = (payload.get("task") or payload.get("name") or "").strip()
        if task_name not in self._library:
            raise ValueError(
                f"Task '{task_name}' not in frozen suite. Allowed: {', '.join(allowed_tasks())}"
            )
        template = deepcopy(self._library[task_name])
        overrides = self._coerce_payload(payload)
        merged = {**template, **overrides}
        if "params" in overrides and isinstance(template.get("params"), dict):
            params = template.get("params", {}).copy()
            params.update(overrides.get("params", {}))
            merged["params"] = params
        merged["task"] = task_name
        merged.setdefault("name", task_name)
        missing = [k for k in self.REQUIRED_KEYS if k not in merged or merged[k] in (None, "")]
        if missing:
            raise ValueError(f"Strategy spec missing required fields: {missing}")
        spec = StrategySpec.model_validate(merged)
        self._enforce_constraints(spec)
        return spec

    def _parse_prompt(self, prompt: str | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(prompt, dict):
            return prompt
        cleaned = prompt.strip()
        if not cleaned:
            raise ValueError("Prompt is empty.")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # fallback to simple "key: value" or "key=value" mini-language
        fields: Dict[str, Any] = {}
        for line in cleaned.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
            elif "=" in line:
                key, value = line.split("=", 1)
            else:
                continue
            fields[key.strip()] = value.strip()
        if fields:
            return fields
        # Final fallback treat raw string as task name.
        return {"task": cleaned}

    def _coerce_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        coerced: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in ("universe", "tools", "required_metrics") and isinstance(value, str):
                coerced[key] = [v.strip() for v in value.split(",") if v.strip()]
            elif key == "params" and isinstance(value, str):
                try:
                    coerced[key] = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Params must be JSON when provided as string: {value}") from exc
            elif key in ("costs_bps", "max_leverage") and isinstance(value, str):
                coerced[key] = float(value)
            elif key in ("start_date", "end_date") and isinstance(value, str):
                coerced[key] = value
            else:
                coerced[key] = value
        return coerced

    def _enforce_constraints(self, spec: StrategySpec) -> None:
        allowed = set(allowed_universe())
        if not set(spec.universe).issubset(allowed):
            raise ValueError("Universe contains symbols outside frozen dataset.")

        bounds_start = dt_date.fromisoformat(DATA_BOUNDS["start_date"])
        bounds_end = dt_date.fromisoformat(DATA_BOUNDS["end_date"])
        if spec.start_date < bounds_start or spec.end_date > bounds_end:
            raise ValueError(
                f"Requested window ({spec.start_date}→{spec.end_date}) exceeds data freeze "
                f"{bounds_start}→{bounds_end}."
            )
        if spec.frequency == "weekly":
            min_days = 26 * 7
            if (spec.end_date - spec.start_date).days < min_days:
                raise ValueError("Weekly strategies require at least 26 weeks of data.")
        if spec.task == "pair_trading":
            mode = spec.params.get("mode", "cointegration")
            if mode not in {"cointegration", "distance"}:
                raise ValueError("pair_trading mode must be 'cointegration' or 'distance'.")
