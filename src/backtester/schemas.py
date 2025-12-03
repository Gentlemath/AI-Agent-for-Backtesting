from __future__ import annotations
from datetime import date
from typing import Literal, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator

class StrategySpec(BaseModel):
    name: str
    task: str
    description: str = ""
    universe: List[str]
    frequency: Literal["daily", "weekly"]
    signal: str  # high-level description or DSL
    rules: Dict[str, str]  # {"entry": "...", "exit": "..."}
    tools: List[str] = Field(default_factory=list)
    required_metrics: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    costs_bps: float = 1.0
    start_date: date
    end_date: date
    seed: int = 42
    max_leverage: float = 1.0

    @field_validator("costs_bps")
    @classmethod
    def _costs_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("costs_bps must be non-negative")
        return v

    @model_validator(mode="after")
    def _validate_dates(self) -> "StrategySpec":
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date")
        return self

class BacktestResult(BaseModel):
    ann_return: float
    ann_vol: float
    sharpe: float
    max_dd: float
    turnover: float
    trades: int
    hit_rate: float
    pf: float  # profit factor
    seed: int
    period_start: date
    period_end: date
    artifact_paths: List[str] = Field(default_factory=list)
    diagnostics: Dict[str, float] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)

class WorkflowState(BaseModel):
    spec: StrategySpec
    code_artifacts: List[str] = Field(default_factory=list)
    run_logs: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    figures: List[str] = Field(default_factory=list)
    verdict: Literal["pass", "fail"] = "fail"
    tool_refs: List[str] = Field(default_factory=list)
    attempts: int = 0
