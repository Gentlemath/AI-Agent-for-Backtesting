from __future__ import annotations

from typing import Any, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from .schemas import WorkflowState, StrategySpec, BacktestResult
from .agents.spec_guard import SpecGuardAgent
from .agents.retriever import RetrieverAgent, ToolSpec
from .agents.coder import CoderAgent
from .agents.code_fixer import CodeFixerAgent
from .agents.code_verifier import CodeVerifierAgent
from .agents.runner import RunnerAgent
from .agents.test_result_verifier import BTVerifierAgent
from .agents.reporter import ReporterAgent
from .utils.data_loader import DataLoader

DEFAULT_TOOLS = ["returns", "sharpe", "drawdown", "normalize_weights", "compute_turnover"]

class GraphState(TypedDict, total=False):
    prompt: Any
    spec: StrategySpec
    tools: List[ToolSpec]
    tool_names: List[str]
    strategy_path: str
    attempt: int
    artifacts: List[str]
    run_logs: List[str]
    last_code: Optional[str]
    last_error: Optional[str]
    failed_checks: List[str]
    result: Optional[BacktestResult]
    verdict: str
    code_ok: bool
    runtime_ok: bool
    verifier_ok: bool
    final_reason: Optional[str]
    workflow: WorkflowState

class Orchestrator:
    def __init__(self, data_path: str, kb_root: str, workdir: str, max_attempts: int = 5):
        self.guard = SpecGuardAgent()
        self.retr = RetrieverAgent(kb_root)
        self.coder = CoderAgent(workdir)
        self.fixr = CodeFixerAgent(self.coder)
        self.codever = CodeVerifierAgent()
        self.loader = DataLoader(data_path)
        self.runr = RunnerAgent(data_loader=self.loader)
        self.verf = BTVerifierAgent()
        self.rept = ReporterAgent(out_dir="reports/adaptive_workflow")
        self.max_attempts = max_attempts
        self._graph = self._build_graph()

    def execute(self, prompt: Any) -> WorkflowState:
        init_state: GraphState = {
            "prompt": prompt,
            "attempt": 1,
            "artifacts": [],
            "run_logs": [],
            "last_error": None,
            "failed_checks": [],
            "verdict": "fail",
        }
        final_state = self._graph.invoke(init_state)
        return final_state["workflow"]

    # --------------------------
    # Graph construction
    # --------------------------
    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("guard", self._node_guard, metadata={"Desc":"Specification Safeguard"})
        graph.add_node("retrieve", self._node_retrieve, metadata={"Desc":"Tool Retriever"})
        graph.add_node("code", self._node_code, metadata={"Desc":"Code Generator with Fixer if repairing"})
        graph.add_node("code_verify", self._node_code_verify, metadata={"Desc":"Code Verifier"})
        graph.add_node("run", self._node_run, metadata={"Desc":"Code Runner"})
        graph.add_node("result_verify", self._node_result_verify, metadata={"Desc":"Backtest Verifier"})
        graph.add_node("report", self._node_report, metadata={"Desc":"Results Reporter"})

        graph.set_entry_point("guard")
        graph.add_edge("guard", "retrieve")
        graph.add_edge("retrieve", "code")
        graph.add_edge("code", "code_verify")
        graph.add_conditional_edges(
            "code_verify",
            self._route_code_verify,
            {
                "ok": "run",
                "retry": "code",
                "stop": "report",
            },
        )
        graph.add_conditional_edges(
            "run",
            self._route_run,
            {
                "ok": "result_verify",
                "retry": "code",
                "stop": "report",
            },
        )
        graph.add_conditional_edges(
            "result_verify",
            self._route_result_verify,
            {
                "ok": "report",
                "retry": "code",
                "stop": "report",
            },
        )
        graph.add_edge("report", END)
        gc = graph.compile()
        png_bytes = gc.get_graph().draw_mermaid_png()
        with open("./pipeline_graph.png", "wb") as f:
            f.write(png_bytes)
        print("Pipeline graph saved as pipeline_graph.png")
        return gc

    def _log_state(self, node: str, state: GraphState, detail: str | None = None) -> None:
        attempt = state.get("attempt", 1)
        verdict = state.get("verdict", "n/a")
        error = state.get("last_error")
        suffix = f" | {detail}" if detail else ""
        status = "ok" if not error else f"error: {error}"
        print(f"[{node}] attempt={attempt} verdict={verdict} status={status}{suffix}")

    # --------------------------
    # Graph nodes
    # --------------------------
    def _node_guard(self, state: GraphState) -> GraphState:
        print("\n----------")
        print("Validating and parsing strategy specification...")
        spec = self.guard.validate_and_struct(state["prompt"])
        state["spec"] = spec
        state["attempt"] = state.get("attempt", 1)
        state["run_logs"] = state.get("run_logs", [])
        state["artifacts"] = state.get("artifacts", [])
        state["failed_checks"] = []
        state["last_error"] = None
        self._log_state("guard", state, detail=f"spec={spec.name}")
        print(f"Strategy spec: {spec}")    

        return state

    def _node_retrieve(self, state: GraphState) -> GraphState:
        print("\n----------")
        print("Retrieving tools for strategy...")
        spec = state["spec"]
        tools = self.retr.fetch(spec.tools or DEFAULT_TOOLS)
        state["tools"] = tools
        state["tool_names"] = [tool.name for tool in tools]
        tool_names = ",".join(state["tool_names"]) if state["tool_names"] else "none"
        self._log_state("retrieve", state, detail=f"tools={tool_names}")
        return state

    def _node_code(self, state: GraphState) -> GraphState:
        print("\n----------")
        print("Generating or repairing strategy code...")
        spec = state["spec"]
        tools = state["tools"]
        attempt = state["attempt"]
        last_code = state.get("last_code")
        last_error = state.get("last_error")
        failed_checks = state.get("failed_checks") or []
        logs = state["run_logs"]
        if attempt == 1 and not last_error and not failed_checks:
            print(f"First attempt: invoking coder")
            path, code = self.coder.write_module(spec, tools, attempt=attempt)
            logs.append(f"attempt {attempt}: coder generated strategy at {path}")
        else:
            print(f"Repair attempt {attempt - 1}: invoking fixer")
            path, hint, code = self.fixr.repair(
                spec,
                tools,
                attempt=attempt,
                error=last_error,
                failed_checks=failed_checks,
            )
            logs.append(f"attempt {attempt}: fixer regenerated strategy with hint '{hint}'")
            state["last_error"] = None
            state["failed_checks"] = []
        state["strategy_path"] = path
        state["last_code"] = code
        state["artifacts"].append(path)
        self._log_state("code", state, detail=f"path={path}")
        return state

    def _node_code_verify(self, state: GraphState) -> GraphState:
        print("\n----------")
        print("Verifying strategy code...")
        attempt = state["attempt"]
        try:
            self.codever.verify(state["strategy_path"])
            state["code_ok"] = True
            state["run_logs"].append(f"attempt {attempt}: code verification passed")
            print("Code verification passed.")
        except Exception as exc:
            state["code_ok"] = False
            message = str(exc)
            state["last_error"] = message
            state["run_logs"].append(f"attempt {attempt}: code verification failed -> {message}")
            self._increment_attempt(state)
            self._bump_seed(state)
        self._log_state("code_verify", state, detail=f"code_ok={state.get('code_ok')}")
        return state

    def _node_run(self, state: GraphState) -> GraphState:
        print("\n----------")
        print("Running backtest...")
        attempt = state["attempt"]
        try:
            result = self.runr.run(state["strategy_path"], state["spec"])
            state["result"] = result
            state["runtime_ok"] = True
            state["run_logs"].append(f"attempt {attempt}: runner executed successfully")
            print("Backtest run succeeded.")
        except Exception as exc:
            state["runtime_ok"] = False
            message = str(exc)
            state["last_error"] = message
            state["run_logs"].append(f"attempt {attempt}: runtime error -> {message}")
            self._increment_attempt(state)
            self._bump_seed(state)
        self._log_state("run", state, detail=f"runtime_ok={state.get('runtime_ok')}")
        return state

    def _node_result_verify(self, state: GraphState) -> GraphState:
        print("\n----------")
        print("Verifying backtest results...")
        result = state.get("result")
        print("-----------\n")
        print("Backtest Results:")
        print(f"Annualized Return: {result.ann_return:.2%}")
        print(f"Annualized Volatility: {result.ann_vol:.2%}")
        print(f"Sharpe Ratio: {result.sharpe:.2f}") 
        print(f"Maximum Drawdown: {result.max_dd:.2%}")
        print(f"Turnover: {result.turnover:.2%}")
        print("-----------\n")

        attempt = state["attempt"]
        if not result:
            state["verifier_ok"] = False
            state["last_error"] = "No backtest result to verify."
            state["run_logs"].append(f"attempt {attempt}: verifier skipped (missing result)")
            self._increment_attempt(state)
            return state
        ok, failures = self.verf.evaluate(result)
        result.issues = failures
        state["failed_checks"] = failures
        if ok:
            state["verifier_ok"] = True
            state["verdict"] = "pass"
            state["run_logs"].append(f"attempt {attempt}: verifier passed")    
            print("Backtest result verification passed.")
        else:
            state["verifier_ok"] = False
            message = f"verifier failed checks: {failures}"
            state["last_error"] = message
            state["run_logs"].append(f"attempt {attempt}: {message}")
            print("Backtest result verification failed.")
            self._increment_attempt(state)
            self._bump_seed(state)
        self._log_state("result_verify", state, detail=f"verifier_ok={state.get('verifier_ok')}")
        return state

    def _node_report(self, state: GraphState) -> GraphState:
        print("\n----------")
        print("Generating report...")
        spec = state["spec"]
        verdict = state.get("verdict", "fail")
        logs = state.get("run_logs", [])
        result = state.get("result")
        figures: List[str] = []
        metrics = {}
        if verdict == "pass" and result:
            summary_path = self.rept.write_summary(spec.name, result)
            figures.append(summary_path)
            metrics = {
                "ann_return": result.ann_return,
                "ann_vol": result.ann_vol,
                "sharpe": result.sharpe,
                "max_dd": result.max_dd,
                "turnover": result.turnover,
            }
        else:
            reason = state.get("final_reason") or state.get("last_error") or "unknown failure"
            summary_path = self.rept.write_failure(spec.name, reason, logs)
            figures.append(summary_path)
        workflow = WorkflowState(
            spec=spec,
            code_artifacts=state.get("artifacts", []),
            run_logs=logs,
            metrics=metrics,
            figures=figures,
            verdict=verdict,
            tool_refs=state.get("tool_names", []),
            attempts=min(state.get("attempt", 1), self.max_attempts),
        )
        state["workflow"] = workflow
        self._log_state("report", state, detail=f"final_verdict={verdict}")
        return state

    # --------------------------
    # Routing helpers
    # --------------------------
    def _route_code_verify(self, state: GraphState) -> str:
        if state.get("code_ok"):
            return "ok"
        if state.get("attempt", 1) > self.max_attempts:
            state["final_reason"] = state.get("last_error") or "code verification failed"
            return "stop"
        return "retry"

    def _route_run(self, state: GraphState) -> str:
        if state.get("runtime_ok"):
            return "ok"
        if state.get("attempt", 1) > self.max_attempts:
            state["final_reason"] = state.get("last_error") or "runtime error"
            return "stop"
        return "retry"

    def _route_result_verify(self, state: GraphState) -> str:
        if state.get("verifier_ok"):
            return "ok"
        if state.get("attempt", 1) > self.max_attempts:
            reason = state.get("last_error") or "verifier failed"
            state["final_reason"] = reason
            state["verdict"] = "fail"
            return "stop"
        return "retry"

    def _increment_attempt(self, state: GraphState) -> None:
        state["attempt"] = state.get("attempt", 1) + 1

    def _bump_seed(self, state: GraphState) -> None:
        spec = state["spec"]
        state["spec"] = spec.model_copy(update={"seed": spec.seed + 1})
