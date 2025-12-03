# Agentic Quant Backtesting

Agentic pipeline for testing whether multi-agent, tool-using workflows beat a single-shot LLM baseline across a frozen suite of strategy specs.

## System Overview
- **Orchestrator** is implemented with a LangGraph state machine so you can visualize/trace the loop `Prompt → Guard → Retriever → Coder → CodeVerifier → Runner → ResultVerifier → Reporter` with automatic repair edges.
- **Spec & Guard Agent** converts prompts (JSON or `task: momentum_daily` text) into a validated `StrategySpec` drawn from the frozen 12-task library.
- **Retriever Agent** resolves requested tools (returns, sharpe, drawdown, walk-forward) into importable modules for the coder to reference.
- **Coder Agent (LLM-backed)** streams prompts to the Argonne inference platform and writes the returned Python module that wires up toolbox snippets plus the knowledge-base `STRATEGY_REGISTRY`.
- **Code Verifier Agent** runs `py_compile` on every generated module before any execution, short-circuiting syntax/import issues and feeding error traces to the fixer.
- **Code Fixer Agent** inspects runtime/verifier failures, crafts repair hints, and re-invokes the coder until verification passes or attempts expire.
- **Data Loader Tool** (yfinance + AlphaVantage) backfills any missing symbols or windows by hitting public APIs and caching the adjusted closes under `data/cache/`.
- **Runner Agent** loads the frozen snapshot (augmenting with the data loader when needed), executes the generated module, and computes metrics (return, Sharpe, drawdown, turnover, hit-rate, profit factor).
- **Verifier Agent** enforces deterministic quality checks (finite metrics, turnover bounds, Sharpe limits, hit-rate in [0,1]); failures trigger the repair loop.
- **Reporter Agent** captures Markdown summaries with metrics, diagnostics, and issues for downstream evaluation.

## Task Suite & Data
- 12 tasks: daily/weekly momentum, mean reversion, breakout, pair trading, volatility targeting, risk parity, ATR stop/TP bandit, weekday mask, regime filter (MA cross), cost sensitivity, walk-forward robustness.
- Deterministic parameters (window lengths, thresholds, holding periods) are encoded inside `src/backtester/tasks.py`.
- Data lives in `data/ETF`, supporting all frozen tasks. When running the agent, the `DataLoader` can also automatically download extra data needed.

## Running the Pipelines
```bash
conda env create -f environment.yml
conda activate agentic_backtester

# Adaptive multi-round agent system (prompt string, file, or task shortcut)
python scripts/run_agentic.py --task momentum_daily
python scripts/run_agentic.py --prompt '{"task":"mean_reversion","start_date":"2016-01-01","end_date":"2022-12-31"}'

# Ablation (single-shot agent)
python scripts/run_single_shot_agent.py --task momentum_daily

# Baseline (single-shot llm, no repair)
python scripts/run_pure_llm.py --task momentum_daily
```

Outputs include generated strategy modules under `.artifacts/`, execution logs, and Markdown summaries in `reports/summary_<task>.md`.

### Argonne Inference Integration
Set the following environment variables before launching the agentic runner so the coder agent can reach the Argonne-hosted model:

- `ARGONNE_INFERENCE_URL` – e.g., `https://inference.argonne.gov/v1`
- `ARGONNE_INFERENCE_KEY` – bearer token issued by Argonne (optional if the endpoint is on a trusted VLAN)
- `ARGONNE_INFERENCE_MODEL` – override the default `meta-llama/Llama-3.1-70B-Instruct` if your deployment exposes a different model ID

The orchestrator will fail fast with a descriptive error if these are missing.

### LangGraph Workflow
The LangGraph state machine exposes the control flow:
1. Guard + Retriever stage freeze the spec and tools.
2. Coder (Argonne LLM) produces code; CodeVerifier (`py_compile`) blocks broken files.
3. Runner executes only verified modules; ResultVerifier enforces economic sanity.
4. Logical or runtime failures route back through the fixer, while persistent issues emit failure reports describing the blocking error/logs.

### Open-API Data Loader
The `DataLoader` helper automatically hydrates missing tickers through:

- **yfinance** (default) for bulk ETF/equity downloads.
- **AlphaVantage** as a fallback when `ALPHAVANTAGE_API_KEY` is exported in your shell.

Fetched data are cached in `data/cache/<symbol>.parquet`, so future runs stay offline unless new tickers/periods are requested.

## Evaluation Harness
- `eval/performance_compare.py`.

## Reproducibility Notes
- All randomness (strategy seeds, bootstrap RNG) routes through the Pydantic specs; any repair bumps the seed deterministically.
- Please avoid mutating the frozen dataset; if you need a new snapshot, document the hash in `data/data_manifest.json` and bump `data_version`.
