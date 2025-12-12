# Agentic Quant Backtesting

Agentic pipeline for testing whether multi-agent, tool-using workflows beat a single-shot LLM baseline across a frozen suite of strategy specs.

## System Overview
- **Orchestrator** is implemented with a LangGraph state machine so you can visualize/trace the loop `Prompt → Guard → Retriever → Coder → CodeVerifier → Runner → ResultVerifier → Reporter` with automatic repair edges.
- **Spec & Guard Agent** converts prompts (JSON or `task: momentum_daily` text) into a validated `StrategySpec` drawn from the frozen 12-task library.
- **Retriever Agent** resolves requested tools (returns, sharpe, drawdown, walk-forward) into importable modules for the coder to reference.
- **Coder Agent (LLM-backed)** streams prompts to the Argonne inference platform and writes the returned Python module that wires up toolbox snippets plus the knowledge-base `STRATEGY_REGISTRY`.
- **Code Verifier Agent** runs `py_compile` on every generated module before any execution, short-circuiting syntax/import issues and feeding error traces to the fixer.
- **Code Fixer Agent** inspects runtime/verifier failures, crafts repair hints, and re-invokes the coder until verification passes or attempts expire.
- **Data Loader Tool** (yfinance + AlphaVantage) backfills any missing symbols or windows by hitting public APIs and caching the adjusted closes under `data/`.
- **Runner Agent** loads the frozen snapshot (augmenting with the data loader when needed), executes the generated module, and computes metrics (return, Sharpe, drawdown, turnover, hit-rate, profit factor).
- **Verifier Agent** enforces deterministic quality checks (finite metrics, turnover bounds, Sharpe limits, hit-rate in [0,1]); failures trigger the repair loop.
- **Reporter Agent** captures Markdown summaries with metrics, diagnostics, and issues for downstream evaluation.

## Task Suite & Data
- 9 tasks: daily/weekly momentum, mean reversion, breakout, pair trading, volatility targeting, risk parity, weekday mask, regime filter (MA cross).
- Deterministic parameters (window lengths, thresholds, holding periods) are encoded inside `src/backtester/tasks.py`.
- Data lives in `data/ETF`, supporting all frozen tasks. When running the agent, the `DataLoader` can also automatically download extra data needed.

## Running the Pipelines
```bash
conda env create -f environment.yml
conda activate agentic_backtester

# Adaptive multi-round agent system (prompt string, file, or task shortcut)
python scripts/run_adaptive_agent.py --task momentum_daily
python scripts/run_adaptive_agent.py --prompt '{"task":"mean_reversion","start_date":"2016-01-01","end_date":"2022-12-31"}'

# Ablation (single-shot agent)
python scripts/run_single_shot_agent.py --task momentum_daily

# Pure LLM baseline (single pass, no internal tools)
python scripts/run_pure_llm.py --task momentum_daily

# Full comparison
python scripts/performance_compare.py --task momentum_daily

```


Outputs generated strategy modules under `.adaptive_workdir/`, `.single_shot_wordir/` and `.pure_llm/`, execution logs in `logs/`, and Markdown summaries in `reports/`.

### Configuring PYTHONPATH
The repository is not packaged; every script expects to find `backtester` under `src/`. Add it to `PYTHONPATH` once and you can run any script without manual tweaks:

```bash
# One-off invocation
PYTHONPATH=src:. python scripts/run_adaptive_agent.py --task momentum_daily

# Persistent shell setup (add to ~/.zshrc or ~/.bashrc)
export PYTHONPATH="$HOME/Documents/.../agentic_backtester/src:$PYTHONPATH"
```

If you use Conda, drop the export into `$CONDA_PREFIX/etc/conda/activate.d/agentic_backtester.sh` so the path is set whenever you activate the environment.

### Argonne Inference Integration
Set the following environment variables before launching the agentic runner so the coder agent can reach the Argonne-hosted model:

- `ARGONNE_INFERENCE_URL` – e.g., `https://inference.argonne.gov/v1`
- `ARGONNE_INFERENCE_KEY` – bearer token issued by Argonne 
- `ARGONNE_INFERENCE_MODEL` – override the default `meta-llama/Llama-3.1-70B-Instruct` if your deployment exposes a different model ID

### Open-API Data Loader
The `DataLoader` helper automatically hydrates missing tickers through:

- **yfinance** (default) for bulk ETF/equity downloads.
- **AlphaVantage** as a fallback when `ALPHAVANTAGE_API_KEY` is exported in your shell.

Fetched data are cached in `data/cache/`, so future runs stay offline unless new tickers/periods are requested.
