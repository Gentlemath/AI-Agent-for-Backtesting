# Failure Report: mean_reversion

            **Reason:** verifier failed checks: ['return_reasonable']

            ## Recent Logs
            - attempt 1: coder generated strategy at .adaptive_workdir/strategy_mean_reversion_attempt1.py
- attempt 1: code verification passed
- attempt 1: runner executed successfully
- attempt 1: verifier failed checks: ['return_reasonable']
- attempt 2: fixer regenerated strategy with hint 'Repair attempt 2 for task mean_reversion | runtime: verifier failed checks: ['return_reasonable'] | verifier: return_reasonable | ensure pandas index alignment, diagnostics scalars (turnover, sharpe, dd), avoid DataFrame masks as column selectors; smooth positions across holding_period, normalize via normalize_weights before computing turnover/returns; leverage <= 1.'
- attempt 2: code verification passed
- attempt 2: runtime error -> Must pass DataFrame or 2-d ndarray with boolean values only
- attempt 3: fixer regenerated strategy with hint 'Repair attempt 3 for task mean_reversion | runtime: Must pass DataFrame or 2-d ndarray with boolean values only | ensure pandas index alignment, diagnostics scalars (turnover, sharpe, dd), avoid DataFrame masks as column selectors; smooth positions across holding_period, normalize via normalize_weights before computing turnover/returns; leverage <= 1.'
- attempt 3: code verification passed
- attempt 3: runner executed successfully
- attempt 3: verifier failed checks: ['return_reasonable']
- attempt 4: fixer regenerated strategy with hint 'Repair attempt 4 for task mean_reversion | runtime: verifier failed checks: ['return_reasonable'] | verifier: return_reasonable | ensure pandas index alignment, diagnostics scalars (turnover, sharpe, dd), avoid DataFrame masks as column selectors; smooth positions across holding_period, normalize via normalize_weights before computing turnover/returns; leverage <= 1.'
- attempt 4: code verification passed
- attempt 4: runner executed successfully
- attempt 4: verifier failed checks: ['non_trivial_performance', 'return_reasonable']
- attempt 5: fixer regenerated strategy with hint 'Repair attempt 5 for task mean_reversion | runtime: verifier failed checks: ['non_trivial_performance', 'return_reasonable'] | verifier: non_trivial_performance, return_reasonable | ensure pandas index alignment, diagnostics scalars (turnover, sharpe, dd), avoid DataFrame masks as column selectors; smooth positions across holding_period, normalize via normalize_weights before computing turnover/returns; leverage <= 1.'
- attempt 5: code verification passed
- attempt 5: runner executed successfully
- attempt 5: verifier failed checks: ['return_reasonable']
