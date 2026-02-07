import numpy as np
import pandas as pd
from config import TRADING_DAYS, WEEKS


def simulate_paths(log_returns, n_years, n_sims, frequency="daily", seed=None):
    if seed is not None:
        np.random.seed(seed)

    periods = TRADING_DAYS * n_years if frequency == "daily" else WEEKS * n_years

    sampled = np.random.choice(
        log_returns.values,
        size=(periods, n_sims),
        replace=True
    )

    return pd.DataFrame(sampled.cumsum(axis=0))


def terminal_wealth(paths, initial_capital=1.0):
    return initial_capital * np.exp(paths.iloc[-1])


def summary_statistics(wealth):
    return {
        "mean": wealth.mean(),
        "median": wealth.median(),
        "p5": wealth.quantile(0.05),
        "p95": wealth.quantile(0.95),
        "prob_loss": (wealth < 1).mean()
    }
