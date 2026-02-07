import numpy as np
import pandas as pd
from config import TRADING_DAYS, WEEKS


def portfolio_volatility(log_returns, weights, frequency="daily"):
    weights = pd.Series(weights, dtype=float)
    weights /= weights.sum()

    aligned = log_returns[weights.index]
    cov = aligned.cov()

    factor = TRADING_DAYS if frequency == "daily" else WEEKS
    cov_annual = cov * factor

    return np.sqrt(weights.T @ cov_annual @ weights)


def max_drawdown(log_returns):
    cumulative = log_returns.cumsum()
    wealth = np.exp(cumulative)

    running_max = wealth.cummax()
    drawdown = (wealth / running_max) - 1
    return drawdown.min()


def cvar(log_returns, alpha=0.05):
    var = log_returns.quantile(alpha)
    return log_returns[log_returns <= var].mean()
