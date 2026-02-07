import numpy as np
import pandas as pd
from config import TRADING_DAYS, WEEKS


def annualized_log_return(log_returns, frequency="daily"):
    factor = TRADING_DAYS if frequency == "daily" else WEEKS
    return log_returns.mean() * factor


def portfolio_log_returns(log_returns, weights):
    weights = pd.Series(weights, dtype=float)
    weights /= weights.sum()

    aligned = log_returns[weights.index]
    return aligned.dot(weights)
