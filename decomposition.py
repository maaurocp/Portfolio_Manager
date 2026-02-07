import numpy as np
import pandas as pd
from config import TRADING_DAYS, WEEKS

def covariance_matrix(log_returns, frequency="daily"):
    factor = TRADING_DAYS if frequency == "daily" else WEEKS
    return log_returns.cov() * factor

def portfolio_volatility_from_cov(cov, weights):
    return np.sqrt(weights.T @ cov @ weights)

def risk_contribution_percentage(log_returns, weights, frequency="daily"):
    weights = pd.Series(weights, dtype=float)
    weights /= weights.sum()

    aligned = log_returns.loc[:, weights.index]
    cov = covariance_matrix(aligned, frequency)

    port_vol = portfolio_volatility_from_cov(cov, weights)
    mrc = cov @ weights / port_vol
    rc = weights * mrc

    return rc / rc.sum()

def concentration_index(log_returns, weights, frequency="daily"):
    rc_pct = risk_contribution_percentage(log_returns, weights, frequency)
    return (rc_pct ** 2).sum()
