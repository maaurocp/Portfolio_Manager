
import pandas as pd
from portfolio import Portfolio

def generate_risk_scenarios(log_returns, base_weights, risk_shifts, frequency="daily"):
    scenarios = {}

    for shift in risk_shifts:
        scale = 1 + shift

        stressed_returns = log_returns * scale

        scenarios[f"risk_{int(shift*100)}%"] = Portfolio(
            base_weights,
            stressed_returns,
            frequency
        )

    return scenarios


def compare_scenarios(portfolios):
    return pd.DataFrame(
        {name: p.summary() for name, p in portfolios.items()}
    ).T
