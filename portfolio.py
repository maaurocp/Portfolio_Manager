import pandas as pd
from returns import annualized_log_return, portfolio_log_returns
from risk import portfolio_volatility, max_drawdown, cvar


class Portfolio:
    def __init__(self, weights, log_returns, frequency="daily"):
        self.weights = pd.Series(weights, dtype=float)
        self.returns = log_returns
        self.frequency = frequency

        self._validate()
        self.weights /= self.weights.sum()

    def _validate(self):
        if not isinstance(self.returns, pd.DataFrame):
            raise TypeError("log_returns debe ser DataFrame")

        missing = set(self.weights.index) - set(self.returns.columns)
        if missing:
            raise ValueError(f"Activos sin datos: {missing}")

        if (self.weights < 0).any():
            raise ValueError("Pesos negativos no permitidos")

    @property
    def portfolio_returns(self):
        return portfolio_log_returns(self.returns, self.weights)

    @property
    def annualized_return(self):
        return annualized_log_return(self.portfolio_returns, self.frequency)

    @property
    def volatility(self):
        return portfolio_volatility(self.returns, self.weights, self.frequency)

    @property
    def max_drawdown(self):
        return max_drawdown(self.portfolio_returns)

    def cvar(self, alpha=0.05):
        return cvar(self.portfolio_returns, alpha)

    def summary(self):
        return pd.Series({
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "cvar_5%": self.cvar(0.05)
        })
