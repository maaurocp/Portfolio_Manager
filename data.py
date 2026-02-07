import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_prices(tickers, start, end=None, interval="1d", filename="prices.csv"):
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No se han descargado datos")

    prices = data["Close"].dropna(how="all")
    prices.to_csv(DATA_DIR / filename)
    return prices


def load_prices(filename="prices.csv"):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.sort_index()


def compute_log_returns(prices):
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")


def align_assets(data):
    return data.dropna(axis=0, how="any")
