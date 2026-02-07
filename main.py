from data import download_prices, load_prices, compute_log_returns, align_assets
from portfolio import Portfolio
from decomposition import risk_contribution_percentage, concentration_index
from scenarios import generate_risk_scenarios, compare_scenarios
from montecarlo import simulate_paths, terminal_wealth, summary_statistics
from portfolios_analizar import PORTFOLIOS

PORTFOLIO_ID = 3  #Portfolio a analizar



def format_summary(series):
    formatted = series.copy()

    formatted["annualized_return"] *= 100
    formatted["volatility"] *= 100
    formatted["cvar_5%"] *= 100

    return formatted.round(2)

def format_montecarlo_summary(summary):
    return {
        "mean_terminal_wealth": round(float(summary["mean"]), 2),
        "median_terminal_wealth": round(float(summary["median"]), 2),
        "p5_terminal_wealth": round(float(summary["p5"]), 2),
        "p95_terminal_wealth": round(float(summary["p95"]), 2),
        "prob_loss_%": round(float(summary["prob_loss"]) * 100, 2)
    }



def main():
    portfolio_cfg = PORTFOLIOS[PORTFOLIO_ID]

    TICKERS = portfolio_cfg["tickers"]
    WEIGHTS = portfolio_cfg["weights"]

    print(f"ANALYZING PORTFOLIO: {portfolio_cfg['name']}")
    print()


    download_prices(TICKERS, "2005-01-01")
    prices = load_prices()
    log_returns = align_assets(compute_log_returns(prices))

    portfolio = Portfolio(WEIGHTS, log_returns)

    print("PORTFOLIO SUMMARY")
    print(format_summary(portfolio.summary()))
    print()

    print("RISK CONTRIBUTION")
    print(risk_contribution_percentage(log_returns, WEIGHTS))
    print()

    print("CONCENTRATION INDEX")
    print(concentration_index(log_returns, WEIGHTS))
    print()

    scenarios = generate_risk_scenarios(
        log_returns, WEIGHTS, [-0.3, 0.0, 0.3]
    )

    print("SCENARIO COMPARISON")
    print(compare_scenarios(scenarios))
    print()

    paths = simulate_paths(
        portfolio.portfolio_returns,
        n_years=10,
        n_sims=20000,
        seed=42
    )

    wealth = terminal_wealth(paths)
    print("MONTE CARLO SUMMARY")
    mc_summary = summary_statistics(wealth)
    print(format_montecarlo_summary(mc_summary))

if __name__ == "__main__":
    main()
