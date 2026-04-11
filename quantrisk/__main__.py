"""
CLI entry point: fetch data and print a portfolio summary.

Usage:
    python -m quantrisk
    python -m quantrisk --tickers AAPL MSFT GLD TLT --weights 0.25 0.25 0.25 0.25
"""

import argparse
import sys


DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "GLD", "TLT", "EEM", "VNQ"]
DEFAULT_WEIGHTS = [0.15, 0.15, 0.10, 0.10, 0.15, 0.15, 0.10, 0.10]
DEFAULT_START = "2015-01-01"


def main():
    parser = argparse.ArgumentParser(description="QuantRisk Portfolio Risk Analytics")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--no-risk", action="store_true", help="Skip risk report")
    args = parser.parse_args()

    if len(args.tickers) != len(args.weights):
        print("ERROR: --tickers and --weights must have the same length")
        sys.exit(1)

    from quantrisk.portfolio.portfolio import Portfolio
    from quantrisk.risk.metrics import RiskReport

    weights = dict(zip(args.tickers, args.weights))
    print(f"\nLoading data for: {list(weights.keys())}")

    portfolio = Portfolio(
        weights=weights,
        start_date=args.start,
        benchmark=args.benchmark,
        name="Demo Portfolio",
    ).load()

    portfolio.print_summary()

    if not args.no_risk:
        print("Computing risk metrics...")
        report = RiskReport(portfolio).compute()
        report.print_report()


if __name__ == "__main__":
    main()
