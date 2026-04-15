"""Tests for the transaction-based portfolio model and Trading 212 loader."""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import pytest

from quantrisk.ingestion.trading212 import (
    _safe_float,
    _safe_str,
    load_multiple_csvs,
    load_trading212_csv,
)
from quantrisk.portfolio.transactions import Holding, Transaction, TransactionPortfolio


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _tx(action, ticker=None, shares=None, total=0.0, date="2024-01-10", isin=None, name=None, price=None):
    return Transaction(
        date=datetime.fromisoformat(date),
        action=action,
        ticker=ticker,
        isin=isin,
        name=name,
        shares=shares,
        price_per_share=price,
        total=total,
    )


@pytest.fixture()
def simple_portfolio():
    """Buy 10 AAPL at £150, then buy 5 more at £160."""
    return TransactionPortfolio([
        _tx("Deposit",     total=2000.0, date="2024-01-01"),
        _tx("Market buy",  ticker="AAPL", shares=10, total=1500.0, price=150.0, date="2024-01-10"),
        _tx("Market buy",  ticker="AAPL", shares=5,  total=800.0,  price=160.0, date="2024-01-20"),
    ])


@pytest.fixture()
def multi_asset_portfolio():
    """Buy AAPL and MSFT."""
    return TransactionPortfolio([
        _tx("Deposit",    total=5000.0, date="2024-01-01"),
        _tx("Market buy", ticker="AAPL", shares=10, total=1500.0, price=150.0, date="2024-01-05"),
        _tx("Market buy", ticker="MSFT", shares=5,  total=1750.0, price=350.0, date="2024-01-06"),
    ])


@pytest.fixture()
def sell_portfolio():
    """Buy 10 AAPL at £150, sell 4 at £170."""
    return TransactionPortfolio([
        _tx("Deposit",      total=2000.0, date="2024-01-01"),
        _tx("Market buy",   ticker="AAPL", shares=10, total=1500.0, price=150.0, date="2024-01-10"),
        _tx("Market sell",  ticker="AAPL", shares=4,  total=680.0,  price=170.0, date="2024-01-20"),
    ])


# ── Holdings ───────────────────────────────────────────────────────────────────

class TestHoldings:
    def test_single_buy(self, simple_portfolio):
        h = simple_portfolio.holdings()
        assert "AAPL" in h
        assert h["AAPL"].shares == pytest.approx(15.0)

    def test_average_cost_two_buys(self, simple_portfolio):
        """Avg cost = (1500 + 800) / 15 = 153.33..."""
        h = simple_portfolio.holdings()
        expected_avg = (1500.0 + 800.0) / 15.0
        assert h["AAPL"].avg_cost == pytest.approx(expected_avg)

    def test_multi_asset_both_held(self, multi_asset_portfolio):
        h = multi_asset_portfolio.holdings()
        assert "AAPL" in h
        assert "MSFT" in h

    def test_sell_reduces_shares(self, sell_portfolio):
        h = sell_portfolio.holdings()
        assert h["AAPL"].shares == pytest.approx(6.0)

    def test_full_sell_removes_holding(self):
        p = TransactionPortfolio([
            _tx("Deposit",     total=2000.0),
            _tx("Market buy",  ticker="AAPL", shares=10, total=1500.0),
            _tx("Market sell", ticker="AAPL", shares=10, total=1700.0, date="2024-01-20"),
        ])
        h = p.holdings()
        assert "AAPL" not in h

    def test_unknown_ticker_not_in_holdings(self):
        p = TransactionPortfolio([_tx("Deposit", total=1000.0)])
        assert p.holdings() == {}

    def test_total_cost_after_partial_sell(self, sell_portfolio):
        """After selling 4/10 shares, cost basis reduces to 60% of original."""
        h = sell_portfolio.holdings()
        expected_cost = 1500.0 * (6.0 / 10.0)
        assert h["AAPL"].total_cost == pytest.approx(expected_cost)


# ── Cash ───────────────────────────────────────────────────────────────────────

class TestCash:
    def test_total_deposited_single(self):
        p = TransactionPortfolio([_tx("Deposit", total=1000.0)])
        assert p.total_deposited() == pytest.approx(1000.0)

    def test_total_deposited_minus_withdrawal(self):
        p = TransactionPortfolio([
            _tx("Deposit",    total=1000.0),
            _tx("Withdrawal", total=-300.0, date="2024-01-15"),
        ])
        assert p.total_deposited() == pytest.approx(700.0)

    def test_cash_balance_after_buy(self, simple_portfolio):
        """Cash = 2000 deposit - 1500 buy - 800 buy = -300 (overspent, unlikely but tested)."""
        assert simple_portfolio.cash_balance() == pytest.approx(2000.0 - 1500.0 - 800.0)

    def test_cash_balance_after_sell(self, sell_portfolio):
        """Cash = 2000 - 1500 (buy) + 680 (sell) = 1180."""
        assert sell_portfolio.cash_balance() == pytest.approx(2000.0 - 1500.0 + 680.0)


# ── P&L ────────────────────────────────────────────────────────────────────────

class TestPnL:
    def test_unrealised_pnl_positive(self, simple_portfolio):
        prices = {"AAPL": 200.0}
        upnl = simple_portfolio.unrealised_pnl(prices)
        avg_cost = (1500.0 + 800.0) / 15.0
        expected = (200.0 - avg_cost) * 15.0
        assert upnl["AAPL"] == pytest.approx(expected)

    def test_unrealised_pnl_negative(self, simple_portfolio):
        prices = {"AAPL": 100.0}   # below avg cost
        upnl = simple_portfolio.unrealised_pnl(prices)
        assert upnl["AAPL"] < 0

    def test_unrealised_pnl_missing_price(self, simple_portfolio):
        """Ticker not in prices dict → not included in result."""
        upnl = simple_portfolio.unrealised_pnl({})
        assert "AAPL" not in upnl

    def test_realised_pnl_sell_higher(self, sell_portfolio):
        """Sold 4 shares at £170, cost was £150 → realised = 4 * (170 - 150) = 80."""
        rpnl = sell_portfolio.realised_pnl()
        assert rpnl.get("AAPL", 0) == pytest.approx(4 * (170.0 - 150.0))

    def test_realised_pnl_no_sells(self, simple_portfolio):
        assert simple_portfolio.realised_pnl() == {}

    def test_current_value(self, simple_portfolio):
        prices = {"AAPL": 200.0}
        equity = 15.0 * 200.0
        cash = 2000.0 - 1500.0 - 800.0
        assert simple_portfolio.current_value(prices) == pytest.approx(equity + cash)

    def test_total_return_positive(self, simple_portfolio):
        prices = {"AAPL": 300.0}
        ret = simple_portfolio.total_return(prices)
        assert ret > 0

    def test_total_return_zero_deposited(self):
        p = TransactionPortfolio([])
        assert p.total_return({}) == pytest.approx(0.0)


# ── DataFrame outputs ──────────────────────────────────────────────────────────

class TestDataFrames:
    def test_holdings_df_columns(self, simple_portfolio):
        df = simple_portfolio.holdings_df({"AAPL": 180.0})
        for col in ["Ticker", "Shares", "Avg Cost", "Current Price", "Unrealised P&L", "P&L %"]:
            assert col in df.columns

    def test_holdings_df_row_count(self, multi_asset_portfolio):
        df = multi_asset_portfolio.holdings_df({"AAPL": 160.0, "MSFT": 360.0})
        assert len(df) == 2

    def test_transaction_df_columns(self, simple_portfolio):
        df = simple_portfolio.transaction_df()
        assert "Action" in df.columns
        assert "Total (GBP)" in df.columns

    def test_transaction_df_row_count(self, simple_portfolio):
        df = simple_portfolio.transaction_df()
        assert len(df) == 3  # 1 deposit + 2 buys


# ── Trading 212 CSV loader ─────────────────────────────────────────────────────

SAMPLE_CSV = """\
Action,Time,ISIN,Ticker,Name,Notes,ID,No. of shares,Price / share,Currency (Price / share),Exchange rate,Total,Currency (Total)
Deposit,2026-04-04 17:59:00,,,,"Transaction ID: abc",id-001,,,,,4000.00,GBP
Market buy,2026-04-07 08:06:05,IE00BK5BQT80,VWRP,"Vanguard FTSE All-World",,id-002,49.67,126.84,GBP,1.00,6300.00,GBP
"""

SAMPLE_SELL_CSV = """\
Action,Time,ISIN,Ticker,Name,Notes,ID,No. of shares,Price / share,Currency (Price / share),Exchange rate,Total,Currency (Total)
Market sell,2026-04-10 10:00:00,IE00BK5BQT80,VWRP,"Vanguard FTSE All-World",,id-003,10.0,130.00,GBP,1.00,1300.00,GBP
"""

CASH_CSV = """\
Action,Time,Notes,ID,Total,Currency (Total)
Deposit,2026-04-04 18:10:39,"Transaction ID: xyz",id-c01,9000.00,GBP
"""


class TestTrading212Loader:
    def test_parses_deposit(self):
        txs = load_trading212_csv(io.StringIO(SAMPLE_CSV))
        deposits = [t for t in txs if t.action == "Deposit"]
        assert len(deposits) == 1
        assert deposits[0].total == pytest.approx(4000.0)

    def test_parses_market_buy(self):
        txs = load_trading212_csv(io.StringIO(SAMPLE_CSV))
        buys = [t for t in txs if t.action == "Market buy"]
        assert len(buys) == 1
        assert buys[0].ticker == "VWRP"
        assert buys[0].shares == pytest.approx(49.67)
        assert buys[0].total == pytest.approx(6300.0)

    def test_parses_isin(self):
        txs = load_trading212_csv(io.StringIO(SAMPLE_CSV))
        buys = [t for t in txs if t.action == "Market buy"]
        assert buys[0].isin == "IE00BK5BQT80"

    def test_sorted_by_date(self):
        txs = load_trading212_csv(io.StringIO(SAMPLE_CSV))
        dates = [t.date for t in txs]
        assert dates == sorted(dates)

    def test_cash_only_csv(self):
        """Cash ISA CSV has fewer columns — should parse without error."""
        txs = load_trading212_csv(io.StringIO(CASH_CSV))
        assert len(txs) == 1
        assert txs[0].total == pytest.approx(9000.0)

    def test_load_multiple_deduplicates(self):
        """Same transaction ID appearing in two files should not be counted twice."""
        txs = load_multiple_csvs([
            io.StringIO(SAMPLE_CSV),
            io.StringIO(SAMPLE_CSV),   # duplicate
        ])
        ids = [t.transaction_id for t in txs if t.transaction_id]
        assert len(ids) == len(set(ids))

    def test_load_multiple_merges_different_files(self):
        txs = load_multiple_csvs([
            io.StringIO(SAMPLE_CSV),
            io.StringIO(SAMPLE_SELL_CSV),
        ])
        actions = {t.action for t in txs}
        assert "Deposit" in actions
        assert "Market buy" in actions
        assert "Market sell" in actions


# ── Helper functions ───────────────────────────────────────────────────────────

class TestHelpers:
    def test_safe_float_valid(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_invalid(self):
        assert _safe_float("abc") is None

    def test_safe_float_default(self):
        assert _safe_float(None, default=1.0) == pytest.approx(1.0)

    def test_safe_str_valid(self):
        assert _safe_str("AAPL") == "AAPL"

    def test_safe_str_none(self):
        assert _safe_str(None) is None

    def test_safe_str_empty(self):
        assert _safe_str("  ") is None

    def test_safe_str_nan(self):
        assert _safe_str(float("nan")) is None


# ── Limit / Stop order actions ─────────────────────────────────────────────────

class TestLimitStopOrders:
    """Limit buy/sell and Stop buy/sell should behave identically to Market orders."""

    def test_limit_buy_adds_holding(self):
        p = TransactionPortfolio([
            _tx("Limit buy", ticker="AAPL", shares=10, total=1500.0),
        ])
        assert "AAPL" in p.holdings()

    def test_stop_buy_adds_holding(self):
        p = TransactionPortfolio([
            _tx("Stop buy", ticker="AAPL", shares=5, total=800.0),
        ])
        assert p.holdings()["AAPL"].shares == pytest.approx(5.0)

    def test_limit_sell_reduces_holding(self):
        p = TransactionPortfolio([
            _tx("Limit buy",  ticker="AAPL", shares=10, total=1500.0, date="2024-01-01"),
            _tx("Limit sell", ticker="AAPL", shares=3,  total=510.0,  date="2024-01-10"),
        ])
        assert p.holdings()["AAPL"].shares == pytest.approx(7.0)

    def test_stop_sell_reduces_holding(self):
        p = TransactionPortfolio([
            _tx("Stop buy",  ticker="AAPL", shares=10, total=1500.0, date="2024-01-01"),
            _tx("Stop sell", ticker="AAPL", shares=4,  total=640.0,  date="2024-01-10"),
        ])
        assert p.holdings()["AAPL"].shares == pytest.approx(6.0)

    def test_cash_balance_limit_buy(self):
        p = TransactionPortfolio([
            _tx("Deposit",   total=2000.0),
            _tx("Limit buy", ticker="AAPL", shares=10, total=1500.0, date="2024-01-10"),
        ])
        assert p.cash_balance() == pytest.approx(500.0)

    def test_cash_balance_limit_sell(self):
        p = TransactionPortfolio([
            _tx("Deposit",    total=2000.0, date="2024-01-01"),
            _tx("Limit buy",  ticker="AAPL", shares=10, total=1500.0, date="2024-01-05"),
            _tx("Limit sell", ticker="AAPL", shares=5,  total=800.0,  date="2024-01-10"),
        ])
        assert p.cash_balance() == pytest.approx(2000.0 - 1500.0 + 800.0)


# ── holdings_df with missing prices ───────────────────────────────────────────

class TestHoldingsDfMissingPrices:
    def test_none_price_gives_null_columns(self, simple_portfolio):
        """When price is not provided, value/P&L columns should be NaN."""
        df = simple_portfolio.holdings_df({})   # no prices
        assert df["Current Price"].isna().all()
        assert df["Unrealised P&L"].isna().all()


# ── value_history ──────────────────────────────────────────────────────────────

class TestValueHistory:
    def _mock_fetcher(self, tickers, start, end):
        """Return a simple 5-row price DataFrame at a fixed price of 200."""
        idx = pd.date_range("2024-01-10", periods=5, freq="B")
        return pd.DataFrame({t: 200.0 for t in tickers}, index=idx)

    def test_value_history_returns_series(self, simple_portfolio):
        s = simple_portfolio.value_history(self._mock_fetcher)
        assert isinstance(s, pd.Series)
        assert len(s) > 0

    def test_value_history_positive(self, simple_portfolio):
        s = simple_portfolio.value_history(self._mock_fetcher)
        assert (s >= 0).all()

    def test_value_history_no_trades_returns_empty(self):
        p = TransactionPortfolio([_tx("Deposit", total=1000.0)])
        s = p.value_history(self._mock_fetcher)
        assert s.empty

    def test_value_history_empty_prices_returns_empty(self, simple_portfolio):
        def empty_fetcher(tickers, start, end):
            return pd.DataFrame()
        s = simple_portfolio.value_history(empty_fetcher)
        assert s.empty


# ── Trading 212 CSV edge cases ─────────────────────────────────────────────────

WITHDRAWAL_CSV = """\
Action,Time,ISIN,Ticker,Name,Notes,ID,No. of shares,Price / share,Currency (Price / share),Exchange rate,Total,Currency (Total)
Withdrawal,2026-04-05 10:00:00,,,,,id-w01,,,,,500.00,GBP
"""

BAD_DATE_CSV = """\
Action,Time,ISIN,Ticker,Name,Notes,ID,No. of shares,Price / share,Currency (Price / share),Exchange rate,Total,Currency (Total)
Market buy,NOT-A-DATE,IE00BK5BQT80,VWRP,"Vanguard FTSE All-World",,id-bad,10.0,126.84,GBP,1.00,1268.40,GBP
"""


class TestTrading212EdgeCases:
    def test_withdrawal_becomes_negative(self):
        txs = load_trading212_csv(io.StringIO(WITHDRAWAL_CSV))
        assert txs[0].total == pytest.approx(-500.0)

    def test_bad_date_row_is_skipped(self):
        txs = load_trading212_csv(io.StringIO(BAD_DATE_CSV))
        assert len(txs) == 0
