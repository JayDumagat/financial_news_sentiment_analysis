"""
Historical Signal Backtester

Evaluates how accurate a BUY or SELL signal for a given PSE-listed stock
would have been if applied historically.  Because we do not store past news
articles, the backtest takes a **price-only** approach:

* Download up to ``lookback_days`` of daily close prices from Yahoo Finance.
* Simulate entering the trade on *every* trading day in that window and
  holding for ``holding_days``.
* Report the fraction of those entries that moved in the signal direction
  (``win_rate``) and the average N-day forward return.

This provides a useful **baseline** — e.g. "historically, buying BDO on any
random day and holding 5 days would have been right 54% of the time."  A
sentiment-filtered signal that consistently beats this baseline has genuine
predictive value.

Additional metrics returned
---------------------------
* ``current_trend``  — ``"UPTREND"`` / ``"DOWNTREND"`` / ``"SIDEWAYS"`` based
  on the 20-day simple moving average.
* ``price_vs_ma20``  — percentage distance of the latest close from the 20-day
  MA (positive = above, negative = below).
* ``recent_return_5d``  — 5-day price change as a percentage.
* ``recent_return_20d`` — 20-day price change as a percentage.

All network / data errors are silently swallowed so callers always receive
a result (possibly with ``None`` fields when data is unavailable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy module-level reference so tests can patch it as `backtester.yf`
try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore[assignment]

# Yahoo Finance PSE suffix
_YF_SUFFIX = ".PS"

# Tolerance band for "SIDEWAYS" classification (MA distance within ±1 %)
_SIDEWAYS_BAND: float = 0.01


@dataclass
class BacktestResult:
    """Summary statistics for a price-based signal backtest."""

    ticker: str
    """PSE ticker (without the .PS suffix)."""
    signal: str
    """The signal direction that was tested: ``"BUY"`` or ``"SELL"``."""
    holding_days: int
    """Number of trading days used as the holding period."""

    win_rate: Optional[float]
    """Fraction of historical entries that moved in the signal direction.
    ``None`` if insufficient data."""
    avg_return: Optional[float]
    """Mean N-day forward return (signed, not abs). ``None`` if insufficient."""
    sample_size: Optional[int]
    """Number of historical entry points used in the calculation."""

    current_trend: Optional[str]
    """``"UPTREND"``, ``"DOWNTREND"``, or ``"SIDEWAYS"`` based on 20-day MA."""
    price_vs_ma20: Optional[float]
    """Latest close relative to 20-day MA as a fraction (e.g. 0.03 = 3% above)."""
    recent_return_5d: Optional[float]
    """5-day price change as a fraction (e.g. -0.02 = −2%)."""
    recent_return_20d: Optional[float]
    """20-day price change as a fraction."""

    def summary(self) -> str:
        """Return a compact human-readable summary string."""
        if self.win_rate is None:
            return f"{self.ticker}: No price data available for backtest."

        trend_str = self.current_trend or "N/A"
        wr_str = f"{self.win_rate:.1%}" if self.win_rate is not None else "N/A"
        avg_str = f"{self.avg_return:+.2%}" if self.avg_return is not None else "N/A"
        r5 = f"{self.recent_return_5d:+.2%}" if self.recent_return_5d is not None else "N/A"
        r20 = f"{self.recent_return_20d:+.2%}" if self.recent_return_20d is not None else "N/A"
        ma_str = f"{self.price_vs_ma20:+.2%}" if self.price_vs_ma20 is not None else "N/A"

        return (
            f"{self.ticker} {self.signal} backtest ({self.holding_days}d hold, "
            f"n={self.sample_size}): win={wr_str}, avg={avg_str}  |  "
            f"Trend={trend_str} (vs MA20 {ma_str})  |  "
            f"5d={r5}  20d={r20}"
        )


def backtest_signal(
    ticker: str,
    signal: str,
    holding_days: int = 5,
    lookback_days: int = 252,
) -> BacktestResult:
    """Run a price-only backtest for a PSE stock signal.

    Args:
        ticker: PSE ticker symbol (e.g. ``"BDO"``).  The ``.PS`` suffix is
                added automatically.
        signal: ``"BUY"`` or ``"SELL"``.
        holding_days: Number of trading days to hold the position.
        lookback_days: Number of calendar days of historical data to download.

    Returns:
        A :class:`BacktestResult` (with ``None`` metric fields when price data
        is unavailable).
    """
    empty = BacktestResult(
        ticker=ticker,
        signal=signal,
        holding_days=holding_days,
        win_rate=None,
        avg_return=None,
        sample_size=None,
        current_trend=None,
        price_vs_ma20=None,
        recent_return_5d=None,
        recent_return_20d=None,
    )

    try:
        if yf is None:
            logger.warning("yfinance is not installed; backtesting is unavailable.")
            return empty

        yf_ticker = ticker + _YF_SUFFIX
        # Download a bit more than needed so we have enough rows after windowing
        period_str = f"{lookback_days + holding_days + 30}d"
        data = yf.download(yf_ticker, period=period_str, progress=False, auto_adjust=True)
        if data is None or data.empty or len(data) < holding_days + 5:
            logger.debug("Insufficient price data for %s", yf_ticker)
            return empty

        close = data["Close"].squeeze()  # ensure 1-D Series

        # ------------------------------------------------------------------
        # Forward-return simulation
        # ------------------------------------------------------------------
        fwd_returns = close.pct_change(holding_days).shift(-holding_days).dropna()
        if len(fwd_returns) < 10:
            return empty

        avg_return = float(fwd_returns.mean())
        if signal == "BUY":
            wins = int((fwd_returns > 0).sum())
        else:  # SELL — win when price falls
            wins = int((fwd_returns < 0).sum())
        sample_size = len(fwd_returns)
        win_rate = wins / sample_size

        # ------------------------------------------------------------------
        # Trend metrics
        # ------------------------------------------------------------------
        recent_return_5d: Optional[float] = None
        recent_return_20d: Optional[float] = None
        price_vs_ma20: Optional[float] = None
        current_trend: Optional[str] = None

        if len(close) >= 2:
            recent_return_5d = float(
                (close.iloc[-1] - close.iloc[max(-6, -len(close))]) / close.iloc[max(-6, -len(close))]
            )
        if len(close) >= 21:
            recent_return_20d = float(
                (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]
            )
            ma20 = float(close.iloc[-20:].mean())
            latest_close = float(close.iloc[-1])
            price_vs_ma20 = (latest_close - ma20) / ma20

            if abs(price_vs_ma20) <= _SIDEWAYS_BAND:
                current_trend = "SIDEWAYS"
            elif price_vs_ma20 > 0:
                current_trend = "UPTREND"
            else:
                current_trend = "DOWNTREND"

        return BacktestResult(
            ticker=ticker,
            signal=signal,
            holding_days=holding_days,
            win_rate=win_rate,
            avg_return=avg_return,
            sample_size=sample_size,
            current_trend=current_trend,
            price_vs_ma20=price_vs_ma20,
            recent_return_5d=recent_return_5d,
            recent_return_20d=recent_return_20d,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Backtest failed for %s: %s", ticker, exc)
        return empty


def backtest_signals(
    tickers_and_signals: list[tuple[str, str]],
    holding_days: int = 5,
    lookback_days: int = 252,
) -> list[BacktestResult]:
    """Run :func:`backtest_signal` for multiple ticker/signal pairs.

    Args:
        tickers_and_signals: List of ``(ticker, signal)`` tuples.
        holding_days: Holding period in trading days.
        lookback_days: Historical window in calendar days.

    Returns:
        List of :class:`BacktestResult` in the same order as the input.
    """
    return [
        backtest_signal(ticker, signal, holding_days=holding_days, lookback_days=lookback_days)
        for ticker, signal in tickers_and_signals
    ]
