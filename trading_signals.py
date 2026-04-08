"""
Trading Signal Generator

Converts FinBERT sentiment results into actionable trading signals for
Philippine Stock Exchange (PSE) listed companies.

For each stock identified as potentially affected by a news item, a
:class:`TradingSignal` is generated that includes:

* The recommended action (BUY / SELL / HOLD).
* A strength qualifier (STRONG / MODERATE / WEAK / HOLD).
* Suggested entry price, take-profit target, and stop-loss level derived
  from the latest available market close price and the signal strength.
* A human-readable reasoning string.

Price data is fetched from Yahoo Finance (PSE tickers use the ``.PS`` suffix).
All price-fetch errors are silently swallowed so that signals are still
generated (without prices) even when the network is unavailable.

Signal / price calculation rules
---------------------------------
* Base risk/reward ratio: 1.5:1 (target = 1.5 × risk from entry).
* Risk per trade (stop distance from entry):

  ============  ========
  Strength      Risk %
  ============  ========
  strong        2.0 %
  moderate      1.5 %
  weak          1.0 %
  ============  ========

* BUY  signal: target = entry × (1 + reward %), stop = entry × (1 − risk %).
* SELL signal: target = entry × (1 − reward %), stop = entry × (1 + risk %).
* HOLD signal: no target / stop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sentiment_analyzer import SentimentResult

logger = logging.getLogger(__name__)

# Risk % per trade keyed by signal strength
_RISK_PCT: dict[str, float] = {
    "strong": 0.020,
    "moderate": 0.015,
    "weak": 0.010,
}

# Reward/risk ratio used to derive the take-profit target
_REWARD_RISK_RATIO: float = 1.5

# Yahoo Finance suffix for PSE-listed stocks
_YF_SUFFIX = ".PS"


@dataclass
class TradingSignal:
    """Encapsulates a single trade recommendation for one PSE stock."""

    ticker: str
    """PSE ticker symbol (e.g. ``"BDO"``)."""
    name: str
    """Full company name."""
    sector: str
    """PSE sector classification."""
    match_type: str
    """How the stock was linked to the article: ``"direct"`` or ``"sector"``."""

    signal: str
    """Recommended action: ``"BUY"``, ``"SELL"``, or ``"HOLD"``."""
    strength: str
    """Signal strength: ``"STRONG"``, ``"MODERATE"``, ``"WEAK"``, or ``"HOLD"``."""

    entry_price: Optional[float]
    """Suggested entry price in PHP (latest close). ``None`` if unavailable."""
    target_price: Optional[float]
    """Take-profit level in PHP. ``None`` if unavailable or HOLD."""
    stop_loss: Optional[float]
    """Stop-loss level in PHP. ``None`` if unavailable or HOLD."""

    sentiment_label: str
    """Underlying sentiment label that drove this signal."""
    sentiment_score: float
    """Model confidence for the underlying sentiment label."""
    reasoning: str
    """Short human-readable explanation of why this signal was generated."""

    def __str__(self) -> str:
        price_str = ""
        if self.entry_price is not None:
            price_str = (
                f"  Entry ₱{self.entry_price:.2f}"
                + (f"  Target ₱{self.target_price:.2f}" if self.target_price else "")
                + (f"  Stop ₱{self.stop_loss:.2f}" if self.stop_loss else "")
            )
        return (
            f"[{self.signal:4s} / {self.strength:8s}]  {self.ticker} ({self.name})"
            f"{price_str}"
        )


def _fetch_latest_price(ticker: str) -> Optional[float]:
    """Return the latest closing price for *ticker* from Yahoo Finance.

    Appends the ``.PS`` suffix automatically for PSE stocks and returns
    ``None`` if the price cannot be retrieved for any reason.
    """
    try:
        import yfinance as yf  # imported here to keep module importable without yfinance

        yf_ticker = ticker + _YF_SUFFIX
        data = yf.download(yf_ticker, period="5d", progress=False, auto_adjust=True)
        if data is None or data.empty:
            logger.debug("No price data returned for %s", yf_ticker)
            return None
        close_series = data["Close"]
        if hasattr(close_series, "iloc"):
            latest = float(close_series.iloc[-1])
        else:
            latest = float(close_series)
        return latest if latest > 0 else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Price fetch failed for %s: %s", ticker, exc)
        return None


def _compute_prices(
    entry: Optional[float],
    signal: str,
    strength: str,
) -> tuple[Optional[float], Optional[float]]:
    """Return (target_price, stop_loss) given entry and signal direction."""
    if entry is None or signal == "HOLD":
        return None, None

    risk_pct = _RISK_PCT.get(strength.lower(), 0.015)
    reward_pct = risk_pct * _REWARD_RISK_RATIO

    if signal == "BUY":
        target = round(entry * (1 + reward_pct), 2)
        stop = round(entry * (1 - risk_pct), 2)
    else:  # SELL
        target = round(entry * (1 - reward_pct), 2)
        stop = round(entry * (1 + risk_pct), 2)

    return target, stop


def generate_signals(result: "SentimentResult") -> list[TradingSignal]:
    """Generate :class:`TradingSignal` objects for every stock in *result*.

    Args:
        result: A :class:`~sentiment_analyzer.SentimentResult` that already
                contains an ``affected_stocks`` list.

    Returns:
        A list of :class:`TradingSignal` objects — one per affected stock.
        Returns an empty list when there are no affected stocks or when the
        sentiment is neutral.
    """
    signals: list[TradingSignal] = []

    for stock in result.affected_stocks:
        ticker = stock["ticker"]
        name = stock["name"]
        sector = stock["sector"]
        match_type = stock.get("match_type", "sector")

        # Determine direction from sentiment
        if result.label == "positive":
            signal = "BUY"
        elif result.label == "negative":
            signal = "SELL"
        else:
            signal = "HOLD"

        # Sector-level matches are generally less reliable → downgrade strength
        if match_type == "sector" and result.strength in ("strong", "moderate"):
            effective_strength = "weak" if result.strength == "moderate" else "moderate"
        else:
            effective_strength = result.strength

        display_strength = "HOLD" if signal == "HOLD" else effective_strength.upper()

        # Fetch current price
        entry = _fetch_latest_price(ticker)
        target, stop = _compute_prices(entry, signal, effective_strength)

        # Build reasoning
        if signal == "HOLD":
            reason = (
                f"Sentiment is neutral ({result.score:.0%} confidence). "
                f"No directional trade recommended for {name}."
            )
        else:
            match_desc = "directly mentioned" if match_type == "direct" else "sector-level trigger"
            reason = (
                f"{name} was {match_desc} in a {result.label} article "
                f"({result.score:.0%} confidence, {result.strength} signal). "
            )
            if entry is not None:
                risk_pct = _RISK_PCT.get(effective_strength, 0.015)
                reward_pct = risk_pct * _REWARD_RISK_RATIO
                reason += (
                    f"Entry at ₱{entry:.2f}. "
                    f"Target: ₱{target:.2f} (+{reward_pct:.1%}). "
                    f"Stop: ₱{stop:.2f} (-{risk_pct:.1%})."
                )
            else:
                reason += "Live price unavailable — check your broker for current levels."

        signals.append(
            TradingSignal(
                ticker=ticker,
                name=name,
                sector=sector,
                match_type=match_type,
                signal=signal,
                strength=display_strength,
                entry_price=entry,
                target_price=target,
                stop_loss=stop,
                sentiment_label=result.label,
                sentiment_score=result.score,
                reasoning=reason,
            )
        )

    return signals
