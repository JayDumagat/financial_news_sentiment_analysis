"""
Trading Signal Generator

Converts FinBERT sentiment results into actionable trading signals for
Philippine Stock Exchange (PSE) listed companies.

For each stock identified as potentially affected by a news item, a
:class:`TradingSignal` is generated that includes:

* The recommended action (BUY / SELL / HOLD).
* A strength qualifier (STRONG / MODERATE / WEAK / HOLD).
* Suggested entry price, take-profit target, and stop-loss level.
* A ``valid_until`` timestamp — signals expire within 4–6 hours of the
  article's publication time because news sentiment has a short half-life.
* Market-hours awareness — when the PSE is closed the signal notes that
  entry should be taken at the **next market open** and uses the open
  price rather than the close.

Price data is fetched from **TradingView** (via the unofficial
``tradingview-ta`` library) using the PSE exchange and the ``philippines``
screener.  Because this is an unofficial API, prices are silently swallowed
on any error so signals are still generated (without prices) when the
network is unavailable.

Signal / price calculation rules
---------------------------------
* Stop-loss is set using the **Average True Range** (14-period ATR) rather
  than a fixed percentage, so position sizing adapts to each stock's
  volatility.  The ATR multipliers are:

  ============  ===========
  Strength      ATR mult.
  ============  ===========
  strong        1.5 ×
  moderate      2.0 ×
  weak          2.5 ×
  ============  ===========

  If ATR data is unavailable the generator falls back to the legacy fixed
  percentage stops (2 % / 1.5 % / 1 %).

* Base risk/reward ratio: 1.5:1 (target = 1.5 × stop distance from entry).
* BUY  signal: target = entry + reward, stop = entry − risk.
* SELL signal: target = entry − reward, stop = entry + risk.
* HOLD signal: no target / stop.

PSE market hours
----------------
The PSE trades Monday–Friday, 09:30–12:00 and 13:30–15:30 Philippine Time
(UTC+8).  When :func:`generate_signals` is called outside these hours the
``entry_note`` field is set to ``"NEXT OPEN"`` and the entry price reflects
the previous session's close (the best available estimate of tomorrow's open).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sentiment_analyzer import SentimentResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PSE market-hours helpers
# ---------------------------------------------------------------------------

# Philippine Standard Time is UTC+8
_PHT = timezone(timedelta(hours=8))

# PSE trading windows (inclusive start, exclusive end) in local PHT time
_PSE_SESSIONS: list[tuple[int, int, int, int]] = [
    # (start_hour, start_min, end_hour, end_min)
    (9, 30, 12, 0),
    (13, 30, 15, 30),
]

_PSE_VALID_WEEKDAYS = range(0, 5)  # Monday=0 … Friday=4


def is_pse_market_open(now: Optional[datetime] = None) -> bool:
    """Return ``True`` when the PSE is currently in a live trading session.

    Args:
        now: Override *now* (in any timezone or naive UTC).  Uses
             ``datetime.now(PHT)`` when omitted.
    """
    if now is None:
        now = datetime.now(_PHT)
    else:
        # Normalise to PHT
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc).astimezone(_PHT)
        else:
            now = now.astimezone(_PHT)

    if now.weekday() not in _PSE_VALID_WEEKDAYS:
        return False

    t = (now.hour, now.minute)
    for sh, sm, eh, em in _PSE_SESSIONS:
        if (sh, sm) <= t < (eh, em):
            return True
    return False


def next_pse_market_open(now: Optional[datetime] = None) -> datetime:
    """Return the datetime of the next PSE session open (in PHT).

    Args:
        now: Override *now*.  Uses ``datetime.now(PHT)`` when omitted.
    """
    if now is None:
        now = datetime.now(_PHT)
    else:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc).astimezone(_PHT)
        else:
            now = now.astimezone(_PHT)

    # Walk forward (at most 7 days) until we find a weekday session that
    # hasn't started yet.
    candidate = now
    for _ in range(7 * 24 * 60):  # iterate by minute for at most 7 days
        if candidate.weekday() in _PSE_VALID_WEEKDAYS:
            for sh, sm, _eh, _em in _PSE_SESSIONS:
                session_open = candidate.replace(
                    hour=sh, minute=sm, second=0, microsecond=0
                )
                if session_open > now:
                    return session_open
        # Move to the next calendar day, aligned to midnight
        candidate = (candidate + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    # Fallback — should never be reached
    return now + timedelta(hours=24)


# ---------------------------------------------------------------------------
# ATR helpers
# ---------------------------------------------------------------------------

# ATR multiplier per signal strength — larger multiplier → wider stop
_ATR_MULTIPLIER: dict[str, float] = {
    "strong": 1.5,
    "moderate": 2.0,
    "weak": 2.5,
}

# Fallback fixed risk % when ATR is unavailable
_FALLBACK_RISK_PCT: dict[str, float] = {
    "strong": 0.020,
    "moderate": 0.015,
    "weak": 0.010,
}

# Reward/risk ratio used to derive the take-profit target
_REWARD_RISK_RATIO: float = 1.5

# Number of periods for ATR calculation (matches TradingView's default ATR(14))
_ATR_PERIOD: int = 14

# Signal validity windows by strength (hours after publication)
_VALID_HOURS: dict[str, int] = {
    "strong": 6,
    "moderate": 5,
    "weak": 4,
    "neutral": 4,
}


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
    """Suggested entry price in PHP (latest close or open). ``None`` if unavailable."""
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

    entry_note: str = "CURRENT"
    """``"CURRENT"`` — enter at the live price shown; ``"NEXT OPEN"`` — market
    is closed, enter at the next PSE session open."""
    valid_until: Optional[datetime] = field(default=None)
    """Expiry timestamp for this signal (4–6 hours after article publication)."""
    atr: Optional[float] = field(default=None)
    """14-period ATR value used to set stop / target, if available."""

    def __str__(self) -> str:
        price_str = ""
        if self.entry_price is not None:
            entry_label = f"{'Next Open' if self.entry_note == 'NEXT OPEN' else 'Entry'}"
            price_str = (
                f"  {entry_label} ₱{self.entry_price:.2f}"
                + (f"  Target ₱{self.target_price:.2f}" if self.target_price else "")
                + (f"  Stop ₱{self.stop_loss:.2f}" if self.stop_loss else "")
            )
        expires = ""
        if self.valid_until:
            expires = f"  [exp {self.valid_until.strftime('%H:%M')}]"
        return (
            f"[{self.signal:4s} / {self.strength:8s}]  {self.ticker} ({self.name})"
            f"{price_str}{expires}"
        )


def _fetch_latest_price(ticker: str) -> Optional[float]:
    """Return the latest closing price for *ticker* from TradingView.

    Uses the unofficial ``tradingview-ta`` library with the PSE exchange and
    ``philippines`` screener.  Returns ``None`` if the price cannot be
    retrieved for any reason (library not installed, network error, etc.).
    """
    try:
        from tradingview_ta import TA_Handler, Interval  # type: ignore[import]

        handler = TA_Handler(
            symbol=ticker,
            screener="philippines",
            exchange="PSE",
            interval=Interval.INTERVAL_1_DAY,
        )
        analysis = handler.get_analysis()
        close = analysis.indicators.get("close")
        if close is None:
            return None
        close = float(close)
        return close if close > 0 else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("TradingView price fetch failed for %s: %s", ticker, exc)
        return None


def _fetch_atr(ticker: str, period: int = _ATR_PERIOD) -> Optional[float]:
    """Return the ATR({period}) value for *ticker* from TradingView.

    Uses the same ``tradingview-ta`` TA_Handler call as :func:`_fetch_latest_price`
    — TradingView computes ATR(14) as a standard indicator so no separate OHLCV
    download is required.  Returns ``None`` on any failure.
    """
    try:
        from tradingview_ta import TA_Handler, Interval  # type: ignore[import]

        handler = TA_Handler(
            symbol=ticker,
            screener="philippines",
            exchange="PSE",
            interval=Interval.INTERVAL_1_DAY,
        )
        analysis = handler.get_analysis()
        atr = analysis.indicators.get("ATR")
        if atr is None:
            return None
        atr = float(atr)
        return atr if atr > 0 else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("TradingView ATR fetch failed for %s: %s", ticker, exc)
        return None


def _compute_prices(
    entry: Optional[float],
    signal: str,
    strength: str,
    atr: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (target_price, stop_loss, atr_used) given entry and signal direction.

    Uses ATR-based stops when ATR data is available; falls back to fixed
    percentage stops otherwise.
    """
    if entry is None or signal == "HOLD":
        return None, None, None

    strength_key = strength.lower()

    if atr is not None and atr > 0:
        multiplier = _ATR_MULTIPLIER.get(strength_key, 2.0)
        risk_amount = atr * multiplier
        reward_amount = risk_amount * _REWARD_RISK_RATIO
        atr_used = atr
    else:
        risk_pct = _FALLBACK_RISK_PCT.get(strength_key, 0.015)
        risk_amount = entry * risk_pct
        reward_amount = risk_amount * _REWARD_RISK_RATIO
        atr_used = None

    if signal == "BUY":
        target = round(entry + reward_amount, 2)
        stop = round(entry - risk_amount, 2)
    else:  # SELL
        target = round(entry - reward_amount, 2)
        stop = round(entry + risk_amount, 2)

    return target, stop, atr_used


def generate_signals(
    result: "SentimentResult",
    published_at: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> list[TradingSignal]:
    """Generate :class:`TradingSignal` objects for every stock in *result*.

    Args:
        result: A :class:`~sentiment_analyzer.SentimentResult` that already
                contains an ``affected_stocks`` list.  When the result was
                produced by aspect-based analysis, each stock entry may carry
                an ``aspect_label`` / ``aspect_score`` that overrides the
                article-level sentiment.
        published_at: Publication datetime of the source article, used to
                      compute the ``valid_until`` expiry.  Defaults to *now*.
        now: Override for the current time (useful in tests).

    Returns:
        A list of :class:`TradingSignal` objects — one per affected stock.
        Returns an empty list when there are no affected stocks or when the
        sentiment is neutral.
    """
    signals: list[TradingSignal] = []

    # Determine market status once for all signals in this batch
    market_open = is_pse_market_open(now)
    entry_note = "CURRENT" if market_open else "NEXT OPEN"

    ref_time = now or datetime.now(_PHT)
    if ref_time.tzinfo is None:
        ref_time = ref_time.replace(tzinfo=timezone.utc).astimezone(_PHT)
    else:
        ref_time = ref_time.astimezone(_PHT)

    pub_time = published_at or ref_time

    for stock in result.affected_stocks:
        ticker = stock["ticker"]
        name = stock["name"]
        sector = stock["sector"]
        match_type = stock.get("match_type", "sector")

        # --- Aspect-based sentiment override ---
        # If the stock entry carries per-ticker sentiment scores (produced by
        # the aspect analyzer), use those; otherwise fall back to article-level.
        label = stock.get("aspect_label") or result.label
        score = stock.get("aspect_score") or result.score
        strength = stock.get("aspect_strength") or result.strength

        # Determine direction from (possibly per-ticker) sentiment
        if label == "positive":
            signal = "BUY"
        elif label == "negative":
            signal = "SELL"
        else:
            signal = "HOLD"

        # Sector-level matches are generally less reliable → downgrade strength
        if match_type == "sector" and strength in ("strong", "moderate"):
            effective_strength = "weak" if strength == "moderate" else "moderate"
        else:
            effective_strength = strength

        display_strength = "HOLD" if signal == "HOLD" else effective_strength.upper()

        # Fetch current price and ATR
        entry = _fetch_latest_price(ticker)
        atr = _fetch_atr(ticker) if entry is not None and signal != "HOLD" else None
        target, stop, atr_used = _compute_prices(entry, signal, effective_strength, atr)

        # Compute valid_until based on signal strength
        valid_hours = _VALID_HOURS.get(effective_strength, 5)
        valid_until = pub_time + timedelta(hours=valid_hours)

        # Build reasoning
        if signal == "HOLD":
            reason = (
                f"Sentiment is neutral ({score:.0%} confidence). "
                f"No directional trade recommended for {name}."
            )
        else:
            match_desc = "directly mentioned" if match_type == "direct" else "sector-level trigger"
            reason = (
                f"{name} was {match_desc} in a {label} article "
                f"({score:.0%} confidence, {effective_strength} signal). "
            )
            if entry is not None:
                open_note = " (next open estimate)" if entry_note == "NEXT OPEN" else ""
                reason += (
                    f"Entry ₱{entry:.2f}{open_note}. "
                    f"Target: ₱{target:.2f}.  Stop: ₱{stop:.2f}."
                )
                if atr_used:
                    reason += f"  ATR({_ATR_PERIOD})=₱{atr_used:.2f}."
            else:
                reason += "Live price unavailable — check your broker for current levels."
            reason += (
                f"  Signal valid until {valid_until.strftime('%Y-%m-%d %H:%M %Z')}."
            )

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
                sentiment_label=label,
                sentiment_score=score,
                reasoning=reason,
                entry_note=entry_note,
                valid_until=valid_until,
                atr=atr_used,
            )
        )

    return signals
