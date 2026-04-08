"""
Webhook Notifier

Sends trading signals to Telegram or Discord so that actionable alerts are
delivered instantly rather than requiring a user to watch console output.

Supported channels
------------------
* **Telegram** — uses the Bot API ``sendMessage`` endpoint.
  Requires a bot token (``--telegram-token``) and a chat/channel ID
  (``--telegram-chat-id``).
* **Discord** — uses an Incoming Webhook URL (``--discord-webhook``).

Both channels are optional and can be used simultaneously.  A :class:`Notifier`
instance is a no-op when neither channel is configured.

Usage example
-------------
.. code-block:: python

    notifier = Notifier(
        telegram_token="...",
        telegram_chat_id="...",
        discord_webhook="https://discord.com/api/webhooks/...",
    )
    notifier.send_signal(signal)

Signal format
-------------
Each notification includes:

* Ticker, signal direction, and strength.
* Entry price (or "next open"), target, and stop-loss.
* ATR value when available.
* Signal expiry time.
* Reasoning.
"""

from __future__ import annotations

import logging
import requests
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trading_signals import TradingSignal

logger = logging.getLogger(__name__)

# Maximum characters in a single Telegram message
_TELEGRAM_MAX_CHARS = 4096


def _format_signal(signal: "TradingSignal") -> str:
    """Return a human-readable text representation of a :class:`TradingSignal`."""
    direction_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(signal.signal, "")
    lines = [
        f"{direction_emoji} *{signal.signal} {signal.ticker}* [{signal.strength}]",
        f"Company: {signal.name} ({signal.sector})",
    ]

    if signal.entry_price is not None:
        entry_label = "Next Open Est." if signal.entry_note == "NEXT OPEN" else "Entry"
        lines.append(f"{entry_label}: ₱{signal.entry_price:.2f}")
        if signal.target_price is not None:
            lines.append(f"Target: ₱{signal.target_price:.2f}")
        if signal.stop_loss is not None:
            lines.append(f"Stop: ₱{signal.stop_loss:.2f}")
        if signal.atr is not None:
            lines.append(f"ATR(14): ₱{signal.atr:.2f}")
    else:
        lines.append("Price: unavailable — check your broker")

    if signal.valid_until:
        lines.append(f"Valid until: {signal.valid_until.strftime('%Y-%m-%d %H:%M %Z')}")

    lines.append(f"Confidence: {signal.sentiment_score:.0%}")
    lines.append(f"_{signal.reasoning}_")

    return "\n".join(lines)


class Notifier:
    """
    Multi-channel webhook notifier for trading signals.

    Instantiate with any combination of Telegram and/or Discord credentials.
    Calls to :meth:`send_signal` and :meth:`send_text` are silently ignored
    when no channels are configured, so the object is always safe to use.

    Args:
        telegram_token: Telegram bot token (the ``XXXXXXX:AAAA…`` string).
        telegram_chat_id: Telegram chat or channel ID to send messages to.
        discord_webhook: Full Discord Incoming Webhook URL.
        timeout: HTTP timeout in seconds for webhook requests (default 10).
    """

    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        discord_webhook: Optional[str] = None,
        timeout: int = 10,
    ):
        self._telegram_token = telegram_token
        self._telegram_chat_id = telegram_chat_id
        self._discord_webhook = discord_webhook
        self._timeout = timeout

        self._has_telegram = bool(telegram_token and telegram_chat_id)
        self._has_discord = bool(discord_webhook)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """Return ``True`` when at least one notification channel is set up."""
        return self._has_telegram or self._has_discord

    def send_signal(self, signal: "TradingSignal") -> None:
        """Format and send a :class:`TradingSignal` to all configured channels.

        Args:
            signal: The trading signal to broadcast.
        """
        if not self.is_configured:
            return
        message = _format_signal(signal)
        self.send_text(message)

    def send_text(self, text: str) -> None:
        """Send a raw text message to all configured channels.

        Args:
            text: Plain text or Markdown-formatted message.
        """
        if self._has_telegram:
            self._send_telegram(text)
        if self._has_discord:
            self._send_discord(text)

    # ------------------------------------------------------------------
    # Private channel implementations
    # ------------------------------------------------------------------

    def _send_telegram(self, text: str) -> None:
        """Send *text* via the Telegram Bot API ``sendMessage`` endpoint."""
        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        # Telegram has a 4096-character hard limit; truncate gracefully.
        payload = {
            "chat_id": self._telegram_chat_id,
            "text": text[:_TELEGRAM_MAX_CHARS],
            "parse_mode": "Markdown",
        }
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            logger.debug("Telegram notification sent.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram send failed: %s", exc)

    def _send_discord(self, text: str) -> None:
        """Send *text* via a Discord Incoming Webhook."""
        # Discord embeds can show Markdown; ``content`` field is plain text.
        # Strip the Markdown bold/italic markers so they render nicely.
        plain = text.replace("*", "**").replace("_", "")
        payload = {"content": plain}
        try:
            resp = requests.post(
                self._discord_webhook,  # type: ignore[arg-type]
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            logger.debug("Discord notification sent.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Discord send failed: %s", exc)
