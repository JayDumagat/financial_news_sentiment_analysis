# Financial News Sentiment Analysis

A Python pipeline that scrapes financial news from **ABS-CBN ANC**
([abs-cbn.com/anc](https://www.abs-cbn.com/anc/anc)), classifies each article's
sentiment with **FinBERT**, identifies affected **PSE-listed stocks**, and
generates actionable **BUY / SELL / HOLD trading signals** — complete with
entry prices, ATR-based stop-loss levels, take-profit targets, signal expiry
times, market-hours awareness, and instant **Telegram / Discord** notifications.

---

## Features

| Feature | Detail |
|---|---|
| **Source** | ABS-CBN ANC & ANC Business section |
| **Sentiment model** | `ProsusAI/finbert` (HuggingFace) |
| **Sentiment labels** | `positive` · `negative` · `neutral` |
| **Noise gate** | Two-layer filter: minimum confidence *and* minimum margin between labels — prevents near-tie outputs from generating false directional calls |
| **Signal strength** | `strong` / `moderate` / `weak` / `neutral` derived from model confidence |
| **Aspect-based sentiment** | Per-ticker context extraction — each PSE stock is scored on the sentences that directly mention it instead of the whole article (avoids conglomerate mismatches) |
| **PSE stock matching** | Direct name / ticker mention detection + sector-level macro triggers |
| **Market-hours awareness** | Checks Philippine Standard Time (UTC+8); when the PSE is closed the signal shows `NEXT OPEN` entry note |
| **ATR-based stops** | 14-period Average True Range sets the stop-loss distance — adapts to each stock's volatility; falls back to fixed % when data is unavailable |
| **Signal expiry** | `valid_until` field: strong = 6 h, moderate = 5 h, weak = 4 h after publication |
| **Trading signals** | BUY / SELL / HOLD with entry price, take-profit target, and ATR stop-loss (via Yahoo Finance live prices) |
| **Notifications** | Telegram Bot API and/or Discord Incoming Webhook — instant actionable alerts via `--telegram-token` / `--discord-webhook` |
| **Relevance filter** | Keyword-score filter + minimum financial keyword count gate (prevents lifestyle articles that mention a brand from being scored as financial) |
| **Fake User-Agent** | `fake-useragent` library rotates browser UA strings so the scraper avoids presenting a static fingerprint; falls back gracefully if not installed |
| **Backtesting** | Historical price-based win-rate, average return, trend indicators, and momentum metrics for each signal |
| **Output** | Console table, CSV, or JSON |
| **Persistence** | Optional SQLite database + data-lake (JSON files by date) |
| **Watch mode** | Poll for new articles on a fixed interval with optional webhook notifications |
| **Offline tests** | Full test suite with mocked HTTP & model — no internet required |

---

## How It Works

```
ABS-CBN ANC news
       │
       ▼
  Relevance Filter  ──── non-financial articles discarded
  ├── Keyword score ≥ threshold
  └── Min. financial keyword count gate (blocks lifestyle brand-mention articles)
       │
       ▼
  FinBERT (ProsusAI)
       │
       ▼
  Two-Gate Noise Filter
  ├── Gate 1: confidence ≥ 0.55
  └── Gate 2: margin (top − runner-up) ≥ 0.15
       │
       ▼
  PSE Stock Matcher
  ├── Direct: ticker / company name in article
  └── Sector: macro keyword triggers (e.g. "BSP rate cut" → all banks)
       │
       ▼
  Aspect-Based Sentiment  (--aspects)
  └── Per-ticker context extraction → individual FinBERT scores per stock
       │
       ▼
  Trading Signal Generator  (--signals)
  ├── Sentiment positive → BUY
  ├── Sentiment negative → SELL
  └── Sentiment neutral  → HOLD
  With:  Entry price · ATR-based stop-loss · Take-profit target
         Market-hours check (CURRENT / NEXT OPEN)
         Signal valid_until (4–6 h after publication)
       │
       ▼
  Historical Backtester  (--backtest)
  ├── Win rate over last ~252 trading days
  ├── Average N-day forward return
  └── Trend (uptrend / downtrend / sideways), 5d & 20d returns
       │
       ├── Telegram / Discord Notification  (--telegram-* / --discord-webhook)
       │
       ▼
  Report  (stdout · CSV · JSON)
```

### Sentiment Noise Reduction

Raw FinBERT scores can be ambiguous. The pipeline applies two gates before
calling a result positive or negative:

1. **Minimum confidence gate** — the winning label's score must be ≥ 0.55.
   Anything below is reported as neutral.
2. **Minimum margin gate** — the gap between the top label and the runner-up
   must be ≥ 0.15.  Near-tie outputs such as `positive=0.56 / negative=0.44`
   are collapsed to neutral.

This combination dramatically reduces noisy directional calls while preserving
genuine high-confidence signals.

### Aspect-Based Sentiment (--aspects)

When `--aspects` is enabled the analyzer scores each PSE ticker individually
rather than applying one article-level sentiment to every matched stock.

For each stock the pipeline:
1. Extracts a ±200-character context window around every direct mention of the
   company name or ticker symbol.
2. Runs FinBERT on that snippet.
3. Attaches `aspect_label`, `aspect_score`, and `aspect_strength` to the stock
   entry; the trading signal uses these per-ticker scores instead of the
   article-level ones.

Stocks with no direct textual mention (sector-level matches only) fall back to
the article-level sentiment, clearly flagged as `aspect_source: "article"`.

### Trading Signal Logic

| Sentiment | Signal | Strength |
|---|---|---|
| positive (≥ 0.80) | BUY | STRONG |
| positive (0.65 – 0.80) | BUY | MODERATE |
| positive (0.55 – 0.65) | BUY | WEAK |
| negative (any) | SELL | (same tiers) |
| neutral | HOLD | — |

**Sector-level matches** (company found via a macro keyword, not by name) are
one tier weaker than direct mentions to reflect lower specificity.

**ATR-based stops** — stop-loss and take-profit are derived from the 14-period
Average True Range rather than a fixed percentage, so they adapt to each
stock's current volatility:

| Strength | ATR mult. (stop) | Reward/risk |
|---|---|---|
| STRONG | 1.5 × ATR | 1.5 : 1 |
| MODERATE | 2.0 × ATR | 1.5 : 1 |
| WEAK | 2.5 × ATR | 1.5 : 1 |

When ATR data is unavailable the generator falls back to fixed percentage
stops (2 % / 1.5 % / 1 %).

**Signal expiry** — every signal carries a `valid_until` timestamp:

| Strength | Validity window |
|---|---|
| STRONG | 6 hours after publication |
| MODERATE | 5 hours |
| WEAK / NEUTRAL | 4 hours |

**Market-hours awareness** — the PSE trades Monday–Friday in two sessions
(09:30–12:00 and 13:30–15:30 Philippine Standard Time, UTC+8).  When a signal
is generated outside these hours the `entry_note` field shows `"NEXT OPEN"` and
the price used is the previous session's close (best available estimate of
tomorrow's open).

### Financial Relevance Filter

The relevance filter prevents lifestyle and entertainment articles from entering
the pipeline, even when they happen to mention a PSE-listed brand:

1. **Score threshold** — total weighted keyword score must reach ≥ 3.0.
2. **Minimum keyword count** — at least 2 distinct financial keywords must
   appear (e.g. `profit`, `revenue`, `interest rate`).  A single brand
   mention (e.g. "celebrity drinks San Miguel") cannot pass on its own.

Articles classified with a financial *category* (e.g. Business, Economy)
bypass the keyword count gate since the category is already a strong signal.

### Backtesting Methodology

The backtester takes a **price-only** approach because historical news
sentiment is not stored.  For each ticker/signal pair it:

1. Downloads up to one year of daily close prices (Yahoo Finance `.PS`).
2. Simulates entering on every trading day in that window and holding for
   `--holding-days` (default 5).
3. Computes:
   - **Win rate** — fraction of entries that moved in the signal direction.
   - **Average return** — mean N-day forward return.
   - **Trend** — UPTREND / DOWNTREND / SIDEWAYS based on the 20-day MA.
   - **Price vs MA20** — distance from the latest close to the 20-day MA.
   - **5-day / 20-day returns** — recent price momentum.

The win rate serves as a **baseline**: a sentiment-filtered signal has genuine
predictive value when it consistently outperforms this random-entry baseline.

---

## Project Structure

```
financial_news_sentiment_analysis/
├── scraper.py            # ABS-CBN ANC web scraper (fake-useragent rotation)
├── sentiment_analyzer.py # FinBERT wrapper with noise gate + aspect analysis
├── pse_stocks.py         # PSE company database & stock matcher
├── relevance_filter.py   # Keyword-based financial relevance filter (min count gate)
├── trading_signals.py    # BUY/SELL/HOLD signal generator (ATR stops, market hours, expiry)
├── backtester.py         # Historical price accuracy backtester
├── notifier.py           # Telegram + Discord webhook notifier
├── main.py               # CLI orchestration script
├── database.py           # SQLite persistence layer
├── data_lake.py          # JSON file data lake
├── requirements.txt      # Python dependencies
└── tests/
    └── test_pipeline.py  # Unit tests (114 tests, no internet required)
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU support:** `torch` will use CUDA automatically if available.
> CPU-only inference works fine for small batches.

### 2. Run the pipeline

```bash
# Basic: print sentiment results to stdout
python main.py

# Fetch full article body for richer analysis (slower)
python main.py --enrich

# Generate BUY/SELL/HOLD trading signals with live prices
python main.py --signals

# Enable aspect-based per-ticker sentiment
python main.py --signals --aspects

# Generate signals + run historical price backtest
python main.py --signals --backtest

# Customize backtest holding period and lookback window
python main.py --signals --backtest --holding-days 10 --lookback-days 180

# Limit articles and save to CSV
python main.py --max-articles 10 --signals --output results.csv

# Save to JSON
python main.py --signals --backtest --output results.json

# Persist to SQLite + data lake
python main.py --signals --db news.db --data-lake data_lake/

# Watch mode — poll every 5 minutes, send Telegram notifications
python main.py --watch --signals --interval 300 \
    --telegram-token <BOT_TOKEN> --telegram-chat-id <CHAT_ID>

# Watch mode with Discord notifications
python main.py --watch --signals --discord-webhook <WEBHOOK_URL>

# Use a different HuggingFace model
python main.py --model yiyanghkust/finbert-tone
```

### 3. Telegram Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) — copy the token.
2. Add the bot to a channel or group and note the chat ID
   (use `@userinfobot` to find it).
3. Pass `--telegram-token <TOKEN> --telegram-chat-id <CHAT_ID>`.

### 4. Discord Setup

1. In your Discord server go to **Channel Settings → Integrations → Webhooks**.
2. Create a webhook and copy the URL.
3. Pass `--discord-webhook <WEBHOOK_URL>`.

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## Command-Line Options

```
usage: main.py [-h] [--max-articles N] [--enrich] [--output FILE]
               [--model MODEL] [--db PATH] [--data-lake DIR]
               [--watch] [--interval SECONDS]
               [--signals] [--aspects] [--backtest]
               [--holding-days N] [--lookback-days N]
               [--telegram-token TOKEN] [--telegram-chat-id CHAT_ID]
               [--discord-webhook URL]
               [--log-level {DEBUG,INFO,WARNING,ERROR}]

options:
  --max-articles N       Maximum number of articles to scrape (default: 20)
  --enrich               Fetch full article body for more accurate analysis
  --output FILE          Save results to FILE (.csv or .json)
  --model MODEL          HuggingFace model identifier (default: ProsusAI/finbert)
  --db PATH              SQLite database path for persistent storage
  --data-lake DIR        Root directory for the JSON data lake
  --watch                Run forever, polling on a fixed interval
  --interval SECONDS     Polling interval for --watch mode (default: 300)
  --signals              Generate BUY/SELL/HOLD trading signals with prices
  --aspects              Enable aspect-based per-ticker sentiment scoring
  --backtest             Backtest each signal against historical price data
  --holding-days N       Holding period (trading days) for backtest (default: 5)
  --lookback-days N      Historical window (calendar days) for backtest (default: 252)
  --telegram-token TOKEN Telegram bot token for signal notifications
  --telegram-chat-id ID  Telegram chat/channel ID for notifications
  --discord-webhook URL  Discord Incoming Webhook URL for notifications
  --log-level            Logging verbosity (default: INFO)
```

---

## Example Output

```
========================================================================
 ABS-CBN ANC Financial News — Sentiment Analysis Report
 Generated: 2024-01-15 09:30:00
========================================================================

[01] ✅ POSITIVE   (93% confidence)  [STRONG]
      BDO Unibank posts record net income for FY2024
      URL : https://www.abs-cbn.com/anc/business/article/...
      Cat : Business
      Pos=0.93  Neg=0.04  Neu=0.03
      📈 PSE stocks (direct): BDO (BDO Unibank)
      ────────────────────────────────────────────────────────────
      📊 TRADING SIGNALS
      🟢 BUY  BDO    (BDO Unibank)  [STRONG]
            Entry ₱132.50  Target ₱136.48  Stop ₱129.85
            ATR ₱1.99  [exp 2024-01-15 15:30]

[02] ❌ NEGATIVE   (87% confidence)  [STRONG]
      Peso weakens sharply amid global trade uncertainty
      URL : https://www.abs-cbn.com/anc/business/article/...
      Cat : Business — Market closed (next open: 2024-01-15 09:30 PHT)
      Pos=0.07  Neg=0.87  Neu=0.06
      🏢 PSE sectors affected: Financials, Services
      ────────────────────────────────────────────────────────────
      📊 TRADING SIGNALS
      🔴 SELL MBT    (Metropolitan Bank & Trust)  [MODERATE]
            Next Open ₱71.20  Target ₱69.57  Stop ₱72.62  ATR ₱0.81

------------------------------------------------------------------------
 Summary: 2 articles  |  Positive: 1  |  Negative: 1  |  Neutral:  0
========================================================================
```

---

## Modules

### `sentiment_analyzer.py`

| Class / Function | Description |
|---|---|
| `FinBERTAnalyzer` | Lazy-loading FinBERT pipeline wrapper |
| `FinBERTAnalyzer.analyze(text)` | Single-text classification |
| `FinBERTAnalyzer.analyze_batch(texts)` | Batched classification with noise gates |
| `FinBERTAnalyzer.analyze_aspects(text, stocks)` | Per-ticker context extraction and scoring |
| `SentimentResult` | Dataclass: label, score, all_scores, strength, affected_stocks |
| `analyze_text(text)` | Convenience function |

### `trading_signals.py`

| Class / Function | Description |
|---|---|
| `TradingSignal` | Dataclass: ticker, signal, strength, entry/target/stop, entry_note, valid_until, atr |
| `generate_signals(result, published_at, now)` | Convert a `SentimentResult` into `TradingSignal` objects |
| `is_pse_market_open(now)` | Check if the PSE is currently in a live session |
| `next_pse_market_open(now)` | Return the next PSE session open datetime |
| `_fetch_latest_price(ticker)` | Fetch latest close from Yahoo Finance (`.PS` suffix) |
| `_fetch_atr(ticker, period)` | Compute 14-period ATR from OHLC data |

### `notifier.py`

| Class / Function | Description |
|---|---|
| `Notifier` | Multi-channel webhook notifier (Telegram + Discord) |
| `Notifier.send_signal(signal)` | Format and broadcast a `TradingSignal` |
| `Notifier.send_text(text)` | Send a raw text message to all configured channels |

### `backtester.py`

| Class / Function | Description |
|---|---|
| `BacktestResult` | Dataclass: win_rate, avg_return, trend, price_vs_ma20, 5d/20d returns |
| `backtest_signal(ticker, signal)` | Run a price-based backtest for one ticker/signal |
| `backtest_signals(pairs)` | Batch wrapper for multiple ticker/signal pairs |

### `pse_stocks.py`

| Class / Function | Description |
|---|---|
| `PSE_COMPANIES` | Database of major PSE blue-chip companies with keywords and sectors |
| `SECTOR_TRIGGERS` | Macro keywords that trigger sector-wide matches |
| `find_affected_stocks(text)` | Scan text for company / ticker / sector matches |

### `scraper.py`

| Class / Function | Description |
|---|---|
| `ANCNewsScraper` | Session-based scraper with rotating User-Agent (`fake-useragent`) |
| `ANCNewsScraper.get_articles(max_articles)` | Scrape listing pages |
| `ANCNewsScraper.enrich_article(article)` | Fetch full article body |
| `scrape_anc_news(max_articles, enrich)` | Convenience function |
| `NewsArticle` | Dataclass: title, URL, summary, content, category, published_at |

### `relevance_filter.py`

| Class / Function | Description |
|---|---|
| `RelevanceFilter` | Keyword-score filter with minimum-count gate, in-memory + DB caching |
| `RelevanceFilter.is_financial(article)` | Returns `True` for financially relevant articles |

### `main.py`

| Function | Description |
|---|---|
| `run(args)` | Full pipeline (scrape → analyse → aspects → signals → backtest → notify → report → save) |
| `build_report(articles, results, signals_map, backtest_map)` | Merge into list of dicts |
| `print_report(report)` | Pretty-print to stdout |
| `save_csv(report, path)` | Write CSV file |
| `save_json(report, path)` | Write JSON file |

---

## Configuration

### Sentiment thresholds (`sentiment_analyzer.py`)

| Constant | Default | Description |
|---|---|---|
| `MIN_DIRECTIONAL_CONFIDENCE` | `0.55` | Gate 1: minimum winning-label score |
| `MIN_CONFIDENCE_MARGIN` | `0.15` | Gate 2: minimum gap between top 2 labels |
| `_STRONG_THRESHOLD` | `0.80` | Score threshold for "strong" signal |
| `_MODERATE_THRESHOLD` | `0.65` | Score threshold for "moderate" signal |

### Signal ATR / fallback risk (`trading_signals.py`)

| Constant | Default | Description |
|---|---|---|
| `_ATR_PERIOD` | `14` | ATR lookback period |
| `_ATR_MULTIPLIER["strong"]` | `1.5` | ATR × multiplier = stop distance (strong) |
| `_ATR_MULTIPLIER["moderate"]` | `2.0` | ATR × multiplier = stop distance (moderate) |
| `_ATR_MULTIPLIER["weak"]` | `2.5` | ATR × multiplier = stop distance (weak) |
| `_FALLBACK_RISK_PCT["strong"]` | `0.020` | Fallback stop % (no ATR data) |
| `_REWARD_RISK_RATIO` | `1.5` | Target = 1.5 × stop distance |

### Signal expiry (`trading_signals.py`)

| Strength | `valid_until` |
|---|---|
| `strong` | 6 h after publication |
| `moderate` | 5 h |
| `weak` / `neutral` | 4 h |

### Backtester defaults (`backtester.py`)

| Constant | Default | Description |
|---|---|---|
| `holding_days` | `5` | Trading days to hold the position |
| `lookback_days` | `252` | Calendar-day historical window (~1 year) |
| `_SIDEWAYS_BAND` | `0.01` | ±1 % MA distance → "SIDEWAYS" trend |

### Relevance filter (`relevance_filter.py`)

| Constant | Default | Description |
|---|---|---|
| `RELEVANCE_THRESHOLD` | `3.0` | Minimum total keyword score |
| `MIN_FINANCIAL_KEYWORD_COUNT` | `2` | Minimum distinct financial keywords (keyword-only articles) |

---

## Limitations and Future Work

- **Historical news dataset** — a true news-aligned backtest would require a
  historical dataset of ANC articles matched to the price action on the same
  date.  The current backtester is price-only (baseline win-rate).
- **PSE trading calendar** — public holidays are not currently tracked; the
  market-hours check uses weekday + session times only.
- **Aspect model granularity** — aspect extraction is sentence-window based;
  a dedicated aspect-BERT model would yield better per-entity scores.

---

## Disclaimer

This tool is provided for **educational and research purposes only**.  The
trading signals and backtest results are derived from a news-sentiment model
and historical price data — they do not constitute financial advice.  Always
conduct your own due diligence before making any investment decisions.

---

## License

MIT — see [LICENSE](LICENSE).
