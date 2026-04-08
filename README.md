# Financial News Sentiment Analysis

A Python pipeline that scrapes financial news from **ABS-CBN ANC**
([abs-cbn.com/anc](https://www.abs-cbn.com/anc/anc)), classifies each article's
sentiment with **FinBERT**, identifies affected **PSE-listed stocks**, and
generates actionable **BUY / SELL / HOLD trading signals** — complete with
entry prices, take-profit targets, stop-loss levels, and a **historical
price-based backtest** to gauge signal accuracy.

---

## Features

| Feature | Detail |
|---|---|
| **Source** | ABS-CBN ANC & ANC Business section |
| **Sentiment model** | `ProsusAI/finbert` (HuggingFace) |
| **Sentiment labels** | `positive` · `negative` · `neutral` |
| **Noise gate** | Two-layer filter: minimum confidence *and* minimum margin between labels — prevents near-tie outputs from generating false directional calls |
| **Signal strength** | `strong` / `moderate` / `weak` / `neutral` derived from model confidence |
| **PSE stock matching** | Direct name / ticker mention detection + sector-level macro triggers |
| **Trading signals** | BUY / SELL / HOLD with entry price, take-profit target, and stop-loss (via Yahoo Finance live prices) |
| **Backtesting** | Historical price-based win-rate, average return, trend indicators, and momentum metrics for each signal |
| **Output** | Console table, CSV, or JSON |
| **Persistence** | Optional SQLite database + data-lake (JSON files by date) |
| **Watch mode** | Poll for new articles on a fixed interval |
| **Offline tests** | Full test suite with mocked HTTP & model — no internet required |

---

## How It Works

```
ABS-CBN ANC news
       │
       ▼
  Relevance Filter  ──── non-financial articles discarded
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
  Trading Signal Generator  (--signals)
  ├── Sentiment positive → BUY
  ├── Sentiment negative → SELL
  └── Sentiment neutral  → HOLD
  With:  Entry price · Take-profit target · Stop-loss
       │
       ▼
  Historical Backtester  (--backtest)
  ├── Win rate over last ~252 trading days
  ├── Average N-day forward return
  └── Trend (uptrend / downtrend / sideways), 5d & 20d returns
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

**Price levels** use a fixed 1.5 : 1 reward/risk ratio:

| Strength | Stop distance | Target distance |
|---|---|---|
| STRONG | 2.0 % | 3.0 % |
| MODERATE | 1.5 % | 2.25 % |
| WEAK | 1.0 % | 1.5 % |

Prices are fetched in real time from Yahoo Finance (PSE tickers: `TICKER.PS`).
The pipeline still generates signals without prices if the network is
unavailable.

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
├── scraper.py            # ABS-CBN ANC web scraper
├── sentiment_analyzer.py # FinBERT wrapper with noise gate
├── pse_stocks.py         # PSE company database & stock matcher
├── relevance_filter.py   # Keyword-based financial relevance filter
├── trading_signals.py    # BUY/SELL/HOLD signal generator
├── backtester.py         # Historical price accuracy backtester
├── main.py               # CLI orchestration script
├── database.py           # SQLite persistence layer
├── data_lake.py          # JSON file data lake
├── requirements.txt      # Python dependencies
└── tests/
    └── test_pipeline.py  # Unit tests (74 tests, no internet required)
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

# Watch mode — poll every 5 minutes
python main.py --watch --signals --interval 300

# Use a different HuggingFace model
python main.py --model yiyanghkust/finbert-tone
```

### 3. Run tests

```bash
python -m pytest tests/ -v
```

---

## Command-Line Options

```
usage: main.py [-h] [--max-articles N] [--enrich] [--output FILE]
               [--model MODEL] [--db PATH] [--data-lake DIR]
               [--watch] [--interval SECONDS]
               [--signals] [--backtest]
               [--holding-days N] [--lookback-days N]
               [--log-level {DEBUG,INFO,WARNING,ERROR}]

options:
  --max-articles N    Maximum number of articles to scrape (default: 20)
  --enrich            Fetch full article body for more accurate analysis
  --output FILE       Save results to FILE (.csv or .json)
  --model MODEL       HuggingFace model identifier (default: ProsusAI/finbert)
  --db PATH           SQLite database path for persistent storage
  --data-lake DIR     Root directory for the JSON data lake
  --watch             Run forever, polling on a fixed interval
  --interval SECONDS  Polling interval for --watch mode (default: 300)
  --signals           Generate BUY/SELL/HOLD trading signals with prices
  --backtest          Backtest each signal against historical price data
  --holding-days N    Holding period (trading days) for backtest (default: 5)
  --lookback-days N   Historical window (calendar days) for backtest (default: 252)
  --log-level         Logging verbosity (default: INFO)
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
      🟢 BUY  BDO    (BDO Unibank)                [STRONG]  Entry ₱132.50  Target ₱136.48  Stop ₱129.85
      ────────────────────────────────────────────────────────────
      🔬 BACKTEST (historical price accuracy)
      BDO    BUY  | Win rate: 55.2% (n=247, 5d)  Avg return: +0.78%  Trend: UPTREND  5d: +1.2%  20d: +3.1%

[02] ❌ NEGATIVE   (87% confidence)  [STRONG]
      Peso weakens sharply amid global trade uncertainty
      URL : https://www.abs-cbn.com/anc/business/article/...
      Cat : Business
      Pos=0.07  Neg=0.87  Neu=0.06
      🏢 PSE sectors affected: Financials, Services
      ────────────────────────────────────────────────────────────
      📊 TRADING SIGNALS
      🔴 SELL MBT    (Metropolitan Bank & Trust)  [MODERATE]  Entry ₱71.20 → Target ₱69.60  Stop ₱72.62
      🔴 SELL BPI    (Bank of the Philippine Is.) [MODERATE]  Entry ₱108.00 → Target ₱105.57  Stop ₱110.16

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
| `SentimentResult` | Dataclass: label, score, all_scores, strength, affected_stocks |
| `analyze_text(text)` | Convenience function |

### `trading_signals.py`

| Class / Function | Description |
|---|---|
| `TradingSignal` | Dataclass: ticker, signal, strength, entry/target/stop prices, reasoning |
| `generate_signals(result)` | Convert a `SentimentResult` into `TradingSignal` objects |
| `_fetch_latest_price(ticker)` | Fetch latest close from Yahoo Finance (`.PS` suffix) |

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
| `ANCNewsScraper` | Session-based scraper with configurable delay |
| `ANCNewsScraper.get_articles(max_articles)` | Scrape listing pages |
| `ANCNewsScraper.enrich_article(article)` | Fetch full article body |
| `scrape_anc_news(max_articles, enrich)` | Convenience function |
| `NewsArticle` | Dataclass: title, URL, summary, content, category, published_at |

### `relevance_filter.py`

| Class / Function | Description |
|---|---|
| `RelevanceFilter` | Keyword-score filter with in-memory + DB caching |
| `RelevanceFilter.is_financial(article)` | Returns `True` for financially relevant articles |

### `main.py`

| Function | Description |
|---|---|
| `run(args)` | Full pipeline (scrape → analyse → signals → backtest → report → save) |
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

### Signal risk/reward (`trading_signals.py`)

| Constant | Default | Description |
|---|---|---|
| `_RISK_PCT["strong"]` | `0.020` | Stop distance for strong signals |
| `_RISK_PCT["moderate"]` | `0.015` | Stop distance for moderate signals |
| `_RISK_PCT["weak"]` | `0.010` | Stop distance for weak signals |
| `_REWARD_RISK_RATIO` | `1.5` | Target = 1.5 × risk |

### Backtester defaults (`backtester.py`)

| Constant | Default | Description |
|---|---|---|
| `holding_days` | `5` | Trading days to hold the position |
| `lookback_days` | `252` | Calendar-day historical window (~1 year) |
| `_SIDEWAYS_BAND` | `0.01` | ±1 % MA distance → "SIDEWAYS" trend |

---

## Disclaimer

This tool is provided for **educational and research purposes only**.  The
trading signals and backtest results are derived from a news-sentiment model
and historical price data — they do not constitute financial advice.  Always
conduct your own due diligence before making any investment decisions.

---

## License

MIT — see [LICENSE](LICENSE).
