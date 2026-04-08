# Financial News Sentiment Analysis

A Python pipeline that scrapes financial news from **ABS-CBN ANC**
([abs-cbn.com/anc](https://www.abs-cbn.com/anc/anc)) and classifies each
article's sentiment using **FinBERT** — a BERT model fine-tuned on financial
text by [ProsusAI](https://huggingface.co/ProsusAI/finbert).

---

## Features

| Feature | Detail |
|---|---|
| **Source** | ABS-CBN ANC & ANC Business section |
| **Model** | `ProsusAI/finbert` (HuggingFace) |
| **Labels** | `positive` · `negative` · `neutral` |
| **Output** | Console table, CSV, or JSON |
| **Offline tests** | Full test suite with mocked HTTP & model |

---

## Project Structure

```
financial_news_sentiment_analysis/
├── scraper.py            # ABS-CBN ANC web scraper
├── sentiment_analyzer.py # FinBERT wrapper
├── main.py               # CLI orchestration script
├── requirements.txt      # Python dependencies
└── tests/
    └── test_pipeline.py  # Unit tests (no internet required)
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
# Print results to stdout (scrapes up to 20 articles)
python main.py

# Fetch full article body for richer analysis (slower)
python main.py --enrich

# Limit articles and save to CSV
python main.py --max-articles 10 --output results.csv

# Save to JSON
python main.py --output results.json

# Use a different HuggingFace model
python main.py --model yiyanghkust/finbert-tone
```

### 3. Run tests

```bash
pytest tests/ -v
```

---

## Command-Line Options

```
usage: main.py [-h] [--max-articles N] [--enrich] [--output FILE]
               [--model MODEL] [--log-level {DEBUG,INFO,WARNING,ERROR}]

options:
  --max-articles N   Maximum number of articles to scrape (default: 20)
  --enrich           Fetch full article body for more accurate analysis
  --output FILE      Save results to FILE (.csv or .json)
  --model MODEL      HuggingFace model identifier (default: ProsusAI/finbert)
  --log-level        Logging verbosity (default: INFO)
```

---

## Example Output

```
========================================================================
 ABS-CBN ANC Financial News — Sentiment Analysis Report
 Generated: 2024-01-15 09:30:00
========================================================================

[01] ✅ POSITIVE   (93% confidence)
      PSEi rises as investors cheer strong Q4 earnings
      URL : https://www.abs-cbn.com/anc/business/article/...
      Cat : Business
      Pos=0.93  Neg=0.04  Neu=0.03

[02] ❌ NEGATIVE   (87% confidence)
      Peso weakens against dollar amid inflation fears
      URL : https://www.abs-cbn.com/anc/business/article/...
      Cat : Business
      Pos=0.07  Neg=0.87  Neu=0.06

------------------------------------------------------------------------
 Summary: 2 articles  |  Positive: 1  |  Negative: 1  |  Neutral:  0
========================================================================
```

---

## Modules

### `scraper.py`

| Class / Function | Description |
|---|---|
| `ANCNewsScraper` | Session-based scraper with configurable delay |
| `ANCNewsScraper.get_articles(max_articles)` | Scrape listing pages |
| `ANCNewsScraper.enrich_article(article)` | Fetch full article body |
| `scrape_anc_news(max_articles, enrich)` | Convenience function |
| `NewsArticle` | Dataclass holding title, URL, summary, content, etc. |

### `sentiment_analyzer.py`

| Class / Function | Description |
|---|---|
| `FinBERTAnalyzer` | Lazy-loading FinBERT pipeline wrapper |
| `FinBERTAnalyzer.analyze(text)` | Single-text classification |
| `FinBERTAnalyzer.analyze_batch(texts)` | Batched classification |
| `SentimentResult` | Dataclass with label, score, and per-class scores |
| `analyze_text(text)` | Convenience function |

### `main.py`

| Function | Description |
|---|---|
| `run(args)` | Full pipeline (scrape → analyse → report → save) |
| `build_report(articles, results)` | Merge into list of dicts |
| `print_report(report)` | Pretty-print to stdout |
| `save_csv(report, path)` | Write CSV file |
| `save_json(report, path)` | Write JSON file |

---

## License

MIT — see [LICENSE](LICENSE).
