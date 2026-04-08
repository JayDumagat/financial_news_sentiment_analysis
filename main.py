"""
Financial News Sentiment Analysis — Main Pipeline

Orchestrates:
  1. Scraping financial news from ABS-CBN ANC
  2. Filtering out non-financial articles (with caching)
  3. Analyzing each article with FinBERT
  4. Reporting which Philippine (PSE-listed) stocks may be affected
  5. Printing / saving a structured report
  6. Optionally persisting raw data to a data lake and results to SQLite

Usage
-----
    # One-shot run
    python main.py [--max-articles N] [--enrich] [--output results.csv]

    # Persist to database and data lake
    python main.py --db news.db --data-lake data_lake/

    # Watch mode — run forever, poll every 5 minutes
    python main.py --watch [--interval 300] [--db news.db] [--data-lake data_lake/]
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from scraper import NewsArticle, scrape_anc_news, run_forever
from sentiment_analyzer import FinBERTAnalyzer, SentimentResult
from relevance_filter import RelevanceFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def build_report(
    articles: list[NewsArticle],
    results: list[SentimentResult],
) -> list[dict]:
    """Merge scraped articles and sentiment results into a list of dicts."""
    report = []
    for article, result in zip(articles, results):
        affected = getattr(result, "affected_stocks", [])
        report.append(
            {
                "title": article.title,
                "url": article.url,
                "category": article.category,
                "published_at": article.published_at or "",
                "sentiment": result.label,
                "confidence": round(result.score, 4),
                "score_positive": round(result.all_scores.get("positive", 0.0), 4),
                "score_negative": round(result.all_scores.get("negative", 0.0), 4),
                "score_neutral": round(result.all_scores.get("neutral", 0.0), 4),
                "analyzed_text": result.text[:200],
                "affected_stocks": affected,
            }
        )
    return report


def print_report(report: list[dict]) -> None:
    """Pretty-print the report to stdout."""
    print("\n" + "=" * 72)
    print(" ABS-CBN ANC Financial News — Sentiment Analysis Report")
    print(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for i, row in enumerate(report, 1):
        label = row["sentiment"]
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

        label_icon = {"positive": "✅", "negative": "❌", "neutral": "➖"}.get(label, "❓")
        print(f"\n[{i:02d}] {label_icon} {label.upper():8s}  ({row['confidence']:.0%} confidence)")
        print(f"      {row['title']}")
        print(f"      URL : {row['url']}")
        if row["category"]:
            print(f"      Cat : {row['category']}")
        print(
            f"      Pos={row['score_positive']:.2f}  "
            f"Neg={row['score_negative']:.2f}  "
            f"Neu={row['score_neutral']:.2f}"
        )
        # Affected PSE stocks
        affected = row.get("affected_stocks", [])
        if affected:
            direct = [s for s in affected if s.get("match_type") == "direct"]
            sector = [s for s in affected if s.get("match_type") == "sector"]
            if direct:
                tickers = ", ".join(
                    f"{s['ticker']} ({s['name']})" for s in direct[:5]
                )
                print(f"      📈 PSE stocks (direct): {tickers}")
            if sector:
                sectors = ", ".join(
                    sorted({s["sector"] for s in sector})
                )
                print(f"      🏢 PSE sectors affected: {sectors}")

    print("\n" + "-" * 72)
    total = len(report)
    print(
        f" Summary: {total} articles  |  "
        f"Positive: {sentiment_counts.get('positive', 0)}  |  "
        f"Negative: {sentiment_counts.get('negative', 0)}  |  "
        f"Neutral:  {sentiment_counts.get('neutral', 0)}"
    )
    print("=" * 72 + "\n")


def save_csv(report: list[dict], path: str) -> None:
    """Save the report as a CSV file."""
    if not report:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Flatten affected_stocks to a JSON string for CSV compatibility
    flat = [
        {**row, "affected_stocks": json.dumps(row.get("affected_stocks", []))}
        for row in report
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    logger.info("Results saved to %s", output_path)


def save_json(report: list[dict], path: str) -> None:
    """Save the report as a JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape ABS-CBN ANC financial news and analyze sentiment with FinBERT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of articles to scrape.",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help="Fetch the full article body for more accurate sentiment analysis.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help=(
            "Save results to FILE.  "
            "Format is inferred from the file extension (.csv or .json). "
            "If omitted, results are only printed to stdout."
        ),
    )
    parser.add_argument(
        "--model",
        default="ProsusAI/finbert",
        metavar="MODEL",
        help="HuggingFace model identifier for sentiment analysis.",
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help=(
            "Path to the SQLite database file for persisting articles and results. "
            "If omitted, no database is written."
        ),
    )
    parser.add_argument(
        "--data-lake",
        default=None,
        metavar="DIR",
        help=(
            "Root directory of the raw data lake.  "
            "If provided, raw articles and processed results are saved as "
            "JSON files organised by date under this directory."
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        default=False,
        help=(
            "Run forever, polling for new articles on a fixed interval.  "
            "Press Ctrl-C to stop."
        ),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Polling interval in seconds for --watch mode.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def _build_stores(args: argparse.Namespace):
    """Instantiate optional database and data lake from parsed arguments."""
    db = None
    lake = None
    if args.db:
        from database import NewsDatabase
        db = NewsDatabase(db_path=args.db)
        logger.info("Database: %s", args.db)
    if args.data_lake:
        from data_lake import DataLake
        lake = DataLake(base_path=args.data_lake)
        logger.info("Data lake: %s", args.data_lake)
    return db, lake


def run(args: argparse.Namespace) -> list[dict]:
    """Execute the full pipeline (one shot) and return the report."""
    db, lake = _build_stores(args)
    relevance = RelevanceFilter(db=db)
    analyzer = FinBERTAnalyzer(model_name=args.model)

    # 1. Scrape
    logger.info("Scraping up to %d articles from ABS-CBN ANC …", args.max_articles)
    articles = scrape_anc_news(max_articles=args.max_articles, enrich=args.enrich)

    if not articles:
        logger.warning("No articles were scraped. The site layout may have changed.")
        return []

    logger.info("Scraped %d article(s).", len(articles))

    # 2. Relevance filter — skip non-financial articles
    financial_articles = [a for a in articles if relevance.is_financial(a)]
    skipped = len(articles) - len(financial_articles)
    if skipped:
        logger.info("Skipped %d non-financial article(s) (cached).", skipped)

    if not financial_articles:
        logger.warning("No financially relevant articles found.")
        return []

    # 3. Save raw articles to data lake
    if lake is not None:
        for article in financial_articles:
            try:
                lake.save_raw_article(article)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Data lake write failed: %s", exc)

    # 4. Save articles to database
    if db is not None:
        for article in financial_articles:
            try:
                db.save_article(article, is_financial=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("DB article write failed: %s", exc)

    # 5. Analyze
    texts = [a.get_text_for_analysis() for a in financial_articles]
    # Pass full text (title+summary+content) as source for stock matching
    source_texts = [
        " ".join(filter(None, [a.title, a.summary, a.content]))
        for a in financial_articles
    ]
    logger.info("Running FinBERT sentiment analysis on %d article(s) …", len(texts))
    results = analyzer.analyze_batch(texts, source_texts=source_texts)

    # 6. Save sentiment results
    if db is not None:
        for article, result in zip(financial_articles, results):
            try:
                db.save_sentiment_result(article.url, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("DB result write failed: %s", exc)

    if lake is not None:
        for article, result in zip(financial_articles, results):
            try:
                lake.save_processed_result(article, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Data lake processed write failed: %s", exc)

    # 7. Report
    report = build_report(financial_articles, results)
    print_report(report)

    # 8. (Optional) Save report
    if args.output:
        ext = Path(args.output).suffix.lower()
        if ext == ".json":
            save_json(report, args.output)
        else:
            save_csv(report, args.output)

    return report


def _watch(args: argparse.Namespace) -> None:
    """Enter watch / daemon mode."""
    db, lake = _build_stores(args)
    relevance = RelevanceFilter(db=db)
    analyzer = FinBERTAnalyzer(model_name=args.model)

    def _process(article: NewsArticle) -> None:
        if not relevance.is_financial(article):
            logger.debug("Watch: skipping non-financial article %s", article.url)
            return

        texts = [article.get_text_for_analysis()]
        source_texts = [
            " ".join(filter(None, [article.title, article.summary, article.content]))
        ]
        results = analyzer.analyze_batch(texts, source_texts=source_texts)
        result = results[0]

        if db is not None:
            try:
                db.save_article(article, is_financial=True)
                db.save_sentiment_result(article.url, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("DB write failed: %s", exc)

        if lake is not None:
            try:
                lake.save_processed_result(article, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Data lake write failed: %s", exc)

        # Print a compact one-liner to stdout
        affected = result.affected_stocks
        tickers = " ".join(s["ticker"] for s in affected if s.get("match_type") == "direct")
        icon = {"positive": "✅", "negative": "❌", "neutral": "➖"}.get(result.label, "❓")
        print(
            f"{icon} [{result.label.upper():8s} {result.score:.0%}]  "
            f"{article.title[:60]}{'…' if len(article.title) > 60 else ''}"
            + (f"  |  PSE: {tickers}" if tickers else "")
        )

    run_forever(
        poll_interval=args.interval,
        max_articles=args.max_articles,
        enrich=args.enrich,
        db=db,
        data_lake=lake,
        on_new_article=_process,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.watch:
        _watch(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
