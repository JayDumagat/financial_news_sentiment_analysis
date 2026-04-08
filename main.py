"""
Financial News Sentiment Analysis — Main Pipeline

Orchestrates:
  1. Scraping financial news from ABS-CBN ANC
  2. Analyzing each article with FinBERT
  3. Printing / saving a structured report

Usage
-----
    python main.py [--max-articles N] [--enrich] [--output results.csv]
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from scraper import NewsArticle, scrape_anc_news
from sentiment_analyzer import FinBERTAnalyzer, SentimentResult

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
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict]:
    """Execute the full pipeline and return the report."""
    # 1. Scrape
    logger.info("Scraping up to %d articles from ABS-CBN ANC …", args.max_articles)
    articles = scrape_anc_news(max_articles=args.max_articles, enrich=args.enrich)

    if not articles:
        logger.warning("No articles were scraped. The site layout may have changed.")
        return []

    logger.info("Scraped %d article(s).", len(articles))

    # 2. Analyze
    analyzer = FinBERTAnalyzer(model_name=args.model)
    texts = [a.get_text_for_analysis() for a in articles]
    logger.info("Running FinBERT sentiment analysis …")
    results = analyzer.analyze_batch(texts)

    # 3. Report
    report = build_report(articles, results)
    print_report(report)

    # 4. (Optional) Save
    if args.output:
        ext = Path(args.output).suffix.lower()
        if ext == ".json":
            save_json(report, args.output)
        else:
            save_csv(report, args.output)

    return report


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args)


if __name__ == "__main__":
    main()
