"""
Financial News Sentiment Analysis — Main Pipeline

Orchestrates:
  1. Scraping financial news from ABS-CBN ANC
  2. Filtering out non-financial articles (with caching)
  3. Analyzing each article with FinBERT
  4. Aspect-based sentiment — per-ticker context scoring
  5. Reporting which Philippine (PSE-listed) stocks may be affected
  6. Generating BUY / SELL / HOLD trading signals with entry / target / stop prices
  7. Optionally backtesting each signal against historical price data
  8. Printing / saving a structured report
  9. Optionally persisting raw data to a data lake and results to SQLite
 10. Sending signals via Telegram or Discord webhook (--telegram-* / --discord-webhook)

Usage
-----
    # One-shot run
    python main.py [--max-articles N] [--enrich] [--output results.csv]

    # With trading signals (live price from Yahoo Finance)
    python main.py --signals

    # With signals + historical backtest (5-day holding period, 1-year window)
    python main.py --signals --backtest [--holding-days 5] [--lookback-days 252]

    # Persist to database and data lake
    python main.py --db news.db --data-lake data_lake/

    # Watch mode — run forever, poll every 5 minutes
    python main.py --watch [--interval 300] [--db news.db] [--data-lake data_lake/]

    # Send signals to Telegram
    python main.py --signals --telegram-token <BOT_TOKEN> --telegram-chat-id <CHAT_ID>

    # Send signals to Discord
    python main.py --signals --discord-webhook <WEBHOOK_URL>
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from scraper import NewsArticle, scrape_anc_news, run_forever
from sentiment_analyzer import FinBERTAnalyzer, SentimentResult
from relevance_filter import RelevanceFilter
from trading_signals import generate_signals, TradingSignal
from notifier import Notifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def build_report(
    articles: list[NewsArticle],
    results: list[SentimentResult],
    signals_map: Optional[dict] = None,
    backtest_map: Optional[dict] = None,
) -> list[dict]:
    """Merge scraped articles and sentiment results into a list of dicts.

    Args:
        articles: Scraped news articles.
        results: Corresponding sentiment results.
        signals_map: Optional mapping of article URL → list of TradingSignal.
        backtest_map: Optional mapping of ticker → BacktestResult.
    """
    report = []
    for article, result in zip(articles, results):
        affected = getattr(result, "affected_stocks", [])
        sigs = signals_map.get(article.url, []) if signals_map else []
        row = {
            "title": article.title,
            "url": article.url,
            "category": article.category,
            "published_at": article.published_at or "",
            "sentiment": result.label,
            "strength": result.strength,
            "confidence": round(result.score, 4),
            "score_positive": round(result.all_scores.get("positive", 0.0), 4),
            "score_negative": round(result.all_scores.get("negative", 0.0), 4),
            "score_neutral": round(result.all_scores.get("neutral", 0.0), 4),
            "analyzed_text": result.text[:200],
            "affected_stocks": affected,
            "trading_signals": [
                {
                    "ticker": s.ticker,
                    "name": s.name,
                    "signal": s.signal,
                    "strength": s.strength,
                    "entry_price": s.entry_price,
                    "target_price": s.target_price,
                    "stop_loss": s.stop_loss,
                    "entry_note": s.entry_note,
                    "atr": s.atr,
                    "valid_until": s.valid_until.isoformat() if s.valid_until else None,
                    "reasoning": s.reasoning,
                }
                for s in sigs
            ],
        }
        if backtest_map:
            row["backtest_results"] = [
                {
                    "ticker": bt.ticker,
                    "signal": bt.signal,
                    "holding_days": bt.holding_days,
                    "win_rate": bt.win_rate,
                    "avg_return": bt.avg_return,
                    "sample_size": bt.sample_size,
                    "current_trend": bt.current_trend,
                    "price_vs_ma20": bt.price_vs_ma20,
                    "recent_return_5d": bt.recent_return_5d,
                    "recent_return_20d": bt.recent_return_20d,
                }
                for bt in [
                    backtest_map[s.ticker]
                    for s in sigs
                    if s.ticker in backtest_map and s.signal != "HOLD"
                ]
            ]
        report.append(row)
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
        strength = row.get("strength", "")
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

        label_icon = {"positive": "✅", "negative": "❌", "neutral": "➖"}.get(label, "❓")
        strength_tag = f"  [{strength.upper()}]" if strength and strength != "neutral" else ""
        print(f"\n[{i:02d}] {label_icon} {label.upper():8s}  ({row['confidence']:.0%} confidence){strength_tag}")
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

        # Trading signals
        sigs = row.get("trading_signals", [])
        if sigs:
            print(f"      {'─' * 60}")
            print("      📊 TRADING SIGNALS")
            for sig in sigs:
                action = sig["signal"]
                sig_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(action, "")
                price_str = ""
                if sig.get("entry_price") is not None:
                    entry_label = (
                        "Next Open" if sig.get("entry_note") == "NEXT OPEN" else "Entry"
                    )
                    price_str = (
                        f"  {entry_label} ₱{sig['entry_price']:.2f}"
                        + (f"  Target ₱{sig['target_price']:.2f}" if sig.get("target_price") else "")
                        + (f"  Stop ₱{sig['stop_loss']:.2f}" if sig.get("stop_loss") else "")
                        + (f"  ATR ₱{sig['atr']:.2f}" if sig.get("atr") else "")
                    )
                expires = ""
                if sig.get("valid_until"):
                    expires = f"  [exp {sig['valid_until'][:16]}]"
                print(
                    f"      {sig_icon} {action:4s} {sig['ticker']:6s} ({sig['name'][:30]})"
                    f"  [{sig['strength']}]{price_str}{expires}"
                )

        # Backtest results
        bts = row.get("backtest_results", [])
        if bts:
            print(f"      {'─' * 60}")
            print("      🔬 BACKTEST (historical price accuracy)")
            for bt in bts:
                wr = f"{bt['win_rate']:.1%}" if bt.get("win_rate") is not None else "N/A"
                avg = f"{bt['avg_return']:+.2%}" if bt.get("avg_return") is not None else "N/A"
                trend = bt.get("current_trend") or "N/A"
                r5 = f"{bt['recent_return_5d']:+.2%}" if bt.get("recent_return_5d") is not None else "N/A"
                r20 = f"{bt['recent_return_20d']:+.2%}" if bt.get("recent_return_20d") is not None else "N/A"
                n = bt.get("sample_size") or "N/A"
                hold = bt.get("holding_days", 5)
                print(
                    f"      {bt['ticker']:6s} {bt['signal']:4s} | "
                    f"Win rate: {wr} (n={n}, {hold}d)  Avg return: {avg}  "
                    f"Trend: {trend}  5d: {r5}  20d: {r20}"
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
        fieldnames = list(report[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in report:
            # Flatten list/dict fields to JSON strings for CSV compatibility
            flat_row = dict(row)
            for list_field in ("affected_stocks", "trading_signals", "backtest_results"):
                if list_field in flat_row:
                    flat_row[list_field] = json.dumps(flat_row[list_field])
            writer.writerow(flat_row)
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
    parser.add_argument(
        "--signals",
        action="store_true",
        default=False,
        help=(
            "Generate BUY / SELL / HOLD trading signals for affected PSE stocks. "
            "Requires internet access to fetch live prices from Yahoo Finance."
        ),
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        default=False,
        help=(
            "Run a historical price-based backtest for each generated signal.  "
            "Implies --signals.  Requires internet access."
        ),
    )
    parser.add_argument(
        "--holding-days",
        type=int,
        default=5,
        metavar="N",
        help="Holding period (trading days) used in the backtest (default: 5).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=252,
        metavar="N",
        help="Historical window (calendar days) used in the backtest (default: 252).",
    )
    parser.add_argument(
        "--aspects",
        action="store_true",
        default=False,
        help=(
            "Enable aspect-based sentiment analysis.  Instead of applying one "
            "article-level sentiment to every matched stock, extract per-ticker "
            "context snippets and score each one independently."
        ),
    )
    # Telegram webhook
    parser.add_argument(
        "--telegram-token",
        default=None,
        metavar="TOKEN",
        help="Telegram bot token for signal notifications.",
    )
    parser.add_argument(
        "--telegram-chat-id",
        default=None,
        metavar="CHAT_ID",
        help="Telegram chat or channel ID to send signal notifications to.",
    )
    # Discord webhook
    parser.add_argument(
        "--discord-webhook",
        default=None,
        metavar="URL",
        help="Discord Incoming Webhook URL for signal notifications.",
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
    notifier = Notifier(
        telegram_token=getattr(args, "telegram_token", None),
        telegram_chat_id=getattr(args, "telegram_chat_id", None),
        discord_webhook=getattr(args, "discord_webhook", None),
    )

    want_signals = getattr(args, "signals", False) or getattr(args, "backtest", False)
    want_backtest = getattr(args, "backtest", False)
    want_aspects = getattr(args, "aspects", False)
    holding_days = getattr(args, "holding_days", 5)
    lookback_days = getattr(args, "lookback_days", 252)

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
                lake.save_preprocessed_article(article)
                lake.save_cleaned_article(article)
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

    # 5b. Aspect-based per-ticker sentiment enrichment
    if want_aspects:
        logger.info("Running aspect-based sentiment analysis …")
        for result, src in zip(results, source_texts):
            if result.affected_stocks:
                result.affected_stocks = analyzer.analyze_aspects(
                    src, stocks=result.affected_stocks
                )

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
                lake.save_analyzed_result(article, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Data lake processed write failed: %s", exc)

    # 7. (Optional) Generate trading signals
    signals_map: dict = {}
    backtest_map: dict = {}

    if want_signals:
        logger.info("Generating trading signals …")
        for article, result in zip(financial_articles, results):
            # Parse published_at for valid_until calculation
            pub_dt: Optional[datetime] = None
            if article.published_at:
                try:
                    pub_dt = datetime.fromisoformat(article.published_at)
                except ValueError:
                    pass
            sigs = generate_signals(result, published_at=pub_dt)
            signals_map[article.url] = sigs

            # Notify each actionable signal
            if notifier.is_configured:
                for sig in sigs:
                    if sig.signal != "HOLD":
                        notifier.send_signal(sig)

        if want_backtest:
            from backtester import backtest_signal

            # Collect unique (ticker, signal) pairs where signal != HOLD
            seen: set[tuple[str, str]] = set()
            pairs: list[tuple[str, str]] = []
            for sigs in signals_map.values():
                for sig in sigs:
                    if sig.signal != "HOLD" and (sig.ticker, sig.signal) not in seen:
                        seen.add((sig.ticker, sig.signal))
                        pairs.append((sig.ticker, sig.signal))

            if pairs:
                logger.info(
                    "Running historical backtest for %d ticker(s) …", len(pairs)
                )
                for ticker, signal in pairs:
                    bt = backtest_signal(
                        ticker,
                        signal,
                        holding_days=holding_days,
                        lookback_days=lookback_days,
                    )
                    backtest_map[ticker] = bt

    # 8. Report
    report = build_report(
        financial_articles,
        results,
        signals_map=signals_map if want_signals else None,
        backtest_map=backtest_map if want_backtest else None,
    )
    print_report(report)

    # 9. (Optional) Save report
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
    notifier = Notifier(
        telegram_token=getattr(args, "telegram_token", None),
        telegram_chat_id=getattr(args, "telegram_chat_id", None),
        discord_webhook=getattr(args, "discord_webhook", None),
    )
    want_signals = getattr(args, "signals", False)
    want_aspects = getattr(args, "aspects", False)

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

        if want_aspects and result.affected_stocks:
            result.affected_stocks = analyzer.analyze_aspects(
                source_texts[0], stocks=result.affected_stocks
            )

        if db is not None:
            try:
                db.save_article(article, is_financial=True)
                db.save_sentiment_result(article.url, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("DB write failed: %s", exc)

        if lake is not None:
            try:
                lake.save_processed_result(article, result)
                lake.save_analyzed_result(article, result)
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

        # Send webhook notifications for actionable signals
        if want_signals and notifier.is_configured:
            pub_dt: Optional[datetime] = None
            if article.published_at:
                try:
                    pub_dt = datetime.fromisoformat(article.published_at)
                except ValueError:
                    pass
            sigs = generate_signals(result, published_at=pub_dt)
            for sig in sigs:
                if sig.signal != "HOLD":
                    notifier.send_signal(sig)

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
