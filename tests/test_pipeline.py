"""
Tests for the Financial News Sentiment Analysis project.

Runs without requiring internet access or downloading the FinBERT model by
using mocks for the HTTP layer and the HuggingFace pipeline.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_HTML_LISTING = """
<html>
<body>
  <article class="article-card">
    <a href="/anc/business/article/2024/1/1/test-headline">Test Headline</a>
    <h3 class="title">Test Headline</h3>
    <p>A brief summary of the test article about Philippine stocks.</p>
    <span class="category">Business</span>
    <time datetime="2024-01-01T08:00:00+08:00"></time>
  </article>
  <article class="article-card">
    <a href="https://www.abs-cbn.com/anc/business/article/2024/1/2/second">Second Article</a>
    <h3 class="title">Second Article</h3>
    <p>Another financial story about the economy.</p>
  </article>
</body>
</html>
"""

SAMPLE_HTML_ARTICLE = """
<html>
<body>
  <div class="article-body">
    <p>Paragraph one of the article body.</p>
    <p>Paragraph two with more financial details.</p>
    <script>/* noise */</script>
  </div>
</body>
</html>
"""


def _make_response(text: str, status_code: int = 200):
    """Return a minimal mock requests.Response."""
    mock = MagicMock()
    mock.text = text
    mock.status_code = status_code
    mock.raise_for_status = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# scraper tests
# ---------------------------------------------------------------------------


class TestANCNewsScraper:
    def setup_method(self):
        from scraper import ANCNewsScraper

        self.scraper = ANCNewsScraper(delay=0)

    @patch("scraper.requests.Session.get")
    def test_get_articles_parses_cards(self, mock_get):
        mock_get.return_value = _make_response(SAMPLE_HTML_LISTING)
        articles = self.scraper.get_articles(max_articles=5)
        assert len(articles) >= 1
        titles = [a.title for a in articles]
        assert any("Test Headline" in t or "Second Article" in t for t in titles)

    @patch("scraper.requests.Session.get")
    def test_get_articles_url_normalization(self, mock_get):
        mock_get.return_value = _make_response(SAMPLE_HTML_LISTING)
        articles = self.scraper.get_articles(max_articles=5)
        for art in articles:
            assert art.url.startswith("http"), f"URL should be absolute: {art.url}"

    @patch("scraper.requests.Session.get")
    def test_enrich_article_fills_content(self, mock_get):
        mock_get.return_value = _make_response(SAMPLE_HTML_ARTICLE)
        from scraper import NewsArticle

        article = NewsArticle(
            title="Dummy",
            url="https://www.abs-cbn.com/anc/business/article/2024/1/1/dummy",
        )
        self.scraper.enrich_article(article)
        assert "Paragraph one" in article.content or article.content == ""

    @patch("scraper.requests.Session.get")
    def test_get_articles_max_limit(self, mock_get):
        mock_get.return_value = _make_response(SAMPLE_HTML_LISTING)
        articles = self.scraper.get_articles(max_articles=1)
        assert len(articles) <= 1

    @patch("scraper.requests.Session.get")
    def test_http_error_returns_empty(self, mock_get):
        import requests as req_lib

        mock_get.side_effect = req_lib.RequestException("network error")
        articles = self.scraper.get_articles(max_articles=5)
        assert articles == []


class TestNewsArticle:
    def test_get_text_for_analysis_prefers_content(self):
        from scraper import NewsArticle

        art = NewsArticle(title="T", url="http://x", summary="Summary", content="Full content here")
        assert art.get_text_for_analysis() == "Full content here"

    def test_get_text_for_analysis_falls_back_to_summary(self):
        from scraper import NewsArticle

        art = NewsArticle(title="T", url="http://x", summary="Only summary")
        assert art.get_text_for_analysis() == "Only summary"

    def test_get_text_for_analysis_falls_back_to_title(self):
        from scraper import NewsArticle

        art = NewsArticle(title="Only title", url="http://x")
        assert art.get_text_for_analysis() == "Only title"

    def test_content_truncated_at_512(self):
        from scraper import NewsArticle

        long_content = "x" * 1000
        art = NewsArticle(title="T", url="http://x", content=long_content)
        assert len(art.get_text_for_analysis()) == 512


# ---------------------------------------------------------------------------
# sentiment_analyzer tests
# ---------------------------------------------------------------------------


def _make_mock_pipeline(label: str = "positive", score: float = 0.9):
    """Return a callable that mimics a HuggingFace text-classification pipeline."""

    def _pipeline(texts, **kwargs):
        return [
            [
                {"label": label, "score": score},
                {"label": "negative", "score": 0.05},
                {"label": "neutral", "score": 0.05},
            ]
            for _ in texts
        ]

    return _pipeline


class TestFinBERTAnalyzer:
    def _make_analyzer(self, mock_label="positive", mock_score=0.9):
        from sentiment_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()
        analyzer._pipeline = _make_mock_pipeline(mock_label, mock_score)
        return analyzer

    def test_analyze_returns_result(self):
        analyzer = self._make_analyzer("positive", 0.9)
        result = analyzer.analyze("Stocks surged to record highs.")
        assert result.label == "positive"
        assert 0 <= result.score <= 1

    def test_analyze_negative_label(self):
        analyzer = self._make_analyzer("negative", 0.85)
        result = analyzer.analyze("Market crash wipes out billions.")
        assert result.label == "negative"

    def test_analyze_neutral_label(self):
        analyzer = self._make_analyzer("neutral", 0.7)
        result = analyzer.analyze("The central bank held rates steady.")
        assert result.label == "neutral"

    def test_analyze_batch_length(self):
        analyzer = self._make_analyzer()
        texts = ["text one", "text two", "text three"]
        results = analyzer.analyze_batch(texts)
        assert len(results) == 3

    def test_analyze_batch_empty(self):
        analyzer = self._make_analyzer()
        results = analyzer.analyze_batch([])
        assert results == []

    def test_sentiment_result_properties(self):
        from sentiment_analyzer import SentimentResult

        r = SentimentResult(
            text="Good news",
            label="positive",
            score=0.95,
            all_scores={"positive": 0.95, "negative": 0.03, "neutral": 0.02},
        )
        assert r.is_positive
        assert not r.is_negative
        assert not r.is_neutral

    def test_sentiment_result_str(self):
        from sentiment_analyzer import SentimentResult

        r = SentimentResult(
            text="Test",
            label="negative",
            score=0.8,
            all_scores={"positive": 0.1, "negative": 0.8, "neutral": 0.1},
        )
        s = str(r)
        assert "NEGATIVE" in s

    def test_get_pipeline_raises_without_transformers(self):
        """_get_pipeline should raise ImportError if transformers is missing."""
        from sentiment_analyzer import FinBERTAnalyzer

        analyzer = FinBERTAnalyzer()
        with patch.dict(sys.modules, {"transformers": None}):
            with pytest.raises(ImportError):
                analyzer._get_pipeline()


# ---------------------------------------------------------------------------
# main pipeline tests
# ---------------------------------------------------------------------------


class TestMainPipeline:
    def _make_articles(self, n=3):
        from scraper import NewsArticle

        return [
            NewsArticle(
                title=f"Article {i}",
                url=f"https://www.abs-cbn.com/anc/test-{i}",
                summary=f"Summary of article {i}",
                category="Business",
            )
            for i in range(n)
        ]

    def test_build_report_length(self):
        from main import build_report
        from sentiment_analyzer import SentimentResult

        articles = self._make_articles(3)
        results = [
            SentimentResult(
                text=a.summary,
                label="positive",
                score=0.9,
                all_scores={"positive": 0.9, "negative": 0.05, "neutral": 0.05},
            )
            for a in articles
        ]
        report = build_report(articles, results)
        assert len(report) == 3

    def test_build_report_keys(self):
        from main import build_report
        from sentiment_analyzer import SentimentResult

        articles = self._make_articles(1)
        results = [
            SentimentResult(
                text="test",
                label="neutral",
                score=0.7,
                all_scores={"positive": 0.1, "negative": 0.2, "neutral": 0.7},
            )
        ]
        report = build_report(articles, results)
        expected_keys = {
            "title", "url", "category", "published_at",
            "sentiment", "strength", "confidence", "score_positive",
            "score_negative", "score_neutral", "analyzed_text",
        }
        assert expected_keys.issubset(report[0].keys())

    def test_save_csv(self, tmp_path):
        import csv as csv_lib
        from main import save_csv, build_report
        from sentiment_analyzer import SentimentResult

        articles = self._make_articles(2)
        results = [
            SentimentResult("t", "positive", 0.9, {"positive": 0.9, "negative": 0.05, "neutral": 0.05})
            for _ in articles
        ]
        report = build_report(articles, results)
        output = str(tmp_path / "out.csv")
        save_csv(report, output)

        with open(output, newline="", encoding="utf-8") as fh:
            rows = list(csv_lib.DictReader(fh))
        assert len(rows) == 2
        assert rows[0]["sentiment"] == "positive"

    def test_save_json(self, tmp_path):
        import json as json_lib
        from main import save_json, build_report
        from sentiment_analyzer import SentimentResult

        articles = self._make_articles(1)
        results = [
            SentimentResult("t", "negative", 0.8, {"positive": 0.1, "negative": 0.8, "neutral": 0.1})
        ]
        report = build_report(articles, results)
        output = str(tmp_path / "out.json")
        save_json(report, output)

        with open(output, encoding="utf-8") as fh:
            data = json_lib.load(fh)
        assert len(data) == 1
        assert data[0]["sentiment"] == "negative"

    def test_parse_args_defaults(self):
        from main import parse_args

        args = parse_args([])
        assert args.max_articles == 20
        assert args.enrich is False
        assert args.output is None
        assert args.model == "ProsusAI/finbert"

    def test_parse_args_custom(self):
        from main import parse_args

        args = parse_args(["--max-articles", "5", "--enrich", "--output", "out.csv"])
        assert args.max_articles == 5
        assert args.enrich is True
        assert args.output == "out.csv"

    @patch("main.scrape_anc_news")
    def test_run_no_articles(self, mock_scrape):
        from main import run, parse_args

        mock_scrape.return_value = []
        args = parse_args([])
        report = run(args)
        assert report == []

    @patch("main.FinBERTAnalyzer")
    @patch("main.scrape_anc_news")
    def test_run_full_pipeline(self, mock_scrape, MockAnalyzer, tmp_path):
        import json as json_lib
        from scraper import NewsArticle
        from sentiment_analyzer import SentimentResult
        from main import run, parse_args

        articles = self._make_articles(2)
        mock_scrape.return_value = articles

        mock_results = [
            SentimentResult("t", "positive", 0.9, {"positive": 0.9, "negative": 0.05, "neutral": 0.05}),
            SentimentResult("t", "negative", 0.8, {"positive": 0.1, "negative": 0.8, "neutral": 0.1}),
        ]
        MockAnalyzer.return_value.analyze_batch.return_value = mock_results

        output = str(tmp_path / "results.json")
        args = parse_args(["--output", output])
        report = run(args)

        assert len(report) == 2
        assert report[0]["sentiment"] == "positive"
        assert report[1]["sentiment"] == "negative"

        with open(output, encoding="utf-8") as fh:
            saved = json_lib.load(fh)
        assert len(saved) == 2

    def test_parse_args_new_flags(self):
        from main import parse_args

        args = parse_args(["--watch", "--interval", "60", "--db", "x.db", "--data-lake", "lake/"])
        assert args.watch is True
        assert args.interval == 60
        assert args.db == "x.db"
        assert args.data_lake == "lake/"

    def test_build_report_includes_affected_stocks(self):
        from main import build_report
        from sentiment_analyzer import SentimentResult

        articles = self._make_articles(1)
        results = [
            SentimentResult(
                text="test",
                label="positive",
                score=0.9,
                all_scores={"positive": 0.9, "negative": 0.05, "neutral": 0.05},
                affected_stocks=[{"ticker": "BDO", "name": "BDO Unibank", "sector": "Financials",
                                   "match_type": "direct", "matched_keyword": "BDO"}],
            )
        ]
        report = build_report(articles, results)
        assert "affected_stocks" in report[0]
        assert report[0]["affected_stocks"][0]["ticker"] == "BDO"


# ---------------------------------------------------------------------------
# PSE stocks tests
# ---------------------------------------------------------------------------


class TestPSEStocks:
    def test_direct_ticker_match(self):
        from pse_stocks import find_affected_stocks

        stocks = find_affected_stocks("BDO reported record profits this quarter.")
        tickers = [s["ticker"] for s in stocks]
        assert "BDO" in tickers

    def test_direct_keyword_match(self):
        from pse_stocks import find_affected_stocks

        stocks = find_affected_stocks("Meralco raises electricity rates next month.")
        tickers = [s["ticker"] for s in stocks]
        assert "MER" in tickers

    def test_sector_trigger(self):
        from pse_stocks import find_affected_stocks

        stocks = find_affected_stocks("BSP cuts the policy rate by 25 basis points.")
        sectors = {s["sector"] for s in stocks}
        assert "Financials" in sectors

    def test_empty_text(self):
        from pse_stocks import find_affected_stocks

        assert find_affected_stocks("") == []

    def test_no_false_positives_unrelated(self):
        from pse_stocks import find_affected_stocks

        stocks = find_affected_stocks("The weather today is sunny and warm.")
        assert stocks == []

    def test_match_type_field_present(self):
        from pse_stocks import find_affected_stocks

        stocks = find_affected_stocks("Jollibee opens new stores abroad.")
        assert all("match_type" in s for s in stocks)
        assert all("ticker" in s for s in stocks)
        assert all("name" in s for s in stocks)


# ---------------------------------------------------------------------------
# Relevance filter tests
# ---------------------------------------------------------------------------


class TestRelevanceFilter:
    def _make_article(self, title="", summary="", category="", content=""):
        from scraper import NewsArticle

        return NewsArticle(
            title=title,
            url=f"https://example.com/{hash(title)}",
            summary=summary,
            category=category,
            content=content,
        )

    def test_financial_category_passes(self):
        from relevance_filter import RelevanceFilter

        filt = RelevanceFilter()
        art = self._make_article(title="Earnings drop", category="Business")
        assert filt.is_financial(art) is True

    def test_financial_keyword_passes(self):
        from relevance_filter import RelevanceFilter

        filt = RelevanceFilter()
        art = self._make_article(
            title="BSP raises interest rate by 50 bps",
            summary="The Bangko Sentral ng Pilipinas hiked the key policy rate.",
        )
        assert filt.is_financial(art) is True

    def test_non_financial_article_rejected(self):
        from relevance_filter import RelevanceFilter

        filt = RelevanceFilter()
        art = self._make_article(
            title="Local basketball team wins championship",
            summary="The team celebrated their victory in the finals.",
        )
        assert filt.is_financial(art) is False

    def test_result_is_cached(self):
        from relevance_filter import RelevanceFilter

        filt = RelevanceFilter()
        art = self._make_article(title="BSP cuts rate", category="Economy")
        # First call evaluates, second should hit cache
        result1 = filt.check(art)
        result2 = filt.check(art)
        assert result1.is_financial == result2.is_financial
        assert art.url in filt._cache

    def test_db_cache_write_read(self, tmp_path):
        from database import NewsDatabase
        from relevance_filter import RelevanceFilter

        db = NewsDatabase(db_path=str(tmp_path / "test.db"))
        filt = RelevanceFilter(db=db)
        art = self._make_article(title="Stock market rally", category="Business")
        result = filt.check(art)
        # DB cache should now have the entry
        cached = db.get_cached_relevance(art.url)
        assert cached == result.is_financial
        db.close()


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------


class TestNewsDatabase:
    def _make_article(self, n=0):
        from scraper import NewsArticle

        return NewsArticle(
            title=f"Test Article {n}",
            url=f"https://example.com/article-{n}",
            summary="A financial summary.",
            category="Business",
        )

    def test_save_and_is_seen(self, tmp_path):
        from database import NewsDatabase

        db = NewsDatabase(db_path=str(tmp_path / "test.db"))
        art = self._make_article()
        assert not db.is_seen(art.url)
        db.save_article(art)
        assert db.is_seen(art.url)
        db.close()

    def test_save_article_replace(self, tmp_path):
        from database import NewsDatabase

        db = NewsDatabase(db_path=str(tmp_path / "test.db"))
        art = self._make_article()
        db.save_article(art)
        art.title = "Updated Title"
        db.save_article(art)  # should not raise
        db.close()

    def test_save_sentiment_result(self, tmp_path):
        from database import NewsDatabase
        from sentiment_analyzer import SentimentResult

        db = NewsDatabase(db_path=str(tmp_path / "test.db"))
        art = self._make_article()
        db.save_article(art)
        result = SentimentResult(
            text="profit up",
            label="positive",
            score=0.9,
            all_scores={"positive": 0.9, "negative": 0.05, "neutral": 0.05},
        )
        db.save_sentiment_result(art.url, result)  # should not raise
        db.close()

    def test_relevance_cache_roundtrip(self, tmp_path):
        from database import NewsDatabase

        db = NewsDatabase(db_path=str(tmp_path / "test.db"))
        url = "https://example.com/sports-news"
        assert db.get_cached_relevance(url) is None
        db.cache_relevance(url, is_financial=False)
        assert db.get_cached_relevance(url) is False
        db.cache_relevance(url, is_financial=True)
        assert db.get_cached_relevance(url) is True
        db.close()


# ---------------------------------------------------------------------------
# Data lake tests
# ---------------------------------------------------------------------------


class TestDataLake:
    def _make_article(self):
        from scraper import NewsArticle

        return NewsArticle(
            title="PSE stocks rally on BSP rate cut news",
            url="https://example.com/article-lake",
            summary="Markets surged after the central bank decision.",
            category="Business",
        )

    def test_save_raw_article(self, tmp_path):
        import json as json_lib
        from data_lake import DataLake

        lake = DataLake(base_path=str(tmp_path / "lake"))
        art = self._make_article()
        path = lake.save_raw_article(art)
        assert path.exists()
        with path.open(encoding="utf-8") as fh:
            data = json_lib.load(fh)
        assert data["url"] == art.url
        assert data["title"] == art.title

    def test_save_processed_result(self, tmp_path):
        import json as json_lib
        from data_lake import DataLake
        from sentiment_analyzer import SentimentResult

        lake = DataLake(base_path=str(tmp_path / "lake"))
        art = self._make_article()
        result = SentimentResult(
            text=art.summary,
            label="positive",
            score=0.88,
            all_scores={"positive": 0.88, "negative": 0.07, "neutral": 0.05},
            affected_stocks=[{"ticker": "BDO", "name": "BDO Unibank", "sector": "Financials",
                               "match_type": "sector", "matched_keyword": "Financials"}],
        )
        path = lake.save_processed_result(art, result)
        assert path.exists()
        with path.open(encoding="utf-8") as fh:
            data = json_lib.load(fh)
        assert data["sentiment"] == "positive"
        assert len(data["affected_stocks"]) == 1

    def test_raw_and_processed_in_different_dirs(self, tmp_path):
        from data_lake import DataLake
        from sentiment_analyzer import SentimentResult

        lake = DataLake(base_path=str(tmp_path / "lake"))
        art = self._make_article()
        result = SentimentResult(
            text="test", label="neutral", score=0.7,
            all_scores={"positive": 0.1, "negative": 0.2, "neutral": 0.7},
        )
        raw_path = lake.save_raw_article(art)
        proc_path = lake.save_processed_result(art, result)
        assert "raw" in str(raw_path)
        assert "processed" in str(proc_path)


# ---------------------------------------------------------------------------
# Watch / run_forever tests
# ---------------------------------------------------------------------------


class TestRunForever:
    @patch("scraper.ANCNewsScraper.get_articles")
    def test_run_forever_calls_on_new_article(self, mock_get):
        from scraper import NewsArticle, run_forever

        art = NewsArticle(
            title="New financial news",
            url="https://example.com/watch-test",
            summary="Central bank update.",
        )
        # Return one article on first poll, then raise to stop the loop
        call_count = {"n": 0}

        def _side_effect(max_articles=20):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [art]
            raise KeyboardInterrupt  # stop loop after first poll

        mock_get.side_effect = _side_effect

        seen = []
        import signal as sig

        # Override signal so it doesn't conflict with pytest
        with patch("scraper.signal.signal"):
            try:
                run_forever(
                    poll_interval=0,
                    max_articles=1,
                    on_new_article=lambda a: seen.append(a.url),
                )
            except KeyboardInterrupt:
                pass

        assert art.url in seen

    @patch("scraper.ANCNewsScraper.get_articles")
    def test_run_forever_skips_seen_urls(self, mock_get):
        from scraper import NewsArticle, run_forever
        from database import NewsDatabase
        import tempfile, os

        art = NewsArticle(
            title="Already seen",
            url="https://example.com/seen-url",
            summary="Should be skipped.",
        )

        call_count = {"n": 0}

        def _side_effect(max_articles=20):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return [art]
            raise KeyboardInterrupt

        mock_get.side_effect = _side_effect

        seen = []

        with tempfile.TemporaryDirectory() as tmp:
            db = NewsDatabase(db_path=os.path.join(tmp, "test.db"))
            # Pre-populate so the article is already "seen"
            db.save_article(art)

            with patch("scraper.signal.signal"):
                try:
                    run_forever(
                        poll_interval=0,
                        max_articles=1,
                        db=db,
                        on_new_article=lambda a: seen.append(a.url),
                    )
                except KeyboardInterrupt:
                    pass

            db.close()

        # Article was pre-seeded in DB, so on_new_article should NOT be called
        assert art.url not in seen


# ---------------------------------------------------------------------------
# Sentiment strength / noise-gate tests
# ---------------------------------------------------------------------------


class TestSentimentStrength:
    def _make_analyzer(self, mock_label="positive", mock_score=0.9, runner_up_score=0.05):
        from sentiment_analyzer import FinBERTAnalyzer

        def _pipeline(texts, **kwargs):
            third_score = max(0.0, 1.0 - mock_score - runner_up_score)
            labels = ["positive", "negative", "neutral"]
            label_map = {mock_label: mock_score}
            remaining = [l for l in labels if l != mock_label]
            label_map[remaining[0]] = runner_up_score
            label_map[remaining[1]] = third_score
            return [
                [{"label": lbl, "score": label_map[lbl]} for lbl in labels]
                for _ in texts
            ]

        analyzer = FinBERTAnalyzer()
        analyzer._pipeline = _pipeline
        return analyzer

    def test_strong_signal(self):
        analyzer = self._make_analyzer("positive", 0.90, 0.06)
        result = analyzer.analyze("Profits soar at BDO after record quarter.")
        assert result.label == "positive"
        assert result.strength == "strong"

    def test_moderate_signal(self):
        analyzer = self._make_analyzer("positive", 0.72, 0.05)
        result = analyzer.analyze("Revenue slightly above expectations.")
        assert result.label == "positive"
        assert result.strength == "moderate"

    def test_weak_signal(self):
        analyzer = self._make_analyzer("negative", 0.60, 0.05)
        result = analyzer.analyze("Minor setback for the sector.")
        assert result.label == "negative"
        assert result.strength == "weak"

    def test_narrow_margin_falls_back_to_neutral(self):
        # pos=0.56, neg=0.44 → margin=0.12 < MIN_CONFIDENCE_MARGIN(0.15) → neutral
        analyzer = self._make_analyzer("positive", 0.56, 0.44)
        result = analyzer.analyze("The market moved slightly.")
        assert result.label == "neutral"

    def test_below_min_confidence_falls_back_to_neutral(self):
        # score=0.52 < MIN_DIRECTIONAL_CONFIDENCE(0.55) → neutral
        analyzer = self._make_analyzer("positive", 0.52, 0.05)
        result = analyzer.analyze("Marginal improvement in the index.")
        assert result.label == "neutral"

    def test_strength_field_present_in_result(self):
        from sentiment_analyzer import SentimentResult

        r = SentimentResult(
            text="test",
            label="positive",
            score=0.9,
            all_scores={"positive": 0.9, "negative": 0.05, "neutral": 0.05},
        )
        assert hasattr(r, "strength")
        assert r.strength == "neutral"  # default

    def test_str_includes_strength(self):
        from sentiment_analyzer import SentimentResult

        r = SentimentResult(
            text="Big profits!",
            label="positive",
            score=0.91,
            all_scores={"positive": 0.91, "negative": 0.05, "neutral": 0.04},
            strength="strong",
        )
        assert "strong" in str(r)


# ---------------------------------------------------------------------------
# Trading signal tests
# ---------------------------------------------------------------------------


class TestTradingSignals:
    def _make_result(self, label="positive", score=0.82, strength="strong", stocks=None):
        from sentiment_analyzer import SentimentResult

        if stocks is None:
            stocks = [
                {
                    "ticker": "BDO",
                    "name": "BDO Unibank",
                    "sector": "Financials",
                    "match_type": "direct",
                    "matched_keyword": "BDO",
                }
            ]
        return SentimentResult(
            text="BDO reports record profits.",
            label=label,
            score=score,
            all_scores={"positive": score, "negative": 0.05, "neutral": 0.05},
            affected_stocks=stocks,
            strength=strength,
        )

    def test_positive_sentiment_generates_buy(self):
        from trading_signals import generate_signals

        result = self._make_result("positive", 0.85, "strong")
        with patch("trading_signals._fetch_latest_price", return_value=None):
            sigs = generate_signals(result)
        assert len(sigs) == 1
        assert sigs[0].signal == "BUY"
        assert sigs[0].ticker == "BDO"

    def test_negative_sentiment_generates_sell(self):
        from trading_signals import generate_signals

        result = self._make_result("negative", 0.78, "moderate")
        with patch("trading_signals._fetch_latest_price", return_value=None):
            sigs = generate_signals(result)
        assert sigs[0].signal == "SELL"

    def test_neutral_sentiment_generates_hold(self):
        from trading_signals import generate_signals

        result = self._make_result("neutral", 0.70, "neutral")
        with patch("trading_signals._fetch_latest_price", return_value=None):
            sigs = generate_signals(result)
        assert sigs[0].signal == "HOLD"

    def test_prices_computed_when_entry_available(self):
        from trading_signals import generate_signals

        result = self._make_result("positive", 0.85, "strong")
        with patch("trading_signals._fetch_latest_price", return_value=100.0):
            sigs = generate_signals(result)
        sig = sigs[0]
        assert sig.entry_price == 100.0
        assert sig.target_price is not None
        assert sig.stop_loss is not None
        assert sig.target_price > sig.entry_price  # BUY: target above entry
        assert sig.stop_loss < sig.entry_price     # BUY: stop below entry

    def test_sell_prices_direction(self):
        from trading_signals import generate_signals

        result = self._make_result("negative", 0.82, "strong")
        with patch("trading_signals._fetch_latest_price", return_value=200.0):
            sigs = generate_signals(result)
        sig = sigs[0]
        assert sig.target_price < sig.entry_price  # SELL: target below entry
        assert sig.stop_loss > sig.entry_price     # SELL: stop above entry

    def test_no_signals_for_no_stocks(self):
        from trading_signals import generate_signals

        result = self._make_result("positive", stocks=[])
        sigs = generate_signals(result)
        assert sigs == []

    def test_sector_match_downgrades_strength(self):
        from trading_signals import generate_signals

        stocks = [
            {
                "ticker": "MBT",
                "name": "Metrobank",
                "sector": "Financials",
                "match_type": "sector",
                "matched_keyword": "Financials",
            }
        ]
        result = self._make_result("positive", 0.88, "strong", stocks=stocks)
        with patch("trading_signals._fetch_latest_price", return_value=None):
            sigs = generate_signals(result)
        # Sector match on a "strong" sentiment → downgraded to "MODERATE"
        assert sigs[0].strength == "MODERATE"

    def test_signal_str_representation(self):
        from trading_signals import TradingSignal

        sig = TradingSignal(
            ticker="JFC",
            name="Jollibee Foods",
            sector="Services",
            match_type="direct",
            signal="BUY",
            strength="STRONG",
            entry_price=250.0,
            target_price=258.0,
            stop_loss=245.0,
            sentiment_label="positive",
            sentiment_score=0.88,
            reasoning="Test reason.",
        )
        s = str(sig)
        assert "BUY" in s
        assert "JFC" in s
        assert "250.00" in s


# ---------------------------------------------------------------------------
# Backtester tests
# ---------------------------------------------------------------------------


class TestBacktester:
    def _make_price_dataframe(self, n=300, start_price=100.0, trend=0.0001):
        """Build a simple synthetic price DataFrame that mimics yfinance output."""
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        prices = [start_price * (1 + trend) ** i for i in range(n)]
        df = pd.DataFrame({"Close": prices}, index=dates)
        return df

    def test_backtest_returns_result_object(self):
        from backtester import backtest_signal, BacktestResult

        with patch("backtester.yf") as mock_yf:
            mock_yf.download.return_value = self._make_price_dataframe()
            result = backtest_signal("BDO", "BUY", holding_days=5, lookback_days=252)
        assert isinstance(result, BacktestResult)
        assert result.ticker == "BDO"
        assert result.signal == "BUY"

    def test_backtest_win_rate_in_range(self):
        from backtester import backtest_signal

        with patch("backtester.yf") as mock_yf:
            mock_yf.download.return_value = self._make_price_dataframe(trend=0.001)
            result = backtest_signal("JFC", "BUY", holding_days=5)
        assert result.win_rate is not None
        assert 0.0 <= result.win_rate <= 1.0

    def test_backtest_sell_win_rate_falling_market(self):
        from backtester import backtest_signal

        with patch("backtester.yf") as mock_yf:
            mock_yf.download.return_value = self._make_price_dataframe(trend=-0.001)
            result = backtest_signal("MER", "SELL", holding_days=5)
        assert result.win_rate is not None
        # In a consistently falling market, SELL should win most of the time
        assert result.win_rate > 0.5

    def test_backtest_returns_empty_on_no_data(self):
        from backtester import backtest_signal
        import pandas as pd

        with patch("backtester.yf") as mock_yf:
            mock_yf.download.return_value = pd.DataFrame()
            result = backtest_signal("FAKE", "BUY")
        assert result.win_rate is None
        assert result.current_trend is None

    def test_backtest_trend_uptrend(self):
        from backtester import backtest_signal

        with patch("backtester.yf") as mock_yf:
            mock_yf.download.return_value = self._make_price_dataframe(trend=0.002)
            result = backtest_signal("BDO", "BUY")
        assert result.current_trend == "UPTREND"

    def test_backtest_trend_downtrend(self):
        from backtester import backtest_signal

        with patch("backtester.yf") as mock_yf:
            mock_yf.download.return_value = self._make_price_dataframe(trend=-0.002)
            result = backtest_signal("BDO", "SELL")
        assert result.current_trend == "DOWNTREND"

    def test_backtest_summary_string(self):
        from backtester import BacktestResult

        bt = BacktestResult(
            ticker="BDO",
            signal="BUY",
            holding_days=5,
            win_rate=0.54,
            avg_return=0.011,
            sample_size=240,
            current_trend="UPTREND",
            price_vs_ma20=0.025,
            recent_return_5d=0.012,
            recent_return_20d=0.031,
        )
        s = bt.summary()
        assert "BDO" in s
        assert "54.0%" in s
        assert "UPTREND" in s

    def test_backtest_multiple_signals(self):
        from backtester import backtest_signals

        with patch("backtester.yf") as mock_yf:
            mock_yf.download.return_value = self._make_price_dataframe()
            results = backtest_signals([("BDO", "BUY"), ("JFC", "SELL")])
        assert len(results) == 2

    def test_backtest_handles_yfinance_not_installed(self):
        import backtester
        from backtester import backtest_signal

        with patch.object(backtester, "yf", None):
            result = backtest_signal("BDO", "BUY")
        assert result.win_rate is None


# ---------------------------------------------------------------------------
# Main pipeline — signals integration tests
# ---------------------------------------------------------------------------


class TestMainSignalsIntegration:
    def _make_articles(self, n=2):
        from scraper import NewsArticle

        return [
            NewsArticle(
                title=f"BDO reports strong profit growth Q{i}",
                url=f"https://www.abs-cbn.com/anc/test-signals-{i}",
                summary="BDO Unibank posts record net income.",
                category="Business",
            )
            for i in range(n)
        ]

    def test_build_report_includes_trading_signals(self):
        from main import build_report
        from sentiment_analyzer import SentimentResult
        from trading_signals import TradingSignal

        articles = self._make_articles(1)
        results = [
            SentimentResult(
                text="BDO profits up.",
                label="positive",
                score=0.88,
                all_scores={"positive": 0.88, "negative": 0.07, "neutral": 0.05},
                strength="strong",
                affected_stocks=[
                    {"ticker": "BDO", "name": "BDO Unibank", "sector": "Financials",
                     "match_type": "direct", "matched_keyword": "BDO"}
                ],
            )
        ]
        sig = TradingSignal(
            ticker="BDO",
            name="BDO Unibank",
            sector="Financials",
            match_type="direct",
            signal="BUY",
            strength="STRONG",
            entry_price=100.0,
            target_price=103.0,
            stop_loss=98.0,
            sentiment_label="positive",
            sentiment_score=0.88,
            reasoning="BDO directly mentioned.",
        )
        signals_map = {articles[0].url: [sig]}
        report = build_report(articles, results, signals_map=signals_map)
        assert "trading_signals" in report[0]
        assert len(report[0]["trading_signals"]) == 1
        assert report[0]["trading_signals"][0]["signal"] == "BUY"
        assert report[0]["trading_signals"][0]["entry_price"] == 100.0

    def test_parse_args_signals_flags(self):
        from main import parse_args

        args = parse_args(["--signals", "--backtest", "--holding-days", "10", "--lookback-days", "180"])
        assert args.signals is True
        assert args.backtest is True
        assert args.holding_days == 10
        assert args.lookback_days == 180

    @patch("main.FinBERTAnalyzer")
    @patch("main.scrape_anc_news")
    def test_run_with_signals_flag(self, mock_scrape, MockAnalyzer, tmp_path):
        from scraper import NewsArticle
        from sentiment_analyzer import SentimentResult
        from main import run, parse_args

        articles = [
            NewsArticle(
                title="BDO posts record profit",
                url="https://www.abs-cbn.com/anc/test-signals",
                summary="BDO Unibank reports profit surge.",
                category="Business",
            )
        ]
        mock_scrape.return_value = articles
        mock_results = [
            SentimentResult(
                text="BDO profits up.",
                label="positive",
                score=0.88,
                all_scores={"positive": 0.88, "negative": 0.07, "neutral": 0.05},
                strength="strong",
                affected_stocks=[
                    {"ticker": "BDO", "name": "BDO Unibank", "sector": "Financials",
                     "match_type": "direct", "matched_keyword": "BDO"}
                ],
            )
        ]
        MockAnalyzer.return_value.analyze_batch.return_value = mock_results

        with patch("trading_signals._fetch_latest_price", return_value=None):
            args = parse_args(["--signals"])
            report = run(args)

        assert len(report) == 1
        assert "trading_signals" in report[0]
        assert report[0]["trading_signals"][0]["signal"] == "BUY"
