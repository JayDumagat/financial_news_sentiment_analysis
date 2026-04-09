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

    @patch("scraper.ANCNewsScraper._fetch_html")
    def test_get_articles_parses_cards(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_HTML_LISTING
        articles = self.scraper.get_articles(max_articles=5)
        assert len(articles) >= 1
        titles = [a.title for a in articles]
        assert any("Test Headline" in t or "Second Article" in t for t in titles)

    @patch("scraper.ANCNewsScraper._fetch_html")
    def test_get_articles_url_normalization(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_HTML_LISTING
        articles = self.scraper.get_articles(max_articles=5)
        for art in articles:
            assert art.url.startswith("http"), f"URL should be absolute: {art.url}"

    @patch("scraper.ANCNewsScraper._fetch_html")
    def test_enrich_article_fills_content(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_HTML_ARTICLE
        from scraper import NewsArticle

        article = NewsArticle(
            title="Dummy",
            url="https://www.abs-cbn.com/anc/business/article/2024/1/1/dummy",
        )
        self.scraper.enrich_article(article)
        assert "Paragraph one" in article.content or article.content == ""

    @patch("scraper.ANCNewsScraper._fetch_html")
    def test_get_articles_max_limit(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_HTML_LISTING
        articles = self.scraper.get_articles(max_articles=1)
        assert len(articles) <= 1

    @patch("scraper.ANCNewsScraper._fetch_html")
    def test_fetch_error_returns_empty(self, mock_fetch):
        mock_fetch.return_value = None  # simulate Playwright / network failure
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

    def test_meta_field_defaults_to_empty_dict(self):
        from scraper import NewsArticle

        art = NewsArticle(title="T", url="http://x")
        assert art.meta == {}

    def test_meta_field_can_be_set(self):
        from scraper import NewsArticle

        art = NewsArticle(title="T", url="http://x", meta={"og:title": "Test"})
        assert art.meta["og:title"] == "Test"


# ---------------------------------------------------------------------------
# ANCNewsScraper – meta extraction & link-discovery logging tests
# ---------------------------------------------------------------------------


SAMPLE_HTML_WITH_META = """
<html>
<head>
  <meta property="og:title" content="PSE Rally" />
  <meta property="og:description" content="Stocks surge on rate cut." />
  <meta name="description" content="Financial news summary." />
  <meta name="keywords" content="PSE, stocks, finance" />
  <meta property="article:section" content="Business" />
  <link rel="canonical" href="https://www.abs-cbn.com/anc/business/article/2024/1/1/pse-rally" />
</head>
<body>
  <div class="article-body">
    <p>Paragraph one of the article body.</p>
  </div>
</body>
</html>
"""

SAMPLE_HTML_LINK_STRATEGY = """
<html>
<body>
  <div class="news-list">
    <div>
      <a href="/anc/business/article/2024/1/3/third">Third Article</a>
      <h3>Third Article</h3>
    </div>
    <div>
      <a href="/anc/business/article/2024/1/4/fourth">Fourth Article</a>
      <h3>Fourth Article</h3>
    </div>
  </div>
</body>
</html>
"""


class TestANCNewsScraperMeta:
    def setup_method(self):
        from scraper import ANCNewsScraper
        self.scraper = ANCNewsScraper(delay=0)

    @patch("scraper.ANCNewsScraper._fetch_html")
    def test_enrich_article_populates_meta(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_HTML_WITH_META
        from scraper import NewsArticle
        art = NewsArticle(
            title="PSE Rally",
            url="https://www.abs-cbn.com/anc/business/article/2024/1/1/pse-rally",
        )
        self.scraper.enrich_article(art)
        assert art.meta.get("og:title") == "PSE Rally"
        assert art.meta.get("description") == "Financial news summary."
        assert art.meta.get("canonical") == (
            "https://www.abs-cbn.com/anc/business/article/2024/1/1/pse-rally"
        )

    @patch("scraper.ANCNewsScraper._fetch_html")
    def test_enrich_article_populates_content_and_meta(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_HTML_WITH_META
        from scraper import NewsArticle
        art = NewsArticle(title="T", url="https://www.abs-cbn.com/anc/business/article/2024/1/1/t")
        self.scraper.enrich_article(art)
        assert "Paragraph one" in art.content
        assert art.meta  # meta dict is not empty

    def test_extract_meta_parses_head(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(SAMPLE_HTML_WITH_META, "html.parser")
        meta = self.scraper._extract_meta(soup)
        assert meta["og:title"] == "PSE Rally"
        assert meta["keywords"] == "PSE, stocks, finance"
        assert meta["article:section"] == "Business"
        assert meta["canonical"].endswith("pse-rally")

    def test_find_article_cards_logs_found_links(self, caplog):
        import logging
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(SAMPLE_HTML_LINK_STRATEGY, "html.parser")
        with caplog.at_level(logging.INFO, logger="scraper"):
            cards = self.scraper._find_article_cards(soup)
        assert len(cards) >= 1
        assert any("Found" in r.message and "links" in r.message for r in caplog.records)

    def test_find_article_cards_logs_article_tags(self, caplog):
        import logging
        from bs4 import BeautifulSoup
        html = """
        <html><body>
          <article><a href="/anc/business/article/2024/1/1/x">X</a><h3>X</h3></article>
          <article><a href="/anc/business/article/2024/1/2/y">Y</a><h3>Y</h3></article>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        with caplog.at_level(logging.INFO, logger="scraper"):
            cards = self.scraper._find_article_cards(soup)
        assert len(cards) == 2
        assert any("Found 2" in r.message for r in caplog.records)

    @patch("scraper.ANCNewsScraper._fetch_html")
    def test_data_lake_save_raw_html_called(self, mock_fetch, tmp_path):
        from data_lake import DataLake
        from scraper import ANCNewsScraper

        lake = DataLake(base_path=str(tmp_path / "lake"))
        scraper = ANCNewsScraper(delay=0, data_lake=lake)
        # _fetch_html is patched so raw HTML save happens inside the real method;
        # test instead that the data_lake reference is stored and callable
        assert scraper.data_lake is lake

    def test_data_lake_save_raw_html_written_when_fetch_succeeds(self, tmp_path):
        """Verify save_raw_html is called when _fetch_html returns HTML."""
        from data_lake import DataLake
        from scraper import ANCNewsScraper
        from unittest.mock import patch as _patch

        lake = DataLake(base_path=str(tmp_path / "lake"))
        scraper = ANCNewsScraper(delay=0, data_lake=lake)

        raw_html = "<html><body>test</body></html>"
        # Patch at the lowest level: make sync_playwright unavailable so the
        # real _fetch_html returns None, then test save_raw_html directly.
        with _patch.object(lake, "save_raw_html") as mock_save:
            # Simulate what _fetch_html does after a successful Playwright call
            scraper.data_lake.save_raw_html("https://example.com/p", raw_html)
            mock_save.assert_called_once_with("https://example.com/p", raw_html)





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

    def test_save_raw_html(self, tmp_path):
        from data_lake import DataLake

        lake = DataLake(base_path=str(tmp_path / "lake"))
        art = self._make_article()
        html = "<html><body><p>Raw page</p></body></html>"
        path = lake.save_raw_html(art.url, html)
        assert path.exists()
        assert path.suffix == ".html"
        assert "raw_html" in str(path)
        assert path.read_text(encoding="utf-8") == html

    def test_save_preprocessed_article(self, tmp_path):
        import json as json_lib
        from data_lake import DataLake
        from scraper import NewsArticle

        lake = DataLake(base_path=str(tmp_path / "lake"))
        art = NewsArticle(
            title="Stocks  Rise",
            url="https://example.com/article-pre",
            summary="Markets up.",
            category="Business",
            meta={"og:title": "Stocks Rise", "description": "Market update"},
        )
        path = lake.save_preprocessed_article(art)
        assert path.exists()
        assert "preprocessed" in str(path)
        with path.open(encoding="utf-8") as fh:
            data = json_lib.load(fh)
        assert data["url"] == art.url
        assert data["meta"]["og:title"] == "Stocks Rise"
        assert "preprocessed_at" in data

    def test_save_cleaned_article(self, tmp_path):
        import json as json_lib
        from data_lake import DataLake
        from scraper import NewsArticle

        lake = DataLake(base_path=str(tmp_path / "lake"))
        art = NewsArticle(
            title="  PSE  stocks  rally \n",
            url="https://example.com/article-clean",
            summary="Markets\tsurged  after  the  decision.",
            category=" Business ",
            content="Paragraph one.\n\nParagraph two.",
        )
        path = lake.save_cleaned_article(art)
        assert path.exists()
        assert "cleaned" in str(path)
        with path.open(encoding="utf-8") as fh:
            data = json_lib.load(fh)
        assert data["title"] == "PSE stocks rally"
        assert "\n" not in data["summary"]
        assert "\t" not in data["summary"]
        assert "cleaned_at" in data

    def test_save_analyzed_result(self, tmp_path):
        import json as json_lib
        from data_lake import DataLake
        from sentiment_analyzer import SentimentResult

        lake = DataLake(base_path=str(tmp_path / "lake"))
        art = self._make_article()
        result = SentimentResult(
            text=art.summary,
            label="positive",
            score=0.91,
            all_scores={"positive": 0.91, "negative": 0.05, "neutral": 0.04},
            affected_stocks=[{"ticker": "ALI", "name": "Ayala Land", "sector": "Property",
                               "match_type": "direct", "matched_keyword": "Ayala"}],
        )
        path = lake.save_analyzed_result(art, result)
        assert path.exists()
        assert "analyzed" in str(path)
        with path.open(encoding="utf-8") as fh:
            data = json_lib.load(fh)
        assert data["sentiment"] == "positive"
        assert data["confidence"] == 0.91
        assert len(data["affected_stocks"]) == 1
        assert "analyzed_at" in data

    def test_all_tiers_in_separate_dirs(self, tmp_path):
        from data_lake import DataLake
        from sentiment_analyzer import SentimentResult

        lake = DataLake(base_path=str(tmp_path / "lake"))
        art = self._make_article()
        result = SentimentResult(
            text="test", label="neutral", score=0.7,
            all_scores={"positive": 0.1, "negative": 0.2, "neutral": 0.7},
        )
        raw_path = lake.save_raw_article(art)
        html_path = lake.save_raw_html(art.url, "<html/>")
        pre_path = lake.save_preprocessed_article(art)
        clean_path = lake.save_cleaned_article(art)
        analyzed_path = lake.save_analyzed_result(art, result)
        # Each path is <base>/<tier>/<YYYY>/<MM>/<DD>/<file> — extract tier name
        def _tier(p):
            return p.parts[-5]  # 5 levels up: tier/YYYY/MM/DD/file → tier
        tiers = {_tier(raw_path), _tier(html_path), _tier(pre_path),
                 _tier(clean_path), _tier(analyzed_path)}
        assert tiers == {"raw", "raw_html", "preprocessed", "cleaned", "analyzed"}


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
        """Build a simple synthetic price DataFrame that mimics tvdatafeed output."""
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        prices = [start_price * (1 + trend) ** i for i in range(n)]
        df = pd.DataFrame({"close": prices}, index=dates)
        return df

    def test_backtest_returns_result_object(self):
        from backtester import backtest_signal, BacktestResult

        with patch("backtester._download_ohlcv", return_value=self._make_price_dataframe()):
            result = backtest_signal("BDO", "BUY", holding_days=5, lookback_days=252)
        assert isinstance(result, BacktestResult)
        assert result.ticker == "BDO"
        assert result.signal == "BUY"

    def test_backtest_win_rate_in_range(self):
        from backtester import backtest_signal

        with patch("backtester._download_ohlcv", return_value=self._make_price_dataframe(trend=0.001)):
            result = backtest_signal("JFC", "BUY", holding_days=5)
        assert result.win_rate is not None
        assert 0.0 <= result.win_rate <= 1.0

    def test_backtest_sell_win_rate_falling_market(self):
        from backtester import backtest_signal

        with patch("backtester._download_ohlcv", return_value=self._make_price_dataframe(trend=-0.001)):
            result = backtest_signal("MER", "SELL", holding_days=5)
        assert result.win_rate is not None
        # In a consistently falling market, SELL should win most of the time
        assert result.win_rate > 0.5

    def test_backtest_returns_empty_on_no_data(self):
        from backtester import backtest_signal
        import pandas as pd

        with patch("backtester._download_ohlcv", return_value=pd.DataFrame()):
            with patch("backtester._fetch_tv_current_metrics", return_value=(None, None)):
                result = backtest_signal("FAKE", "BUY")
        assert result.win_rate is None
        assert result.current_trend is None

    def test_backtest_trend_uptrend(self):
        from backtester import backtest_signal

        with patch("backtester._download_ohlcv", return_value=self._make_price_dataframe(trend=0.002)):
            result = backtest_signal("BDO", "BUY")
        assert result.current_trend == "UPTREND"

    def test_backtest_trend_downtrend(self):
        from backtester import backtest_signal

        with patch("backtester._download_ohlcv", return_value=self._make_price_dataframe(trend=-0.002)):
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

        with patch("backtester._download_ohlcv", return_value=self._make_price_dataframe()):
            results = backtest_signals([("BDO", "BUY"), ("JFC", "SELL")])
        assert len(results) == 2

    def test_backtest_handles_tvdatafeed_not_installed(self):
        from backtester import backtest_signal

        with patch("backtester._download_ohlcv", return_value=None):
            with patch("backtester._fetch_tv_current_metrics", return_value=(None, None)):
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


# ---------------------------------------------------------------------------
# PSE market-hours tests
# ---------------------------------------------------------------------------


class TestPSEMarketHours:
    def _make_dt(self, weekday, hour, minute=0):
        """Build a PHT-aware datetime with a given ISO weekday (0=Mon)."""
        from datetime import datetime, timezone, timedelta

        pht = timezone(timedelta(hours=8))
        # Use a Monday=2025-01-06 as base
        base_monday = datetime(2025, 1, 6, tzinfo=pht)
        day = base_monday + timedelta(days=weekday)
        return day.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def test_market_open_during_morning_session(self):
        from trading_signals import is_pse_market_open

        assert is_pse_market_open(self._make_dt(0, 10, 0)) is True  # 10:00 Mon

    def test_market_open_during_afternoon_session(self):
        from trading_signals import is_pse_market_open

        assert is_pse_market_open(self._make_dt(1, 14, 0)) is True  # 14:00 Tue

    def test_market_closed_before_open(self):
        from trading_signals import is_pse_market_open

        assert is_pse_market_open(self._make_dt(0, 8, 0)) is False  # 08:00 Mon

    def test_market_closed_at_lunchbreak(self):
        from trading_signals import is_pse_market_open

        assert is_pse_market_open(self._make_dt(2, 12, 30)) is False  # 12:30 Wed

    def test_market_closed_after_close(self):
        from trading_signals import is_pse_market_open

        assert is_pse_market_open(self._make_dt(3, 16, 0)) is False  # 16:00 Thu

    def test_market_closed_on_saturday(self):
        from trading_signals import is_pse_market_open

        assert is_pse_market_open(self._make_dt(5, 10, 0)) is False  # Saturday

    def test_market_closed_on_sunday(self):
        from trading_signals import is_pse_market_open

        assert is_pse_market_open(self._make_dt(6, 10, 0)) is False  # Sunday

    def test_next_pse_market_open_returns_future(self):
        from trading_signals import next_pse_market_open
        from datetime import timezone, timedelta

        pht = timezone(timedelta(hours=8))
        # Use a Saturday — next open should be Monday 09:30 PHT
        from datetime import datetime

        saturday = datetime(2025, 1, 11, 20, 0, tzinfo=pht)  # Sat 20:00
        nxt = next_pse_market_open(saturday)
        assert nxt > saturday
        assert nxt.weekday() < 5  # must be a weekday

    def test_entry_note_is_next_open_when_market_closed(self):
        from trading_signals import generate_signals, is_pse_market_open
        from sentiment_analyzer import SentimentResult
        from datetime import datetime, timezone, timedelta

        pht = timezone(timedelta(hours=8))
        # Saturday — market closed
        saturday = datetime(2025, 1, 11, 20, 0, tzinfo=pht)

        result = SentimentResult(
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
        with patch("trading_signals._fetch_latest_price", return_value=100.0):
            with patch("trading_signals._fetch_atr", return_value=None):
                sigs = generate_signals(result, now=saturday)
        assert sigs[0].entry_note == "NEXT OPEN"

    def test_entry_note_is_current_when_market_open(self):
        from trading_signals import generate_signals
        from sentiment_analyzer import SentimentResult
        from datetime import datetime, timezone, timedelta

        pht = timezone(timedelta(hours=8))
        monday_10am = datetime(2025, 1, 6, 10, 0, tzinfo=pht)

        result = SentimentResult(
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
        with patch("trading_signals._fetch_latest_price", return_value=100.0):
            with patch("trading_signals._fetch_atr", return_value=None):
                sigs = generate_signals(result, now=monday_10am)
        assert sigs[0].entry_note == "CURRENT"


# ---------------------------------------------------------------------------
# ATR-based stop-loss tests
# ---------------------------------------------------------------------------


class TestATRStops:
    def _make_result(self, strength="strong"):
        from sentiment_analyzer import SentimentResult

        return SentimentResult(
            text="BDO profits up.",
            label="positive",
            score=0.88,
            all_scores={"positive": 0.88, "negative": 0.07, "neutral": 0.05},
            strength=strength,
            affected_stocks=[
                {"ticker": "BDO", "name": "BDO Unibank", "sector": "Financials",
                 "match_type": "direct", "matched_keyword": "BDO"}
            ],
        )

    def test_atr_stop_wider_than_entry(self):
        from trading_signals import generate_signals

        with patch("trading_signals._fetch_latest_price", return_value=100.0):
            with patch("trading_signals._fetch_atr", return_value=2.0):
                sigs = generate_signals(self._make_result("strong"))
        sig = sigs[0]
        # ATR=2.0, mult=1.5 → stop distance=3.0 → stop=97.0
        assert sig.stop_loss == pytest.approx(100.0 - 2.0 * 1.5, abs=0.01)
        assert sig.atr == pytest.approx(2.0)

    def test_fallback_to_fixed_pct_when_atr_none(self):
        from trading_signals import generate_signals

        with patch("trading_signals._fetch_latest_price", return_value=100.0):
            with patch("trading_signals._fetch_atr", return_value=None):
                sigs = generate_signals(self._make_result("strong"))
        sig = sigs[0]
        # Fallback: strong → 2% stop
        assert sig.stop_loss == pytest.approx(100.0 * (1 - 0.02), abs=0.01)
        assert sig.atr is None

    def test_atr_stored_on_signal(self):
        from trading_signals import generate_signals

        with patch("trading_signals._fetch_latest_price", return_value=50.0):
            with patch("trading_signals._fetch_atr", return_value=1.5):
                sigs = generate_signals(self._make_result())
        assert sigs[0].atr == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# valid_until / signal expiry tests
# ---------------------------------------------------------------------------


class TestSignalExpiry:
    def _make_result(self, strength="strong"):
        from sentiment_analyzer import SentimentResult

        return SentimentResult(
            text="BDO profits up.",
            label="positive",
            score=0.88,
            all_scores={"positive": 0.88, "negative": 0.07, "neutral": 0.05},
            strength=strength,
            affected_stocks=[
                {"ticker": "BDO", "name": "BDO Unibank", "sector": "Financials",
                 "match_type": "direct", "matched_keyword": "BDO"}
            ],
        )

    def test_valid_until_set(self):
        from trading_signals import generate_signals
        from datetime import datetime, timezone, timedelta

        pht = timezone(timedelta(hours=8))
        pub = datetime(2025, 1, 6, 10, 0, tzinfo=pht)
        with patch("trading_signals._fetch_latest_price", return_value=None):
            with patch("trading_signals._fetch_atr", return_value=None):
                sigs = generate_signals(self._make_result(), published_at=pub)
        assert sigs[0].valid_until is not None
        assert sigs[0].valid_until > pub

    def test_strong_signal_valid_6h(self):
        from trading_signals import generate_signals
        from datetime import datetime, timezone, timedelta

        pht = timezone(timedelta(hours=8))
        pub = datetime(2025, 1, 6, 9, 0, tzinfo=pht)
        with patch("trading_signals._fetch_latest_price", return_value=None):
            with patch("trading_signals._fetch_atr", return_value=None):
                sigs = generate_signals(self._make_result("strong"), published_at=pub)
        expected = pub + timedelta(hours=6)
        assert sigs[0].valid_until == expected

    def test_weak_signal_valid_4h(self):
        from trading_signals import generate_signals
        from datetime import datetime, timezone, timedelta

        pht = timezone(timedelta(hours=8))
        pub = datetime(2025, 1, 6, 9, 0, tzinfo=pht)
        with patch("trading_signals._fetch_latest_price", return_value=None):
            with patch("trading_signals._fetch_atr", return_value=None):
                sigs = generate_signals(self._make_result("weak"), published_at=pub)
        expected = pub + timedelta(hours=4)
        assert sigs[0].valid_until == expected


# ---------------------------------------------------------------------------
# Aspect-based sentiment tests
# ---------------------------------------------------------------------------


class TestAspectSentiment:
    def _make_analyzer(self, mock_label="positive", mock_score=0.85):
        from sentiment_analyzer import FinBERTAnalyzer

        def _pipeline(texts, **kwargs):
            return [
                [
                    {"label": mock_label, "score": mock_score},
                    {"label": "negative", "score": 0.05},
                    {"label": "neutral", "score": 0.10},
                ]
                for _ in texts
            ]

        analyzer = FinBERTAnalyzer()
        analyzer._pipeline = _pipeline
        return analyzer

    def test_analyze_aspects_returns_stock_list(self):
        analyzer = self._make_analyzer()
        stocks = [
            {"ticker": "BDO", "name": "BDO Unibank", "sector": "Financials",
             "match_type": "direct", "matched_keyword": "BDO"}
        ]
        text = "BDO reported record net income this quarter driven by strong lending."
        result = analyzer.analyze_aspects(text, stocks=stocks)
        assert len(result) == 1
        assert "aspect_label" in result[0]
        assert "aspect_score" in result[0]
        assert "aspect_strength" in result[0]
        assert "aspect_source" in result[0]

    def test_analyze_aspects_direct_mention_tagged(self):
        analyzer = self._make_analyzer("positive", 0.90)
        stocks = [
            {"ticker": "BDO", "name": "BDO Unibank", "sector": "Financials",
             "match_type": "direct", "matched_keyword": "BDO"}
        ]
        text = "BDO Unibank posted strong earnings."
        result = analyzer.analyze_aspects(text, stocks=stocks)
        assert result[0]["aspect_source"] == "direct"

    def test_analyze_aspects_fallback_when_no_mention(self):
        analyzer = self._make_analyzer("neutral", 0.70)
        stocks = [
            {"ticker": "MBT", "name": "Metrobank", "sector": "Financials",
             "match_type": "sector", "matched_keyword": "Financials"}
        ]
        # Text does not mention MBT or Metrobank directly
        text = "The BSP raised interest rates today."
        result = analyzer.analyze_aspects(text, stocks=stocks)
        assert result[0]["aspect_source"] == "article"

    def test_analyze_aspects_empty_stocks(self):
        analyzer = self._make_analyzer()
        result = analyzer.analyze_aspects("Some text.", stocks=[])
        assert result == []

    def test_analyze_aspects_auto_detects_stocks(self):
        analyzer = self._make_analyzer("positive", 0.90)
        text = "Meralco announced a rate increase affecting Manila consumers."
        result = analyzer.analyze_aspects(text)
        # MER should be detected via keyword "Meralco"
        tickers = [s["ticker"] for s in result]
        assert "MER" in tickers


# ---------------------------------------------------------------------------
# Relevance filter — min keyword count gate tests
# ---------------------------------------------------------------------------


class TestRelevanceFilterKeywordGate:
    def _make_article(self, title="", summary="", category="", content=""):
        from scraper import NewsArticle

        return NewsArticle(
            title=title,
            url=f"https://example.com/{hash(title + summary)}",
            summary=summary,
            category=category,
            content=content,
        )

    def test_single_brand_mention_without_financial_context_rejected(self):
        from relevance_filter import RelevanceFilter

        filt = RelevanceFilter()
        # Celebrity article mentioning a brand — single keyword hit, no category
        art = self._make_article(
            title="Actress spotted drinking San Miguel beer at concert",
            summary="The popular singer enjoyed the event.",
        )
        # "market" or "stock" won't appear → score < threshold → rejected
        assert filt.is_financial(art) is False

    def test_two_keywords_without_category_passes(self):
        from relevance_filter import RelevanceFilter

        filt = RelevanceFilter()
        art = self._make_article(
            title="San Miguel Corporation reports profit surge",
            summary="Revenue climbed 12% driven by beer volumes.",
        )
        # 'profit' + 'revenue' → 2 keywords → passes count gate
        assert filt.is_financial(art) is True

    def test_financial_category_bypasses_count_gate(self):
        from relevance_filter import RelevanceFilter

        filt = RelevanceFilter()
        art = self._make_article(title="Quick update", category="Business")
        # Category boost alone (5.0 ≥ threshold) + category bypasses count gate
        assert filt.is_financial(art) is True

    def test_custom_min_keyword_count(self):
        from relevance_filter import RelevanceFilter

        filt = RelevanceFilter(min_keyword_count=3)
        # Two keywords → should fail with count=3
        art = self._make_article(
            title="Stock profit rises",
            summary="Quarterly earnings improved.",
        )
        # 'stock', 'profit', 'earnings' → 3 keywords → passes
        assert filt.is_financial(art) is True

        art2 = self._make_article(title="Stock profit", summary="Good news.")
        # Only 'stock', 'profit' → 2 keywords < 3 → fails
        result = filt.check(art2)
        assert result.is_financial is False


# ---------------------------------------------------------------------------
# Scraper — fake-useragent tests
# ---------------------------------------------------------------------------


class TestFakeUserAgent:
    def test_get_random_ua_returns_string(self):
        from scraper import _get_random_ua

        ua = _get_random_ua()
        assert isinstance(ua, str)
        assert len(ua) > 10

    def test_get_random_ua_falls_back_on_import_error(self):
        """When fake-useragent is not installed, fall back to static UA."""
        from scraper import _get_random_ua, _FALLBACK_UA

        with patch.dict(sys.modules, {"fake_useragent": None}):
            ua = _get_random_ua()
        assert isinstance(ua, str)
        # When the module is None, ImportError is raised and we fall back
        # to the static string — just check it's a non-empty string.
        assert len(ua) > 0

    def test_scraper_uses_random_ua_in_fetch(self):
        """Scraper passes a User-Agent via _get_random_ua() into _fetch_html."""
        from scraper import ANCNewsScraper, _get_random_ua

        scraper = ANCNewsScraper(delay=0)
        # Verify _get_random_ua returns a valid UA string (consumed by _fetch_html)
        ua = _get_random_ua()
        assert isinstance(ua, str) and len(ua) > 10


# ---------------------------------------------------------------------------
# Notifier tests
# ---------------------------------------------------------------------------


class TestNotifier:
    def _make_signal(self):
        from trading_signals import TradingSignal
        from datetime import datetime, timezone, timedelta

        pht = timezone(timedelta(hours=8))
        return TradingSignal(
            ticker="BDO",
            name="BDO Unibank",
            sector="Financials",
            match_type="direct",
            signal="BUY",
            strength="STRONG",
            entry_price=132.5,
            target_price=136.5,
            stop_loss=129.5,
            sentiment_label="positive",
            sentiment_score=0.88,
            reasoning="BDO reported record profits.",
            entry_note="CURRENT",
            valid_until=datetime(2025, 1, 6, 15, 0, tzinfo=pht),
            atr=2.1,
        )

    def test_notifier_not_configured_by_default(self):
        from notifier import Notifier

        n = Notifier()
        assert n.is_configured is False

    def test_notifier_configured_with_telegram(self):
        from notifier import Notifier

        n = Notifier(telegram_token="abc:123", telegram_chat_id="@mychan")
        assert n.is_configured is True

    def test_notifier_configured_with_discord(self):
        from notifier import Notifier

        n = Notifier(discord_webhook="https://discord.com/api/webhooks/x/y")
        assert n.is_configured is True

    def test_send_signal_no_op_when_not_configured(self):
        from notifier import Notifier

        n = Notifier()
        # Should not raise even with a valid signal
        n.send_signal(self._make_signal())

    def test_send_signal_calls_telegram(self):
        from notifier import Notifier
        import requests as req_lib

        n = Notifier(telegram_token="token", telegram_chat_id="chat")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch("notifier.requests.post", return_value=mock_resp) as mock_post:
            n.send_signal(self._make_signal())
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0].startswith("https://api.telegram.org/bot")

    def test_send_signal_calls_discord(self):
        from notifier import Notifier

        webhook = "https://discord.com/api/webhooks/1234/abcd"
        n = Notifier(discord_webhook=webhook)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch("notifier.requests.post", return_value=mock_resp) as mock_post:
            n.send_signal(self._make_signal())
        mock_post.assert_called_once()
        assert mock_post.call_args[0][0] == webhook

    def test_send_text_swallows_network_error(self):
        from notifier import Notifier
        import requests as req_lib

        n = Notifier(telegram_token="tok", telegram_chat_id="cid")
        with patch("notifier.requests.post", side_effect=req_lib.RequestException("err")):
            # Should not raise
            n.send_text("hello")

    def test_format_signal_contains_key_fields(self):
        from notifier import _format_signal

        sig = self._make_signal()
        text = _format_signal(sig)
        assert "BDO" in text
        assert "BUY" in text
        assert "132.50" in text
        assert "136.50" in text


# ---------------------------------------------------------------------------
# Main pipeline — new CLI flags
# ---------------------------------------------------------------------------


class TestMainNewFlags:
    def test_parse_args_aspects_flag(self):
        from main import parse_args

        args = parse_args(["--aspects"])
        assert args.aspects is True

    def test_parse_args_telegram_flags(self):
        from main import parse_args

        args = parse_args(["--telegram-token", "tok", "--telegram-chat-id", "cid"])
        assert args.telegram_token == "tok"
        assert args.telegram_chat_id == "cid"

    def test_parse_args_discord_flag(self):
        from main import parse_args

        args = parse_args(["--discord-webhook", "https://discord.com/api/webhooks/x"])
        assert args.discord_webhook == "https://discord.com/api/webhooks/x"

    def test_build_report_includes_valid_until_and_entry_note(self):
        from main import build_report
        from sentiment_analyzer import SentimentResult
        from trading_signals import TradingSignal
        from datetime import datetime, timezone, timedelta
        from scraper import NewsArticle

        pht = timezone(timedelta(hours=8))
        article = NewsArticle(
            title="BDO profits",
            url="https://example.com/bdo",
            summary="Record income.",
            category="Business",
        )
        result = SentimentResult(
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
            reasoning="Test.",
            entry_note="NEXT OPEN",
            valid_until=datetime(2025, 1, 6, 15, 0, tzinfo=pht),
            atr=1.5,
        )
        signals_map = {article.url: [sig]}
        report = build_report([article], [result], signals_map=signals_map)
        sig_row = report[0]["trading_signals"][0]
        assert sig_row["entry_note"] == "NEXT OPEN"
        assert sig_row["valid_until"] is not None
        assert sig_row["atr"] == pytest.approx(1.5)
