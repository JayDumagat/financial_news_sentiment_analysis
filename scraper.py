"""
ABS-CBN ANC Financial News Scraper

Scrapes financial/business news articles from the ABS-CBN ANC website
(https://www.abs-cbn.com/anc/anc).

Watch / daemon mode
-------------------
:func:`run_forever` polls the news listing on a configurable interval,
saves new articles to the optional database and data lake, and runs
until interrupted (SIGINT / SIGTERM).  It intentionally does *not*
scrape aggressively — a minimum :data:`REQUEST_DELAY` is always
observed between HTTP requests, and a configurable inter-poll interval
prevents hammering the target server.
"""

import signal
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import requests
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from database import NewsDatabase
    from data_lake import DataLake

logger = logging.getLogger(__name__)

ANC_URL = "https://www.abs-cbn.com/anc/anc"
ANC_BUSINESS_URL = "https://www.abs-cbn.com/anc/business"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_DELAY = 1.5  # seconds between requests to be polite


@dataclass
class NewsArticle:
    """Represents a scraped news article."""

    title: str
    url: str
    summary: str = ""
    content: str = ""
    published_at: Optional[str] = None
    category: str = ""
    tags: list = field(default_factory=list)

    def get_text_for_analysis(self) -> str:
        """Return the best available text for sentiment analysis."""
        if self.content:
            # Truncate to avoid exceeding model token limits
            return self.content[:512]
        if self.summary:
            return self.summary
        return self.title


def _class_contains(keyword: str):
    """Return a BeautifulSoup class-filter callable that checks for *keyword*."""

    def _filter(css_classes):
        if not css_classes:
            return False
        classes = css_classes if isinstance(css_classes, list) else [css_classes]
        return any(keyword in c.lower() for c in classes)

    return _filter


class ANCNewsScraper:
    """Scraper for ABS-CBN ANC financial news articles."""

    def __init__(self, delay: float = REQUEST_DELAY):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_articles(self, max_articles: int = 20) -> list[NewsArticle]:
        """
        Fetch the latest articles from ABS-CBN ANC.

        Tries the business section first; falls back to the main ANC
        landing page so we always return something useful.

        Args:
            max_articles: Maximum number of articles to return.

        Returns:
            List of :class:`NewsArticle` objects.
        """
        articles = self._scrape_listing(ANC_BUSINESS_URL, max_articles)

        if len(articles) < max_articles:
            remaining = max_articles - len(articles)
            seen_urls = {a.url for a in articles}
            extra = self._scrape_listing(ANC_URL, remaining, exclude_urls=seen_urls)
            articles.extend(extra)

        logger.info("Scraped %d articles", len(articles))
        return articles[:max_articles]

    def enrich_article(self, article: NewsArticle) -> NewsArticle:
        """
        Fetch and attach the full body text of *article* (in-place).

        Args:
            article: Article whose ``content`` field will be populated.

        Returns:
            The same article instance, now with ``content`` filled.
        """
        try:
            time.sleep(self.delay)
            response = self.session.get(article.url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            article.content = self._extract_body(soup)
        except requests.RequestException as exc:
            logger.warning("Could not fetch %s: %s", article.url, exc)
        return article

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scrape_listing(
        self,
        url: str,
        max_articles: int,
        exclude_urls: set[str] | None = None,
    ) -> list[NewsArticle]:
        """Scrape article cards from a listing page."""
        exclude_urls = exclude_urls or set()
        articles: list[NewsArticle] = []

        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to fetch listing page %s: %s", url, exc)
            return articles

        soup = BeautifulSoup(response.text, "html.parser")
        cards = self._find_article_cards(soup)

        for card in cards:
            if len(articles) >= max_articles:
                break
            article = self._parse_card(card)
            if article and article.url not in exclude_urls:
                articles.append(article)

        return articles

    def _find_article_cards(self, soup: BeautifulSoup) -> list:
        """
        Return a list of tag elements that look like article cards.

        The ABS-CBN website uses several layouts; we try the most common
        selectors in order.
        """
        selectors = [
            "article.article-card",
            "div.article-card",
            "div.story-card",
            "li.article-item",
            "div.news-item",
            # generic fallback: any <a> inside <article>
            "article",
        ]
        for selector in selectors:
            cards = soup.select(selector)
            if cards:
                return cards
        return []

    def _parse_card(self, card) -> Optional[NewsArticle]:
        """Extract title, URL, and summary from an article card element."""
        # --- URL & Title ---
        link = card.find("a", href=True)
        if not link:
            return None

        href = link["href"]
        if not href.startswith("http"):
            href = "https://www.abs-cbn.com" + href

        title_tag = card.find(["h1", "h2", "h3", "h4", "span"], class_=_class_contains("title"))
        title = title_tag.get_text(strip=True) if title_tag else link.get_text(strip=True)

        if not title:
            return None

        # --- Summary / Teaser ---
        summary_tag = card.find("p")
        summary = summary_tag.get_text(strip=True) if summary_tag else ""

        # --- Category ---
        cat_tag = card.find(class_=_class_contains("category"))
        category = cat_tag.get_text(strip=True) if cat_tag else ""

        # --- Published date ---
        time_tag = card.find("time")
        published_at = time_tag.get("datetime") if time_tag else None

        return NewsArticle(
            title=title,
            url=href,
            summary=summary,
            category=category,
            published_at=published_at,
        )

    def _extract_body(self, soup: BeautifulSoup) -> str:
        """Extract the main body text from an article page."""
        # Try common article body selectors used by ABS-CBN
        body_selectors = [
            "div.article-body",
            "div.story-body",
            "div.entry-content",
            "div[class*='article-content']",
            "div[class*='story-content']",
            "article",
        ]
        for selector in body_selectors:
            body = soup.select_one(selector)
            if body:
                # Remove script/style noise
                for tag in body.find_all(["script", "style", "figure", "aside"]):
                    tag.decompose()
                paragraphs = body.find_all("p")
                if paragraphs:
                    return " ".join(p.get_text(strip=True) for p in paragraphs)
                text = body.get_text(separator=" ", strip=True)
                if len(text) > 100:
                    return text

        return ""


def scrape_anc_news(max_articles: int = 20, enrich: bool = False) -> list[NewsArticle]:
    """
    Convenience function: scrape ANC news and optionally enrich with full text.

    Args:
        max_articles: Maximum number of articles to return.
        enrich: If ``True``, fetch the full body text for every article.
                This makes one extra HTTP request per article.

    Returns:
        List of :class:`NewsArticle` objects.
    """
    scraper = ANCNewsScraper()
    articles = scraper.get_articles(max_articles=max_articles)

    if enrich:
        for article in articles:
            scraper.enrich_article(article)

    return articles


def run_forever(
    poll_interval: int = 300,
    max_articles: int = 20,
    enrich: bool = False,
    db: "Optional[NewsDatabase]" = None,
    data_lake: "Optional[DataLake]" = None,
    on_new_article=None,
) -> None:
    """
    Poll ABS-CBN ANC for new articles indefinitely.

    New articles (URLs not yet seen in *db* or the in-memory seen set) are
    saved to the database and data lake when those stores are provided, and
    passed to *on_new_article* for further processing.

    The function returns only when interrupted via SIGINT or SIGTERM.

    Args:
        poll_interval: Seconds to wait between successive scraping rounds.
                       Defaults to 300 s (5 minutes).  The effective delay
                       between individual HTTP requests is always at least
                       :data:`REQUEST_DELAY`.
        max_articles:  Number of articles to fetch per poll round.
        enrich:        Fetch the full article body on each new article.
        db:            Optional :class:`~database.NewsDatabase` instance.
                       Used to track seen URLs across process restarts.
        data_lake:     Optional :class:`~data_lake.DataLake` instance.
                       New raw articles are persisted here.
        on_new_article: Optional callable ``(article: NewsArticle) -> None``
                        invoked for every genuinely new article.
    """
    scraper = ANCNewsScraper()
    seen_urls: set[str] = set()
    _stop = {"flag": False}

    def _handle_signal(signum, frame):  # noqa: ANN001
        logger.info("Received signal %s — stopping watch loop.", signum)
        _stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "Watch mode started.  Polling every %d s for up to %d articles.",
        poll_interval,
        max_articles,
    )

    while not _stop["flag"]:
        try:
            articles = scraper.get_articles(max_articles=max_articles)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error during scraping: %s", exc)
            articles = []

        new_articles: list[NewsArticle] = []
        for article in articles:
            already_seen = article.url in seen_urls or (
                db is not None and db.is_seen(article.url)
            )
            if already_seen:
                continue

            seen_urls.add(article.url)

            if enrich:
                scraper.enrich_article(article)

            # Persist raw article to the data lake
            if data_lake is not None:
                try:
                    data_lake.save_raw_article(article)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Data lake write failed: %s", exc)

            # Save to database (relevance flag determined later by pipeline)
            if db is not None:
                try:
                    db.save_article(article)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("DB write failed: %s", exc)

            new_articles.append(article)

        if new_articles:
            logger.info("Watch: %d new article(s) found.", len(new_articles))
            if on_new_article is not None:
                for article in new_articles:
                    try:
                        on_new_article(article)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("on_new_article callback raised: %s", exc)
        else:
            logger.debug("Watch: no new articles this round.")

        if _stop["flag"]:
            break

        # Politely wait before the next round
        logger.debug("Watch: sleeping %d s until next poll …", poll_interval)
        for _ in range(poll_interval):
            if _stop["flag"]:
                break
            time.sleep(1)

    logger.info("Watch mode stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = scrape_anc_news(max_articles=5)
    for i, art in enumerate(results, 1):
        print(f"\n[{i}] {art.title}")
        print(f"    URL     : {art.url}")
        print(f"    Category: {art.category}")
        print(f"    Summary : {art.summary[:120]}{'...' if len(art.summary) > 120 else ''}")
