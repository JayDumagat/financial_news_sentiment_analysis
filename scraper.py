"""
ABS-CBN ANC Financial News Scraper

Scrapes financial/business news articles from the ABS-CBN ANC website
(https://www.abs-cbn.com/anc/anc).

The ABS-CBN ANC website is a JavaScript-rendered (dynamic) single-page
application.  Plain ``requests`` cannot retrieve rendered article cards
because the content is injected by JavaScript after the initial HTML load.
This scraper uses a **headless Chromium browser** (via Playwright) to fully
render each page before parsing the resulting DOM with BeautifulSoup.

To install the required Chromium binary after ``pip install playwright``,
run once::

    playwright install chromium

Watch / daemon mode
-------------------
:func:`run_forever` polls the news listing on a configurable interval,
saves new articles to the optional database and data lake, and runs
until interrupted (SIGINT / SIGTERM).  It intentionally does *not*
scrape aggressively — a minimum :data:`REQUEST_DELAY` is always
observed between HTTP requests, and a configurable inter-poll interval
prevents hammering the target server.
"""

import re
import signal
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from database import NewsDatabase
    from data_lake import DataLake

logger = logging.getLogger(__name__)

ANC_URL = "https://www.abs-cbn.com/anc/anc"
ANC_BUSINESS_URL = "https://www.abs-cbn.com/anc/business"

# Fallback static User-Agent used when fake-useragent is not installed
_FALLBACK_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

REQUEST_DELAY = 1.5  # seconds between requests to be polite

# Regex for ABS-CBN article URL paths (e.g. /anc/business/article/2024/1/15/...)
_ARTICLE_URL_RE = re.compile(r"/(?:anc|news)/[\w/-]*article/\d{4}/")

# Maximum number of parent elements to traverse when grouping article links
_MAX_PARENT_WALK_DEPTH = 4


def _get_random_ua() -> str:
    """Return a random browser User-Agent string.

    Uses the ``fake-useragent`` library when available so that the scraper
    does not present an obviously static header to the target server.
    Falls back gracefully to a hard-coded Chrome string if the library is
    not installed.
    """
    try:
        from fake_useragent import UserAgent  # type: ignore[import]

        return UserAgent().random
    except Exception:  # library absent or network error during UA update
        return _FALLBACK_UA


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
    meta: dict = field(default_factory=dict)

    def get_text_for_analysis(self) -> str:
        """Return the best available text for sentiment analysis."""
        if self.content:
            # Truncate to avoid exceeding model token limits
            return self.content[:512]
        if self.summary:
            return self.summary
        return self.title



def _has_card_class(tag, keyword: str) -> bool:
    """Return True when *tag* has a CSS class containing *keyword*."""
    classes = tag.get("class") or []
    return any(keyword in c.lower() for c in classes)


class ANCNewsScraper:
    """Scraper for ABS-CBN ANC financial news articles.

    Uses a headless Chromium browser (Playwright) to render the JavaScript-
    driven ABS-CBN ANC pages before extracting article data.  Run
    ``playwright install chromium`` once after installing the package.
    """

    def __init__(self, delay: float = REQUEST_DELAY, data_lake=None):
        self.delay = delay
        self.data_lake = data_lake

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
        Fetch and attach the full body text and head metadata of *article*
        (in-place).

        Args:
            article: Article whose ``content`` and ``meta`` fields will be
                     populated.

        Returns:
            The same article instance, now with ``content`` and ``meta`` filled.
        """
        try:
            time.sleep(self.delay)
            html = self._fetch_html(article.url)
            if html is None:
                return article
            soup = BeautifulSoup(html, "html.parser")
            article.content = self._extract_body(soup)
            article.meta = self._extract_meta(soup)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch %s: %s", article.url, exc)
        return article

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_html(self, url: str) -> Optional[str]:
        """Return the fully-rendered HTML for *url* using a headless browser.

        Uses Playwright's Chromium in headless mode so that JavaScript-driven
        content is executed before the DOM is captured.  Returns ``None`` on
        any failure (network error, Playwright not installed, timeout, …).

        When a :class:`~data_lake.DataLake` was provided at construction time
        the raw HTML is saved via :meth:`~data_lake.DataLake.save_raw_html`
        before being returned.

        This method is intentionally thin so that tests can patch it easily::

            with patch("scraper.ANCNewsScraper._fetch_html", return_value=html):
                articles = scraper.get_articles()
        """
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import]

            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                try:
                    ctx = browser.new_context(
                        user_agent=_get_random_ua(),
                        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                    )
                    page = ctx.new_page()
                    page.goto(url, timeout=30_000, wait_until="domcontentloaded")
                    # Wait for JS to populate the article list; fall through on
                    # timeout rather than raising so we still get whatever is there.
                    try:
                        page.wait_for_load_state("networkidle", timeout=15_000)
                    except Exception:  # noqa: BLE001
                        pass
                    html = page.content()
                finally:
                    browser.close()
        except ImportError:
            logger.error(
                "Playwright is not installed.  "
                "Run: pip install playwright && playwright install chromium"
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Playwright fetch failed for %s: %s", url, exc)
            return None

        if html and self.data_lake is not None:
            try:
                self.data_lake.save_raw_html(url, html)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Data lake raw HTML save failed for %s: %s", url, exc)

        return html

    def _scrape_listing(
        self,
        url: str,
        max_articles: int,
        exclude_urls: set[str] | None = None,
    ) -> list[NewsArticle]:
        """Scrape article cards from a listing page."""
        exclude_urls = exclude_urls or set()
        articles: list[NewsArticle] = []

        time.sleep(self.delay)
        html = self._fetch_html(url)
        if html is None:
            logger.error("Failed to fetch listing page %s", url)
            return articles

        soup = BeautifulSoup(html, "html.parser")
        cards = self._find_article_cards(soup)

        for card in cards:
            if len(articles) >= max_articles:
                break
            article = self._parse_card(card)
            if article and article.url not in exclude_urls:
                articles.append(article)

        return articles

    def _find_article_cards(self, soup: BeautifulSoup) -> list:
        """Return a list of tag elements that look like article cards.

        Tries several strategies in order so the scraper degrades gracefully
        if ABS-CBN's CSS class names change:

        1. Standard ``<article>`` semantic elements.
        2. Elements whose class name contains common card keywords.
        3. URL-pattern matching — find ``<a>`` tags whose ``href`` looks like
           an article path and return their nearest meaningful parent container.
        """
        # Strategy 1: semantic <article> tags (most reliable)
        cards = soup.find_all("article")
        if cards:
            logger.info("Found %d article elements via <article> tags", len(cards))
            return cards

        # Strategy 2: elements whose CSS class contains a card keyword
        card_keywords = [
            "article-card", "story-card", "news-card",
            "article-item", "news-item", "content-card",
            "card--article", "post-card",
        ]
        for keyword in card_keywords:
            cards = soup.find_all(
                lambda tag, kw=keyword: _has_card_class(tag, kw)
            )
            if cards:
                logger.info(
                    "Found %d article elements via CSS class keyword '%s'",
                    len(cards), keyword,
                )
                return cards

        # Strategy 3: links matching ABS-CBN article URL patterns
        links = soup.find_all("a", href=_ARTICLE_URL_RE)
        if links:
            logger.info(
                "Found %d links matching article URL pattern — collecting containers",
                len(links),
            )
            seen_ids: set[int] = set()
            containers = []
            for link in links:
                # Walk up to find a meaningful container element
                node = link.parent
                for _ in range(_MAX_PARENT_WALK_DEPTH):
                    if node is None or node.name in ("body", "html", "[document]"):
                        break
                    if id(node) not in seen_ids:
                        seen_ids.add(id(node))
                        containers.append(node)
                        break
                    node = node.parent
            if containers:
                return containers

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

        # Look for a dedicated title element (heading or labelled span/div)
        title_tag = (
            card.find(["h1", "h2", "h3", "h4"], attrs={"class": True})
            or card.find(["h1", "h2", "h3", "h4"])
            or card.find(
                ["span", "div", "p"],
                class_=lambda c: c and any(
                    kw in " ".join(c).lower()
                    for kw in ("title", "headline", "heading", "story-name")
                ),
            )
        )
        title = title_tag.get_text(strip=True) if title_tag else link.get_text(strip=True)

        if not title:
            return None

        # --- Summary / Teaser ---
        summary_tag = card.find(
            ["p", "div", "span"],
            class_=lambda c: c and any(
                kw in " ".join(c).lower()
                for kw in ("summary", "description", "teaser", "excerpt", "blurb")
            ),
        ) or card.find("p")
        summary = summary_tag.get_text(strip=True) if summary_tag else ""

        # --- Category ---
        cat_tag = card.find(
            class_=lambda c: c and any(
                kw in " ".join(c).lower()
                for kw in ("category", "section", "tag", "label")
            )
        )
        category = cat_tag.get_text(strip=True) if cat_tag else ""

        # --- Published date ---
        time_tag = card.find("time")
        published_at = None
        if time_tag:
            published_at = time_tag.get("datetime") or time_tag.get_text(strip=True) or None

        return NewsArticle(
            title=title,
            url=href,
            summary=summary,
            category=category,
            published_at=published_at,
        )

    @staticmethod
    def _extract_meta(soup: BeautifulSoup) -> dict:
        """Extract metadata from ``<head>`` meta / link tags.

        Collects standard, Open Graph (``og:``), Twitter Card (``twitter:``),
        and article-namespace (``article:``) properties as well as the
        canonical URL.

        Returns:
            A flat dict mapping lowercased property/name to content string.
        """
        meta: dict = {}
        for tag in soup.find_all("meta"):
            name = tag.get("property") or tag.get("name") or ""
            content = tag.get("content", "")
            if name and content:
                meta[name.lower()] = content
        canonical = soup.find("link", rel="canonical")
        if canonical:
            meta["canonical"] = canonical.get("href", "")
        return meta

    def _extract_body(self, soup: BeautifulSoup) -> str:
        """Extract the main body text from an article page."""
        # Ordered list of CSS selectors to try for the article body.
        # ABS-CBN and common news CMS class patterns are included.
        body_selectors = [
            "[class*='article-body']",
            "[class*='story-body']",
            "[class*='article-content']",
            "[class*='story-content']",
            "[class*='entry-content']",
            "[class*='post-content']",
            "[class*='news-content']",
            "div.content",
            "main article",
            "article",
        ]
        for selector in body_selectors:
            body = soup.select_one(selector)
            if body:
                # Remove script/style/ad noise
                for tag in body.find_all(["script", "style", "figure", "aside", "nav"]):
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
    scraper = ANCNewsScraper(data_lake=data_lake)
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
                    data_lake.save_preprocessed_article(article)
                    data_lake.save_cleaned_article(article)
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
