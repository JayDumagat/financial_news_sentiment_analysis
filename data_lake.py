"""
Raw data lake for persisting scraped news articles and processed results as
JSON files, organised by ingestion date.

Directory layout
----------------
::

    <base_path>/
        raw_html/
            <YYYY>/<MM>/<DD>/
                <source>_<url_hash>.html   ← raw Playwright-rendered HTML
        raw/
            <YYYY>/<MM>/<DD>/
                <source>_<url_hash>.json   ← raw article data (JSON)
        preprocessed/
            <YYYY>/<MM>/<DD>/
                <source>_<url_hash>.json   ← structured extraction with
                                             head meta and body fields
        cleaned/
            <YYYY>/<MM>/<DD>/
                <source>_<url_hash>.json   ← cleaned / normalised text fields
        analyzed/
            <YYYY>/<MM>/<DD>/
                <source>_<url_hash>.json   ← sentiment result + article metadata
        processed/
            <YYYY>/<MM>/<DD>/
                <source>_<url_hash>.json   ← legacy alias for analyzed/
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scraper import NewsArticle
    from sentiment_analyzer import SentimentResult

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y/%m/%d"


def _url_hash(url: str) -> str:
    """Return a short, filesystem-safe hash of *url*."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _clean_text(text: str) -> str:
    """Return a normalised copy of *text*.

    Applies NFKC unicode normalisation, collapses runs of whitespace and
    newlines to a single space, and strips leading/trailing whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


class DataLake:
    """
    Stores raw scraped articles and processed sentiment results as JSON files
    organised by date under *base_path*.

    Args:
        base_path: Root directory for the data lake.  Defaults to
                   ``data_lake`` in the current working directory.
    """

    def __init__(self, base_path: str = "data_lake"):
        self.base_path = Path(base_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_raw_article(self, article: "NewsArticle", source: str = "anc") -> Path:
        """
        Persist a raw scraped article to the data lake.

        Args:
            article: The article to persist.
            source:  Short identifier for the data source (e.g. ``"anc"``).

        Returns:
            :class:`~pathlib.Path` of the written file.
        """
        dest = self._dest_path("raw", article.url, source, ext="json")
        payload = {
            "url": article.url,
            "title": article.title,
            "summary": article.summary,
            "content": article.content,
            "category": article.category,
            "published_at": article.published_at,
            "tags": getattr(article, "tags", []),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        self._write(dest, payload)
        logger.debug("Data lake: raw article → %s", dest)
        return dest

    def save_raw_html(self, url: str, html: str, source: str = "anc") -> Path:
        """
        Persist the raw Playwright-rendered HTML for *url* to the data lake.

        Args:
            url:    The URL the HTML was fetched from.
            html:   Full HTML string returned by the browser.
            source: Short identifier for the data source.

        Returns:
            :class:`~pathlib.Path` of the written file.
        """
        dest = self._dest_path("raw_html", url, source, ext="html")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(html, encoding="utf-8")
        logger.debug("Data lake: raw HTML → %s", dest)
        return dest

    def save_preprocessed_article(
        self, article: "NewsArticle", source: str = "anc"
    ) -> Path:
        """
        Persist structured extraction data (including head ``<meta>`` tags)
        for *article* to the ``preprocessed/`` tier.

        Args:
            article: The article to persist.
            source:  Short identifier for the data source.

        Returns:
            :class:`~pathlib.Path` of the written file.
        """
        dest = self._dest_path("preprocessed", article.url, source, ext="json")
        payload = {
            "url": article.url,
            "title": article.title,
            "summary": article.summary,
            "content": article.content,
            "category": article.category,
            "published_at": article.published_at,
            "tags": getattr(article, "tags", []),
            "meta": getattr(article, "meta", {}),
            "preprocessed_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        self._write(dest, payload)
        logger.debug("Data lake: preprocessed article → %s", dest)
        return dest

    def save_cleaned_article(
        self, article: "NewsArticle", source: str = "anc"
    ) -> Path:
        """
        Persist cleaned and normalised text fields for *article* to the
        ``cleaned/`` tier.

        Text fields (title, summary, content, category) are passed through
        :func:`_clean_text` to normalise unicode, collapse whitespace, and
        strip leading/trailing characters.

        Args:
            article: The article whose text fields will be cleaned.
            source:  Short identifier for the data source.

        Returns:
            :class:`~pathlib.Path` of the written file.
        """
        dest = self._dest_path("cleaned", article.url, source, ext="json")
        payload = {
            "url": article.url,
            "title": _clean_text(article.title),
            "summary": _clean_text(article.summary),
            "content": _clean_text(article.content),
            "category": _clean_text(article.category),
            "published_at": article.published_at,
            "tags": [_clean_text(t) for t in getattr(article, "tags", [])],
            "meta": getattr(article, "meta", {}),
            "cleaned_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        self._write(dest, payload)
        logger.debug("Data lake: cleaned article → %s", dest)
        return dest

    def save_analyzed_result(
        self,
        article: "NewsArticle",
        result: "SentimentResult",
        source: str = "anc",
    ) -> Path:
        """
        Persist a sentiment analysis result alongside article metadata to the
        ``analyzed/`` tier.

        Args:
            article: The source article.
            result:  The corresponding :class:`~sentiment_analyzer.SentimentResult`.
            source:  Short identifier for the data source.

        Returns:
            :class:`~pathlib.Path` of the written file.
        """
        dest = self._dest_path("analyzed", article.url, source, ext="json")
        payload = {
            "url": article.url,
            "title": article.title,
            "summary": article.summary,
            "category": article.category,
            "published_at": article.published_at,
            "sentiment": result.label,
            "confidence": result.score,
            "strength": getattr(result, "strength", ""),
            "all_scores": result.all_scores,
            "affected_stocks": getattr(result, "affected_stocks", []),
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        self._write(dest, payload)
        logger.debug("Data lake: analyzed result → %s", dest)
        return dest

    def save_processed_result(
        self,
        article: "NewsArticle",
        result: "SentimentResult",
        source: str = "anc",
    ) -> Path:
        """
        Persist a processed sentiment result alongside its article metadata.

        .. deprecated::
            Prefer :meth:`save_analyzed_result` which writes to the
            ``analyzed/`` tier with a richer payload.  This method is kept
            for backwards compatibility and continues to write to the legacy
            ``processed/`` tier.

        Args:
            article: The source article.
            result:  The corresponding :class:`~sentiment_analyzer.SentimentResult`.
            source:  Short identifier for the data source.

        Returns:
            :class:`~pathlib.Path` of the written file.
        """
        dest = self._dest_path("processed", article.url, source, ext="json")
        payload = {
            "url": article.url,
            "title": article.title,
            "summary": article.summary,
            "category": article.category,
            "published_at": article.published_at,
            "sentiment": result.label,
            "confidence": result.score,
            "all_scores": result.all_scores,
            "affected_stocks": getattr(result, "affected_stocks", []),
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        self._write(dest, payload)
        logger.debug("Data lake: processed result → %s", dest)
        return dest

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dest_path(self, tier: str, url: str, source: str, ext: str = "json") -> Path:
        """Build the destination file path for *tier*."""
        date_str = datetime.now(timezone.utc).strftime(_DATE_FMT)
        filename = f"{source}_{_url_hash(url)}.{ext}"
        return self.base_path / tier / date_str / filename

    @staticmethod
    def _write(dest: Path, payload: dict) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
