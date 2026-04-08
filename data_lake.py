"""
Raw data lake for persisting scraped news articles and processed results as
JSON files, organised by ingestion date.

Directory layout
----------------
::

    <base_path>/
        raw/
            <YYYY>/<MM>/<DD>/
                <source>_<url_hash>.json   ← raw article data
        processed/
            <YYYY>/<MM>/<DD>/
                <source>_<url_hash>.json   ← sentiment result + article metadata
"""

from __future__ import annotations

import hashlib
import json
import logging
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
        dest = self._dest_path("raw", article.url, source)
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

    def save_processed_result(
        self,
        article: "NewsArticle",
        result: "SentimentResult",
        source: str = "anc",
    ) -> Path:
        """
        Persist a processed sentiment result alongside its article metadata.

        Args:
            article: The source article.
            result:  The corresponding :class:`~sentiment_analyzer.SentimentResult`.
            source:  Short identifier for the data source.

        Returns:
            :class:`~pathlib.Path` of the written file.
        """
        dest = self._dest_path("processed", article.url, source)
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

    def _dest_path(self, tier: str, url: str, source: str) -> Path:
        """Build the destination file path for *tier* (``"raw"`` or ``"processed"``)."""
        date_str = datetime.now(timezone.utc).strftime(_DATE_FMT)
        filename = f"{source}_{_url_hash(url)}.json"
        return self.base_path / tier / date_str / filename

    @staticmethod
    def _write(dest: Path, payload: dict) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
