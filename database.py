"""
SQLite database handler for the financial news sentiment analysis pipeline.

Tables
------
articles
    Raw scraped article data, keyed by URL.

sentiment_results
    One row per analysis run, foreign-keyed to ``articles.url``.

relevance_cache
    Persists the relevance-filter decision for each URL so non-financial
    articles are skipped on subsequent runs without re-scoring.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scraper import NewsArticle
    from sentiment_analyzer import SentimentResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    url          TEXT    PRIMARY KEY,
    title        TEXT    NOT NULL,
    summary      TEXT    DEFAULT '',
    content      TEXT    DEFAULT '',
    category     TEXT    DEFAULT '',
    published_at TEXT    DEFAULT '',
    tags         TEXT    DEFAULT '[]',
    scraped_at   TEXT    NOT NULL,
    is_financial INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sentiment_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_url     TEXT    NOT NULL,
    label           TEXT    NOT NULL,
    score           REAL    NOT NULL,
    all_scores      TEXT    NOT NULL,
    affected_stocks TEXT    DEFAULT '[]',
    analyzed_at     TEXT    NOT NULL,
    FOREIGN KEY (article_url) REFERENCES articles (url)
);

CREATE TABLE IF NOT EXISTS relevance_cache (
    url          TEXT    PRIMARY KEY,
    is_financial INTEGER NOT NULL,
    score        REAL    DEFAULT 0.0,
    cached_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_articles_scraped_at
    ON articles (scraped_at);

CREATE INDEX IF NOT EXISTS idx_results_article_url
    ON sentiment_results (article_url);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class NewsDatabase:
    """
    Lightweight SQLite-backed store for news articles and sentiment results.

    The database file is created automatically on first use.  All writes are
    committed immediately so the store can be safely read by other processes.

    Args:
        db_path: Path to the SQLite file.  Defaults to ``news.db`` in the
                 current working directory.
    """

    def __init__(self, db_path: str = "news.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    # ------------------------------------------------------------------
    # Article operations
    # ------------------------------------------------------------------

    def save_article(self, article: "NewsArticle", is_financial: bool = True) -> None:
        """Insert or replace a scraped article record."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO articles
                (url, title, summary, content, category, published_at,
                 tags, scraped_at, is_financial)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article.url,
                article.title,
                article.summary,
                article.content,
                article.category,
                article.published_at or "",
                json.dumps(getattr(article, "tags", [])),
                now,
                int(is_financial),
            ),
        )
        conn.commit()
        logger.debug("DB: saved article %s", article.url)

    def is_seen(self, url: str) -> bool:
        """Return ``True`` if the URL already exists in the ``articles`` table."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM articles WHERE url = ?", (url,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Sentiment result operations
    # ------------------------------------------------------------------

    def save_sentiment_result(
        self,
        article_url: str,
        result: "SentimentResult",
    ) -> None:
        """Insert a sentiment analysis result linked to an article URL."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        affected = getattr(result, "affected_stocks", [])
        conn.execute(
            """
            INSERT INTO sentiment_results
                (article_url, label, score, all_scores, affected_stocks, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                article_url,
                result.label,
                result.score,
                json.dumps(result.all_scores),
                json.dumps(affected),
                now,
            ),
        )
        conn.commit()
        logger.debug("DB: saved sentiment result for %s", article_url)

    # ------------------------------------------------------------------
    # Relevance cache operations
    # ------------------------------------------------------------------

    def get_cached_relevance(self, url: str) -> Optional[bool]:
        """
        Return the cached relevance decision for *url*, or ``None`` if the
        URL has not been cached yet.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT is_financial FROM relevance_cache WHERE url = ?", (url,)
        ).fetchone()
        if row is None:
            return None
        return bool(row["is_financial"])

    def cache_relevance(self, url: str, is_financial: bool, score: float = 0.0) -> None:
        """Persist the relevance decision for *url*."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO relevance_cache (url, is_financial, score, cached_at)
            VALUES (?, ?, ?, ?)
            """,
            (url, int(is_financial), score, now),
        )
        conn.commit()
