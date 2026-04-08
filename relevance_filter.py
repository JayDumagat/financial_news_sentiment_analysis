"""
Financial news relevance filter.

Classifies a :class:`~scraper.NewsArticle` as financially relevant or not
using lightweight keyword scoring.  Non-relevant articles are cached in memory
(and optionally persisted to a :class:`~database.NewsDatabase`) so that the
same URL is never scored twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scraper import NewsArticle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword dictionary  (keyword, weight)
# Higher weight → stronger signal of financial relevance
# ---------------------------------------------------------------------------

FINANCIAL_KEYWORDS: list[tuple[str, float]] = [
    # Macro / monetary
    ("BSP", 3.0),
    ("Bangko Sentral", 3.0),
    ("interest rate", 3.0),
    ("monetary policy", 3.0),
    ("inflation", 2.5),
    ("GDP", 3.0),
    ("gross domestic product", 3.0),
    ("balance of payments", 3.0),
    ("foreign direct investment", 3.0),
    ("FDI", 2.5),
    ("OFW remittance", 2.5),
    ("forex", 2.5),
    ("foreign exchange", 2.5),
    ("peso depreciation", 3.0),
    ("USD/PHP", 2.5),
    # Markets / securities
    ("stock", 3.0),
    ("shares", 2.5),
    ("PSE", 3.0),
    ("IPO", 2.5),
    ("listing", 1.5),
    ("dividend", 2.5),
    ("earnings", 2.5),
    # Corporate
    ("profit", 2.5),
    ("revenue", 2.5),
    ("net income", 2.5),
    ("quarterly", 1.5),
    ("annual report", 2.5),
    ("merger", 2.5),
    ("acquisition", 2.5),
    ("bankruptcy", 2.5),
    # Banks / credit
    ("bank", 2.0),
    ("banking", 2.0),
    ("credit", 1.5),
    ("loan", 1.5),
    ("debt", 1.5),
    ("bond", 2.0),
    ("treasury", 2.0),
    # Government / regulation
    ("budget", 2.0),
    ("fiscal", 2.5),
    ("tax", 1.5),
    ("BIR", 2.0),
    ("DTI", 2.0),
    ("DOF", 2.0),
    ("NEDA", 2.0),
    ("SEC Philippines", 2.5),
    ("tariff", 2.0),
    ("trade deficit", 2.5),
    # General economic
    ("economy", 2.0),
    ("economic", 2.0),
    ("investment", 2.0),
    ("market", 1.5),
    ("financial", 2.0),
    ("peso", 2.0),
    ("remittance", 2.0),
    ("export", 1.5),
    ("import", 1.5),
    ("REIT", 2.5),
    ("real estate", 1.5),
    ("energy cost", 2.0),
    ("oil price", 2.0),
    ("power rate", 2.0),
]

# News categories that strongly indicate financial content
FINANCIAL_CATEGORIES: frozenset[str] = frozenset(
    {
        "business",
        "economy",
        "finance",
        "markets",
        "money",
        "investing",
        "stocks",
        "banking",
        "trade",
        "economic",
    }
)

# Score must reach this threshold for an article to be considered financial
RELEVANCE_THRESHOLD: float = 3.0

# Minimum number of *distinct* financial keywords that must appear in the text.
# A single brand mention (e.g. "San Miguel") can push the score above the
# threshold via PSE-stock keywords, but without at least this many genuine
# financial-context words the article is treated as non-financial.
MIN_FINANCIAL_KEYWORD_COUNT: int = 2


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RelevanceResult:
    """Outcome of a relevance check."""

    is_financial: bool
    score: float
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Filter class
# ---------------------------------------------------------------------------


class RelevanceFilter:
    """
    Determines whether a news article is financially relevant.

    Results are cached by URL so repeated calls for the same article are free.
    An optional :class:`~database.NewsDatabase` instance may be supplied; if
    provided, the filter also reads from / writes to the database's
    ``relevance_cache`` table so decisions survive process restarts.

    Args:
        threshold: Minimum keyword score to be classified as financial.
                   Defaults to :data:`RELEVANCE_THRESHOLD`.
        db: Optional database instance for persistent caching.
    """

    def __init__(
        self,
        threshold: float = RELEVANCE_THRESHOLD,
        min_keyword_count: int = MIN_FINANCIAL_KEYWORD_COUNT,
        db=None,
    ):
        self.threshold = threshold
        self.min_keyword_count = min_keyword_count
        self._db = db
        self._cache: dict[str, RelevanceResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, article: "NewsArticle") -> RelevanceResult:
        """
        Return a :class:`RelevanceResult` for *article*.

        Hits the in-memory cache first, then the DB cache (if configured),
        and finally runs a full evaluation.
        """
        # 1. In-memory cache
        if article.url in self._cache:
            logger.debug("Relevance cache hit (memory) for %s", article.url)
            return self._cache[article.url]

        # 2. DB cache
        if self._db is not None:
            cached = self._db.get_cached_relevance(article.url)
            if cached is not None:
                result = RelevanceResult(is_financial=cached, score=0.0, reasons=["db-cache"])
                self._cache[article.url] = result
                logger.debug("Relevance cache hit (db) for %s", article.url)
                return result

        # 3. Full evaluation
        result = self._evaluate(article)
        self._cache[article.url] = result

        if self._db is not None:
            self._db.cache_relevance(article.url, result.is_financial, result.score)

        logger.debug(
            "Relevance '%s': %s (score=%.2f)",
            article.title[:60],
            "financial" if result.is_financial else "non-financial",
            result.score,
        )
        return result

    def is_financial(self, article: "NewsArticle") -> bool:
        """Convenience wrapper — returns ``True`` if the article is financial."""
        return self.check(article).is_financial

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate(self, article: "NewsArticle") -> RelevanceResult:
        score = 0.0
        reasons: list[str] = []
        keyword_hits: int = 0
        has_financial_category = False

        # Category boost
        if article.category and article.category.lower() in FINANCIAL_CATEGORIES:
            score += 5.0
            has_financial_category = True
            reasons.append(f"category={article.category!r}")

        # Keyword scoring over title + summary + content prefix
        combined = " ".join(
            filter(None, [article.title, article.summary, (article.content or "")[:500]])
        )
        combined_lower = combined.lower()

        for keyword, weight in FINANCIAL_KEYWORDS:
            if keyword.lower() in combined_lower:
                score += weight
                keyword_hits += 1
                reasons.append(f"keyword={keyword!r}")

        # The minimum-keyword-count gate guards against lifestyle / entertainment
        # articles that mention a brand (e.g. "celebrity drinks San Miguel") but
        # contain no genuine financial vocabulary.  When the article's *category*
        # is already classified as financial we trust that classification and skip
        # the count gate — it only applies to keyword-only passes.
        passes_keyword_count = has_financial_category or keyword_hits >= self.min_keyword_count

        return RelevanceResult(
            is_financial=score >= self.threshold and passes_keyword_count,
            score=score,
            reasons=reasons,
        )
