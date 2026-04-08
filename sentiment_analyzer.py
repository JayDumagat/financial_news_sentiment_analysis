"""
FINBERT-based Financial News Sentiment Analyzer

Uses the ProsusAI/finbert model (a BERT model fine-tuned on financial text)
to classify news text as positive, negative, or neutral.

Enhancements over the base FinBERT wrapper
------------------------------------------
* ``SentimentResult`` carries an ``affected_stocks`` list that identifies
  PSE-listed companies likely impacted by the news item.
* :meth:`FinBERTAnalyzer.analyze_batch` accepts an optional ``source_texts``
  argument (the original, un-truncated texts) used for stock matching while
  the truncated version is fed to the model.
* A two-gate noise filter is applied:
  1. Minimum confidence threshold — borderline results fall back to neutral.
  2. Minimum margin check — the winning label must lead the runner-up by at
     least ``MIN_CONFIDENCE_MARGIN`` so that near-tie outputs are not forced
     into a positive/negative call.
* Each result includes a ``strength`` field (``"strong"`` / ``"moderate"`` /
  ``"weak"`` / ``"neutral"``) derived from the winning score and margin.
* **Aspect-based sentiment** — :meth:`FinBERTAnalyzer.analyze_aspects` scores
  the sentiment for each PSE ticker *individually* by extracting the sentences
  that mention that ticker and analysing only that context.  This avoids the
  conglomerate problem where a positive BDO article incorrectly influences the
  sentiment of every other bank triggered by a sector keyword.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from pse_stocks import find_affected_stocks

logger = logging.getLogger(__name__)

# Default HuggingFace model identifier for FinBERT
FINBERT_MODEL = "ProsusAI/finbert"

# Sentiment label mapping returned by FinBERT
SENTIMENT_LABELS = {"positive", "negative", "neutral"}

# Gate 1 — minimum directional confidence.
# If the top label's score is below this the result is reported as neutral.
MIN_DIRECTIONAL_CONFIDENCE: float = 0.55

# Gate 2 — minimum margin between the top label and the runner-up.
# Prevents near-tie outputs (e.g. pos=0.56, neg=0.44) from generating
# positive/negative calls.
MIN_CONFIDENCE_MARGIN: float = 0.15

# Strength thresholds
_STRONG_THRESHOLD: float = 0.80
_MODERATE_THRESHOLD: float = 0.65


@dataclass
class SentimentResult:
    """Holds the sentiment analysis result for a single text input."""

    text: str
    label: str          # 'positive' | 'negative' | 'neutral'
    score: float        # confidence score in [0, 1]
    all_scores: dict    # {'positive': float, 'negative': float, 'neutral': float}
    affected_stocks: list = field(default_factory=list)
    """PSE-listed stocks potentially impacted by this news item."""
    strength: str = "neutral"
    """Signal strength: 'strong' | 'moderate' | 'weak' | 'neutral'."""

    @property
    def is_positive(self) -> bool:
        return self.label == "positive"

    @property
    def is_negative(self) -> bool:
        return self.label == "negative"

    @property
    def is_neutral(self) -> bool:
        return self.label == "neutral"

    def __str__(self) -> str:
        return (
            f"[{self.label.upper():8s}  {self.score:.2%}  {self.strength}]  "
            f"{self.text[:80]}{'...' if len(self.text) > 80 else ''}"
        )


class FinBERTAnalyzer:
    """
    Wrapper around the ProsusAI/finbert HuggingFace model.

    The model and tokenizer are loaded lazily on first use so that importing
    this module does not trigger a slow model download.

    Args:
        model_name: HuggingFace model identifier. Defaults to
                    ``'ProsusAI/finbert'``.
        device: ``'cpu'``, ``'cuda'``, or ``None`` (auto-detect).
        batch_size: Number of texts to process in one forward pass.
        max_length: Maximum token length passed to the tokenizer.
    """

    def __init__(
        self,
        model_name: str = FINBERT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self._pipeline = None  # loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze the sentiment of a single text.

        Args:
            text: The text to analyze.

        Returns:
            A :class:`SentimentResult` with label, confidence scores, and
            a list of potentially affected PSE-listed stocks.
        """
        results = self.analyze_batch([text])
        return results[0]

    def analyze_batch(
        self,
        texts: list[str],
        source_texts: Optional[list[str]] = None,
    ) -> list[SentimentResult]:
        """
        Analyze the sentiment of multiple texts in batches.

        Args:
            texts: List of texts to analyze.  Long strings will be truncated
                   before being passed to the model.
            source_texts: Optional list of *original* (un-truncated) texts
                          used solely for PSE stock matching.  Must be the same
                          length as *texts* when provided.  Defaults to *texts*.

        Returns:
            List of :class:`SentimentResult` objects in the same order.
            Each result includes an ``affected_stocks`` list of PSE tickers
            identified in the corresponding text.
        """
        if not texts:
            return []

        if source_texts is None:
            source_texts = texts

        pipeline = self._get_pipeline()
        results: list[SentimentResult] = []

        # Process in chunks to respect batch_size
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            src_chunk = source_texts[start : start + self.batch_size]

            # Truncate long texts to avoid exceeding token limits
            truncated = [t[: self.max_length * 4] for t in chunk]

            raw_outputs = pipeline(
                truncated,
                truncation=True,
                max_length=self.max_length,
                top_k=None,  # return scores for all labels
            )

            for text, src_text, label_scores in zip(chunk, src_chunk, raw_outputs):
                all_scores = {item["label"].lower(): item["score"] for item in label_scores}
                sorted_scores = sorted(label_scores, key=lambda x: x["score"], reverse=True)
                best = sorted_scores[0]
                label = best["label"].lower()
                score = best["score"]

                # Gate 2 requires a runner-up; if the model returned only one
                # label (unusual but defensive), treat margin as zero.
                if len(sorted_scores) >= 2:
                    margin = score - sorted_scores[1]["score"]
                else:
                    margin = 0.0

                # Gate 1 — minimum directional confidence: if the winning
                # label's confidence doesn't clearly clear the threshold,
                # report as neutral to avoid noisy positive/negative calls.
                # Gate 2 — minimum margin: if the top two labels are too close
                # together the signal is unreliable; fall back to neutral.
                if label in ("positive", "negative") and (
                    score < MIN_DIRECTIONAL_CONFIDENCE
                    or margin < MIN_CONFIDENCE_MARGIN
                ):
                    label = "neutral"
                    score = all_scores.get("neutral", score)

                # Compute signal strength from confidence score
                if label == "neutral":
                    strength = "neutral"
                elif score >= _STRONG_THRESHOLD:
                    strength = "strong"
                elif score >= _MODERATE_THRESHOLD:
                    strength = "moderate"
                else:
                    strength = "weak"

                stocks = find_affected_stocks(src_text)

                results.append(
                    SentimentResult(
                        text=text,
                        label=label,
                        score=score,
                        all_scores=all_scores,
                        affected_stocks=stocks,
                        strength=strength,
                    )
                )

        return results

    def analyze_aspects(
        self,
        text: str,
        stocks: Optional[list[dict]] = None,
        context_chars: int = 400,
    ) -> list[dict]:
        """Perform aspect-based sentiment analysis — one score per PSE ticker.

        Instead of applying a single sentiment score to the entire article
        (which can be misleading for conglomerates), this method:

        1. For each stock in *stocks* (or from :func:`~pse_stocks.find_affected_stocks`),
           extracts the sentences / window of text that **directly mention** that
           ticker or its company name.
        2. Runs FinBERT on each ticker-specific context snippet.
        3. Attaches the per-ticker sentiment as ``aspect_label``,
           ``aspect_score``, and ``aspect_strength`` keys on the stock dict.

        Stocks for which no direct mention is found receive the article-level
        sentiment as a fallback (same as before), clearly marked with
        ``aspect_source = "article"``.

        Args:
            text: Full article text (un-truncated).
            stocks: List of stock dicts as returned by
                    :func:`~pse_stocks.find_affected_stocks`.  When ``None``
                    the stocks are detected automatically.
            context_chars: Characters of text to extract around each ticker
                           mention (default 400 — roughly 2–3 sentences).

        Returns:
            A copy of *stocks* with ``aspect_label``, ``aspect_score``,
            ``aspect_strength``, and ``aspect_source`` fields added.
        """
        if stocks is None:
            stocks = find_affected_stocks(text)

        if not stocks:
            return stocks

        # Article-level sentiment used as fallback
        article_result = self.analyze(text)

        enriched: list[dict] = []
        for stock in stocks:
            stock_copy = dict(stock)

            # Collect mention keywords for this stock
            from pse_stocks import PSE_COMPANIES  # noqa: PLC0415

            ticker = stock["ticker"]
            info = PSE_COMPANIES.get(ticker, {})
            keywords = [ticker] + list(info.get("keywords", []))

            # Extract context windows around each mention
            snippets: list[str] = []
            for kw in keywords:
                for m in re.finditer(re.escape(kw), text, re.IGNORECASE):
                    start = max(0, m.start() - context_chars // 2)
                    end = min(len(text), m.end() + context_chars // 2)
                    snippets.append(text[start:end])
                if snippets:
                    break  # stop at first keyword that has hits

            if snippets:
                # Deduplicate overlapping snippets by joining them
                context = " … ".join(dict.fromkeys(snippets))
                aspect_result = self.analyze(context)
                stock_copy["aspect_label"] = aspect_result.label
                stock_copy["aspect_score"] = aspect_result.score
                stock_copy["aspect_strength"] = aspect_result.strength
                stock_copy["aspect_source"] = "direct"
            else:
                # No direct mention — use article-level sentiment
                stock_copy["aspect_label"] = article_result.label
                stock_copy["aspect_score"] = article_result.score
                stock_copy["aspect_strength"] = article_result.strength
                stock_copy["aspect_source"] = "article"

            enriched.append(stock_copy)

        return enriched

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_pipeline(self):
        """Load and cache the HuggingFace text-classification pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        # Import here so the module can be imported without transformers installed
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for sentiment analysis. "
                "Install it with:  pip install transformers torch"
            ) from exc

        import torch

        if self.device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = self.device

        logger.info(
            "Loading FinBERT model '%s' on device '%s' …",
            self.model_name,
            resolved_device,
        )

        self._pipeline = hf_pipeline(
            task="text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            device=0 if resolved_device == "cuda" else -1,
        )

        logger.info("Model loaded.")
        return self._pipeline


def analyze_text(text: str, model_name: str = FINBERT_MODEL) -> SentimentResult:
    """
    Convenience function: analyze a single text with FinBERT.

    Args:
        text: Financial news text to classify.
        model_name: HuggingFace model identifier.

    Returns:
        :class:`SentimentResult`
    """
    analyzer = FinBERTAnalyzer(model_name=model_name)
    return analyzer.analyze(text)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    samples = [
        "The company reported record profits and raised its full-year guidance.",
        "Shares plummeted after the firm announced massive layoffs and revenue shortfall.",
        "The central bank kept interest rates unchanged at its quarterly meeting.",
    ]

    analyzer = FinBERTAnalyzer()
    for sample in samples:
        result = analyzer.analyze(sample)
        print(result)
