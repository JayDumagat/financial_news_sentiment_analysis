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
* A minimum confidence threshold is applied so borderline results are
  reported as neutral rather than forced into a positive/negative bucket.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from pse_stocks import find_affected_stocks

logger = logging.getLogger(__name__)

# Default HuggingFace model identifier for FinBERT
FINBERT_MODEL = "ProsusAI/finbert"

# Sentiment label mapping returned by FinBERT
SENTIMENT_LABELS = {"positive", "negative", "neutral"}

# If neither positive nor negative confidence exceeds this threshold the
# result is reported as neutral regardless of the raw model output.
MIN_DIRECTIONAL_CONFIDENCE: float = 0.55


@dataclass
class SentimentResult:
    """Holds the sentiment analysis result for a single text input."""

    text: str
    label: str          # 'positive' | 'negative' | 'neutral'
    score: float        # confidence score in [0, 1]
    all_scores: dict    # {'positive': float, 'negative': float, 'neutral': float}
    affected_stocks: list = field(default_factory=list)
    """PSE-listed stocks potentially impacted by this news item."""

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
            f"[{self.label.upper():8s}  {self.score:.2%}]  "
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
                best = max(label_scores, key=lambda x: x["score"])
                label = best["label"].lower()
                score = best["score"]

                # Apply minimum directional confidence: if the winning
                # label's confidence doesn't clearly clear the threshold,
                # report as neutral to avoid noisy positive/negative calls.
                if label in ("positive", "negative") and score < MIN_DIRECTIONAL_CONFIDENCE:
                    label = "neutral"
                    score = all_scores.get("neutral", score)

                stocks = find_affected_stocks(src_text)

                results.append(
                    SentimentResult(
                        text=text,
                        label=label,
                        score=score,
                        all_scores=all_scores,
                        affected_stocks=stocks,
                    )
                )

        return results

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
