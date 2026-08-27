"""
Local Transformer Sentiment Analysis Provider.

Implements sentiment analysis using a genuine local pretrained Hugging Face
Transformers model running on CPU/GPU with model caching.
"""

import logging
import os
import threading
from typing import Any, Dict, Optional
from .base import BaseSentimentProvider

logger = logging.getLogger(__name__)

# Default 3-class English sentiment model (Positive, Neutral, Negative)
DEFAULT_LOCAL_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class LocalTransformerProvider(BaseSentimentProvider):
    """
    Local sentiment provider utilizing Hugging Face Transformers pipeline.
    Implements singleton-style model caching to prevent redundant downloads/loads.
    """

    _pipeline = None
    _lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = (
            model_name or
            os.getenv("LOCAL_MODEL_NAME", DEFAULT_LOCAL_MODEL)
        )

    @property
    def provider_name(self) -> str:
        return "local"

    def _load_pipeline(self):
        """Loads and caches the transformers pipeline in a thread-safe manner."""
        if LocalTransformerProvider._pipeline is None:
            with LocalTransformerProvider._lock:
                if LocalTransformerProvider._pipeline is None:
                    logger.info(
                        "Loading local sentiment model: %s "
                        "(this may take a few seconds on first run)...",
                        self.model_name
                    )
                    try:
                        # Lazy import to avoid overhead when using other providers
                        from transformers import pipeline  # pylint: disable=import-outside-toplevel
                        LocalTransformerProvider._pipeline = pipeline(
                            "sentiment-analysis",
                            model=self.model_name,
                            top_k=None,
                            truncation=True,
                            max_length=512
                        )
                        logger.info(
                            "Local sentiment model '%s' loaded successfully.",
                            self.model_name
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to load local transformers model '%s': %s",
                            self.model_name,
                            exc
                        )
                        raise exc
        return LocalTransformerProvider._pipeline

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        """
        Normalizes various model label representations to POSITIVE, NEGATIVE, NEUTRAL.
        """
        clean = raw_label.strip().upper()
        if "POS" in clean:
            return "POSITIVE"
        if "NEG" in clean:
            return "NEGATIVE"
        if "NEU" in clean:
            return "NEUTRAL"
        return clean

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Executes local neural sentiment analysis and normalizes the prediction.
        """
        fallback_result: Dict[str, Any] = {
            "label": None,
            "score": None,
            "provider": self.provider_name,
            "status": "service_unavailable"
        }

        try:
            classifier = self._load_pipeline()
            # Run inference
            # pylint: disable=not-callable
            predictions = classifier(text)

            # Extract top prediction
            if isinstance(predictions, list) and predictions:
                first_item = predictions[0]
                # If top_k=None returned list of dicts, find highest score
                if isinstance(first_item, list):
                    best_pred = max(first_item, key=lambda x: x.get("score", 0))
                elif isinstance(first_item, dict):
                    best_pred = first_item
                else:
                    best_pred = {"label": "NEUTRAL", "score": 0.5}

                norm_label = self._normalize_label(best_pred.get("label", ""))
                score = round(float(best_pred.get("score", 0.0)), 4)

                return {
                    "label": norm_label,
                    "score": score,
                    "provider": self.provider_name,
                    "status": "success"
                }

            return {
                "label": None,
                "score": None,
                "provider": self.provider_name,
                "status": "error"
            }

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Inference failure on local sentiment provider: %s", exc)
            return fallback_result
