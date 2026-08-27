"""
Sentiment Service Dispatcher and Input Validator.

Manages provider lifecycle and routes analysis requests to the configured backend.
"""

import logging
import os
from typing import Any, Dict, Optional
from .providers.base import BaseSentimentProvider
from .providers.local_provider import LocalTransformerProvider
from .providers.watson_provider import WatsonNLPProvider

logger = logging.getLogger(__name__)

# Maximum permissible input characters to guard against DOS / memory exhaustion
MAX_TEXT_LENGTH = 5000


class SentimentService:
    """
    Central orchestration service for sentiment analysis.
    """

    def __init__(self, provider: Optional[BaseSentimentProvider] = None):
        if provider is not None:
            self._provider = provider
        else:
            self._provider = self._resolve_provider()

    @staticmethod
    def _resolve_provider() -> BaseSentimentProvider:
        provider_key = os.getenv("SENTIMENT_PROVIDER", "").strip().lower()
        if not provider_key:
            if os.getenv("VERCEL") and os.getenv("SENTIMENT_ANALYSIS_API_URL"):
                provider_key = "local"
            elif os.getenv("VERCEL"):
                provider_key = "watson"
            else:
                provider_key = "local"
            
        if provider_key == "local":
            logger.info("Initializing SentimentService with 'LocalTransformerProvider'.")
            return LocalTransformerProvider()
        if provider_key == "watson":
            logger.info("Initializing SentimentService with 'WatsonNLPProvider'.")
            return WatsonNLPProvider()

        error_msg = (
            f"Invalid SENTIMENT_PROVIDER configured: '{provider_key}'. "
            "Supported values are 'local' or 'watson'."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    @property
    def provider(self) -> BaseSentimentProvider:
        """Returns the active provider instance."""
        return self._provider

    @property
    def provider_name(self) -> str:
        """Returns active provider identifier."""
        return self._provider.provider_name

    def validate_input(self, text: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Validates text input before dispatching to provider.
        Returns error dict if invalid, else None.
        """
        if text is None or not isinstance(text, str) or not text.strip():
            logger.warning("Rejected empty or non-string input.")
            return {
                "label": None,
                "score": None,
                "provider": self.provider_name,
                "status": "invalid_input"
            }

        if len(text) > MAX_TEXT_LENGTH:
            logger.warning("Input exceeds max length (%d > %d).", len(text), MAX_TEXT_LENGTH)
            return {
                "label": None,
                "score": None,
                "provider": self.provider_name,
                "status": "invalid_input"
            }

        return None

    def analyze(self, text: Optional[str]) -> Dict[str, Any]:
        """
        Validates input and performs sentiment analysis using the configured provider.
        """
        validation_error = self.validate_input(text)
        if validation_error is not None:
            return validation_error

        # Guaranteed valid non-empty string
        sanitized_text = text.strip()  # type: ignore[union-attr]
        return self._provider.analyze(sanitized_text)


_service_instance: Optional[SentimentService] = None


def get_sentiment_service(reload_config: bool = False) -> SentimentService:
    """
    Returns the singleton SentimentService instance.
    """
    global _service_instance  # pylint: disable=global-statement
    if _service_instance is None or reload_config:
        _service_instance = SentimentService()
    return _service_instance
