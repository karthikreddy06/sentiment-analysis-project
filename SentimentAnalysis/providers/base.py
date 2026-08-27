"""
Base Sentiment Provider Abstract Interface.

Defines the contract for all sentiment analysis providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseSentimentProvider(ABC):
    """
    Abstract Base Class for sentiment analysis providers.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the unique identifier of the provider."""

    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the provided text.

        Args:
            text: Validated string to analyze.

        Returns:
            Dict containing:
                - 'label': str | None ('POSITIVE', 'NEGATIVE', 'NEUTRAL')
                - 'score': float | None
                - 'provider': str
                - 'status': str ('success', 'invalid_input',
                  'service_unavailable', 'error')
        """
