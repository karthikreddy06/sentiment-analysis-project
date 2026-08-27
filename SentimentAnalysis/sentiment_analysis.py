"""
Public Sentiment Analysis Interface.

Exposes the unified sentiment_analyzer() entrypoint routing to the
configured backend provider (Local Transformer or Watson NLP).
"""

from typing import Any, Dict, Optional
from .service import get_sentiment_service


def sentiment_analyzer(text_to_analyse: Optional[str]) -> Dict[str, Any]:
    """
    Analyzes sentiment of the supplied text using the actively configured provider.

    Args:
        text_to_analyse: String to evaluate.

    Returns:
        dict: Standardized sentiment dictionary containing:
            - 'label': 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' | None
            - 'score': float | None
            - 'provider': 'local' | 'watson'
            - 'status': 'success' | 'invalid_input' | 'service_unavailable' | 'error'
    """
    service = get_sentiment_service()
    return service.analyze(text_to_analyse)
