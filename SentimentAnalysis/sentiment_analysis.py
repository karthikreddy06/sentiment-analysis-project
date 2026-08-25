"""
Watson NLP Sentiment Analysis Module.

This module provides an interface to analyze sentiment using the Watson NLP
BERT model over HTTP POST requests.
"""

import json
import logging
from typing import Any, Dict, Optional
import requests

# Configure module-level logger
logger = logging.getLogger(__name__)

# Watson NLP Sentiment Analysis BERT Service Endpoint & Header Configuration
WATSON_URL = (
    "https://sn-watson-sentiment-bert.labs.skills.network"
    "/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict"
)
WATSON_HEADERS = {
    "grpc-metadata-mm-model-id": (
        "sentiment_aggregated-bert-workflow_lang_multi_stock"
    ),
    "Content-Type": "application/json",
}
REQUEST_TIMEOUT = 10  # Timeout in seconds


def sentiment_analyzer(text_to_analyse: Optional[str]) -> Dict[str, Any]:
    """
    Analyzes the sentiment of the provided text using Watson NLP BERT.

    Args:
        text_to_analyse: A string containing text to be evaluated.

    Returns:
        dict: A dictionary containing:
            - 'label': Sentiment label (e.g. 'SENT_POSITIVE', 'SENT_NEGATIVE',
              'SENT_NEUTRAL' or 'POSITIVE', etc.) or None on failure.
            - 'score': Confidence score float (e.g. 0.987) or None on failure.
    """
    result = {"label": None, "score": None}

    # Validate input
    if not text_to_analyse or not isinstance(text_to_analyse, str):
        logger.warning("Empty or invalid input provided to sentiment_analyzer.")
        return result

    payload = {
        "raw_document": {
            "text": text_to_analyse
        }
    }

    try:
        response = requests.post(
            WATSON_URL,
            headers=WATSON_HEADERS,
            data=json.dumps(payload),
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 500:
            logger.error("Watson API returned status 500 (Internal Server Error).")
        elif response.status_code != 200:
            logger.warning(
                "Watson API returned unexpected status code: %d",
                response.status_code
            )
        else:
            response_data = response.json()
            doc_sentiment = response_data.get("documentSentiment", {})
            label = doc_sentiment.get("label")
            score = doc_sentiment.get("score")

            if label is not None and score is not None:
                result = {"label": label, "score": score}
            else:
                logger.warning(
                    "Sentiment fields missing from response: %s", response_data
                )

    except requests.exceptions.Timeout:
        logger.error("Request to Watson Sentiment API timed out.")
    except requests.exceptions.ConnectionError:
        logger.error("Connection error occurred while connecting to Watson API.")
    except requests.exceptions.RequestException as err:
        logger.error("Watson API request exception encountered: %s", err)
    except (ValueError, KeyError, TypeError) as parse_err:
        logger.error("Error parsing Watson API JSON response: %s", parse_err)

    return result
