"""
Watson NLP Sentiment Analysis Module.

This module provides an interface to analyze sentiment using the Watson NLP
BERT model over HTTP POST requests with robust error discrimination and
configurable environment parameters.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple
import requests

# Configure module-level logger
logger = logging.getLogger(__name__)

# Configurable Watson Service Settings
DEFAULT_WATSON_URL = (
    "https://sn-watson-sentiment-bert.labs.skills.network"
    "/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict"
)
DEFAULT_MODEL_ID = "sentiment_aggregated-bert-workflow_lang_multi_stock"
DEFAULT_TIMEOUT_SEC = 10


def get_watson_config() -> Tuple[str, Dict[str, str], int]:
    """
    Retrieve Watson service configuration from environment variables or defaults.

    Returns:
        tuple containing (url, headers, timeout_seconds).
    """
    url = os.getenv("WATSON_SENTIMENT_URL", DEFAULT_WATSON_URL)
    model_id = os.getenv("WATSON_MODEL_ID", DEFAULT_MODEL_ID)
    try:
        timeout = int(os.getenv("WATSON_TIMEOUT", str(DEFAULT_TIMEOUT_SEC)))
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SEC

    headers = {
        "grpc-metadata-mm-model-id": model_id,
        "Content-Type": "application/json",
    }
    return url, headers, timeout


def sentiment_analyzer(text_to_analyse: Optional[str]) -> Dict[str, Any]:
    """
    Analyzes the sentiment of the provided text using Watson NLP BERT.

    Args:
        text_to_analyse: A string containing text to be evaluated.

    Returns:
        dict: A dictionary containing:
            - 'label': Sentiment label or None.
            - 'score': Confidence score float or None.
            - 'status': Status indicator ('SUCCESS', 'INVALID_INPUT', 'TIMEOUT',
              'CONNECTION_ERROR', 'API_ERROR', 'INVALID_RESPONSE').
    """
    # Validate input
    if (text_to_analyse is None or not isinstance(text_to_analyse, str) or
            not text_to_analyse.strip()):
        logger.warning("Empty or non-string input provided to sentiment_analyzer.")
        return {"label": None, "score": None, "status": "INVALID_INPUT"}

    url, headers, timeout = get_watson_config()
    payload = {"raw_document": {"text": text_to_analyse}}
    result: Dict[str, Any] = {"label": None, "score": None, "status": "API_ERROR"}

    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=timeout,
        )

        if response.status_code == 200:
            response_data = response.json()
            doc_sentiment = response_data.get("documentSentiment", {})
            label = doc_sentiment.get("label")
            score = doc_sentiment.get("score")

            if label is not None and score is not None:
                result = {"label": label, "score": score, "status": "SUCCESS"}
            else:
                logger.warning("Sentiment fields missing in response: %s", response_data)
                result = {"label": None, "score": None, "status": "INVALID_RESPONSE"}
        elif response.status_code == 400:
            logger.warning("Watson API returned 400 Bad Request.")
            result = {"label": None, "score": None, "status": "INVALID_INPUT"}
        else:
            logger.error("Watson API returned HTTP %d", response.status_code)
            result = {"label": None, "score": None, "status": "API_ERROR"}

    except requests.exceptions.Timeout:
        logger.error("Request to Watson Sentiment API timed out after %d seconds.", timeout)
        result = {"label": None, "score": None, "status": "TIMEOUT"}
    except requests.exceptions.ConnectionError as conn_err:
        logger.error("Connection error while reaching Watson Sentiment API: %s", conn_err)
        result = {"label": None, "score": None, "status": "CONNECTION_ERROR"}
    except requests.exceptions.RequestException as req_err:
        logger.error("Watson API request exception: %s", req_err)
        result = {"label": None, "score": None, "status": "API_ERROR"}
    except (ValueError, KeyError, TypeError) as parse_err:
        logger.error("Failed to parse Watson API JSON response: %s", parse_err)
        result = {"label": None, "score": None, "status": "INVALID_RESPONSE"}

    return result
