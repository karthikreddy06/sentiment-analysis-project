"""
Watson NLP Sentiment Analysis Provider.

Implements sentiment analysis by dispatching HTTP POST requests to the
IBM Watson NLP BERT sentiment predict runtime service.
"""

import json
import logging
import os
from typing import Any, Dict, Tuple
import requests
from .base import BaseSentimentProvider

logger = logging.getLogger(__name__)

DEFAULT_WATSON_URL = (
    "https://sn-watson-sentiment-bert.labs.skills.network"
    "/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict"
)
DEFAULT_MODEL_ID = "sentiment_aggregated-bert-workflow_lang_multi_stock"
DEFAULT_TIMEOUT_SEC = 10


class WatsonNLPProvider(BaseSentimentProvider):
    """
    Watson NLP BERT sentiment prediction provider over REST API.
    """

    @property
    def provider_name(self) -> str:
        return "watson"

    def _get_config(self) -> Tuple[str, Dict[str, str], int]:
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

    @staticmethod
    def _fallback_result(provider_name: str) -> Dict[str, Any]:
        """Returns standardized fallback result for service unavailable."""
        return {
            "label": None,
            "score": None,
            "provider": provider_name,
            "status": "service_unavailable"
        }

    @staticmethod
    def _error_result(provider_name: str) -> Dict[str, Any]:
        """Returns standardized error result."""
        return {
            "label": None,
            "score": None,
            "provider": provider_name,
            "status": "error"
        }

    @staticmethod
    def _invalid_input_result(provider_name: str) -> Dict[str, Any]:
        """Returns standardized invalid input result."""
        return {
            "label": None,
            "score": None,
            "provider": provider_name,
            "status": "invalid_input"
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Calls Watson NLP API and normalizes sentiment response.
        """
        # pylint: disable=too-many-locals
        provider = self.provider_name
        url, headers, timeout = self._get_config()
        payload = {"raw_document": {"text": text}}
        result: Dict[str, Any] = self._fallback_result(provider)

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
                raw_label = doc_sentiment.get("label")
                score = doc_sentiment.get("score")

                if raw_label is not None and score is not None:
                    clean_label = raw_label.replace("SENT_", "").strip().upper()
                    result = {
                        "label": clean_label,
                        "score": round(float(score), 4),
                        "provider": provider,
                        "status": "success"
                    }
                else:
                    logger.warning(
                        "Watson response missing documentSentiment fields: %s",
                        response_data
                    )
                    result = self._error_result(provider)

            elif response.status_code == 400:
                logger.warning("Watson returned HTTP 400 Bad Request.")
                result = self._invalid_input_result(provider)
            else:
                logger.error("Watson returned HTTP status %d", response.status_code)

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as net_err:
            logger.error("Watson network connectivity failure: %s", net_err)
        except requests.exceptions.RequestException as req_err:
            logger.error("Watson request exception: %s", req_err)
        except (ValueError, KeyError, TypeError) as parse_err:
            logger.error("Failed to parse Watson response: %s", parse_err)
            result = self._error_result(provider)

        return result
