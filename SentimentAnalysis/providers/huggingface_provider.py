"""
Hugging Face Inference API Sentiment Analysis Provider.

Implements sentiment analysis by dispatching HTTP POST requests to the
Hugging Face Inference API for a 3-class sentiment model.
"""

import json
import logging
import os
from typing import Any, Dict, Tuple
import requests
from .base import BaseSentimentProvider

logger = logging.getLogger(__name__)

DEFAULT_HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_HF_API_URL = f"https://api-inference.huggingface.co/models/{DEFAULT_HF_MODEL}"
DEFAULT_TIMEOUT_SEC = 30


class HuggingFaceProvider(BaseSentimentProvider):
    """
    Hugging Face Inference API sentiment prediction provider over REST API.
    Uses a model that genuinely supports POSITIVE, NEGATIVE, NEUTRAL classes.
    """

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def _get_config(self) -> Tuple[str, Dict[str, str], int]:
        model_id = os.getenv("HF_MODEL_ID", DEFAULT_HF_MODEL)
        api_url = os.getenv("SENTIMENT_ANALYSIS_API_URL", "").strip()
        if not api_url:
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        api_token = os.getenv("SENTIMENT_API_TOKEN", "").strip()
        
        try:
            timeout = int(os.getenv("HF_TIMEOUT", str(DEFAULT_TIMEOUT_SEC)))
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SEC

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        return api_url, headers, timeout

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

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        """Normalizes HF model labels to POSITIVE, NEGATIVE, NEUTRAL."""
        clean = raw_label.strip().lower()
        if clean in ("positive", "label_2", "pos"):
            return "POSITIVE"
        if clean in ("negative", "label_0", "neg"):
            return "NEGATIVE"
        if clean in ("neutral", "label_1", "neu"):
            return "NEUTRAL"
        return clean.upper()

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Calls Hugging Face Inference API and normalizes sentiment response.
        """
        provider = self.provider_name
        url, headers, timeout = self._get_config()
        payload = {"inputs": text, "options": {"wait_for_model": True}}
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
                
                # HF Inference API returns list of predictions
                if isinstance(response_data, list) and response_data:
                    predictions = response_data[0] if isinstance(response_data[0], list) else response_data
                    
                    if isinstance(predictions, list) and predictions:
                        # Find highest scoring prediction
                        best_pred = max(predictions, key=lambda x: x.get("score", 0))
                        raw_label = best_pred.get("label")
                        score = best_pred.get("score")

                        if raw_label is not None and score is not None:
                            clean_label = self._normalize_label(raw_label)
                            result = {
                                "label": clean_label,
                                "score": round(float(score), 4),
                                "provider": provider,
                                "status": "success"
                            }
                        else:
                            logger.warning("HF response missing label/score fields: %s", response_data)
                            result = self._error_result(provider)
                    else:
                        logger.warning("HF response has empty predictions: %s", response_data)
                        result = self._error_result(provider)
                else:
                    logger.warning("Unexpected HF response format: %s", response_data)
                    result = self._error_result(provider)

            elif response.status_code == 400:
                logger.warning("HF returned HTTP 400 Bad Request: %s", response.text)
                result = self._invalid_input_result(provider)
            elif response.status_code == 401:
                logger.error("HF returned HTTP 401 Unauthorized - check SENTIMENT_API_TOKEN")
                result = self._error_result(provider)
            elif response.status_code == 404:
                logger.error("HF returned HTTP 404 - model not found: %s", url)
                result = self._error_result(provider)
            elif response.status_code == 503:
                logger.warning("HF returned HTTP 503 - model loading: %s", response.text)
                result = self._fallback_result(provider)
            else:
                logger.error("HF returned HTTP status %d: %s", response.status_code, response.text)
                result = self._error_result(provider)

        except requests.exceptions.Timeout as net_err:
            logger.error("HF network timeout: %s", net_err)
        except requests.exceptions.ConnectionError as net_err:
            logger.error("HF network connectivity failure: %s", net_err)
        except requests.exceptions.RequestException as req_err:
            logger.error("HF request exception: %s", req_err)
        except (ValueError, KeyError, TypeError) as parse_err:
            logger.error("Failed to parse HF response: %s", parse_err)
            result = self._error_result(provider)

        return result