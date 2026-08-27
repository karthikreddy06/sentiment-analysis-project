"""
Hugging Face Inference API Sentiment Analysis Provider.

Implements sentiment analysis by dispatching HTTP POST requests to the
Hugging Face Inference API for a 3-class sentiment model.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple
import requests
from .base import BaseSentimentProvider

logger = logging.getLogger(__name__)

DEFAULT_HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{DEFAULT_HF_MODEL}"
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
        model_id = os.getenv("HF_MODEL_ID", DEFAULT_HF_MODEL).strip() or DEFAULT_HF_MODEL
        api_url = os.getenv("SENTIMENT_ANALYSIS_API_URL", "").strip()
        if "api-inference.huggingface.co" in api_url:
            api_url = api_url.replace("https://api-inference.huggingface.co/models/", "https://router.huggingface.co/hf-inference/models/")
            api_url = api_url.replace("api-inference.huggingface.co", "router.huggingface.co/hf-inference")
        if not api_url:
            api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        
        # Read API token from any common environment variable alias
        raw_token = ""
        for env_var in [
            "SENTIMENT_API_TOKEN",
            "HF_TOKEN",
            "HUGGINGFACE_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HF_API_TOKEN",
        ]:
            val = os.getenv(env_var, "").strip()
            if val:
                # Strip outer quotes if wrapped in single or double quotes
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1].strip()
                val = val.strip("\"' \t\r\n")
                if val:
                    raw_token = val
                    break
        
        try:
            timeout = int(os.getenv("HF_TIMEOUT", str(DEFAULT_TIMEOUT_SEC)))
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SEC

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-wait-for-model": "true"
        }
        if raw_token:
            headers["Authorization"] = f"Bearer {raw_token}"
        else:
            logger.warning("No Hugging Face token detected in environment variables (SENTIMENT_API_TOKEN/HF_TOKEN).")

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

    @classmethod
    def _extract_best_prediction(cls, data: Any) -> Optional[Tuple[str, float]]:
        """
        Robustly extracts the best prediction (label, score) from diverse HF response shapes.
        """
        if not data:
            return None

        # Dict response
        if isinstance(data, dict):
            if "label" in data and "score" in data:
                return str(data["label"]), float(data["score"])
            if "error" in data:
                return None
            for nested_key in ("predictions", "output", "data", "results"):
                if nested_key in data and data[nested_key]:
                    return cls._extract_best_prediction(data[nested_key])
            return None

        # List response
        if isinstance(data, list):
            items = data
            if len(data) > 0 and isinstance(data[0], list):
                items = data[0]

            valid_preds = [
                p for p in items
                if isinstance(p, dict) and "label" in p and "score" in p
            ]
            if valid_preds:
                best = max(valid_preds, key=lambda x: float(x.get("score", 0.0)))
                return str(best["label"]), float(best["score"])

        return None

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Calls Hugging Face Inference API and normalizes sentiment response.
        """
        provider = self.provider_name
        url, headers, timeout = self._get_config()
        payload = {"inputs": text}
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
                extracted = self._extract_best_prediction(response_data)
                
                if extracted:
                    raw_label, score = extracted
                    clean_label = self._normalize_label(raw_label)
                    result = {
                        "label": clean_label,
                        "score": round(float(score), 4),
                        "provider": provider,
                        "status": "success"
                    }
                else:
                    logger.warning("Failed to extract predictions from HF response: %s", response_data)
                    result = self._error_result(provider)

            elif response.status_code == 400:
                logger.warning("HF returned HTTP 400 Bad Request: %s", response.text)
                result = self._invalid_input_result(provider)
            elif response.status_code == 401:
                logger.error("HF returned HTTP 401 Unauthorized - invalid or missing Hugging Face API token.")
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