"""
Unit tests for the public sentiment_analyzer function and SentimentService.
Tests provider dispatching, Watson failure modes, input validation, and boundary conditions.
"""

import unittest
from unittest.mock import patch, MagicMock
import requests
from SentimentAnalysis.sentiment_analysis import sentiment_analyzer
from SentimentAnalysis.service import SentimentService, get_sentiment_service
from SentimentAnalysis.providers.watson_provider import WatsonNLPProvider


class TestSentimentAnalysis(unittest.TestCase):
    """Test suite for public sentiment analysis API and services."""

    def test_empty_and_whitespace_input(self):
        """Test input validation for empty, None, and whitespace strings."""
        res_none = sentiment_analyzer(None)
        self.assertEqual(res_none["status"], "invalid_input")
        self.assertIsNone(res_none["label"])

        res_empty = sentiment_analyzer("")
        self.assertEqual(res_empty["status"], "invalid_input")
        self.assertIsNone(res_empty["label"])

        res_ws = sentiment_analyzer("    ")
        self.assertEqual(res_ws["status"], "invalid_input")
        self.assertIsNone(res_ws["label"])

    def test_excessively_long_input(self):
        """Test validation when input exceeds max allowed characters."""
        long_text = "A" * 6000
        result = sentiment_analyzer(long_text)
        self.assertEqual(result["status"], "invalid_input")
        self.assertIsNone(result["label"])

    def test_invalid_provider_configuration(self):
        """Test that configuring an unknown provider raises a clean ValueError."""
        with patch.dict("os.environ", {"SENTIMENT_PROVIDER": "unknown_provider"}):
            with self.assertRaises(ValueError):
                SentimentService()

    @patch.object(WatsonNLPProvider, "analyze")
    def test_watson_provider_selection(self, mock_watson_analyze):
        """Test that SENTIMENT_PROVIDER=watson dispatches to WatsonNLPProvider."""
        mock_watson_analyze.return_value = {
            "label": "POSITIVE",
            "score": 0.99,
            "provider": "watson",
            "status": "success"
        }
        with patch.dict("os.environ", {"SENTIMENT_PROVIDER": "watson"}):
            service = SentimentService()
            self.assertEqual(service.provider_name, "watson")
            result = service.analyze("Great service!")
            self.assertEqual(result["provider"], "watson")
            self.assertEqual(result["label"], "POSITIVE")

    @patch("SentimentAnalysis.providers.watson_provider.requests.post")
    def test_watson_provider_success(self, mock_post):
        """Test Watson NLP provider successful response handling."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "documentSentiment": {
                "label": "SENT_POSITIVE",
                "score": 0.9876
            }
        }
        mock_post.return_value = mock_resp

        provider = WatsonNLPProvider()
        result = provider.analyze("I love this!")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["label"], "POSITIVE")
        self.assertAlmostEqual(result["score"], 0.9876, places=4)
        self.assertEqual(result["provider"], "watson")

    @patch("SentimentAnalysis.providers.watson_provider.requests.post")
    def test_watson_provider_timeout(self, mock_post):
        """Test Watson NLP provider timeout reporting."""
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        provider = WatsonNLPProvider()
        result = provider.analyze("Sample timeout text")
        self.assertEqual(result["status"], "service_unavailable")
        self.assertIsNone(result["label"])
        self.assertEqual(result["provider"], "watson")

    @patch("SentimentAnalysis.providers.watson_provider.requests.post")
    def test_watson_provider_connection_error(self, mock_post):
        """Test Watson NLP provider connection error reporting."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Unreachable network")

        provider = WatsonNLPProvider()
        result = provider.analyze("Sample connection error text")
        self.assertEqual(result["status"], "service_unavailable")
        self.assertIsNone(result["label"])
        self.assertEqual(result["provider"], "watson")

    @patch("SentimentAnalysis.providers.watson_provider.requests.post")
    def test_watson_provider_http_500(self, mock_post):
        """Test Watson NLP provider HTTP 500 error reporting."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp

        provider = WatsonNLPProvider()
        result = provider.analyze("Sample server error text")
        self.assertEqual(result["status"], "service_unavailable")
        self.assertIsNone(result["label"])

    @patch("SentimentAnalysis.providers.watson_provider.requests.post")
    def test_watson_provider_malformed_json(self, mock_post):
        """Test Watson NLP provider malformed JSON handling."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Corrupt JSON payload")
        mock_post.return_value = mock_resp

        provider = WatsonNLPProvider()
        result = provider.analyze("Sample malformed text")
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["label"])


if __name__ == "__main__":
    unittest.main()
