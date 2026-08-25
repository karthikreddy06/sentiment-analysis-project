"""
Unit tests for the Watson NLP Sentiment Analysis package.
Tests positive, negative, neutral, timeout, connection failure, HTTP 500,
malformed JSON, and empty input handling using mocked responses.
"""

import unittest
from unittest.mock import patch, MagicMock
import requests
from SentimentAnalysis.sentiment_analysis import sentiment_analyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Test suite for the sentiment_analyzer function."""

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_positive(self, mock_post):
        """Test sentiment analyzer with a positive statement."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "documentSentiment": {
                "label": "SENT_POSITIVE",
                "score": 0.987654
            }
        }
        mock_post.return_value = mock_response

        result = sentiment_analyzer("I love working with Python and building awesome web apps!")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["label"], "SENT_POSITIVE")
        self.assertAlmostEqual(result["score"], 0.987654, places=4)

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_negative(self, mock_post):
        """Test sentiment analyzer with a negative statement."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "documentSentiment": {
                "label": "SENT_NEGATIVE",
                "score": 0.912345
            }
        }
        mock_post.return_value = mock_response

        result = sentiment_analyzer("I am really disappointed with the slow performance and bugs.")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["label"], "SENT_NEGATIVE")
        self.assertAlmostEqual(result["score"], 0.912345, places=4)

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_neutral(self, mock_post):
        """Test sentiment analyzer with a neutral statement."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "documentSentiment": {
                "label": "SENT_NEUTRAL",
                "score": 0.500000
            }
        }
        mock_post.return_value = mock_response

        result = sentiment_analyzer("The server is scheduled to restart at midnight.")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["label"], "SENT_NEUTRAL")
        self.assertAlmostEqual(result["score"], 0.500000, places=4)

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_server_error_500(self, mock_post):
        """Test handling when API returns HTTP 500 error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = sentiment_analyzer("Trigger server failure")
        self.assertEqual(result["status"], "API_ERROR")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["score"])

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_timeout(self, mock_post):
        """Test handling when request times out."""
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        result = sentiment_analyzer("Timeout test message")
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["score"])

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_connection_error(self, mock_post):
        """Test handling when network connection fails."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Network unreachable")

        result = sentiment_analyzer("Connection error message")
        self.assertEqual(result["status"], "CONNECTION_ERROR")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["score"])

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_malformed_json(self, mock_post):
        """Test handling when response JSON is malformed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response

        result = sentiment_analyzer("Malformed response test")
        self.assertEqual(result["status"], "INVALID_RESPONSE")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["score"])

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_missing_fields(self, mock_post):
        """Test handling when response is valid JSON but missing sentiment fields."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unrelated": 123}
        mock_post.return_value = mock_response

        result = sentiment_analyzer("Missing fields test")
        self.assertEqual(result["status"], "INVALID_RESPONSE")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["score"])

    def test_sentiment_analyzer_empty_input(self):
        """Test analyzer with empty string and None."""
        result_empty = sentiment_analyzer("")
        self.assertEqual(result_empty["status"], "INVALID_INPUT")
        self.assertIsNone(result_empty["label"])
        self.assertIsNone(result_empty["score"])

        result_spaces = sentiment_analyzer("   ")
        self.assertEqual(result_spaces["status"], "INVALID_INPUT")
        self.assertIsNone(result_spaces["label"])

        result_none = sentiment_analyzer(None)
        self.assertEqual(result_none["status"], "INVALID_INPUT")
        self.assertIsNone(result_none["label"])


if __name__ == "__main__":
    unittest.main()
