"""
Unit tests for the Watson NLP Sentiment Analysis package.
Tests positive, negative, neutral, and error handling scenarios using mocked responses.
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
        self.assertIsNotNone(result)
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
        self.assertIsNotNone(result)
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
        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "SENT_NEUTRAL")
        self.assertAlmostEqual(result["score"], 0.500000, places=4)

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_server_error(self, mock_post):
        """Test handling when API returns 500 error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = sentiment_analyzer("Trigger server failure")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["score"])

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_timeout(self, mock_post):
        """Test handling when request times out."""
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        result = sentiment_analyzer("Timeout test message")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["score"])

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_connection_error(self, mock_post):
        """Test handling when network connection fails."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Network unreachable")

        result = sentiment_analyzer("Connection error message")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["score"])

    @patch("SentimentAnalysis.sentiment_analysis.requests.post")
    def test_sentiment_analyzer_malformed_json(self, mock_post):
        """Test handling when response contains malformed JSON or missing fields."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unrelated_field": "data"}
        mock_post.return_value = mock_response

        result = sentiment_analyzer("Malformed response test")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["score"])

    def test_sentiment_analyzer_empty_input(self):
        """Test analyzer with empty string and None."""
        result_empty = sentiment_analyzer("")
        self.assertIsNone(result_empty["label"])
        self.assertIsNone(result_empty["score"])

        result_none = sentiment_analyzer(None)
        self.assertIsNone(result_none["label"])
        self.assertIsNone(result_none["score"])


if __name__ == "__main__":
    unittest.main()
