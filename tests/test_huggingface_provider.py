"""
Unit tests for the Hugging Face Inference API sentiment analysis provider.
"""

import unittest
from unittest.mock import patch, MagicMock
import requests
from SentimentAnalysis.providers.huggingface_provider import HuggingFaceProvider


class TestHuggingFaceProvider(unittest.TestCase):
    """Test suite for HuggingFaceProvider."""

    def setUp(self):
        self.provider = HuggingFaceProvider()

    def test_provider_name(self):
        """Test provider identification name."""
        self.assertEqual(self.provider.provider_name, "huggingface")

    def test_normalize_labels(self):
        """Test normalization of various raw model output labels."""
        self.assertEqual(self.provider._normalize_label("positive"), "POSITIVE")
        self.assertEqual(self.provider._normalize_label("negative"), "NEGATIVE")
        self.assertEqual(self.provider._normalize_label("neutral"), "NEUTRAL")
        self.assertEqual(self.provider._normalize_label("POS"), "POSITIVE")
        self.assertEqual(self.provider._normalize_label("NEG"), "NEGATIVE")
        self.assertEqual(self.provider._normalize_label("NEU"), "NEUTRAL")
        self.assertEqual(self.provider._normalize_label("LABEL_2"), "POSITIVE")
        self.assertEqual(self.provider._normalize_label("LABEL_0"), "NEGATIVE")
        self.assertEqual(self.provider._normalize_label("LABEL_1"), "NEUTRAL")

    @patch("SentimentAnalysis.providers.huggingface_provider.requests.post")
    def test_analyze_positive(self, mock_post):
        """Test analysis with mocked positive prediction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [
                {"label": "positive", "score": 0.9854},
                {"label": "neutral", "score": 0.0100},
                {"label": "negative", "score": 0.0046}
            ]
        ]
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"SENTIMENT_ANALYSIS_API_URL": "https://api-inference.huggingface.co/models/test"}):
            result = self.provider.analyze("I love this application!")
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["label"], "POSITIVE")
            self.assertAlmostEqual(result["score"], 0.9854, places=4)
            self.assertEqual(result["provider"], "huggingface")

    @patch("SentimentAnalysis.providers.huggingface_provider.requests.post")
    def test_analyze_negative(self, mock_post):
        """Test analysis with mocked negative prediction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [
                {"label": "negative", "score": 0.9521},
                {"label": "neutral", "score": 0.0320},
                {"label": "positive", "score": 0.0159}
            ]
        ]
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"SENTIMENT_ANALYSIS_API_URL": "https://api-inference.huggingface.co/models/test"}):
            result = self.provider.analyze("I hate this terrible bug!")
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["label"], "NEGATIVE")
            self.assertAlmostEqual(result["score"], 0.9521, places=4)
            self.assertEqual(result["provider"], "huggingface")

    @patch("SentimentAnalysis.providers.huggingface_provider.requests.post")
    def test_analyze_neutral(self, mock_post):
        """Test analysis with mocked neutral prediction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [
                {"label": "neutral", "score": 0.8800},
                {"label": "positive", "score": 0.0700},
                {"label": "negative", "score": 0.0500}
            ]
        ]
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"SENTIMENT_ANALYSIS_API_URL": "https://api-inference.huggingface.co/models/test"}):
            result = self.provider.analyze("The document was printed on Friday.")
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["label"], "NEUTRAL")
            self.assertAlmostEqual(result["score"], 0.8800, places=4)
            self.assertEqual(result["provider"], "huggingface")

    @patch("SentimentAnalysis.providers.huggingface_provider.requests.post")
    def test_timeout(self, mock_post):
        """Test HF provider timeout handling."""
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        with patch.dict("os.environ", {"SENTIMENT_ANALYSIS_API_URL": "https://api-inference.huggingface.co/models/test"}):
            result = self.provider.analyze("Sample timeout text")
            self.assertEqual(result["status"], "service_unavailable")
            self.assertIsNone(result["label"])
            self.assertEqual(result["provider"], "huggingface")

    @patch("SentimentAnalysis.providers.huggingface_provider.requests.post")
    def test_connection_error(self, mock_post):
        """Test HF provider connection error handling."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Unreachable network")

        with patch.dict("os.environ", {"SENTIMENT_ANALYSIS_API_URL": "https://api-inference.huggingface.co/models/test"}):
            result = self.provider.analyze("Sample connection error text")
            self.assertEqual(result["status"], "service_unavailable")
            self.assertIsNone(result["label"])
            self.assertEqual(result["provider"], "huggingface")

    @patch("SentimentAnalysis.providers.huggingface_provider.requests.post")
    def test_http_500(self, mock_post):
        """Test HF provider HTTP 500 error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"SENTIMENT_ANALYSIS_API_URL": "https://api-inference.huggingface.co/models/test"}):
            result = self.provider.analyze("Sample server error text")
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["label"])

    @patch("SentimentAnalysis.providers.huggingface_provider.requests.post")
    def test_http_401(self, mock_post):
        """Test HF provider HTTP 401 unauthorized handling."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"SENTIMENT_ANALYSIS_API_URL": "https://api-inference.huggingface.co/models/test"}):
            result = self.provider.analyze("Sample auth error text")
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["label"])

    @patch("SentimentAnalysis.providers.huggingface_provider.requests.post")
    def test_http_503_model_loading(self, mock_post):
        """Test HF provider HTTP 503 model loading handling."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Model is loading"
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"SENTIMENT_ANALYSIS_API_URL": "https://api-inference.huggingface.co/models/test"}):
            result = self.provider.analyze("Sample loading text")
            self.assertEqual(result["status"], "service_unavailable")
            self.assertIsNone(result["label"])

    @patch("SentimentAnalysis.providers.huggingface_provider.requests.post")
    def test_malformed_json(self, mock_post):
        """Test HF provider malformed JSON handling."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Corrupt JSON payload")
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"SENTIMENT_ANALYSIS_API_URL": "https://api-inference.huggingface.co/models/test"}):
            result = self.provider.analyze("Sample malformed text")
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["label"])

    @patch("SentimentAnalysis.providers.huggingface_provider.requests.post")
    def test_empty_predictions(self, mock_post):
        """Test HF provider empty predictions handling."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [[]]
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"SENTIMENT_ANALYSIS_API_URL": "https://api-inference.huggingface.co/models/test"}):
            result = self.provider.analyze("Sample text")
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["label"])


if __name__ == "__main__":
    unittest.main()