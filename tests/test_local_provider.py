"""
Unit tests for the Local Transformer sentiment analysis provider.
"""

import unittest
from unittest.mock import patch, MagicMock
from SentimentAnalysis.providers.local_provider import LocalTransformerProvider


class TestLocalProvider(unittest.TestCase):
    """Test suite for LocalTransformerProvider."""

    def setUp(self):
        self.provider = LocalTransformerProvider()

    def test_provider_name(self):
        """Test provider identification name."""
        self.assertEqual(self.provider.provider_name, "local")

    def test_normalize_labels(self):
        """Test normalization of various raw model output labels."""
        self.assertEqual(self.provider._normalize_label("positive"), "POSITIVE")
        self.assertEqual(self.provider._normalize_label("LABEL_2"), "LABEL_2")
        self.assertEqual(self.provider._normalize_label("POS"), "POSITIVE")
        self.assertEqual(self.provider._normalize_label("negative"), "NEGATIVE")
        self.assertEqual(self.provider._normalize_label("NEG"), "NEGATIVE")
        self.assertEqual(self.provider._normalize_label("neutral"), "NEUTRAL")
        self.assertEqual(self.provider._normalize_label("NEU"), "NEUTRAL")

    @patch.object(LocalTransformerProvider, "_load_pipeline")
    def test_analyze_positive(self, mock_load):
        """Test analysis with mocked positive prediction."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = [
            [
                {"label": "positive", "score": 0.9854},
                {"label": "neutral", "score": 0.0100},
                {"label": "negative", "score": 0.0046}
            ]
        ]
        mock_load.return_value = mock_classifier

        result = self.provider.analyze("I love this application!")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["label"], "POSITIVE")
        self.assertAlmostEqual(result["score"], 0.9854, places=4)
        self.assertEqual(result["provider"], "local")

    @patch.object(LocalTransformerProvider, "_load_pipeline")
    def test_analyze_negative(self, mock_load):
        """Test analysis with mocked negative prediction."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = [
            [
                {"label": "negative", "score": 0.9521},
                {"label": "neutral", "score": 0.0320},
                {"label": "positive", "score": 0.0159}
            ]
        ]
        mock_load.return_value = mock_classifier

        result = self.provider.analyze("I hate this terrible bug!")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["label"], "NEGATIVE")
        self.assertAlmostEqual(result["score"], 0.9521, places=4)
        self.assertEqual(result["provider"], "local")

    @patch.object(LocalTransformerProvider, "_load_pipeline")
    def test_analyze_neutral(self, mock_load):
        """Test analysis with mocked neutral prediction."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = [
            [
                {"label": "neutral", "score": 0.8800},
                {"label": "positive", "score": 0.0700},
                {"label": "negative", "score": 0.0500}
            ]
        ]
        mock_load.return_value = mock_classifier

        result = self.provider.analyze("The document was printed on Friday.")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["label"], "NEUTRAL")
        self.assertAlmostEqual(result["score"], 0.8800, places=4)
        self.assertEqual(result["provider"], "local")

    @patch.object(LocalTransformerProvider, "_load_pipeline")
    def test_model_failure_graceful_handling(self, mock_load):
        """Test graceful failure when underlying pipeline fails."""
        mock_load.side_effect = RuntimeError("Model loading failed")

        result = self.provider.analyze("Any text")
        self.assertEqual(result["status"], "service_unavailable")
        self.assertIsNone(result["label"])
        self.assertIsNone(result["score"])
        self.assertEqual(result["provider"], "local")


if __name__ == "__main__":
    unittest.main()
