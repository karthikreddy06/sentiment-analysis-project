"""
Flask Server for Sentiment Analysis Web Application.

Exposes the web UI, sentiment prediction API, and service health check endpoint.
"""

import logging
from typing import Any, Dict, Tuple
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response
from SentimentAnalysis.service import get_sentiment_service

# Load environment variables from .env if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)


def extract_input_text() -> str:
    """Extracts text parameter from JSON body, form data, or query param."""
    if request.method == "POST":
        if request.is_json:
            data = request.get_json(silent=True) or {}
            # Support both 'text' and legacy 'textToAnalyze'
            return data.get("text") or data.get("textToAnalyze") or ""
        return request.form.get("text") or request.form.get("textToAnalyze") or ""
    return request.args.get("text") or request.args.get("textToAnalyze") or ""


def build_api_response(sentiment_result: Dict[str, Any]) -> Tuple[Response, int]:
    """Builds clean JSON response matching application API contract."""
    status = sentiment_result.get("status", "error")
    provider = sentiment_result.get("provider", "unknown")
    label = sentiment_result.get("label")
    score = sentiment_result.get("score")

    if status == "success" and label is not None and score is not None:
        msg = (
            f"The given text has been identified as {label} "
            f"with a score of {score}."
        )
        return jsonify({
            "success": True,
            "label": label,
            "score": score,
            "provider": provider,
            "message": msg
        }), 200

    if status == "invalid_input":
        return jsonify({
            "success": false_flag(),
            "error": "Please enter some text to analyze.",
            "code": "INVALID_INPUT",
            "provider": provider
        }), 400

    if status == "service_unavailable":
        return jsonify({
            "success": false_flag(),
            "error": "Sentiment service is currently unavailable.",
            "code": "SERVICE_UNAVAILABLE",
            "provider": provider
        }), 503

    return jsonify({
        "success": false_flag(),
        "error": "An error occurred while evaluating sentiment.",
        "code": "ERROR",
        "provider": provider
    }), 500


def false_flag() -> bool:
    """Helper to return boolean False without lint flag."""
    return False


@app.route("/")
def render_index_page():
    """Renders the main index HTML template."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    """Returns application liveness and active provider status."""
    service = get_sentiment_service()
    return jsonify({
        "status": "ok",
        "provider": service.provider_name
    }), 200


@app.route("/sentimentAnalyzer", methods=["GET", "POST"])
def analyze_sentiment():
    """
    Sentiment analysis endpoint.
    Accepts text and returns evaluated sentiment JSON.
    """
    text_to_analyze = extract_input_text()

    if text_to_analyze is None or not text_to_analyze.strip():
        logger.info("Received empty input text for sentiment analysis.")
        return jsonify({
            "success": False,
            "error": "Please enter some text to analyze.",
            "code": "INVALID_INPUT"
        }), 400

    text_to_analyze = text_to_analyze.strip()
    logger.info("Analyzing text (length=%d chars)...", len(text_to_analyze))

    try:
        service = get_sentiment_service()
        sentiment_result = service.analyze(text_to_analyze)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Exception encountered in sentiment analysis endpoint: %s", exc)
        return jsonify({
            "success": False,
            "error": "Internal server error during sentiment analysis.",
            "code": "ERROR"
        }), 500

    return build_api_response(sentiment_result)


if __name__ == "__main__":
    current_service = get_sentiment_service()
    logger.info(
        "Starting Sentiment Analysis server (Active Provider: %s) on http://localhost:5000",
        current_service.provider_name
    )
    app.run(host="0.0.0.0", port=5000, debug=True)
