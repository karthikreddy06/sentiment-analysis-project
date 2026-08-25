"""
Flask Server for Watson NLP Sentiment Analysis Web Application.

Provides the web UI at '/' and the sentiment analysis endpoint at '/sentimentAnalyzer'.
"""

import logging
from typing import Any, Dict, Tuple
from flask import Flask, render_template, request, jsonify, Response
from SentimentAnalysis.sentiment_analysis import sentiment_analyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)


def format_sentiment_label(raw_label: str) -> str:
    """
    Format raw sentiment label by removing 'SENT_' prefix if present.

    Args:
        raw_label: Raw label string (e.g. 'SENT_POSITIVE', 'POSITIVE').

    Returns:
        Formatted label in uppercase (e.g. 'POSITIVE', 'NEGATIVE', 'NEUTRAL').
    """
    if not raw_label:
        return ""
    return raw_label.replace("SENT_", "").strip().upper()


def extract_input_text() -> str:
    """Extracts textToAnalyze parameter from GET or POST request."""
    if request.method == "POST":
        if request.is_json:
            data = request.get_json(silent=True) or {}
            return data.get("textToAnalyze", "")
        return request.form.get("textToAnalyze", "")
    return request.args.get("textToAnalyze", "")


def create_response(
    message: str,
    status_code: int,
    status_type: str,
    label: Any = None,
    score: Any = None
) -> Tuple[Response, int] | Tuple[str, int]:
    """Builds unified JSON or text response based on request headers."""
    if request.headers.get("Accept") == "application/json" or request.is_json:
        return jsonify({
            "message": message,
            "label": label,
            "score": score,
            "status": status_type
        }), status_code
    return message, status_code


def process_sentiment_status(sentiment_result: Dict[str, Any]) -> Tuple[Any, int]:
    """Maps sentiment analyzer result to appropriate HTTP response and code."""
    status = sentiment_result.get("status", "API_ERROR")
    label = sentiment_result.get("label")
    score = sentiment_result.get("score")

    if status in ("TIMEOUT", "CONNECTION_ERROR"):
        logger.warning("Watson service unreachable or timed out (status=%s).", status)
        msg = "Sentiment service is currently unavailable. Please try again later."
        return create_response(msg, 503, status)

    if status == "INVALID_INPUT":
        logger.warning("Invalid input received for sentiment analysis.")
        return create_response("Invalid input! Try again.", 200, "INVALID_INPUT")

    if status in ("API_ERROR", "INVALID_RESPONSE") or label is None or score is None:
        logger.warning("Sentiment analysis failed with status=%s.", status)
        msg = "Sentiment service is currently unavailable. Please try again later."
        return create_response(msg, 502, status)

    display_label = format_sentiment_label(label)
    rounded_score = round(float(score), 4)
    msg = f"The given text has been identified as {display_label} with a score of {rounded_score}."
    return create_response(msg, 200, "SUCCESS", display_label, rounded_score)


@app.route("/")
def render_index_page():
    """Renders the main index HTML template."""
    return render_template("index.html")


@app.route("/sentimentAnalyzer", methods=["GET", "POST"])
def analyze_sentiment():
    """
    Endpoint that processes user-submitted text and returns sentiment evaluation.
    Supports GET (query param 'textToAnalyze') and POST (JSON or form body).
    """
    text_to_analyze = extract_input_text()

    # Check for empty or whitespace-only input
    if text_to_analyze is None or not text_to_analyze.strip():
        logger.info("Received empty input text for sentiment analysis.")
        return create_response("Please enter some text to analyze.", 400, "EMPTY_INPUT")

    text_to_analyze = text_to_analyze.strip()
    logger.info("Analyzing text (length=%d characters)...", len(text_to_analyze))

    try:
        sentiment_result = sentiment_analyzer(text_to_analyze)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Unexpected exception during sentiment analysis: %s", exc)
        sentiment_result = {"label": None, "score": None, "status": "API_ERROR"}

    return process_sentiment_status(sentiment_result)


if __name__ == "__main__":
    logger.info("Starting Sentiment Analysis Flask server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
