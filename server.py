"""
Flask Server for Watson NLP Sentiment Analysis Web Application.

Provides the web UI at '/' and the sentiment analysis endpoint at '/sentimentAnalyzer'.
"""

import logging
from flask import Flask, render_template, request, jsonify
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
    cleaned = raw_label.replace("SENT_", "").strip()
    return cleaned.upper()


@app.route("/")
def render_index_page():
    """
    Renders the main index HTML template.
    """
    return render_template("index.html")


@app.route("/sentimentAnalyzer", methods=["GET", "POST"])
def analyze_sentiment():
    """
    Endpoint that processes user-submitted text and returns sentiment evaluation.
    Supports GET (query param 'textToAnalyze') and POST (JSON or form body).
    """
    text_to_analyze = ""

    if request.method == "POST":
        if request.is_json:
            data = request.get_json(silent=True) or {}
            text_to_analyze = data.get("textToAnalyze", "")
        else:
            text_to_analyze = request.form.get("textToAnalyze", "")
    else:
        text_to_analyze = request.args.get("textToAnalyze", "")

    # Check for empty or whitespace-only input
    if text_to_analyze is None or not text_to_analyze.strip():
        logger.info("Received empty input text for sentiment analysis.")
        response_msg = "Please enter some text to analyze."
        if request.headers.get("Accept") == "application/json" or request.is_json:
            return jsonify({
                "message": response_msg,
                "label": None,
                "score": None,
                "status": "empty"
            }), 400
        return response_msg, 400

    text_to_analyze = text_to_analyze.strip()
    logger.info("Analyzing text (length=%d characters)...", len(text_to_analyze))

    try:
        sentiment_result = sentiment_analyzer(text_to_analyze)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Unexpected exception during sentiment analysis: %s", exc)
        sentiment_result = {"label": None, "score": None}

    label = sentiment_result.get("label")
    score = sentiment_result.get("score")

    # Handle invalid or failed analysis
    if label is None or score is None:
        logger.warning("Sentiment analyzer returned None for label or score.")
        response_msg = "Invalid input! Try again."
        if request.headers.get("Accept") == "application/json" or request.is_json:
            return jsonify({
                "message": response_msg,
                "label": None,
                "score": None,
                "status": "invalid"
            }), 200
        return response_msg, 200

    # Format sentiment result
    display_label = format_sentiment_label(label)
    rounded_score = round(float(score), 4)

    response_msg = (
        f"The given text has been identified as {display_label} "
        f"with a score of {rounded_score}."
    )

    if request.headers.get("Accept") == "application/json" or request.is_json:
        return jsonify({
            "message": response_msg,
            "label": display_label,
            "score": rounded_score,
            "status": "success"
        }), 200

    return response_msg, 200


if __name__ == "__main__":
    logger.info("Starting Sentiment Analysis Flask server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
