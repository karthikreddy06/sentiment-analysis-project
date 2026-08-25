# Watson NLP Sentiment Analysis Project

An enterprise-ready, standalone Python sentiment analysis web application and reusable library. This project integrates the IBM Watson NLP BERT deep learning model to evaluate emotion, sentiment polarity, and confidence scores from arbitrary text inputs in real time.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Project Directory Structure](#project-directory-structure)
- [Python Setup & Installation](#python-setup--installation)
- [Running the Application](#running-the-application)
- [Running Unit Tests](#running-unit-tests)
- [Code Quality & Pylint](#code-quality--pylint)
- [Watson NLP API Integration](#watson-nlp-api-integration)
- [API Endpoints Specification](#api-endpoints-specification)
- [Error Handling Strategy](#error-handling-strategy)
- [Example Usage](#example-usage)
- [Future Improvements](#future-improvements)

---

## Overview

The **Sentiment Analysis Project** is built from scratch as an end-to-end Python system demonstrating:
- **Modular Python Packaging**: Clean separation between core machine learning inference logic and application delivery layers.
- **Microservices & External Model Ingestion**: HTTP POST integration with Watson NLP BERT service.
- **RESTful API Service**: Lightweight Flask server with multi-content-type response negotiation.
- **Modern Responsive Frontend**: Accessible, glassmorphism-themed UI with real-time feedback, sample prompts, and confidence score visualizers.
- **Deterministic Testing**: Isolated unit tests with request mocking (`unittest.mock`).
- **Code Standards**: 100% compliant with PEP 8 and checked with Pylint.

---

## Features

- **Real-Time Sentiment Classification**: Detects `POSITIVE`, `NEGATIVE`, and `NEUTRAL` sentiment with high-precision confidence scores.
- **Robust Exception Handling**: Gracefully handles network timeouts, DNS failures, HTTP 500 status codes, and malformed JSON payloads without server crashes.
- **Modern Responsive Web UI**:
  - Dark-mode glassmorphism interface.
  - Character counter with instant feedback.
  - Quick sample prompt chips for one-click testing.
  - Accessible visual sentiment badges with both color and distinct iconography.
  - Interactive animated confidence score bar.
- **Full Test Coverage**: Unit tests covering positive, negative, neutral, error, and boundary conditions without live network dependencies.
- **Strict Linting Standards**: Clean code verified with `pylint`.

---

## Architecture & Tech Stack

- **Language**: Python 3.11
- **Backend Framework**: Flask 3.x
- **HTTP Client**: Requests 2.31.x
- **NLP Model**: Watson NLP BERT Aggregated Multilingual Model (`sentiment_aggregated-bert-workflow_lang_multi_stock`)
- **Frontend**: Semantic HTML5, Vanilla Modern CSS3 (Glassmorphism & Flexbox/Grid), Modern Vanilla JavaScript (Fetch API)
- **Testing**: Python standard `unittest` & `unittest.mock`
- **Code Quality**: `pylint`

---

## Project Directory Structure

```text
sentiment_analysis_project/
├── SentimentAnalysis/
│   ├── __init__.py
│   └── sentiment_analysis.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── tests/
│   └── test_sentiment_analysis.py
├── server.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Python Setup & Installation

### 1. Prerequisites
Ensure Python 3.11 is installed on your system.

### 2. Create Virtual Environment

On macOS/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirements.txt
```

---

## Running the Application

To start the Flask development server:

```bash
python3.11 server.py
```

Once started, navigate to:
```text
http://127.0.0.1:5000/
```

---

## Running Unit Tests

The test suite validates sentiment classification logic and mocked network failure modes.

Run tests using Python's discover runner:

```bash
python3.11 -m unittest discover
```

Or run the specific test module:

```bash
python3.11 -m unittest tests/test_sentiment_analysis.py
```

---

## Code Quality & Pylint

To verify code quality against PEP 8 and static analysis rules:

```bash
pylint server.py
pylint SentimentAnalysis/sentiment_analysis.py
```

---

## Watson NLP API Integration

The core analyzer (`SentimentAnalysis/sentiment_analysis.py`) connects to the Watson NLP BERT Sentiment Predict runtime:

- **Endpoint**:
  ```text
  https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict
  ```
- **Required Header**:
  ```http
  grpc-metadata-mm-model-id: sentiment_aggregated-bert-workflow_lang_multi_stock
  ```
- **Payload Schema**:
  ```json
  {
    "raw_document": {
      "text": "Your text to analyze here"
    }
  }
  ```
- **Response Extraction**:
  The function parses `response["documentSentiment"]["label"]` and `response["documentSentiment"]["score"]`.
  On failure or invalid status codes, it returns:
  ```python
  {"label": None, "score": None}
  ```

---

## API Endpoints Specification

### 1. Web Interface
- **Route**: `GET /`
- **Description**: Delivers the single-page application interface.

### 2. Sentiment Analyzer
- **Route**: `POST /sentimentAnalyzer` or `GET /sentimentAnalyzer?textToAnalyze=<text>`
- **Headers**:
  - `Content-Type: application/json` (optional for POST)
  - `Accept: application/json` (optional)
- **JSON Request Body** (for POST):
  ```json
  {
    "textToAnalyze": "I am thrilled with the great service!"
  }
  ```
- **Success Response (200 OK)**:
  ```json
  {
    "label": "POSITIVE",
    "score": 0.9876,
    "message": "The given text has been identified as POSITIVE with a score of 0.9876.",
    "status": "success"
  }
  ```
- **Empty Input (400 Bad Request)**:
  ```json
  {
    "label": null,
    "score": null,
    "message": "Please enter some text to analyze.",
    "status": "empty"
  }
  ```
- **Invalid Input / API Failure (200 OK)**:
  ```json
  {
    "label": null,
    "score": null,
    "message": "Invalid input! Try again.",
    "status": "invalid"
  }
  ```

---

## Error Handling Strategy

1. **Empty / Blank Inputs**: Checked prior to API dispatch. Prompts user with `"Please enter some text to analyze."`
2. **Network Failures & Timeouts**: `requests.exceptions.Timeout` and `requests.exceptions.ConnectionError` are caught, logged server-side, and safely return `{ "label": None, "score": None }`.
3. **HTTP 500 / Non-200 Responses**: Intercepted and returned as `Invalid input! Try again.` without leaking stack traces.
4. **Malformed Responses / Missing Keys**: Safely parsed using `.get()` guards to avoid `KeyError` or `TypeError`.
5. **Server Resilience**: The Flask app isolates per-request exceptions so the service remains 100% available.

---

## Example Usage

### As a Python Package:

```python
from SentimentAnalysis.sentiment_analysis import sentiment_analyzer

result = sentiment_analyzer("The product quality is outstanding!")
print(result)
# Output: {'label': 'SENT_POSITIVE', 'score': 0.9942}
```

### Via cURL (POST JSON):

```bash
curl -X POST http://127.0.0.1:5000/sentimentAnalyzer \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"textToAnalyze": "Super fast delivery and great packaging!"}'
```

### Via cURL (GET Query):

```bash
curl "http://127.0.0.1:5000/sentimentAnalyzer?textToAnalyze=Terrible%20experience"
```

---

## Future Improvements

- **Batch Sentiment Analysis**: Support multi-document array inputs in a single HTTP request.
- **Aspect-Based Sentiment Extraction**: Highlight specific keywords/phrases driving positive or negative polarity.
- **History & Exporting**: Allow saving analysis history to CSV or JSON formats.
- **Multi-language Auto-detection**: Enhance UI indicators with automatic source language detection.
