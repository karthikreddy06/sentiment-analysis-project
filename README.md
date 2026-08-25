# Watson NLP Sentiment Analysis Project

An enterprise-ready, standalone Python sentiment analysis web application and reusable library. This project integrates the IBM Watson NLP BERT deep learning model to evaluate emotion, sentiment polarity, and confidence scores from arbitrary text inputs in real time.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Project Directory Structure](#project-directory-structure)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [Network Diagnostics & Connectivity](#network-diagnostics--connectivity)
- [Python Setup & Installation](#python-setup--installation)
- [Running the Application](#running-the-application)
- [Running Unit Tests](#running-unit-tests)
- [Code Quality & Pylint](#code-quality--pylint)
- [Watson NLP API Integration](#watson-nlp-api-integration)
- [API Endpoints Specification](#api-endpoints-specification)
- [Error Handling Strategy](#error-handling-strategy)
- [Example Usage](#example-usage)

---

## Overview

The **Sentiment Analysis Project** is built from scratch as an end-to-end Python system demonstrating:
- **Modular Python Packaging**: Clean separation between core machine learning inference logic and application delivery layers.
- **Microservices & External Model Ingestion**: HTTP POST integration with Watson NLP BERT service.
- **Granular Status Discrimination**: Differentiates `SUCCESS`, `INVALID_INPUT`, `TIMEOUT`, `CONNECTION_ERROR`, `API_ERROR`, and `INVALID_RESPONSE`.
- **RESTful API Service**: Lightweight Flask server with multi-content-type response negotiation.
- **Modern Responsive Frontend**: Accessible, glassmorphism-themed UI with real-time feedback, sample prompts, and confidence score visualizers.
- **Deterministic Testing**: Isolated unit tests with request mocking (`unittest.mock`).
- **Code Standards**: 100% compliant with PEP 8 and checked with Pylint.

---

## Features

- **Real-Time Sentiment Classification**: Detects `POSITIVE`, `NEGATIVE`, and `NEUTRAL` sentiment with high-precision confidence scores.
- **Service-Level Error Discrimination**: Distinguishes upstream network timeouts / private lab endpoint unreachable states from invalid user input.
- **Modern Responsive Web UI**:
  - Dark-mode glassmorphism interface.
  - Character counter with instant feedback.
  - Quick sample prompt chips for one-click testing.
  - Accessible visual sentiment badges with both color and distinct iconography.
  - Interactive animated confidence score bar.
- **Diagnostic Tool**: Included `scripts/test_watson_connection.py` utility to inspect DNS resolution, TCP handshake, and API latency.
- **Full Test Coverage**: Unit tests covering positive, negative, neutral, error, timeout, and boundary conditions without live network dependencies.
- **Strict Linting Standards**: Clean code verified with `pylint`.

---

## Project Directory Structure

```text
sentiment_analysis_project/
├── .git/
├── .gitignore
├── README.md
├── requirements.txt
├── server.py
├── SentimentAnalysis/
│   ├── __init__.py
│   └── sentiment_analysis.py
├── scripts/
│   └── test_watson_connection.py
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
└── tests/
    ├── __init__.py
    └── test_sentiment_analysis.py
```

---

## Configuration & Environment Variables

The Watson service parameters can be configured via environment variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `WATSON_SENTIMENT_URL` | `https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict` | Watson NLP BERT Sentiment Predict endpoint URL |
| `WATSON_MODEL_ID` | `sentiment_aggregated-bert-workflow_lang_multi_stock` | Watson NLP model identifier |
| `WATSON_TIMEOUT` | `10` | Request timeout in seconds |

---

## Network Diagnostics & Connectivity

The Watson endpoint `sn-watson-sentiment-bert.labs.skills.network` resides within the IBM Skills Network internal VPC (`10.x.x.x` private IP space).

To test endpoint reachability from your current environment, run:

```bash
python scripts/test_watson_connection.py
```

---

## Python Setup & Installation

### 1. Prerequisites
Ensure Python 3.11+ is installed on your system.

### 2. Create Virtual Environment

On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Running the Application

To start the Flask development server:

```bash
python server.py
```

Once started, navigate to:
```text
http://127.0.0.1:5000/
```

---

## Running Unit Tests

The test suite validates sentiment classification logic and mocked network failure modes without internet access.

```bash
python -m unittest discover
```

---

## Code Quality & Pylint

To verify code quality against PEP 8 and static analysis rules:

```bash
python -m pylint server.py SentimentAnalysis/sentiment_analysis.py scripts/test_watson_connection.py
```

---

## API Endpoints Specification

### 1. Web Interface
- **Route**: `GET /`
- **Description**: Delivers the single-page application interface.

### 2. Sentiment Analyzer
- **Route**: `POST /sentimentAnalyzer` or `GET /sentimentAnalyzer?textToAnalyze=<text>`
- **Headers**:
  - `Content-Type: application/json` (for POST)
  - `Accept: application/json` (optional)
- **JSON Request Body**:
  ```json
  {
    "textToAnalyze": "I am thrilled with the great service!"
  }
  ```
- **Success (200 OK)**:
  ```json
  {
    "label": "POSITIVE",
    "score": 0.9876,
    "message": "The given text has been identified as POSITIVE with a score of 0.9876.",
    "status": "SUCCESS"
  }
  ```
- **Service Unavailable / Timeout (503 Service Unavailable)**:
  ```json
  {
    "label": null,
    "score": null,
    "message": "Sentiment service is currently unavailable. Please try again later.",
    "status": "TIMEOUT"
  }
  ```
- **Empty Input (400 Bad Request)**:
  ```json
  {
    "label": null,
    "score": null,
    "message": "Please enter some text to analyze.",
    "status": "EMPTY_INPUT"
  }
  ```
- **Invalid Input (200 OK / 400)**:
  ```json
  {
    "label": null,
    "score": null,
    "message": "Invalid input! Try again.",
    "status": "INVALID_INPUT"
  }
  ```

---

## Error Handling Strategy

1. **Empty / Blank Inputs**: Returns `"Please enter some text to analyze."` with HTTP 400.
2. **Network Failures & Timeouts**: `requests.exceptions.Timeout` and `requests.exceptions.ConnectionError` are caught, logged server-side, and return `"Sentiment service is currently unavailable. Please try again later."` with HTTP 503.
3. **HTTP 500 / Non-200 Responses**: Intercepted and returned cleanly as service unavailable with HTTP 502.
4. **Malformed Responses / Missing Keys**: Safely parsed using `.get()` guards returning `INVALID_RESPONSE`.
5. **Server Resilience**: The Flask app isolates per-request exceptions so the service remains 100% available.
