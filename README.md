# Sentiment AI – Hybrid Sentiment Analysis Service

An enterprise-ready, standalone Python sentiment analysis web application and reusable library. This project features a modular provider-based architecture allowing seamless switching between a **local pretrained Hugging Face RoBERTa model** (the default) and the **IBM Watson NLP BERT deep learning model**.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Project Directory Structure](#project-directory-structure)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [Python Setup & Installation](#python-setup--installation)
- [Running the Application](#running-the-application)
- [Running Unit Tests](#running-unit-tests)
- [Watson NLP Diagnostic Script](#watson-nlp-diagnostic-script)
- [API Endpoints Specification](#api-endpoints-specification)
- [Error Handling Strategy](#error-handling-strategy)

---

## Overview

The **Sentiment AI Service** is designed to demonstrate modern, decoupled software patterns in Python:
- **Modular Provider Architecture**: The analysis interface is defined by a common base class, isolating model initialization and API calls into distinct providers (`local` vs `watson`).
- **Hybrid Backends**: 
  - **Local**: Runs `cardiffnlp/twitter-roberta-base-sentiment-latest` in-memory. Zero external API calls, highly performant and private.
  - **Watson**: Integrates with upstream Watson NLP sentiment aggregates.
- **Granular Status Tracking**: Response structure includes descriptive states: `success`, `invalid_input`, `service_unavailable`, `api_error`, and `invalid_response`.
- **RESTful API Service**: Lightweight Flask server with robust JSON exchange formats.
- **Modern Responsive Frontend**: Accessible, premium dark-mode Sage & Cream UI with example inputs, real-time counter, and clean result displays showing confidence levels.

---

## Features

- **Real-Time Sentiment Classification**: Categorizes text into `POSITIVE`, `NEGATIVE`, or `NEUTRAL` with confidence scores.
- **Lazy Initialization / Model Caching**: On the local provider, the transformer pipeline is initialized once upon the first request, then cached for subsequent evaluations.
- **Service-Level Resilience**: upstream network failures or parsing issues are handled gracefully without terminating the server.
- **Premium Responsive Web UI**:
  - Warm sage, cream, and terracotta color theme.
  - Grid-based two-column desktop hero layout putting the title and vase on the left and the analyzer card on the right.
  - Real-time character counter.
  - Sample chips for testing positive, negative, or neutral sentiment.
  - Clean animated confidence meters showing the model's accuracy.
- **Deterministic Offline Testing**: Extensive test coverage using mocks to simulate provider responses and failures offline.

---

## Project Directory Structure

```text
sentiment_analysis_project/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── server.py
│
├── SentimentAnalysis/
│   ├── __init__.py
│   ├── sentiment_analysis.py
│   ├── service.py
│   └── providers/
│       ├── __init__.py
│       ├── base.py
│       ├── local_provider.py
│       └── watson_provider.py
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── tests/
│   ├── __init__.py
│   ├── test_local_provider.py
│   └── test_sentiment_analysis.py
│
└── scripts/
    └── test_watson_connection.py
```

---

## Configuration & Environment Variables

All configuration is managed through environment variables loaded from a `.env` file via `python-dotenv`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `SENTIMENT_PROVIDER` | `local` | Active provider backend (`local` or `watson`) |
| `WATSON_SENTIMENT_URL` | `https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict` | Upstream Watson endpoint |
| `WATSON_MODEL_ID` | `sentiment_aggregated-bert-workflow_lang_multi_stock` | Watson NLP model identifier |
| `WATSON_TIMEOUT` | `10` | Upstream timeout in seconds |

---

## Python Setup & Installation

### 1. Prerequisites
Ensure Python 3.11+ is installed on your system.

### 2. Create Virtual Environment

On Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:
```bash
python3 -m venv .venv
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
http://localhost:5000/
```

The first request to the local provider will trigger the download/caching of the RoBERTa model. Subscriptions are saved locally inside Hugging Face's cache directory and loaded instantly on subsequent requests.

---

## Running Unit Tests

To run the complete test suite:

```bash
python -m unittest discover -v
```

---

## Watson NLP Diagnostic Script

Since the internal Watson NLP aggregates endpoint may be unreachable from external networks, you can diagnose DNS and HTTP connectivity using:

```bash
python scripts/test_watson_connection.py
```

---

## API Endpoints Specification

### 1. Health Status
- **Route**: `GET /health`
- **Description**: Returns server state and the active provider name.
- **Response**:
  ```json
  {
    "status": "ok",
    "provider": "local"
  }
  ```

### 2. Sentiment Analyzer
- **Route**: `POST /sentimentAnalyzer`
- **Headers**:
  - `Content-Type: application/json`
- **JSON Request Body**:
  ```json
  {
    "text": "I absolutely love this application. It is fantastic!"
  }
  ```
- **Success Response (200 OK)**:
  ```json
  {
    "success": true,
    "label": "POSITIVE",
    "score": 0.9861,
    "provider": "local",
    "status": "success",
    "message": "The given text has been identified as POSITIVE with a score of 0.9861."
  }
  ```
- **Service Unavailable (503 Service Unavailable)**:
  ```json
  {
    "success": false,
    "code": "SERVICE_UNAVAILABLE",
    "error": "Sentiment service is currently unavailable. Please try again later."
  }
  ```
- **Invalid/Empty Input (400 Bad Request)**:
  ```json
  {
    "success": false,
    "code": "INVALID_INPUT",
    "error": "Please enter some text to analyze."
  }
  ```

---

## Error Handling Strategy

1. **Empty or Whitespace Inputs**: Triggers a fast-fail validator returning `INVALID_INPUT` and HTTP 400.
2. **Upstream Network Failures**: Connection errors and timeouts from the Watson API return `SERVICE_UNAVAILABLE` and HTTP 503 instead of crashing the server.
3. **Model Errors**: Gracefully handles Hugging Face pipeline initialization/inference faults by falling back to descriptive status codes.
