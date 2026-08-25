/**
 * Sentiment Analysis Web Client
 * Handles real-time input validation, backend API communication,
 * loading states, visual state indicators, and error handling.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const textarea = document.getElementById("textToAnalyze");
    const charCountBadge = document.getElementById("charCount");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const clearBtn = document.getElementById("clearBtn");
    const alertBox = document.getElementById("alertBox");
    const alertIcon = document.getElementById("alertIcon");
    const alertMessage = document.getElementById("alertMessage");
    const resultCard = document.getElementById("resultCard");
    const systemOutputText = document.getElementById("systemOutputText");
    const sentimentBadge = document.getElementById("sentimentBadge");
    const indicatorIcon = document.getElementById("indicatorIcon");
    const indicatorLabel = document.getElementById("indicatorLabel");
    const displaySentiment = document.getElementById("displaySentiment");
    const displayScore = document.getElementById("displayScore");
    const confidenceBar = document.getElementById("confidenceBar");
    const timestampText = document.getElementById("timestampText");
    const sampleButtons = document.querySelectorAll(".chip-btn");

    /**
     * Update character counter in real time
     */
    function updateCharCount() {
        const count = textarea.value.length;
        charCountBadge.textContent = `${count} character${count === 1 ? "" : "s"}`;
    }

    textarea.addEventListener("input", updateCharCount);

    /**
     * Sample prompt population
     */
    sampleButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const sampleText = btn.getAttribute("data-sample");
            if (sampleText) {
                textarea.value = sampleText;
                updateCharCount();
                textarea.focus();
                hideAlert();
                hideResult();
            }
        });
    });

    /**
     * Clear & reset form
     */
    clearBtn.addEventListener("click", () => {
        textarea.value = "";
        updateCharCount();
        hideAlert();
        hideResult();
        textarea.focus();
    });

    /**
     * Show Alert message
     */
    function showAlert(message, type = "warning") {
        alertBox.className = "alert-box";
        if (type === "error") {
            alertBox.classList.add("alert-error");
            alertIcon.textContent = "⚠️";
        } else if (type === "info") {
            alertBox.classList.add("alert-info");
            alertIcon.textContent = "ℹ️";
        } else {
            alertIcon.textContent = "⚠️";
        }
        alertMessage.textContent = message;
        alertBox.classList.remove("hidden");
    }

    /**
     * Hide Alert message
     */
    function hideAlert() {
        alertBox.classList.add("hidden");
    }

    /**
     * Hide Result Card
     */
    function hideResult() {
        resultCard.classList.add("hidden");
    }

    /**
     * Set loading state on button
     */
    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            analyzeBtn.classList.add("loading");
        } else {
            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove("loading");
        }
    }

    /**
     * Render the sentiment result in the UI
     */
    function renderResult(data) {
        hideAlert();

        const rawLabel = (data.label || "").toUpperCase();
        const score = typeof data.score === "number" ? data.score : parseFloat(data.score) || 0;
        const percentage = Math.round(score * 100 * 10) / 10;

        systemOutputText.textContent = data.message;
        displaySentiment.textContent = rawLabel || "UNKNOWN";
        displayScore.textContent = `${percentage}% (${score.toFixed(4)})`;
        confidenceBar.style.width = `${Math.min(Math.max(percentage, 5), 100)}%`;

        // Update timestamp
        const now = new Date();
        timestampText.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        // Reset badge classes
        sentimentBadge.className = "sentiment-indicator-badge";
        resultCard.className = "result-card";

        if (rawLabel.includes("POSITIVE")) {
            sentimentBadge.classList.add("positive");
            resultCard.classList.add("positive");
            indicatorIcon.textContent = "🟢 😊";
            indicatorLabel.textContent = "Positive Sentiment";
        } else if (rawLabel.includes("NEGATIVE")) {
            sentimentBadge.classList.add("negative");
            resultCard.classList.add("negative");
            indicatorIcon.textContent = "🔴 😞";
            indicatorLabel.textContent = "Negative Sentiment";
        } else if (rawLabel.includes("NEUTRAL")) {
            sentimentBadge.classList.add("neutral");
            resultCard.classList.add("neutral");
            indicatorIcon.textContent = "🔵 😐";
            indicatorLabel.textContent = "Neutral Sentiment";
        } else {
            indicatorIcon.textContent = "⚪";
            indicatorLabel.textContent = rawLabel || "Evaluated";
        }

        resultCard.classList.remove("hidden");
        resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    /**
     * Analyze Sentiment Handler
     */
    async function handleAnalyze() {
        const text = textarea.value.trim();

        // Validate empty input locally
        if (!text) {
            hideResult();
            showAlert("Please enter some text to analyze.", "warning");
            textarea.focus();
            return;
        }

        hideAlert();
        hideResult();
        setLoading(true);

        try {
            const response = await fetch("/sentimentAnalyzer", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({ textToAnalyze: text })
            });

            const contentType = response.headers.get("content-type") || "";
            let resultData;

            if (contentType.includes("application/json")) {
                resultData = await response.json();
            } else {
                const textResponse = await response.text();
                resultData = {
                    message: textResponse,
                    status: response.ok ? "SUCCESS" : "API_ERROR"
                };
            }

            const status = resultData.status || "";

            // Handle service timeouts and connection errors
            if (response.status === 503 || status === "TIMEOUT" || status === "CONNECTION_ERROR") {
                showAlert(resultData.message || "Sentiment service is currently unavailable. Please try again later.", "error");
            } else if (response.status === 400 || status === "EMPTY_INPUT") {
                showAlert(resultData.message || "Please enter some text to analyze.", "warning");
            } else if (status === "INVALID_INPUT" || resultData.message === "Invalid input! Try again.") {
                showAlert("Invalid input! Try again.", "error");
            } else if (response.status === 502 || status === "API_ERROR" || status === "INVALID_RESPONSE") {
                showAlert(resultData.message || "Sentiment service is currently unavailable. Please try again later.", "error");
            } else if (resultData.label && resultData.score !== undefined) {
                renderResult(resultData);
            } else {
                showAlert(resultData.message || "Sentiment service is currently unavailable. Please try again later.", "info");
            }

        } catch (error) {
            console.error("Sentiment analysis fetch error:", error);
            showAlert("Failed to connect to the server. Please ensure the Flask app is running.", "error");
        } finally {
            setLoading(false);
        }
    }

    analyzeBtn.addEventListener("click", handleAnalyze);

    // Allow Ctrl+Enter or Cmd+Enter to submit
    textarea.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            handleAnalyze();
        }
    });

    // Initial character count
    updateCharCount();
});
