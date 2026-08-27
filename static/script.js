/**
 * Sentiment Analysis Web Client
 * Interacts with Flask backend, handles active provider status,
 * loading animations, and dynamic result rendering.
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
    const progressPercent = document.getElementById("progressPercent");
    const timestampText = document.getElementById("timestampText");
    const metaProviderText = document.getElementById("metaProviderText");
    const providerBadgeText = document.getElementById("providerBadgeText");
    const sampleButtons = document.querySelectorAll(".chip-btn");

    let isRequestActive = false;

    // Provider badge
    async function updateProviderBadge() {
        try {
            const res = await fetch("/health");
            if (res.ok) {
                const data = await res.json();
                providerBadgeText.textContent = data.provider === "watson" ? "● WATSON NLP" : "● LOCAL AI ENGINE";
            }
        } catch (err) {
            console.warn("Could not fetch provider status from /health", err);
        }
    }
    updateProviderBadge();

    // Character counter
    function updateCharCount() {
        const count = textarea.value.length;
        charCountBadge.textContent = `${count} / 2000 characters`;
    }
    textarea.addEventListener("input", updateCharCount);

    // Sample prompts
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

    // Clear
    clearBtn.addEventListener("click", () => {
        textarea.value = "";
        updateCharCount();
        hideAlert();
        hideResult();
        textarea.focus();
    });

    // Alert helpers
    function showAlert(message, type = "warning") {
        alertBox.className = `alert ${type}`;
        alertBox.classList.remove("hidden");
        alertMessage.textContent = message;
        // SVG icons for alert types
        if (type === "error") {
            alertIcon.innerHTML = `<circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line>`;
        } else {
            alertIcon.innerHTML = `<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>`;
        }
    }
    function hideAlert() { alertBox.classList.add("hidden"); }
    function hideResult() { resultCard.classList.add("hidden"); }
    function showResult() { resultCard.classList.remove("hidden"); }

    function setLoading(isLoading) {
        isRequestActive = isLoading;
        analyzeBtn.disabled = isLoading;
        clearBtn.disabled = isLoading;
        analyzeBtn.classList.toggle("loading", isLoading);
    }

    // Render result
    function renderResult(data) {
        hideAlert();

        const rawLabel = (data.label || "").toUpperCase();
        const score = typeof data.score === "number" ? data.score : parseFloat(data.score) || 0;
        const percentage = Math.round(score * 100 * 10) / 10;
        const providerName = data.provider === "watson" ? "Watson NLP BERT" : "Local Transformer";

        systemOutputText.textContent = data.message || `Identified as ${rawLabel} with score ${score}.`;
        displaySentiment.textContent = rawLabel || "UNKNOWN";
        displayScore.textContent = `${percentage}% (${score.toFixed(4)})`;

        // Progress bar
        confidenceBar.style.width = `${percentage}%`;
        progressPercent.textContent = `${percentage}%`;
        confidenceBar.className = "progress-fill";
        if (rawLabel.includes("POSITIVE")) confidenceBar.classList.add("positive");
        else if (rawLabel.includes("NEGATIVE")) confidenceBar.classList.add("negative");
        else if (rawLabel.includes("NEUTRAL")) confidenceBar.classList.add("neutral");

        metaProviderText.textContent = `Provider: ${providerName}`;

        const now = new Date();
        timestampText.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

        // Sentiment badge
        sentimentBadge.className = "sentiment-badge";
        if (rawLabel.includes("POSITIVE")) {
            sentimentBadge.classList.add("positive");
            indicatorIcon.textContent = "🟢 😊";
            indicatorLabel.textContent = "Positive Sentiment";
        } else if (rawLabel.includes("NEGATIVE")) {
            sentimentBadge.classList.add("negative");
            indicatorIcon.textContent = "🔴 😞";
            indicatorLabel.textContent = "Negative Sentiment";
        } else if (rawLabel.includes("NEUTRAL")) {
            sentimentBadge.classList.add("neutral");
            indicatorIcon.textContent = "🔵 😐";
            indicatorLabel.textContent = "Neutral Sentiment";
        } else {
            indicatorIcon.textContent = "⚪";
            indicatorLabel.textContent = rawLabel || "Evaluated";
        }

        showResult();
        resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Analyze handler
    async function handleAnalyze() {
        if (isRequestActive) return;

        const text = textarea.value.trim();

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
                headers: { "Content-Type": "application/json", "Accept": "application/json" },
                body: JSON.stringify({ text })
            });

            const resultData = await response.json();

            if (response.ok && resultData.success) {
                renderResult(resultData);
            } else if (resultData.code === "INVALID_INPUT") {
                showAlert(resultData.error || "We couldn't determine the sentiment of this text. Please try another sentence.", "warning");
            } else if (resultData.code === "SERVICE_UNAVAILABLE" || response.status === 503) {
                showAlert(resultData.error || "The sentiment service is temporarily unavailable. Please try again later.", "error");
            } else {
                showAlert(resultData.error || "An error occurred while analyzing sentiment.", "error");
            }
        } catch (error) {
            console.error("Sentiment analysis fetch error:", error);
            showAlert("Failed to connect to the server. Please ensure the Flask app is running.", "error");
        } finally {
            setLoading(false);
        }
    }

    analyzeBtn.addEventListener("click", handleAnalyze);

    // Ctrl/Cmd + Enter shortcut
    textarea.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") handleAnalyze();
    });

    updateCharCount();
});