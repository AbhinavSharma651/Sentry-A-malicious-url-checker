const form = document.getElementById("check-form");
const input = document.getElementById("url-input");
const btn = document.getElementById("check-btn");
const scanIndicator = document.getElementById("scan-indicator");
const resultBox = document.getElementById("result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = input.value.trim();
  if (!url) {
    input.focus();
    return;
  }

  btn.disabled = true;
  resultBox.classList.add("hidden");
  scanIndicator.classList.remove("hidden");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();

    if (!res.ok) {
      renderError(data.error || "Something went wrong.");
    } else {
      renderResult(data);
    }
  } catch (err) {
    renderError("Could not reach the classifier. Please try again.");
  } finally {
    scanIndicator.classList.add("hidden");
    btn.disabled = false;
  }
});

function renderError(message) {
  resultBox.innerHTML = `<div class="error-box">${escapeHtml(message)}</div>`;
  resultBox.classList.remove("hidden");
}

function renderResult(data) {
  const pct = Math.round(data.confidence * 100);

  const probRows = Object.entries(data.class_probabilities)
    .map(([cls, p]) => {
      const width = Math.max(Math.round(p * 100), 1);
      const color = colorFor(cls, data);
      return `
        <div class="prob-row">
          <span class="name">${escapeHtml(cls)}</span>
          <div class="prob-track">
            <div class="prob-fill" style="width:${width}%; background:${color};"></div>
          </div>
          <span class="prob-pct">${Math.round(p * 100)}%</span>
        </div>`;
    })
    .join("");

  resultBox.innerHTML = `
    <div class="result-head">
      <div class="verdict">
        <span class="verdict-dot" style="background:${data.color}; box-shadow:0 0 8px ${data.color};"></span>
        <div class="verdict-text">
          <div class="label">${escapeHtml(data.prediction_label)}</div>
          <div class="confidence">${pct}% confidence</div>
        </div>
      </div>
      <div class="checked-url">${escapeHtml(data.url)}</div>
    </div>
    <div class="result-desc">${escapeHtml(data.description)}</div>
    <div class="prob-list">${probRows}</div>
  `;
  resultBox.classList.remove("hidden");
}

function colorFor(cls, data) {
  const map = {
    benign: "#2FBF71",
    phishing: "#E4572E",
    malware: "#7a4fe0",
    defacement: "#E3B341",
  };
  return map[cls] || data.color || "#4C8DFF";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
