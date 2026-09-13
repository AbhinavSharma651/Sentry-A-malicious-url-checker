from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import json
import os

from features import extract_features, FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

clf = joblib.load(os.path.join(MODEL_DIR, "url_classifier.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
    feature_names = json.load(f)
with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    metrics = json.load(f)

CLASS_INFO = {
    "benign": {
        "label": "Benign",
        "color": "#1a7f37",
        "desc": "No malicious indicators detected in this URL's structure.",
    },
    "phishing": {
        "label": "Phishing",
        "color": "#c0392b",
        "desc": "This URL shares structural patterns with known phishing links (e.g. credential-harvesting pages impersonating trusted sites).",
    },
    "malware": {
        "label": "Malware",
        "color": "#7a1fa2",
        "desc": "This URL shares structural patterns with links historically used to distribute malware.",
    },
    "defacement": {
        "label": "Defacement",
        "color": "#b8860b",
        "desc": "This URL shares structural patterns with sites that have been compromised / defaced.",
    },
}


@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "Please enter a URL."}), 400
    if len(url) > 2048:
        return jsonify({"error": "URL is too long."}), 400

    feats = extract_features(url)
    X = pd.DataFrame([feats], columns=feature_names)

    pred_idx = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    class_probs = {
        label_encoder.inverse_transform([i])[0]: float(p)
        for i, p in enumerate(proba)
    }
    class_probs = dict(sorted(class_probs.items(), key=lambda kv: -kv[1]))

    info = CLASS_INFO.get(pred_label, {"label": pred_label, "color": "#333", "desc": ""})

    return jsonify({
        "url": url,
        "prediction": pred_label,
        "prediction_label": info["label"],
        "color": info["color"],
        "description": info["desc"],
        "confidence": class_probs[pred_label],
        "class_probabilities": class_probs,
        "is_malicious": pred_label != "benign",
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_metrics": {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "classes": metrics["classes"],
    }})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
