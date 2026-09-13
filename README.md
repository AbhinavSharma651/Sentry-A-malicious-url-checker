# Sentry — Malicious URL Checker

A Flask web app that classifies a URL as **benign**, **phishing**, **malware**, or
**defacement**, using a Random Forest model trained on 651,191 labeled URLs
(`malicious_phish.csv`). Classification is purely **lexical/structural** — it
reads the shape of the URL text (length, digit ratio, hyphens, suspicious
words, entropy, etc.) and never fetches or visits the link.

## Project structure

```
url_checker/
├── app.py              Flask server + /api/predict endpoint
├── features.py          Feature extraction (shared by training + serving)
├── train.py              Training pipeline (run once to produce model/)
├── requirements.txt
├── model/
│   ├── url_classifier.joblib   Trained RandomForestClassifier (~39MB)
│   ├── label_encoder.joblib
│   ├── feature_names.json
│   └── metrics.json            Held-out test metrics from training
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Running it

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

The trained model is already included in `model/`, so you can run the app
immediately without retraining.

## Retraining on your own data

`train.py` expects a CSV with two columns, `url` and `type`, at
`/mnt/user-data/uploads/malicious_phish.csv` (edit `DATA_PATH` at the top of
`train.py` to point elsewhere). Run:

```bash
python train.py
```

This re-extracts features, retrains, evaluates on a held-out 20% split, and
overwrites the files in `model/`.

## Model performance

On a held-out test set (130,239 URLs, stratified 80/20 split):

| Metric | Score |
|---|---|
| Accuracy | 87.0% |
| Macro F1 | 0.853 |

Per-class F1: benign 0.91, defacement 0.85, malware 0.93, phishing 0.72.
Phishing is the hardest class to separate from benign — that mirrors published
research on lexical-only URL classifiers, since a well-crafted phishing URL
can be lexically indistinguishable from a legitimate link.

## Known limitations (read before treating this as production-grade)

This is a **lexical classifier**, not a live threat-intelligence service. It
has no access to:
- Domain age / WHOIS registration data
- TLS certificate validity
- Real-time blocklists (Google Safe Browsing, PhishTank, VirusTotal, etc.)
- Page content, redirects, or actual network behavior

Because of that, it can be fooled by:
- A well-made phishing link that just doesn't look structurally unusual
- A legitimate but unusually-shaped URL (short query strings, uncommon
  paths) getting flagged as suspicious

**A dataset bias we found and fixed:** in the raw training data, ~96–100% of
malware/defacement URLs included an `http://`/`https://` prefix, but only ~8%
of benign URLs did (a scraping artifact, not a real-world pattern). Left
unfixed, the model would have learned "has a scheme prefix" as a proxy for
"malicious" — which would flag nearly every real-world URL a user pastes in,
since real URLs almost always include the scheme. `features.py` strips the
scheme before computing any feature, for both training and inference, to
remove that spurious signal. This is why the honest post-fix accuracy (87%)
is lower than the naive pre-fix number would have suggested (~95%) — 95%
was measuring the leak, not real skill.

**To make this genuinely production-ready**, the natural next steps are:
1. Add domain-reputation features (age, registrar, DNS record history)
2. Cross-reference against a live blocklist API as a second signal
3. Add a human-in-the-loop review queue for low-confidence predictions
   (the app already exposes full class probabilities for this)
