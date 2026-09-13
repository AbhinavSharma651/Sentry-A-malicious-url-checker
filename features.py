"""
Lexical feature extraction for malicious URL detection.
Pure string/structural analysis of the URL - no network calls, no
external lookups (WHOIS, DNS, etc). This means predictions are
instant and work fully offline, at the cost of not using
reputation/age-based signals.
"""

import re
import math
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure",
    "account", "update", "confirm", "bank", "password", "pay", "billing",
    "webscr", "wp-admin", "invoice", "click", "urgent", "suspend",
    "unlock", "reset", "free", "bonus", "gift", "prize",
]

ABUSED_TLDS = (
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click", ".link",
)

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "shorte.st", "cutt.ly", "rb.gy",
}

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)$"
)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _safe_has_port(parsed) -> int:
    try:
        return 1 if parsed.port else 0
    except (ValueError, Exception):
        return 0


_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _strip_scheme(url: str) -> str:
    """Remove any leading scheme (http://, https://, ftp://, ...).

    IMPORTANT: the training dataset has a strong collection artifact —
    ~96-100% of malware/defacement rows include a scheme prefix, but
    only ~8% of benign rows do (they were scraped without one). If we
    left the scheme in, the model would learn "has http(s)://" as a
    proxy for "malicious" - which falls apart on any real-world benign
    URL a user pastes in (which almost always includes the scheme).
    Stripping it uniformly here, for both training and inference,
    removes that spurious signal so the model has to rely on genuine
    structural/lexical patterns instead.
    """
    return _SCHEME_RE.sub("", url)


def _ensure_scheme(url: str) -> str:
    """urlparse needs *some* scheme to correctly split netloc from path."""
    if not _SCHEME_RE.match(url):
        return "http://" + url
    return url


def extract_features(raw_url: str) -> dict:
    raw_url = (raw_url or "").strip()
    raw_url = _strip_scheme(raw_url)
    url_for_parse = _ensure_scheme(raw_url)

    try:
        parsed = urlparse(url_for_parse)
        hostname = parsed.hostname or ""
    except Exception:
        parsed = urlparse("http://invalid")
        hostname = ""

    try:
        path = parsed.path or ""
    except Exception:
        path = ""
    try:
        query = parsed.query or ""
    except Exception:
        query = ""

    full = raw_url
    length = len(full)

    digits = sum(c.isdigit() for c in full)
    letters = sum(c.isalpha() for c in full)

    feats = {
        "url_length": length,
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": full.count("."),
        "num_hyphens": full.count("-"),
        "num_underscores": full.count("_"),
        "num_slashes": full.count("/"),
        "num_question_marks": full.count("?"),
        "num_equal_signs": full.count("="),
        "num_ampersands": full.count("&"),
        "num_at_signs": full.count("@"),
        "num_percent": full.count("%"),
        "num_digits": digits,
        "num_letters": letters,
        "digit_ratio": digits / length if length else 0.0,
        "has_ip_host": 1 if IP_PATTERN.match(hostname) else 0,
        "num_subdomains": max(hostname.count(".") - 1, 0) if hostname else 0,
        "hostname_hyphens": hostname.count("-"),
        "hostname_digits": sum(c.isdigit() for c in hostname),
        "has_abused_tld": 1 if hostname.lower().endswith(ABUSED_TLDS) else 0,
        "is_shortener": 1 if hostname.lower() in SHORTENERS else 0,
        "num_suspicious_words": sum(
            1 for w in SUSPICIOUS_WORDS if w in full.lower()
        ),
        "has_port": _safe_has_port(parsed),
        "entropy": _shannon_entropy(full),
        "path_depth": path.count("/"),
        "tld_length": len(hostname.split(".")[-1]) if "." in hostname else 0,
    }
    return feats


FEATURE_NAMES = list(extract_features("http://example.com/test").keys())


def extract_features_batch(urls):
    """Vectorized-ish helper for training: returns list of dicts."""
    return [extract_features(u) for u in urls]
