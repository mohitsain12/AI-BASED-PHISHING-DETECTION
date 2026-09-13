import re
import math
import requests
import ipaddress
from urllib.parse import urlparse
from collections import Counter


# ============================================================
# 22 FEATURES USED BY THE AI PHISHING DETECTION MODEL
# ============================================================

ADDRESS_BAR_FEATURES = [
    "URLLength",
    "IsDomainIP",
    "HasObfuscation",
    "NoOfSubDomain",
    "IsHTTPS",
    "URLSimilarityIndex",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "SpacialCharRatioInURL"
]

DOMAIN_FEATURES = [
    "DomainLength",
    "TLDLegitimateProb",
    "CharContinuationRate",
    "URLCharProb",
    "TLDLength"
]

HTML_JS_FEATURES = [
    "NoOfiFrame",
    "NoOfPopup",
    "HasHiddenFields",
    "NoOfSelfRedirect",
    "HasExternalFormSubmit",
    "HasSubmitButton",
    "HasPasswordField",
    "HasSocialNet"
]

ALL_FEATURES = (
    ADDRESS_BAR_FEATURES
    + DOMAIN_FEATURES
    + HTML_JS_FEATURES
)


# ============================================================
# BASIC URL FUNCTIONS
# ============================================================

def get_domain(url):
    """Extract domain name from URL."""

    try:
        parsed = urlparse(url)

        domain = parsed.netloc.lower()

        # Remove username/password if present
        if "@" in domain:
            domain = domain.split("@")[-1]

        # Remove port number
        domain = domain.split(":")[0]

        return domain

    except Exception:
        return ""


def get_tld(domain):
    """Extract top-level domain."""

    try:
        parts = domain.split(".")

        if len(parts) >= 2:
            return parts[-1]

        return ""

    except Exception:
        return ""


# ============================================================
# ADDRESS BAR FEATURES
# ============================================================

def url_length(url):
    """Feature 1: Length of URL."""

    return len(url)


def is_domain_ip(url):
    """Feature 2: Check whether domain is an IP address."""

    domain = get_domain(url)

    try:
        ipaddress.ip_address(domain)
        return 1

    except ValueError:
        return 0


def has_obfuscation(url):
    """
    Feature 3: Detect URL obfuscation.

    Looks for:
    - Excessive hexadecimal encoding
    - @ symbol
    - Multiple backslashes
    - Suspicious percent encoding
    """

    hex_encoding = len(re.findall(r"%[0-9a-fA-F]{2}", url))
    at_symbol = "@" in url
    backslashes = url.count("\\")

    if hex_encoding >= 3 or at_symbol or backslashes >= 2:
        return 1

    return 0


def number_of_subdomains(url):
    """Feature 4: Number of subdomains."""

    domain = get_domain(url)

    if not domain:
        return 0

    parts = domain.split(".")

    # Remove empty components
    parts = [part for part in parts if part]

    # domain + TLD = minimum 2 parts
    if len(parts) <= 2:
        return 0

    return len(parts) - 2


def is_https(url):
    """Feature 5: Check whether URL uses HTTPS."""

    return 1 if url.lower().startswith("https://") else 0


def url_similarity_index(url):
    """
    Feature 6: URL similarity index.

    Measures similarity between characters in the URL.
    Higher repetition can indicate suspicious URLs.
    """

    if not url:
        return 0.0

    characters = Counter(url)

    total = len(url)

    repeated = sum(
        count
        for count in characters.values()
        if count > 1
    )

    return round((repeated / total) * 100, 4)


def number_of_equals(url):
    """Feature 7: Number of '=' characters."""

    return url.count("=")


def number_of_question_marks(url):
    """Feature 8: Number of '?' characters."""

    return url.count("?")


def special_character_ratio(url):
    """
    Feature 9: Ratio of special characters in URL.

    Special characters include:
    ! @ # $ % ^ & * ( ) - _ = + [ ] { } ; : ' " , < > / ?
    """

    if not url:
        return 0.0

    special_chars = re.findall(
        r"[^a-zA-Z0-9]",
        url
    )

    return round(
        len(special_chars) / len(url),
        4
    )


# ============================================================
# DOMAIN FEATURES
# ============================================================

def domain_length(url):
    """Feature 10: Length of domain."""

    domain = get_domain(url)

    return len(domain)


def tld_legitimate_probability(url):
    """
    Feature 11: TLD legitimacy probability.

    Common legitimate TLDs receive higher values.
    """

    domain = get_domain(url)
    tld = get_tld(domain)

    legitimate_tlds = {
        "com": 0.95,
        "org": 0.90,
        "net": 0.85,
        "edu": 0.98,
        "gov": 0.99,
        "in": 0.90,
        "co": 0.80,
        "uk": 0.90,
        "de": 0.88,
        "au": 0.88,
        "ca": 0.88,
        "us": 0.85,
        "info": 0.55,
        "biz": 0.50,
        "xyz": 0.35,
        "top": 0.30,
        "online": 0.35,
        "site": 0.35,
        "click": 0.25
    }

    return legitimate_tlds.get(tld, 0.50)


def character_continuation_rate(url):
    """
    Feature 12: Character continuation rate.

    Measures repeated consecutive characters.
    """

    if len(url) < 2:
        return 0.0

    continuation = 0

    for i in range(1, len(url)):
        if url[i] == url[i - 1]:
            continuation += 1

    return round(
        continuation / (len(url) - 1),
        4
    )


def url_character_probability(url):
    """
    Feature 13: Character probability.

    Estimates how unusual the URL characters are.
    """

    if not url:
        return 0.0

    allowed_chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "./_-?=&:"
    )

    valid = sum(
        1 for char in url
        if char in allowed_chars
    )

    return round(
        valid / len(url),
        4
    )


def tld_length(url):
    """Feature 14: Length of TLD."""

    domain = get_domain(url)
    tld = get_tld(domain)

    return len(tld)


# ============================================================
# HTML / JAVASCRIPT FEATURES
# ============================================================

def fetch_html(url):
    """
    Download webpage HTML.

    Returns empty string if webpage cannot be accessed.
    """

    try:
        response = requests.get(
            url,
            timeout=(3, 3),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120 Safari/537.36"
                )
            }
        )

        if response.status_code == 200:
            return response.text

    except Exception:
        pass

    return ""


def number_of_iframes(html):
    """Feature 15: Number of iframe tags."""

    if not html:
        return 0

    return len(
        re.findall(
            r"<iframe\b",
            html,
            re.IGNORECASE
        )
    )


def number_of_popups(html):
    """
    Feature 16: Number of popup indicators.
    """

    if not html:
        return 0

    popup_patterns = [
        r"window\.open\s*\(",
        r"alert\s*\(",
        r"prompt\s*\(",
        r"confirm\s*\("
    ]

    count = 0

    for pattern in popup_patterns:
        count += len(
            re.findall(
                pattern,
                html,
                re.IGNORECASE
            )
        )

    return count


def has_hidden_fields(html):
    """
    Feature 17: Detect hidden HTML form fields.
    """

    if not html:
        return 0

    pattern = (
        r'<input[^>]+'
        r'type\s*=\s*["\']hidden["\']'
    )

    return 1 if re.search(
        pattern,
        html,
        re.IGNORECASE
    ) else 0


def number_of_self_redirects(html, url):
    """
    Feature 18: Number of redirects/links
    pointing back to the same domain.
    """

    if not html:
        return 0

    domain = get_domain(url)

    if not domain:
        return 0

    links = re.findall(
        r'href\s*=\s*["\']([^"\']+)["\']',
        html,
        re.IGNORECASE
    )

    count = 0

    for link in links:

        if domain in link:
            count += 1

    return count


def has_external_form_submit(html, url):
    """
    Feature 19: Detect forms submitting data
    to an external domain.
    """

    if not html:
        return 0

    domain = get_domain(url)

    forms = re.findall(
        r'<form[^>]*action\s*=\s*["\']([^"\']+)["\']',
        html,
        re.IGNORECASE
    )

    for action in forms:

        action_domain = get_domain(action)

        if action_domain and action_domain != domain:
            return 1

    return 0


def has_submit_button(html):
    """
    Feature 20: Detect submit buttons.
    """

    if not html:
        return 0

    patterns = [
        r'<input[^>]+type\s*=\s*["\']submit["\']',
        r'<button[^>]*type\s*=\s*["\']submit["\']',
        r'<button[^>]*>'
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            html,
            re.IGNORECASE
        ):
            return 1

    return 0


def has_password_field(html):
    """
    Feature 21: Detect password input field.
    """

    if not html:
        return 0

    pattern = (
        r'<input[^>]+'
        r'type\s*=\s*["\']password["\']'
    )

    return 1 if re.search(
        pattern,
        html,
        re.IGNORECASE
    ) else 0


def has_social_network(html):
    """
    Feature 22: Detect links to social networks.
    """

    if not html:
        return 0

    social_domains = [
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "x.com",
        "linkedin.com",
        "youtube.com",
        "tiktok.com",
        "pinterest.com",
        "reddit.com"
    ]

    html_lower = html.lower()

    for social_domain in social_domains:

        if social_domain in html_lower:
            return 1

    return 0


# ============================================================
# MAIN FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(url):
    """
    Extract all 22 phishing detection features.

    Returns:
        Dictionary containing all 22 features.
    """

    # Make sure URL has a scheme
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    # Fetch webpage
    html = fetch_html(url)

    features = {

        # ----------------------------------------------------
        # Address Bar Features
        # ----------------------------------------------------

        "URLLength":
            url_length(url),

        "IsDomainIP":
            is_domain_ip(url),

        "HasObfuscation":
            has_obfuscation(url),

        "NoOfSubDomain":
            number_of_subdomains(url),

        "IsHTTPS":
            is_https(url),

        "URLSimilarityIndex":
            url_similarity_index(url),

        "NoOfEqualsInURL":
            number_of_equals(url),

        "NoOfQMarkInURL":
            number_of_question_marks(url),

        "SpacialCharRatioInURL":
            special_character_ratio(url),

        # ----------------------------------------------------
        # Domain Features
        # ----------------------------------------------------

        "DomainLength":
            domain_length(url),

        "TLDLegitimateProb":
            tld_legitimate_probability(url),

        "CharContinuationRate":
            character_continuation_rate(url),

        "URLCharProb":
            url_character_probability(url),

        "TLDLength":
            tld_length(url),

        # ----------------------------------------------------
        # HTML / JavaScript Features
        # ----------------------------------------------------

        "NoOfiFrame":
            number_of_iframes(html),

        "NoOfPopup":
            number_of_popups(html),

        "HasHiddenFields":
            has_hidden_fields(html),

        "NoOfSelfRedirect":
            number_of_self_redirects(
                html,
                url
            ),

        "HasExternalFormSubmit":
            has_external_form_submit(
                html,
                url
            ),

        "HasSubmitButton":
            has_submit_button(html),

        "HasPasswordField":
            has_password_field(html),

        "HasSocialNet":
            has_social_network(html)
    }

    return features


# ============================================================
# RETURN FEATURES IN MODEL ORDER
# ============================================================

def extract_feature_vector(url):
    """
    Extract features and return them as a list
    in the exact order expected by the ML model.
    """

    features = extract_features(url)

    return [
        features[name]
        for name in ALL_FEATURES
    ]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_url = "https://www.google.com"

    print("\nAI PHISHING DETECTION")
    print("=" * 50)

    features = extract_features(test_url)

    for number, name in enumerate(ALL_FEATURES, start=1):

        print(
            f"{number:02d}. "
            f"{name:<25} : "
            f"{features[name]}"
        )

    print("=" * 50)

    print("\nFeature Vector:")
    print(extract_feature_vector(test_url))