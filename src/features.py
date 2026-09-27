import re
import numpy as np

from rapidfuzz import fuzz


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_string(value):
    """Convert missing values into empty strings."""

    if value is None:
        return ""

    if isinstance(value, float):

        if np.isnan(value):
            return ""

    return str(value).strip().lower()


def token_set(text):
    """Return normalized token set."""

    text = safe_string(text)

    if not text:
        return set()

    return set(
        token
        for token in text.split()
        if token
    )


# ============================================================
# STRING SIMILARITY
# ============================================================

def similarity_features(a, b, prefix):
    """
    Generate multiple fuzzy matching features.
    """

    a = safe_string(a)
    b = safe_string(b)

    if not a or not b:

        return {
            f"{prefix}_ratio": 0.0,
            f"{prefix}_partial_ratio": 0.0,
            f"{prefix}_token_sort": 0.0,
            f"{prefix}_token_set": 0.0,
        }

    return {

        # Normal edit similarity
        f"{prefix}_ratio":
            fuzz.ratio(a, b) / 100.0,

        # Useful when one representation is shorter
        f"{prefix}_partial_ratio":
            fuzz.partial_ratio(a, b) / 100.0,

        # Handles token ordering
        f"{prefix}_token_sort":
            fuzz.token_sort_ratio(a, b) / 100.0,

        # Handles extra/missing tokens
        f"{prefix}_token_set":
            fuzz.token_set_ratio(a, b) / 100.0,
    }


# ============================================================
# TOKEN FEATURES
# ============================================================

def token_features(a, b, prefix):

    tokens_a = token_set(a)
    tokens_b = token_set(b)

    if not tokens_a or not tokens_b:

        return {
            f"{prefix}_jaccard": 0.0,
            f"{prefix}_overlap": 0.0,
        }

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    jaccard = (
        len(intersection) /
        len(union)
        if union else 0.0
    )

    overlap = (
        len(intersection) /
        min(len(tokens_a), len(tokens_b))
        if min(len(tokens_a), len(tokens_b)) > 0
        else 0.0
    )

    return {
        f"{prefix}_jaccard": jaccard,
        f"{prefix}_overlap": overlap,
    }


# ============================================================
# EXACT MATCH
# ============================================================

def exact_features(a, b, prefix):

    a = safe_string(a)
    b = safe_string(b)

    return {
        f"{prefix}_exact": int(
            bool(a) and a == b
        )
    }


# ============================================================
# HOUSE NUMBER
# ============================================================

def extract_house_number(address):

    address = safe_string(address)

    if not address:
        return ""

    match = re.match(
        r"^\s*(\d+[a-zA-Z]?)",
        address
    )

    if match:
        return match.group(1)

    return ""


def house_number_features(address1, address2):

    h1 = extract_house_number(address1)
    h2 = extract_house_number(address2)

    return {
        "house_number_match": int(
            bool(h1)
            and bool(h2)
            and h1 == h2
        ),

        "house_number_missing": int(
            not h1 or not h2
        ),
    }


# ============================================================
# ADDRESS NUMBER + TEXT
# ============================================================

def address_numeric_features(address1, address2):

    numbers1 = set(
        re.findall(
            r"\b\d+[a-zA-Z]?\b",
            safe_string(address1)
        )
    )

    numbers2 = set(
        re.findall(
            r"\b\d+[a-zA-Z]?\b",
            safe_string(address2)
        )
    )

    if not numbers1 or not numbers2:

        return {
            "address_number_overlap": 0.0
        }

    intersection = numbers1 & numbers2

    return {
        "address_number_overlap":
            len(intersection) /
            len(numbers1 | numbers2)
    }


# ============================================================
# FIRST TOKEN
# ============================================================

def first_token(text):

    tokens = safe_string(text).split()

    return tokens[0] if tokens else ""


def first_token_features(name1, name2):

    a = first_token(name1)
    b = first_token(name2)

    return {
        "name_first_token_match": int(
            bool(a)
            and bool(b)
            and a == b
        )
    }


# ============================================================
# LENGTH FEATURES
# ============================================================

def length_features(a, b, prefix):

    a = safe_string(a)
    b = safe_string(b)

    return {

        f"{prefix}_length_difference":
            abs(len(a) - len(b)),

        f"{prefix}_length_ratio":
            (
                min(len(a), len(b)) /
                max(len(a), len(b))
                if max(len(a), len(b)) > 0
                else 0.0
            ),
    }


# ============================================================
# COMPLETE PAIR FEATURES
# ============================================================

def generate_pair_features(
    record1,
    record2
):
    """
    Generate all features for a Source 1 / candidate pair.
    """

    features = {}

    # --------------------------------------------------------
    # Names
    # --------------------------------------------------------

    name1 = record1.get(
        "name_normalized",
        record1.get("name", "")
    )

    name2 = record2.get(
        "name_normalized",
        record2.get("name", "")
    )

    name_core1 = record1.get(
        "name_core",
        name1
    )

    name_core2 = record2.get(
        "name_core",
        name2
    )

    features.update(
        similarity_features(
            name1,
            name2,
            "name"
        )
    )

    features.update(
        similarity_features(
            name_core1,
            name_core2,
            "name_core"
        )
    )

    features.update(
        token_features(
            name1,
            name2,
            "name"
        )
    )

    features.update(
        exact_features(
            name1,
            name2,
            "name"
        )
    )

    features.update(
        first_token_features(
            name_core1,
            name_core2
        )
    )

    features.update(
        length_features(
            name1,
            name2,
            "name"
        )
    )

    # --------------------------------------------------------
    # Addresses
    # --------------------------------------------------------

    address1 = record1.get(
        "address_normalized",
        record1.get("address", "")
    )

    address2 = record2.get(
        "address_normalized",
        record2.get("address", "")
    )

    features.update(
        similarity_features(
            address1,
            address2,
            "address"
        )
    )

    features.update(
        token_features(
            address1,
            address2,
            "address"
        )
    )

    features.update(
        exact_features(
            address1,
            address2,
            "address"
        )
    )

    features.update(
        length_features(
            address1,
            address2,
            "address"
        )
    )

    # --------------------------------------------------------
    # House number
    # --------------------------------------------------------

    features.update(
        house_number_features(
            address1,
            address2
        )
    )

    features.update(
        address_numeric_features(
            address1,
            address2
        )
    )

    return features
