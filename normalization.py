import re
import unicodedata


# ============================================================
# COMPANY NAME NORMALIZATION
# ============================================================

COMPANY_SUFFIXES = {
    "incorporated": "inc",
    "inc": "inc",

    "corporation": "corp",
    "corp": "corp",

    "limited": "ltd",
    "ltd": "ltd",

    "company": "co",
    "co": "co",

    "llc": "llc",
    "l.l.c": "llc",

    "plc": "plc",

    "limited liability company": "llc",
}


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

ADDRESS_ABBREVIATIONS = {
    "street": "st",
    "st": "st",

    "road": "rd",
    "rd": "rd",

    "avenue": "ave",
    "ave": "ave",

    "boulevard": "blvd",
    "blvd": "blvd",

    "drive": "dr",
    "dr": "dr",

    "lane": "ln",
    "ln": "ln",

    "place": "pl",
    "pl": "pl",

    "parkway": "pkwy",
    "pkwy": "pkwy",

    "highway": "hwy",
    "hwy": "hwy",

    "court": "ct",
    "ct": "ct",

    "circle": "cir",
    "cir": "cir",

    "terrace": "ter",
    "ter": "ter",

    "square": "sq",
    "sq": "sq",
}


# ============================================================
# GENERIC TEXT NORMALIZATION
# ============================================================

def basic_normalize(text):
    """
    Basic normalization shared by names and addresses.

    Example:
        '  ACME Robotics, Inc.  '
        ->
        'acme robotics inc'
    """

    if text is None:
        return ""

    # Handle NaN safely
    if isinstance(text, float):
        try:
            if text != text:
                return ""
        except Exception:
            pass

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_name(name):
    """
    Returns multiple representations of a company/person name.

    We intentionally keep:
        - original normalized form
        - suffix-normalized form
        - core name without legal suffix
        - tokens
    """

    normalized = basic_normalize(name)

    if not normalized:
        return {
            "normalized": "",
            "core": "",
            "tokens": [],
        }

    tokens = normalized.split()

    # Normalize company suffixes
    normalized_tokens = []

    for token in tokens:
        normalized_tokens.append(
            COMPANY_SUFFIXES.get(token, token)
        )

    normalized_name = " ".join(normalized_tokens)

    # Remove legal/company suffixes for core representation
    suffixes = {
        "inc",
        "corp",
        "ltd",
        "co",
        "llc",
        "plc",
    }

    core_tokens = [
        token
        for token in normalized_tokens
        if token not in suffixes
    ]

    core_name = " ".join(core_tokens)

    return {
        "normalized": normalized_name,
        "core": core_name,
        "tokens": normalized_tokens,
    }


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_address(address):
    """
    Normalize an address while preserving useful information.
    """

    normalized = basic_normalize(address)

    if not normalized:
        return {
            "normalized": "",
            "tokens": [],
            "house_number": "",
        }

    tokens = normalized.split()

    # Normalize street/address abbreviations
    normalized_tokens = [
        ADDRESS_ABBREVIATIONS.get(token, token)
        for token in tokens
    ]

    normalized_address = " ".join(normalized_tokens)

    # Extract house/building number from beginning
    house_number = ""

    if normalized_tokens:
        match = re.match(
            r"^(\d+[a-zA-Z]?)",
            normalized_tokens[0]
        )

        if match:
            house_number = match.group(1)

    return {
        "normalized": normalized_address,
        "tokens": normalized_tokens,
        "house_number": house_number,
    }


# ============================================================
# TOKEN SET
# ============================================================

def get_token_set(text):
    """
    Convert text into a set of tokens.

    Useful for calculating token overlap later.
    """

    normalized = basic_normalize(text)

    if not normalized:
        return set()

    return set(normalized.split())


# ============================================================
# COMPLETE RECORD NORMALIZATION
# ============================================================

def normalize_record(name, address):
    """
    Normalize one complete entity record.
    """

    name_data = normalize_name(name)
    address_data = normalize_address(address)

    return {
        "name_normalized": name_data["normalized"],
        "name_core": name_data["core"],
        "name_tokens": name_data["tokens"],

        "address_normalized": address_data["normalized"],
        "address_tokens": address_data["tokens"],
        "house_number": address_data["house_number"],
    }


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    name = "Acme Robotics, Incorporated"
    address = "500 Market Street, San Jose CA"

    result = normalize_record(name, address)

    print("Name normalized:")
    print(result["name_normalized"])

    print("\nName core:")
    print(result["name_core"])

    print("\nAddress normalized:")
    print(result["address_normalized"])

    print("\nHouse number:")
    print(result["house_number"])
