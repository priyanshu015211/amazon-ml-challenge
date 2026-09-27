import re
import pandas as pd
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

# Number of characters used for name prefix blocking
NAME_PREFIX_LENGTH = 4


# ============================================================
# HELPERS
# ============================================================

def clean_block_value(value):
    """Convert a value into a safe blocking key."""

    if value is None:
        return ""

    value = str(value).lower().strip()

    # Keep only alphanumeric characters
    value = re.sub(r"[^a-z0-9]", "", value)

    return value


def get_name_prefix(name, length=NAME_PREFIX_LENGTH):
    """Return first N characters of normalized name."""

    value = clean_block_value(name)

    if not value:
        return ""

    return value[:length]


def get_first_token(name):
    """Return first token of normalized name."""

    if not name:
        return ""

    tokens = str(name).lower().split()

    return clean_block_value(tokens[0]) if tokens else ""


def get_house_number(address):
    """Extract leading house/building number."""

    if not address:
        return ""

    match = re.match(
        r"^\s*(\d+[a-zA-Z]?)",
        str(address)
    )

    return match.group(1).lower() if match else ""


def get_address_tokens(address):
    """Return useful address tokens."""

    if not address:
        return []

    return [
        clean_block_value(token)
        for token in str(address).split()
        if clean_block_value(token)
    ]


# ============================================================
# BLOCKING KEY GENERATION
# ============================================================

def generate_block_keys(row):
    """
    Generate multiple blocking keys for one record.

    Multiple keys are used because no single blocking strategy
    should determine whether a true match is considered.
    """

    name = row.get("name_normalized", "")
    name_core = row.get("name_core", "")
    address = row.get("address_normalized", "")

    keys = []

    # --------------------------------------------------------
    # 1. Name prefix
    # --------------------------------------------------------

    name_prefix = get_name_prefix(name)

    if name_prefix:
        keys.append(
            ("name_prefix", name_prefix)
        )

    # --------------------------------------------------------
    # 2. First name token
    # --------------------------------------------------------

    first_token = get_first_token(name_core or name)

    if first_token:
        keys.append(
            ("first_token", first_token)
        )

    # --------------------------------------------------------
    # 3. House number + name prefix
    # --------------------------------------------------------

    house_number = row.get(
        "house_number",
        ""
    )

    if not house_number:
        house_number = get_house_number(address)

    if house_number and name_prefix:

        keys.append(
            (
                "house_name",
                f"{house_number}_{name_prefix}"
            )
        )

    # --------------------------------------------------------
    # 4. House number + first name token
    # --------------------------------------------------------

    if house_number and first_token:

        keys.append(
            (
                "house_first_token",
                f"{house_number}_{first_token}"
            )
        )

    # --------------------------------------------------------
    # 5. Address token blocks
    # --------------------------------------------------------

    address_tokens = get_address_tokens(address)

    # Only use reasonably informative tokens
    stop_tokens = {
        "st",
        "street",
        "rd",
        "road",
        "ave",
        "avenue",
        "dr",
        "drive",
        "ln",
        "lane",
        "blvd",
        "boulevard",
        "city",
        "the",
    }

    useful_tokens = [
        token
        for token in address_tokens
        if len(token) >= 4
        and token not in stop_tokens
        and not token.isdigit()
    ]

    for token in useful_tokens[:3]:

        keys.append(
            ("address_token", token)
        )

    return keys


# ============================================================
# BUILD BLOCK INDEX
# ============================================================

def build_block_index(df):
    """
    Build an inverted index.

    Example:

        ("name_prefix", "acme")
                ->
        [S2-1, S2-5, S2-19]
    """

    index = defaultdict(set)

    for idx, row in df.iterrows():

        keys = generate_block_keys(row)

        for key_type, key_value in keys:

            index[
                (key_type, key_value)
            ].add(idx)

    return index


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    source1,
    source2,
    source3
):
    """
    Generate candidate pairs between Source 1 and
    Source 2 / Source 3.

    Returns:

        candidate_pairs dataframe
    """

    source2_index = build_block_index(source2)
    source3_index = build_block_index(source3)

    candidates = []

    # --------------------------------------------------------
    # Source 1 -> Source 2
    # --------------------------------------------------------

    for s1_idx, s1_row in source1.iterrows():

        keys = generate_block_keys(s1_row)

        matched_s2 = set()

        for key in keys:

            matched_s2.update(
                source2_index.get(key, set())
            )

        for s2_idx in matched_s2:

            candidates.append({
                "source1_idx": s1_idx,
                "source2_idx": s2_idx,
                "source": "source2",
            })

    # --------------------------------------------------------
    # Source 1 -> Source 3
    # --------------------------------------------------------

    for s1_idx, s1_row in source1.iterrows():

        keys = generate_block_keys(s1_row)

        matched_s3 = set()

        for key in keys:

            matched_s3.update(
                source3_index.get(key, set())
            )

        for s3_idx in matched_s3:

            candidates.append({
                "source1_idx": s1_idx,
                "source3_idx": s3_idx,
                "source": "source3",
            })

    return pd.DataFrame(candidates)


# ============================================================
# AUDIT INFORMATION
# ============================================================

def add_candidate_ids(
    candidates,
    source1,
    source2,
    source3
):
    """
    Add actual entity IDs to candidate pairs.
    """

    output = candidates.copy()

    output["source1_id"] = output[
        "source1_idx"
    ].map(source1["id"])

    output["candidate_id"] = ""

    mask2 = output["source"] == "source2"

    output.loc[mask2, "candidate_id"] = (
        output.loc[mask2, "source2_idx"]
        .map(source2["id"])
    )

    mask3 = output["source"] == "source3"

    output.loc[mask3, "candidate_id"] = (
        output.loc[mask3, "source3_idx"]
        .map(source3["id"])
    )

    return output[
        [
            "source1_id",
            "candidate_id",
            "source"
        ]
    ].drop_duplicates()
