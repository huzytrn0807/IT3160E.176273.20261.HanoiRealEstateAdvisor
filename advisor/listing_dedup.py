"""Group likely reposts for display while retaining the selected source record."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import district_key


BLOCK_FIELDS = ("property_group", "district_name", "price_vnd", "area_m2")
TEXT_FIELDS = ("ward_name", "street_name", "project_name", "house_direction", "balcony_direction")
NUMERIC_FIELDS = ("bedroom_count", "bathroom_count", "floor_count", "frontage_m", "road_width_m", "depth_m")
DETAIL_FIELDS = (*TEXT_FIELDS, *NUMERIC_FIELDS)
RELATED_COUNT = "related_listing_count"
AREA_PATTERN = re.compile(r"(?<=\d)\s*m(?:\s*\^?2\b|\b)")
BED_PATTERN = re.compile(r"(?<=\d)\s*(?:phong\s+ngu|pn|ngu)\b")
BATH_PATTERN = re.compile(r"(?<=\d)\s*(?:phong\s+tam|ve\s+sinh|wc)\b")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.,]\d+)?")
MEASUREMENT_ROLES = ("m2", "pn", "wc", "tang")
UNIT_ID_PATTERN = re.compile(
    r"\b(?:ma\s+can|can(?:\s+ho)?\s+so|can)\s*[:#]?\s*"
    r"([a-z]{1,4}(?:[-. ]?[a-z]{1,3})?[-. ]?\d{2,5})(?![a-z0-9])"
)
COMMON_TITLE_WORDS = frozenset(
    "can ban chinh chu gap can ho nha mat dat chung cu dep gia chi ty trieu "
    "quan huyen khu vuc phong ngu m2 pn wc lien he moi full noi that".split()
)


@lru_cache(maxsize=8192)
def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold().replace("đ", "d"))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _text_key(value: object) -> str | None:
    text = _string(value)
    if not text:
        return None
    return " ".join(TOKEN_PATTERN.findall(_fold(text))) or None


def _number_key(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@dataclass(frozen=True)
class _Title:
    text: str
    tokens: frozenset[str]
    numbers: tuple[str, ...]
    unit_ids: frozenset[str]
    measurements: tuple[frozenset[str], ...]


def _title(value: object) -> _Title:
    folded = _fold(_string(value))
    units = frozenset(re.sub(r"[^a-z0-9]", "", match) for match in UNIT_ID_PATTERN.findall(folded))
    text = AREA_PATTERN.sub(" m2", folded)
    text = BED_PATTERN.sub(" pn", text)
    text = BATH_PATTERN.sub(" wc", text)
    words = tuple(token.replace(",", ".") for token in TOKEN_PATTERN.findall(text))
    numbers = tuple(sorted(word for word in words if word[0].isdigit()))
    measured = {
        role: set() for role in MEASUREMENT_ROLES
    }
    for number, role in zip(words, words[1:]):
        if role in measured and number[0].isdigit():
            measured[role].add(number)
    return _Title(
        " ".join(words), frozenset(words), numbers, units,
        tuple(frozenset(measured[role]) for role in MEASUREMENT_ROLES),
    )


@dataclass(frozen=True)
class _Listing:
    position: int
    title: _Title
    details: tuple
    completeness: int
    bedrooms_known: bool
    date: int
    listing_id: str


def _compatible(left: _Listing, right: _Listing) -> bool:
    return all(a is None or b is None or a == b for a, b in zip(left.details, right.details))


def _same_listing(left: _Listing, right: _Listing) -> bool:
    if not _compatible(left, right):
        return False
    a, b = left.title, right.title
    if a.unit_ids and b.unit_ids and a.unit_ids != b.unit_ids:
        return False
    if a.numbers != b.numbers:
        return False
    if any(x and y and x != y for x, y in zip(a.measurements, b.measurements)):
        return False
    if len(a.tokens & b.tokens) < 5:
        return False
    shared_terms = {
        word for word in (a.tokens & b.tokens) - COMMON_TITLE_WORDS
        if len(word) > 1 and word[0].isalpha()
    }
    if len(shared_terms) < 2 and not (a.unit_ids & b.unit_ids):
        return False
    if a.text == b.text or a.tokens == b.tokens:
        return True
    similarity = len(a.tokens & b.tokens) / len(a.tokens | b.tokens)
    return similarity >= 0.78 and SequenceMatcher(None, a.text, b.text, autojunk=False).ratio() >= 0.86


def _candidates(frame: pd.DataFrame, positions: np.ndarray) -> list[_Listing]:
    columns = ["listing_id", "name", "published_at", *DETAIL_FIELDS]
    values = frame.iloc[positions].reindex(columns=columns)
    dates = pd.to_datetime(values["published_at"], format="mixed", utc=True, errors="coerce")
    records = []
    for position, date, row in zip(positions, dates, values.itertuples(index=False, name=None)):
        listing_id, title, _, *details = row
        text = tuple(_text_key(value) for value in details[:len(TEXT_FIELDS)])
        numbers = tuple(_number_key(value) for value in details[len(TEXT_FIELDS):])
        known = (*text, *numbers)
        records.append(_Listing(
            int(position), _title(title), known, sum(value is not None for value in known),
            numbers[0] is not None, date.value if pd.notna(date) else -2**63, _string(listing_id),
        ))
    return records


def _groups(candidates: list[_Listing]) -> list[list[_Listing]]:
    ranked = sorted(candidates, key=lambda row: (
        not row.bedrooms_known, -row.completeness, -row.date, row.listing_id, row.position
    ))
    groups: list[list[_Listing]] = []
    for row in ranked:
        for group in groups:
            # Comparing all members prevents a missing field from bridging two conflicts.
            if all(_same_listing(row, member) for member in group):
                group.append(row)
                break
        else:
            groups.append([row])
    return groups


def deduplicate_listings(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one original row per likely repost group for customer-facing results.

    The group requires the same property type, district, source price and area.
    Titles must be similar and any two nonempty details must agree. A group
    retains the row with known bedrooms and the most detail, then the latest date.
    """
    required = {"name", *BLOCK_FIELDS}
    if missing := required - set(frame.columns):
        raise ValueError(f"Thiếu cột để gom tin: {', '.join(sorted(missing))}")

    counts = np.ones(len(frame), dtype=np.int64)
    keep = np.ones(len(frame), dtype=bool)
    blocks = frame.loc[:, list(BLOCK_FIELDS)].reset_index(drop=True).copy()
    districts = blocks["district_name"]
    labels = {name: district_key(name) for name in districts.dropna().unique() if _string(name)}
    blocks["district_name"] = districts.map(labels)
    valid = blocks["district_name"].fillna("").ne("") & blocks[list(BLOCK_FIELDS)].notna().all(axis=1)
    repeated = valid & blocks.duplicated(keep=False)
    positions = np.flatnonzero(repeated.to_numpy())
    candidates = {row.position: row for row in _candidates(frame, positions)}

    for _, block in blocks.loc[repeated].groupby(list(BLOCK_FIELDS), sort=False, observed=True):
        for group in _groups([candidates[position] for position in block.index]):
            representative, *duplicates = group
            counts[representative.position] = len(group)
            for duplicate in duplicates:
                keep[duplicate.position] = False

    result = frame.iloc[np.flatnonzero(keep)].copy()
    result[RELATED_COUNT] = counts[keep]
    return result
