"""Build a UTF-8 Hanoi sale-listing CSV from the supplied Kaggle export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor.property_types import COLLECTIVE_HOUSING, property_group_for_title

SOURCE = "Kaggle: yashdev01/vietnam-real-estates"
SOURCE_URL = "https://www.kaggle.com/datasets/yashdev01/vietnam-real-estates"
TYPE_GROUPS = {
    "can ho chung cu": "apartment",
    "nha tap the": COLLECTIVE_HOUSING,
    "can ho tap the": COLLECTIVE_HOUSING,
    "tap the": COLLECTIVE_HOUSING,
    "nha": "house",
    "biet thu/nha lien ke": "villa_townhouse",
    "shophouse": "shophouse",
}
LOCATION = ("province_name", "district_name", "ward_name", "street_name", "project_name")
COUNTS = ("floor_count", "bedroom_count", "bathroom_count")
DIMENSIONS = {"frontage_width": "frontage_m", "house_depth": "depth_m", "road_width": "road_width_m"}
DIRECTIONS = ("house_direction", "balcony_direction")
CSV_TEXT_FIELDS = ("name", "property_type_name", *LOCATION, *DIRECTIONS)
REQUIRED = {
    "name", "description", "property_type_name", "price", "area", "published_at",
    *LOCATION, *COUNTS, *DIMENSIONS, *DIRECTIONS,
}
FIELDS = (
    "listing_id", "name", "property_type_name", "property_group", *LOCATION,
    "price_vnd", "area_m2", "price_per_m2_vnd", *COUNTS, *DIMENSIONS.values(), *DIRECTIONS,
    "published_at", "price_type", "listing_purpose", "purpose_evidence", "source",
    "source_url", "data_origin", "dac_diem_noi_bat", "quality_flags", "source_record",
)

# Signals from normalized Vietnamese text. Unclear or mixed intent is held out.
SELL_START = re.compile(
    r"^(?:(?:chinh chu|can|muon|can gap|nhan|khach|toi|gia dinh|chu nha|"
    r"gia that|ban chinh chu)\s+)*(?:ban\b(?!\s+(?:giao|cong|kinh|ngay\s+thang)\b)|chuyen nhuong\b)"
)
SELL_TEXT = re.compile(
    r"\b(?:gia ban|can ban|chinh chu ban|ban gap|rao ban|chuyen nhuong|"
    r"chu nha ban|gui ban|mo ban|ban (?:can ho|nha|dat|biet thu|shophouse|chung cu))\b"
)
RENT_START = re.compile(r"^(?:chinh chu\s+|can\s+|gap\s+|dang\s+)*(?:cho thue|thue)\b")
RENT_PRICE = re.compile(
    r"\b(?:gia (?:cho )?thue|cho thue voi gia)\b|"
    r"\b\d+(?:[.,]\d+)?\s*(?:trieu|tr|k|nghin|ngan)\s*(?:dong\s*)?(?:/|moi\s+)\s*thang\b"
)
HIGHLIGHTS = (
    (r"\bcan goc\b", "Căn góc"),
    (r"\b(?:full (?:noi that|do)|day du noi that|noi that day du)\b", "Đầy đủ nội thất"),
    (r"\b(?:cho do o to|slot o to|cho dau o to)\b", "Có nhắc chỗ đỗ ô tô"),
    (r"\b(?:so do|so hong)\b", "Có nhắc sổ đỏ/sổ hồng"),
    (r"\b(?:tang cao|tang trung|tang thap)\b", "Có mô tả nhóm tầng"),
    (r"\b(?:gan truong|gan benh vien|gan sieu thi|gan cho)\b", "Có nhắc tiện ích gần"),
)
CONTACT = re.compile(
    r"(?i)\[[^\]]*(?:phone|email)[^\]]*\]|"
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|"
    r"(?<!\w)(?:\+?84|0)(?:[. ()-]*\d){9}(?![. ()-]*\d)"
)


def clean_text(value: str | None) -> str:
    return " ".join(unicodedata.normalize("NFC", value or "").split())


def normalized(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value).casefold().replace("đ", "d"))
    return "".join(char for char in text if not unicodedata.combining(char))


def redact(value: str | None) -> str:
    return CONTACT.sub("[liên hệ]", clean_text(value))


def safe_csv_row(listing: dict) -> dict:
    """Keep source text from being evaluated as a formula when opened in Excel."""
    return {
        **listing,
        **{field: "'" + listing[field] for field in CSV_TEXT_FIELDS if listing[field].startswith(("=", "+", "-", "@"))},
    }


def purpose(title: str, description: str) -> tuple[str, str]:
    title_key = re.sub(r"[^a-z0-9/., ]", " ", normalized(title)).strip()
    title_key = " ".join(title_key.split())
    description_key = normalized(description) if description else ""
    sale_in_title = bool(SELL_START.search(title_key) or SELL_TEXT.search(title_key))
    rental_in_title = bool(RENT_START.search(title_key) or RENT_PRICE.search(title_key))
    if sale_in_title and rental_in_title:
        return "ambiguous", "mixed_title_signals"
    if rental_in_title:
        return "rent", "title_rent_signal"
    if sale_in_title:
        if RENT_PRICE.search(description_key) and not SELL_START.search(title_key):
            return "ambiguous", "sale_title_rent_price"
        return "sale", "title_sale_signal"
    if RENT_PRICE.search(description_key):
        return "ambiguous", "rent_price_without_clear_title"
    if SELL_TEXT.search(description_key):
        return "sale", "description_sale_signal"
    return "unknown", "no_clear_sale_signal"


def number(raw: str | None, *, integer: bool = False, allow_zero: bool = False) -> int | float | None:
    value = clean_text(raw)
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed < 0 or (parsed == 0 and not allow_zero):
        return None
    if integer and not parsed.is_integer():
        return None
    return int(parsed) if integer else parsed


def published_date(raw: str | None) -> str:
    try:
        return datetime.fromisoformat(clean_text(raw)).isoformat()
    except ValueError:
        return ""


def fingerprint(listing: dict, description: str) -> str:
    # Include the body and optional fields to avoid collapsing similar apartments.
    keys = ("name", "property_group", *LOCATION, "price_vnd", "area_m2", *COUNTS, *DIMENSIONS.values(), *DIRECTIONS)
    payload = [normalized(str(listing[key])) if listing[key] is not None else "" for key in keys]
    payload.append(normalized(redact(description)))
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def prepare(row: dict, record: int, group: str, evidence: str) -> tuple[dict | None, str]:
    price = number(row["price"], integer=True)
    area = number(row["area"])
    if price is None or area is None:
        return None, "missing_price_or_area" if not clean_text(row["price"]) or not clean_text(row["area"]) else "invalid_price_or_area"
    unit_price = price / area
    if not math.isfinite(unit_price):
        return None, "invalid_price_or_area"
    title = redact(row["name"])
    resolved_group = property_group_for_title(group, title)
    title_key = normalized(title)
    listing = {key: redact(row[key]) for key in (*LOCATION, *DIRECTIONS)}
    listing["province_name"] = "Hà Nội"
    listing["published_at"] = published_date(row["published_at"])
    flags = []
    if not listing["district_name"]:
        flags.append("missing_district")
    if not listing["published_at"]:
        flags.append("missing_or_invalid_published_at")
    if evidence == "description_sale_signal":
        flags.append("sale_inferred_from_description")
    if unit_price < 1_000_000 or unit_price > 1_000_000_000:
        flags.append("unit_price_review")
    if area < 10 or (resolved_group in {"apartment", COLLECTIVE_HOUSING} and area > 500):
        flags.append("area_review")
    for key in COUNTS:
        listing[key] = number(row[key], integer=True, allow_zero=key != "floor_count")
        if clean_text(row[key]) and listing[key] is None:
            flags.append(f"invalid_{key}")
    for source, target in DIMENSIONS.items():
        listing[target] = number(row[source])
        if clean_text(row[source]) and listing[target] is None:
            flags.append(f"invalid_{source}")
    listing.update(
        name=title, property_type_name=clean_text(row["property_type_name"]), property_group=group,
        price_vnd=price, area_m2=area, price_per_m2_vnd=round(unit_price, 2),
        price_type="asking", listing_purpose="sale", purpose_evidence=evidence, source=SOURCE,
        source_url=SOURCE_URL, data_origin="kaggle_listing_dataset",
        dac_diem_noi_bat=" | ".join(label for pattern, label in HIGHLIGHTS if re.search(pattern, title_key)),
        quality_flags="|".join(flags), source_record=record,
    )
    listing["listing_id"] = fingerprint(listing, row["description"])
    # Preserve IDs generated from the structured source group in older builds.
    listing["property_group"] = resolved_group
    return listing, ""


def build(input_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    groups: Counter = Counter()
    hanoi_types: Counter = Counter()
    coverage: dict[str, Counter] = {}
    districts: Counter = Counter()
    quality: Counter = Counter()
    evidence_counts: Counter = Counter()
    earliest = latest = ""
    seen: set[str] = set()
    with tempfile.TemporaryDirectory(prefix=".hanoi-", dir=output_dir) as tmp:
        staging = Path(tmp)
        with (
            input_path.open(encoding="utf-8-sig", newline="") as source,
            (staging / "hanoi_properties_final.csv").open("w", encoding="utf-8-sig", newline="") as destination,
        ):
            reader = csv.DictReader(source)
            missing = REQUIRED - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Thiếu cột nguồn: {', '.join(sorted(missing))}")
            writer = csv.DictWriter(destination, fieldnames=FIELDS)
            writer.writeheader()
            for record, row in enumerate(reader, 1):
                counts["input_rows"] += 1
                if normalized(row["province_name"]) not in {"ha noi", "thanh pho ha noi", "tp ha noi", "tp. ha noi"}:
                    continue
                counts["hanoi_rows"] += 1
                hanoi_types[clean_text(row["property_type_name"]) or "(blank)"] += 1
                group = TYPE_GROUPS.get(normalized(row["property_type_name"]))
                if group is None:
                    counts["out_of_scope_property_type"] += 1
                    continue
                kind, evidence = purpose(row["name"], row["description"])
                counts[f"purpose_{kind}"] += 1
                if kind != "sale":
                    continue
                listing, reason = prepare(row, record, group, evidence)
                if listing is None:
                    counts[reason] += 1
                    continue
                listing_id = listing["listing_id"]
                if listing_id in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(listing_id)
                writer.writerow(safe_csv_row(listing))
                counts["output_rows"] += 1
                groups[listing["property_group"]] += 1
                evidence_counts[evidence] += 1
                districts[listing["district_name"] or "(blank)"] += 1
                quality.update(flag for flag in listing["quality_flags"].split("|") if flag)
                by_group = coverage.setdefault(listing["property_group"], Counter())
                by_group.update(key for key in FIELDS if listing[key] is None or listing[key] == "")
                date = listing["published_at"][:10]
                if date:
                    earliest = min(earliest, date) if earliest else date
                    latest = max(latest, date) if latest else date
        report = {
            "input_file": input_path.name, "source_url": SOURCE_URL, "include_land": False,
            "counts": counts, "hanoi_raw_property_types": hanoi_types, "output_property_groups": groups,
            "districts_as_recorded": districts, "purpose_evidence": evidence_counts,
            "quality_flags": quality, "missing_by_group": coverage,
            "published_date_range": [earliest or None, latest or None],
            "encoding": "utf-8-sig", "delimiter": ",",
        }
        (staging / "quality_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for name in ("hanoi_properties_final.csv", "quality_report.json"):
            (staging / name).replace(output_dir / name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    report = build(args.input_csv, args.output_dir)
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
