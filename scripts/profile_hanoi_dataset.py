"""Validate the generated CSV and summarize price and feature coverage by property group."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from build_hanoi_dataset import COLLECTIVE_HOUSING, FIELDS


FEATURES = {
    "apartment": ("area_m2", "district_name", "ward_name", "project_name",
                  "bedroom_count", "bathroom_count", "balcony_direction"),
    COLLECTIVE_HOUSING: ("area_m2", "district_name", "ward_name", "bedroom_count", "bathroom_count"),
    "house": ("area_m2", "district_name", "ward_name", "floor_count",
              "bedroom_count", "bathroom_count", "frontage_m", "road_width_m", "house_direction"),
    "villa_townhouse": ("area_m2", "district_name", "ward_name", "project_name",
                        "floor_count", "bedroom_count", "bathroom_count", "frontage_m", "road_width_m"),
    "shophouse": ("area_m2", "district_name", "ward_name", "project_name",
                  "floor_count", "frontage_m", "road_width_m"),
}
MEASURES = ("price_vnd", "area_m2", "price_per_m2_vnd")


def describe(values: list[float]) -> dict:
    percentiles = statistics.quantiles(values, n=100, method="inclusive") if len(values) > 1 else values * 99
    return {
        "min": min(values), "p05": percentiles[4], "median": percentiles[49],
        "p95": percentiles[94], "max": max(values),
    }


def profile(source: Path) -> dict:
    ids: set[str] = set()
    group_count: Counter = Counter()
    missing: dict[str, Counter] = defaultdict(Counter)
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    ready: Counter = Counter()
    months: Counter = Counter()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Header không khớp với pipeline. Kiểm tra tệp đầu vào và dấu phân cách.")
        for record, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Dòng dữ liệu {record} không khớp số cột.")
            group = row["property_group"]
            if group not in FEATURES or row["listing_purpose"] != "sale" or row["price_type"] != "asking":
                raise ValueError(f"Dòng dữ liệu {record} nằm ngoài phạm vi giá chào bán nhà và căn hộ.")
            listing_id = row["listing_id"]
            if not listing_id or listing_id in ids:
                raise ValueError(f"listing_id thiếu hoặc trùng tại dòng dữ liệu {record}.")
            ids.add(listing_id)
            price, area, unit_price = (float(row[key]) for key in MEASURES)
            if not all(math.isfinite(v) and v > 0 for v in (price, area, unit_price)):
                raise ValueError(f"Giá hoặc diện tích không hợp lệ tại dòng dữ liệu {record}.")
            if abs(price / area - unit_price) > 0.02:
                raise ValueError(f"Giá/m² không khớp tại dòng dữ liệu {record}.")
            group_count[group] += 1
            months[row["published_at"][:7] or "(blank)"] += 1
            for key, value in zip(MEASURES, (price, area, unit_price)):
                values[group][key].append(value)
            missing[group].update(key for key in FEATURES[group] if not row[key])
            flags = set(row["quality_flags"].split("|"))
            if row["published_at"] and row["district_name"] and not flags.intersection(
                {"area_review", "unit_price_review", "sale_inferred_from_description"}
            ):
                ready[group] += 1
    return {
        "input_csv": source.name,
        "rows": len(ids),
        "target": "price_vnd",
        "group_summary": {
            group: {
                "rows": count,
                "rows_without_major_quality_flags_and_with_district_and_date": ready[group],
                "suggested_features": FEATURES[group],
                "feature_missing_pct": {
                    feature: round(100 * missing[group][feature] / count, 2)
                    for feature in FEATURES[group]
                },
                "distributions": {key: describe(values[group][key]) for key in MEASURES},
            } for group, count in group_count.items()
        },
        "listings_by_month": dict(sorted(months.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/eda_summary.json"))
    args = parser.parse_args()
    report = profile(args.input_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Số dòng kiểm tra: {report['rows']:,}")


if __name__ == "__main__":
    main()
