"""Shared CSV reader and dataset contract for the app."""
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from advisor.property_types import COLLECTIVE_HOUSING, property_group_for_title

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_CSV = DEFAULT_PROCESSED / "hanoi_properties_final.csv"
GROUP_LABELS = {
    "apartment": "Căn hộ chung cư",
    COLLECTIVE_HOUSING: "Nhà tập thể",
    "house": "Nhà",
    "villa_townhouse": "Biệt thự / nhà liền kề",
    "shophouse": "Shophouse",
}
NUMERIC_COLUMNS = (
    "price_vnd", "area_m2", "price_per_m2_vnd", "floor_count", "bedroom_count",
    "bathroom_count", "frontage_m", "depth_m", "road_width_m",
)
REQUIRED_COLUMNS = {
    "listing_id", "name", "property_group", "province_name", "district_name",
    "property_type_name", "published_at", "price_type", "listing_purpose",
    "ward_name", "street_name", "project_name", "house_direction", "balcony_direction",
    "source", "source_url", "quality_flags", "dac_diem_noi_bat", *NUMERIC_COLUMNS,
}


def district_key(value: str) -> str:
    """Match explicit source labels without guessing district from an address."""
    folded = unicodedata.normalize("NFKD", str(value).casefold().replace("đ", "d"))
    ascii_key = " ".join("".join(ch for ch in folded if not unicodedata.combining(ch)).split())
    return re.sub(r"^(?:quan|huyen|thi xa|q\.|h\.|tx\.)\s+", "", ascii_key)


@lru_cache(maxsize=2)
def _read(path: Path, version: tuple[int, int]) -> pd.DataFrame:
    del version  # The cache key includes the file's modification time and size.
    try:
        table = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={"listing_id": "string"})
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("CSV không đọc được. Hãy xuất lại bằng pipeline ở định dạng UTF-8, dấu phẩy.") from exc
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {', '.join(sorted(missing))}")
    if table.empty:
        raise ValueError("CSV không có dòng dữ liệu.")
    if table["listing_id"].isna().any() or not table["listing_id"].is_unique:
        raise ValueError("CSV có listing_id thiếu hoặc trùng.")
    for name in NUMERIC_COLUMNS:
        values = pd.to_numeric(table[name], errors="coerce")
        if (table[name].notna() & ~np.isfinite(values)).any():
            raise ValueError(f"Cột {name} có dữ liệu số không hợp lệ.")
        table[name] = values
    if not table[["price_vnd", "area_m2", "price_per_m2_vnd"]].gt(0).all().all():
        raise ValueError("CSV có giá hoặc diện tích không hợp lệ.")
    if not table["price_type"].eq("asking").all() or not table["listing_purpose"].eq("sale").all():
        raise ValueError("CSV cần chứa giá chào bán của tin bán.")
    if not table["property_group"].isin(GROUP_LABELS).all():
        raise ValueError("CSV có nhóm bất động sản ngoài phạm vi dự án.")
    table["property_group"] = [
        property_group_for_title(group, title) for group, title in zip(table["property_group"], table["name"])
    ]
    return table


def load_listings(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Load a local generated CSV. Callers should treat the returned table as read-only."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Chưa có CSV tại {source}. Hãy chạy scripts/build_hanoi_dataset.py trước.")
    info = source.stat()
    return _read(source, (info.st_mtime_ns, info.st_size))
