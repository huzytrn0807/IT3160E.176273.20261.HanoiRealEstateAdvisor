"""Hard filters, explainable ranking, and a bounded backtracking example."""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from heapq import nsmallest
from math import isfinite
from typing import Any
import unicodedata


OUTLIER_FLAGS = frozenset({"unit_price_review", "area_review"})
DETAILS_TO_CHECK = ("district_name", "ward_name", "street_name", "bedroom_count", "published_at")


def _text(value: Any) -> str:
    if value is None:
        return ""
    clean = str(value).strip()
    return "" if clean.casefold() in {"nan", "<na>", "nat"} else clean


def _district_key(value: Any) -> str:
    key = unicodedata.normalize("NFKD", _text(value).casefold().replace("đ", "d"))
    key = " ".join("".join(c for c in key if not unicodedata.combining(c)).split())
    for prefix in ("quan ", "huyen ", "thi xa "):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _number(value: Any, *, allow_zero: bool = False) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        return None
    return number


def _records(data: Any) -> Iterator[dict]:
    if hasattr(data, "itertuples"):
        columns = tuple(data.columns)
        for values in data.itertuples(index=False, name=None):
            yield dict(zip(columns, values))
        return
    for row in data:
        yield dict(row)


@dataclass(frozen=True)
class SearchCriteria:
    max_price_vnd: int | None = None
    districts: tuple[str, ...] = ()
    property_groups: tuple[str, ...] = ()
    min_area_m2: float | None = None
    min_bedrooms: int | None = None
    preferred_price_vnd: int | None = None
    preferred_area_m2: float | None = None
    preferred_districts: tuple[str, ...] = ()
    price_weight: float = 0.45
    area_weight: float = 0.35
    district_weight: float = 0.20
    exclude_quality_outliers: bool = True

    def __post_init__(self) -> None:
        checks = {
            "ngân sách": self.max_price_vnd,
            "diện tích tối thiểu": self.min_area_m2,
            "giá mong muốn": self.preferred_price_vnd,
            "diện tích mong muốn": self.preferred_area_m2,
        }
        for field, value in checks.items():
            if value is not None and _number(value) is None:
                raise ValueError(f"{field.capitalize()} phải là số dương hữu hạn.")
        beds = self.min_bedrooms
        if beds is not None and (not isinstance(beds, int) or beds < 0):
            raise ValueError("Số phòng ngủ tối thiểu phải là số nguyên không âm.")
        for weight in (self.price_weight, self.area_weight, self.district_weight):
            if _number(weight, allow_zero=True) is None:
                raise ValueError("Trọng số phải là số hữu hạn không âm.")


def _weighted_score(components: Mapping[str, float | None], criteria: SearchCriteria) -> float:
    weights = {
        "price": criteria.price_weight,
        "area": criteria.area_weight,
        "district": criteria.district_weight,
    }
    active = [(float(value), weights[key]) for key, value in components.items() if value is not None]
    total_weight = sum(weight for _, weight in active)
    if not total_weight:
        return 0.0
    return round(100 * sum(value * weight for value, weight in active) / total_weight, 4)


def _explain(
    row: dict, criteria: SearchCriteria, price: float, area: float, beds: float | None,
    district_matches: bool | None,
) -> list[str]:
    reasons = [f"Giá chào bán {price / 1e9:,.3f} tỷ đồng, diện tích {area:g} m²."]
    if criteria.max_price_vnd is not None:
        gap = (criteria.max_price_vnd - price) / 1e9
        reasons.append(f"Đạt giới hạn ngân sách, còn {gap:,.3f} tỷ đồng so với mức tối đa.")
    if criteria.min_area_m2 is not None:
        reasons.append(f"Đạt diện tích tối thiểu {criteria.min_area_m2:g} m².")
    if criteria.min_bedrooms is not None:
        reasons.append(f"Đạt số phòng ngủ tối thiểu, tin đăng ghi {beds:g} phòng ngủ.")
    if criteria.property_groups:
        reasons.append("Thuộc loại bất động sản đã chọn.")
    if criteria.districts:
        reasons.append(f"Thuộc địa bàn đã chọn ({_text(row.get('district_name'))}).")
    if district_matches is not None:
        if district_matches:
            reasons.append("Khớp khu vực ưu tiên.")
        elif not _text(row.get("district_name")):
            reasons.append("Thiếu quận/huyện nên không có điểm khu vực ưu tiên.")
        else:
            reasons.append("Ngoài khu vực ưu tiên.")
    if criteria.preferred_price_vnd is not None:
        gap = abs(criteria.preferred_price_vnd - price) / 1e9
        reasons.append(f"Chênh {gap:,.3f} tỷ đồng so với giá mong muốn.")
    if criteria.preferred_area_m2 is not None:
        gap = abs(criteria.preferred_area_m2 - area)
        reasons.append(f"Chênh {gap:g} m² so với diện tích mong muốn.")
    return reasons


def search_listings(records: Iterable[Mapping] | Any, criteria: SearchCriteria, limit: int = 30) -> list[dict]:
    """Rank records passing every requested hard filter.

    A preferred numeric value scores min(actual, preferred)/max(actual, preferred).
    Otherwise, budget scores 1-price/max and minimum area scores min(1, area/(2*min)).
    Preferred district scores 1 for a match and 0 otherwise. Inactive criteria are
    excluded from the weighted mean. Missing hard-filter values fail that filter.
    """
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Giới hạn kết quả phải là số nguyên dương.")
    districts = {_district_key(item) for item in criteria.districts if _text(item)}
    preferred = {_district_key(item) for item in criteria.preferred_districts if _text(item)}
    groups = {_text(group).casefold() for group in criteria.property_groups if _text(group)}

    def passing() -> Iterator[tuple[float, float, str, dict, float, float | None, dict]]:
        for row in _records(records):
            price = _number(row.get("price_vnd"))
            area = _number(row.get("area_m2"))
            if price is None or area is None:
                continue
            if criteria.max_price_vnd is not None and price > criteria.max_price_vnd:
                continue
            if criteria.min_area_m2 is not None and area < criteria.min_area_m2:
                continue
            if groups and _text(row.get("property_group")).casefold() not in groups:
                continue
            flags = _text(row.get("quality_flags")).split("|")
            if criteria.exclude_quality_outliers and not OUTLIER_FLAGS.isdisjoint(flags):
                continue
            district = _district_key(row.get("district_name")) if districts or preferred else ""
            if districts and district not in districts:
                continue
            beds = _number(row.get("bedroom_count"), allow_zero=True)
            if criteria.min_bedrooms is not None and (beds is None or beds < criteria.min_bedrooms):
                continue
            price_score = None
            if criteria.preferred_price_vnd is not None:
                target = criteria.preferred_price_vnd
                price_score = min(price, target) / max(price, target)
            elif criteria.max_price_vnd is not None:
                price_score = 1 - price / criteria.max_price_vnd
            area_score = None
            if criteria.preferred_area_m2 is not None:
                target = criteria.preferred_area_m2
                area_score = min(area, target) / max(area, target)
            elif criteria.min_area_m2 is not None:
                area_score = min(1.0, area / (2 * criteria.min_area_m2))
            components = {
                "price": price_score,
                "area": area_score,
                "district": float(district in preferred) if preferred else None,
            }
            score = _weighted_score(components, criteria)
            yield (-score, price, _text(row.get("listing_id")), row, area, beds, components)

    ranked = []
    best = nsmallest(limit, passing(), key=lambda entry: entry[:3])
    for negative_score, price, _, row, area, beds, components in best:
        district_match = components["district"]
        ranked.append({
            **row,
            "score": -negative_score,
            "score_components": components,
            "reasons": _explain(
                row, criteria, price, area, beds, bool(district_match) if district_match is not None else None
            ),
            "missing_fields": [key for key in DETAILS_TO_CHECK if not _text(row.get(key))],
        })
    return ranked


def compare_listings(records: Iterable[Mapping] | Any, listing_ids: Iterable[str]) -> list[dict]:
    """Return requested rows in the user's selection order."""
    requested = tuple(dict.fromkeys(listing_ids))
    remaining = set(requested)
    by_id = {}
    for row in _records(records):
        listing_id = _text(row.get("listing_id"))
        if listing_id in remaining:
            by_id[listing_id] = row
            remaining.remove(listing_id)
        if not remaining:
            break
    return [by_id[key] for key in requested if key in by_id]


def solve_csp(
    ranked_candidates: Iterable[Mapping] | Any, *, count: int, total_budget_vnd: int,
    candidate_limit: int = 16, node_limit: int = 20_000, trace_limit: int = 50,
) -> dict:
    """Find one set of distinct listings under a shared budget by backtracking.

    Each slot is a variable and each ranked listing is a domain value. Later
    slots take later candidates so equivalent permutations are explored once.
    An infeasible result applies only to the supplied, bounded domain.
    """
    if not isinstance(count, int) or not 1 <= count <= 8:
        raise ValueError("Ví dụ CSP hỗ trợ chọn từ 1 đến 8 tin.")
    budget = _number(total_budget_vnd)
    if budget is None or not budget.is_integer():
        raise ValueError("Ngân sách CSP phải là số nguyên dương tính bằng đồng.")
    for label, limit in (("ứng viên", candidate_limit), ("lượt thử", node_limit), ("dòng nhật ký", trace_limit)):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"Giới hạn {label} phải là số nguyên dương.")
    candidates: list[tuple[str, int, dict]] = []
    seen: set[str] = set()
    truncated = False
    for row in _records(ranked_candidates):
        listing_id = _text(row.get("listing_id"))
        price = _number(row.get("price_vnd"))
        if not listing_id or listing_id in seen or price is None or not price.is_integer():
            continue
        if len(candidates) == candidate_limit:
            truncated = True
            break
        seen.add(listing_id)
        candidates.append((listing_id, int(price), row))

    trace: list[str] = []
    chosen: list[tuple[str, int, dict]] = []
    nodes = backtracks = 0
    stopped = False

    def record(message: str) -> None:
        if len(trace) < trace_limit:
            trace.append(message)

    def backtrack(start: int, remaining_budget: int) -> bool:
        nonlocal nodes, backtracks, stopped
        slots = count - len(chosen)
        if slots == 0:
            record("Đã gán đủ vị trí trong ngân sách.")
            return True
        if len(candidates) - start < slots:
            record(f"Cần thêm {slots} tin nhưng miền còn quá ít ứng viên.")
            return False
        for index in range(start, len(candidates) - slots + 1):
            if nodes == node_limit:
                stopped = True
                record("Đã đạt giới hạn lượt thử.")
                return False
            listing_id, price, row = candidates[index]
            nodes += 1
            label = f"ứng viên #{index + 1} ({listing_id[:10]})"
            if price > remaining_budget:
                record(f"Bỏ {label} vì vượt ngân sách còn lại.")
                continue
            chosen.append((listing_id, price, row))
            record(f"Thử {label} cho vị trí {len(chosen)}.")
            if backtrack(index + 1, remaining_budget - price):
                return True
            chosen.pop()
            if stopped:
                return False
            backtracks += 1
            record(f"Quay lui khỏi {label} để thử lựa chọn tiếp theo.")
        return False

    solved = backtrack(0, int(budget))
    return {
        "status": "solved" if solved else ("node_limit" if stopped else "infeasible"),
        "selected": [row for _, _, row in chosen],
        "total_price_vnd": sum(price for _, price, _ in chosen),
        "trace": trace,
        "nodes": nodes,
        "backtracks": backtracks,
        "candidate_count": len(candidates),
        "truncated": truncated,
    }
