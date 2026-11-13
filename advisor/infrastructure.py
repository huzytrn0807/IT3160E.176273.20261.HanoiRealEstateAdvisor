"""Score documented infrastructure separately from a listing's needs fit."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from .data import district_key
from .knowledge import development_insights
from .search import SearchCriteria, search_listings


PRIORITY_LABELS = {
    "mobility": "Đi lại hiện tại",
    "development": "Phát triển tương lai",
}
STATUS_POINTS = {
    "mobility": {
        "operational_at_source_date": 100,
        "works_reported_at_source_date": 60,
        "alignment_announced": 30,
    },
    "development": {
        "operational_at_source_date": 60,
        "works_reported_at_source_date": 100,
        "alignment_announced": 80,
    },
}
UNKNOWN_REASON = "Chưa đủ hồ sơ có nguồn cho khu vực này."
AREA_SCOPE = "Điểm dựa trên hồ sơ cấp khu vực. Khoảng cách, thời gian đi lại và ảnh hưởng đến từng căn cần được đối chiếu riêng."


def _normalized_district(value: object) -> str:
    if not isinstance(value, str) or value.strip().casefold() in {"", "nan", "<na>", "nat"}:
        return ""
    return district_key(value)


def _points(priority: str) -> dict[str, int]:
    try:
        return STATUS_POINTS[priority]
    except KeyError as error:
        raise ValueError(f"Ưu tiên hạ tầng không hợp lệ: {priority}") from error


@lru_cache(maxsize=256)
def _scored_facts(key: str, priority: str) -> tuple[dict, ...]:
    points = _points(priority)
    facts = (
        {**fact, "priority_points": points[fact["status_code"]]}
        for fact in development_insights(key)
        if fact.get("status_code") in points and fact.get("source_url")
    )
    return tuple(sorted(facts, key=lambda fact: -fact["priority_points"]))


def score_infrastructure(district: object, priority: str = "mobility") -> dict:
    """Return the best documented match, retaining all supporting area facts."""
    facts = _scored_facts(_normalized_district(district), priority)
    best = facts[0] if facts else None
    return {
        "priority": priority,
        "score": best["priority_points"] if best else None,
        "reason": f"{best['title']} · {best['status']}." if best else UNKNOWN_REASON,
        "facts": [{**fact, "districts": list(fact["districts"])} for fact in facts],
        "scope": AREA_SCOPE,
    }


def _infrastructure_score(district: object, priority: str) -> int | None:
    facts = _scored_facts(_normalized_district(district), priority)
    return facts[0]["priority_points"] if facts else None


def _partitions(records, priority: str):
    """Yield groups from highest evidence score through missing evidence."""
    if hasattr(records, "itertuples") and "district_name" in records:
        districts = records["district_name"]
        district_scores = {
            name: _infrastructure_score(name, priority)
            for name in districts.dropna().unique()
        }
        scores = districts.map(district_scores)
        for score in sorted(scores.dropna().unique(), reverse=True):
            yield records.loc[scores.eq(score)]
        yield records.loc[scores.isna()]
        return

    groups = defaultdict(list)
    for record in records:
        groups[_infrastructure_score(record.get("district_name"), priority)].append(record)
    for score in sorted((score for score in groups if score is not None), reverse=True):
        yield groups[score]
    yield groups.get(None, ())


def rank_listings(
    records,
    criteria: SearchCriteria,
    limit: int = 30,
    *,
    sort_by: str = "needs",
    priority: str = "mobility",
) -> list[dict]:
    """Filter first, then order by needs or by sourced area infrastructure.

    Within an infrastructure score tier, use the existing needs ranking.
    The infrastructure order considers all eligible records, including listings
    outside the top results by needs fit.
    """
    _points(priority)
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Giới hạn kết quả phải là số nguyên dương.")
    if sort_by == "needs":
        ranked = search_listings(records, criteria, limit=limit)
    elif sort_by == "infrastructure":
        ranked = []
        for partition in _partitions(records, priority):
            ranked.extend(search_listings(partition, criteria, limit=limit - len(ranked)))
            if len(ranked) >= limit:
                break
    else:
        raise ValueError(f"Cách sắp xếp không hợp lệ: {sort_by}")
    return [
        {**row, "infrastructure": score_infrastructure(row.get("district_name"), priority)}
        for row in ranked
    ]
