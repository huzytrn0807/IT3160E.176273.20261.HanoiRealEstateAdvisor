import pandas as pd
import pytest

from advisor.infrastructure import rank_listings, score_infrastructure
from advisor.search import SearchCriteria


def _record(listing_id, district, price):
    return {
        "listing_id": listing_id,
        "district_name": district,
        "price_vnd": price,
        "area_m2": 70,
        "bedroom_count": 2,
        "property_group": "apartment",
    }


def test_score_reports_unknown_and_cites_matching_area():
    assert score_infrastructure("Tây Hồ")["score"] is None
    assert score_infrastructure(pd.NA)["facts"] == []
    result = score_infrastructure("Quận Cầu Giấy")
    assert result["score"] == 100
    assert result["facts"][0]["source_url"].startswith("https://")
    assert result["facts"][0]["status_code"] == "operational_at_source_date"
    assert "cấp khu vực" in result["scope"]
    assert score_infrastructure("Hai Bà Trưng", "mobility")["facts"][0]["fact_id"] == "ring-road-2-vinh-tuy-nga-tu-so"
    assert score_infrastructure("Hai Bà Trưng", "development")["facts"][0]["fact_id"] == "tran-hung-dao-bridge-groundbreaking"


def test_priority_distinguishes_project_stages_without_stacking_facts():
    assert score_infrastructure("Hoài Đức", "mobility")["score"] == 60
    result = score_infrastructure("Hoài Đức", "development")
    assert result["score"] == 100
    assert [fact["priority_points"] for fact in result["facts"]] == [100, 80]
    assert score_infrastructure("Bắc Từ Liêm", "development")["score"] == 80


@pytest.mark.parametrize("as_frame", [False, True])
def test_infrastructure_sort_reaches_beyond_needs_top_k(as_frame):
    records = [
        _record("cheap", "Tây Hồ", 2_000_000_000),
        _record("work", "Hoài Đức", 3_000_000_000),
        _record("metro", "Cầu Giấy", 4_000_000_000),
        _record("too-expensive", "Hà Đông", 6_000_000_000),
    ]
    if as_frame:
        records = pd.DataFrame(records)
    criteria = SearchCriteria(max_price_vnd=5_000_000_000)
    needs = rank_listings(records, criteria, limit=1)
    infrastructure = rank_listings(records, criteria, limit=1, sort_by="infrastructure")
    all_ranked = rank_listings(records, criteria, limit=3, sort_by="infrastructure")
    assert needs[0]["listing_id"] == "cheap"
    assert infrastructure[0]["listing_id"] == "metro"
    assert [row["listing_id"] for row in all_ranked] == ["metro", "work", "cheap"]
    assert all_ranked[-1]["infrastructure"]["score"] is None
    assert all_ranked[-1]["score"] == needs[0]["score"]
