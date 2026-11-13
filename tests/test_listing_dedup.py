import pandas as pd

from advisor.listing_dedup import deduplicate_listings


PENTHOUSE = "Bán căn hộ penthouse 120m2, 2 ngủ sân thượng riêng giá chỉ 2.25 tỷ Hai Bà Trưng"


def _listing(listing_id, **updates):
    return {
        "listing_id": listing_id,
        "name": PENTHOUSE,
        "property_group": "apartment",
        "district_name": "Hai Bà Trưng",
        "price_vnd": 2_250_000_000,
        "area_m2": 100.0,
        "bedroom_count": 2,
        "bathroom_count": 2,
        "ward_name": "",
        "published_at": "2025-09-30T14:39:25.902000",
        **updates,
    }


def test_merges_area_unit_and_case_variants_in_penthouse_example():
    source = pd.DataFrame([
        _listing("666888"),
        _listing("891287", name=PENTHOUSE.replace("120m2", "120m").lower(),
                 ward_name="Bạch Đằng", published_at="2025-09-30T16:17:38.389000"),
    ])
    result = deduplicate_listings(source)
    assert result["listing_id"].tolist() == ["891287"]
    assert result["area_m2"].tolist() == [100.0]
    assert result["related_listing_count"].tolist() == [2]
    assert "related_listing_count" not in source
    assert len(source) == 2


def test_near_title_uses_complete_then_newest_source_row():
    source = pd.DataFrame([
        _listing("old", ward_name="Bạch Đằng"),
        _listing("new", name=PENTHOUSE + " view đẹp", ward_name="Bạch Đằng",
                 published_at="2025-10-01"),
        _listing("newest_but_missing_beds", bedroom_count=None, ward_name="Bạch Đằng",
                 published_at="2025-11-01"),
    ])
    result = deduplicate_listings(source)
    assert result["listing_id"].tolist() == ["new"]
    assert result["related_listing_count"].tolist() == [3]


def test_missing_ward_cannot_bridge_distinct_wards():
    source = pd.DataFrame([
        _listing("unknown"),
        _listing("bach-dang", ward_name="Bạch Đằng"),
        _listing("minh-khai", ward_name="Minh Khai"),
    ])
    result = deduplicate_listings(source)
    assert set(result["listing_id"]) == {"bach-dang", "minh-khai"}
    assert sorted(result["related_listing_count"]) == [1, 2]


def test_keeps_distinct_units_and_rows_with_no_district():
    source = pd.DataFrame([
        _listing("one", name=PENTHOUSE + " mã căn A1203"),
        _listing("two", name=PENTHOUSE + " mã căn B1203"),
        _listing("unknown-1", district_name=None),
        _listing("unknown-2", district_name=None),
    ])
    result = deduplicate_listings(source)
    assert result["listing_id"].tolist() == ["one", "two", "unknown-1", "unknown-2"]
    assert result["related_listing_count"].tolist() == [1, 1, 1, 1]


def test_generic_titles_and_differing_prices_stay_separate():
    source = pd.DataFrame([
        _listing("generic-1", name="Bán căn hộ chung cư 2 phòng ngủ"),
        _listing("generic-2", name="Bán căn hộ chung cư 2 phòng ngủ"),
        _listing("price-1"),
        _listing("price-2", price_vnd=2_300_000_000),
    ])
    assert len(deduplicate_listings(source)) == 4


def test_reordered_numbers_with_different_room_meanings_stay_separate():
    common = "Bán penthouse sân thượng riêng tại Hai Bà Trưng"
    source = pd.DataFrame([
        _listing("two-bed", name=common + " 2 phòng ngủ 3 WC", bedroom_count=None, bathroom_count=None),
        _listing("three-bed", name=common + " 3 phòng ngủ 2 WC", bedroom_count=None, bathroom_count=None),
    ])
    assert len(deduplicate_listings(source)) == 2
