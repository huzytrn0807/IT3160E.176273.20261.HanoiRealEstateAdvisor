import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_hanoi_dataset import REQUIRED, build
from profile_hanoi_dataset import profile
from advisor.data import load_listings
from advisor.knowledge import development_insights
from advisor.search import SearchCriteria, search_listings, solve_csp


def source_row(title, kind, **changes):
    row = dict.fromkeys(REQUIRED, "")
    row.update(name=title, property_type_name=kind, province_name="Hà Nội",
               district_name="Cầu Giấy", price="5200000000", area="75",
               bedroom_count="2", published_at="2025-08-01")
    row.update(changes)
    return row


def test_pipeline_to_search(tmp_path):
    input_path = tmp_path / "source.csv"
    apt = source_row("=Bán căn hộ có chỗ đỗ ô tô", "Căn hộ chung cư")
    rows = [
        apt, apt, source_row("Cho thuê căn hộ", "Căn hộ chung cư"),
        source_row("Bán đất", "Đất"), source_row("Căn hộ đẹp", "Căn hộ chung cư"),
        source_row("Bán nhà", "Nhà", district_name="", bedroom_count=""),
        source_row("Bán nhà Quận 1", "Nhà", province_name="Hồ Chí Minh"),
        source_row("Bán nhà mới", "Nhà", area=""),
    ]
    with input_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED))
        writer.writeheader()
        writer.writerows(rows)
    report = build(input_path, tmp_path)
    counts = report["counts"]
    assert counts["output_rows"] == 2
    assert counts["duplicates"] == counts["purpose_unknown"] == counts["purpose_rent"] == 1
    assert counts["missing_price_or_area"] == counts["out_of_scope_property_type"] == 1
    output = tmp_path / "hanoi_properties_final.csv"
    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    assert profile(output)["rows"] == 2
    table = load_listings(output)
    assert table.iloc[0]["name"].startswith("'=")
    assert table.iloc[0]["dac_diem_noi_bat"] == "Có nhắc chỗ đỗ ô tô"
    ranked = search_listings(table, SearchCriteria(districts=("Cầu Giấy",), min_bedrooms=2))
    assert [item["property_group"] for item in ranked] == ["apartment"]
    assert ranked[0]["reasons"]


def test_reader_reports_missing_schema(tmp_path):
    malformed = tmp_path / "wrong.csv"
    malformed.write_text("name;price\nX;1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu cột"):
        load_listings(malformed)


def test_collective_housing_is_classified_before_search(tmp_path):
    source = tmp_path / "source.csv"
    rows = [
        source_row("Bán nhà tập thể gần Học Viện Tài Chính 100m2, 3 ngủ", "Căn hộ chung cư"),
        source_row("Chính chủ bán gấp căn hộ tập thể", "Nhà"),
        source_row("Bán căn hộ gần khu tập thể", "Căn hộ chung cư"),
        source_row("Bán nhà riêng đối diện nhà tập thể", "Nhà"),
        source_row("Bán nhà khu tập thể, ngõ rộng", "Nhà"),
    ]
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED))
        writer.writeheader()
        writer.writerows(rows)
    assert build(source, tmp_path)["output_property_groups"] == {
        "collective_housing": 2, "apartment": 1, "house": 2,
    }
    output = tmp_path / "hanoi_properties_final.csv"
    assert profile(output)["group_summary"]["collective_housing"]["rows"] == 2
    table = load_listings(output)
    assert table.iloc[0]["property_type_name"] == "Căn hộ chung cư"
    for group, expected_ids in (
        ("collective_housing", table.iloc[:2]["listing_id"]),
        ("apartment", table.iloc[2:3]["listing_id"]),
        ("house", table.iloc[3:]["listing_id"]),
    ):
        ranked = search_listings(table, SearchCriteria(property_groups=(group,)))
        assert {item["listing_id"] for item in ranked} == set(expected_ids)

    legacy = tmp_path / "legacy.csv"
    table.assign(property_group=["apartment", "house", "apartment", "house", "house"]).to_csv(
        legacy, index=False, encoding="utf-8-sig"
    )
    assert load_listings(legacy)["property_group"].tolist() == table["property_group"].tolist()


def test_csp_can_backtrack():
    candidates = [
        {"listing_id": "A", "name": "A", "price_vnd": 8},
        {"listing_id": "B", "name": "B", "price_vnd": 4},
        {"listing_id": "C", "name": "C", "price_vnd": 4},
    ]
    result = solve_csp(candidates, count=2, total_budget_vnd=8)
    assert result["status"] == "solved"
    assert result["backtracks"] > 0
    assert {item["listing_id"] for item in result["selected"]} == {"B", "C"}


def test_knowledge_keeps_citation_and_scope():
    results = development_insights("Quận Cầu Giấy")
    assert results[0]["source_url"].startswith("https://")
    assert results[0]["matched_rule"].startswith("IF ")
    assert development_insights("Khu vực không có trong nguồn") == []
