from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from advisor.modeling import FEATURES, _metrics, _read_candidates, _split_dates, predict_listing


def test_chronological_splits_keep_complete_days() -> None:
    days = pd.date_range("2025-06-01", periods=10, freq="D")
    rows = pd.DataFrame({"_day": days.repeat(40)})
    train, validation, test = _split_dates(rows)
    assert sum(map(len, (train, validation, test))) == len(rows)
    assert train["_day"].max() < validation["_day"].min()
    assert validation["_day"].max() < test["_day"].min()


def test_newer_relist_with_different_price_is_held_out_of_splits(tmp_path: Path) -> None:
    fields = set().union(*FEATURES.values())
    listing = {
        **dict.fromkeys(fields, ""),
        "listing_id": "first",
        "name": "Căn hộ 2 phòng ngủ",
        "property_group": "apartment",
        "price_type": "asking",
        "listing_purpose": "sale",
        "price_vnd": 7_000_000_000,
        "area_m2": 83,
        "bedroom_count": 2,
        "district_name": "Hai Bà Trưng",
        "project_name": "Times City",
        "street_name": "Minh Khai",
        "published_at": "2025-06-01",
        "quality_flags": "",
    }
    relist = dict(listing, listing_id="later", name=" CĂN HỘ 2 PHÒNG NGỦ ",
                  price_vnd=7_500_000_000, published_at="2025-09-01")
    bad_quality = dict(listing, listing_id="bad", quality_flags="unit_price_review")
    source = tmp_path / "listings.csv"
    pd.DataFrame([relist, listing, bad_quality]).to_csv(source, index=False, encoding="utf-8-sig")
    selected, counts = _read_candidates(source)
    assert selected["listing_id"].tolist() == ["first"]
    assert counts["same_listing_or_title_and_attributes_rows_removed"] == 1
    assert counts["excluded_quality_flag_rows"] == 1


def test_metrics_use_actual_vnd_scale() -> None:
    metrics = _metrics([3_000_000_000, 7_000_000_000], [4_000_000_000, 6_000_000_000])
    assert metrics == {"rows": 2, "mae_vnd": 1_000_000_000, "rmse_vnd": 1_000_000_000, "r2": 0.75}


class AreaPriceModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["area_m2"].to_numpy() * 0.1


def test_prediction_preserves_price_units_and_surfaces_weak_r2() -> None:
    bundle = {
        "models": {
            "apartment": {
                "model": AreaPriceModel(), "model_name": "linear_regression", "features": FEATURES["apartment"],
                "interval_radius_vnd": 1_000_000_000, "reference_values": {"area_m2": 70},
                "known_districts": ["hai bà trưng"], "area_p01_p99_m2": [30, 150],
            },
        },
        "report": {"groups": {"apartment": {"test_metrics": {"linear_regression": {"r2": -0.2}}}}},
    }
    result = predict_listing(bundle, "apartment", {"district_name": "Hai Bà Trưng", "area_m2": 80})
    assert result["predicted_price_vnd"] == 8_000_000_000
    assert result["predicted_price_per_m2_vnd"] == 100_000_000
    assert result["interval_vnd"] == [7_000_000_000, 9_000_000_000]
    assert result["effects"] == [{"feature": "area_m2", "reference_value": 70, "price_difference_vnd": 1_000_000_000}]
    assert any("R²" in warning for warning in result["warnings"])
    with pytest.raises(ValueError, match="Diện tích"):
        predict_listing(bundle, "apartment", {"area_m2": 0})
