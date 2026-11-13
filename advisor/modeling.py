"""Train and serve asking-price models from the processed Hanoi CSV."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from advisor.property_types import COLLECTIVE_HOUSING, property_group_for_title

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
NUMERIC_FEATURES = {"area_m2", "floor_count", "bedroom_count", "bathroom_count", "frontage_m", "road_width_m"}
MODEL_FILE = "price_models.joblib"
REPORT_FILE = "evaluation.json"
FORMAT_VERSION = 1
TARGET_SCALE = 1_000_000_000
MIN_TRAIN = 100
MIN_EVALUATION = 30
SUBGROUP_MIN = 25
AREA_BINS = [0, 50, 75, 100, 150, np.inf]
AREA_LABELS = ["≤50 m²", ">50–75 m²", ">75–100 m²", ">100–150 m²", ">150 m²"]
EXCLUDED_FLAGS = {"area_review", "unit_price_review", "sale_inferred_from_description"}


def _key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _prepare_features(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    prepared = pd.DataFrame(index=frame.index)
    for feature in features:
        column = frame[feature]
        if feature in NUMERIC_FEATURES:
            prepared[feature] = pd.to_numeric(column, errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            prepared[feature] = column.map(
                lambda value: (_key(value) or np.nan) if isinstance(value, str) else np.nan
            )
    return prepared


def _read_candidates(path: Path) -> tuple[pd.DataFrame, dict]:
    feature_cols = set().union(*FEATURES.values())
    columns = {
        "listing_id", "name", "property_group", "price_type", "listing_purpose", "price_vnd",
        "published_at", "quality_flags", "street_name", *feature_cols,
    }
    header = set(pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns)
    if missing := columns - header:
        raise ValueError(f"CSV thiếu các cột sau: {', '.join(sorted(missing))}")
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in columns, dtype=str)
    frame["property_group"] = [
        property_group_for_title(group, title) for group, title in zip(frame["property_group"], frame["name"])
    ]
    frame["_day"] = pd.to_datetime(frame["published_at"].str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    frame["price_vnd"] = pd.to_numeric(frame["price_vnd"], errors="coerce")
    frame["area_m2"] = pd.to_numeric(frame["area_m2"], errors="coerce")
    mask = (
        frame["property_group"].isin(FEATURES) & frame["price_type"].eq("asking")
        & frame["listing_purpose"].eq("sale") & frame["listing_id"].notna()
        & frame["_day"].notna() & frame["district_name"].fillna("").str.strip().ne("")
        & np.isfinite(frame["price_vnd"]) & frame["price_vnd"].gt(0)
        & np.isfinite(frame["area_m2"]) & frame["area_m2"].gt(0)
    )
    flags = frame["quality_flags"].fillna("").str.split("|").map(lambda items: bool(EXCLUDED_FLAGS.intersection(items)))
    eligible = frame.loc[mask & ~flags].copy()
    # Keep the earliest record for every exact ID and for same-title listings with matching
    # structured location/area/bedrooms. Price and posting time are deliberately absent.
    location = eligible[["district_name", "ward_name", "street_name", "project_name"]].fillna("")
    location = location.apply(lambda column: column.map(_key))
    title = eligible["name"].fillna("").map(_key)
    bedrooms = pd.to_numeric(eligible["bedroom_count"], errors="coerce").fillna(-1)
    identity = location.assign(
        group=eligible["property_group"], title=title,
        area=eligible["area_m2"].round(1), bedrooms=bedrooms,
    )
    signature = pd.util.hash_pandas_object(identity, index=False).astype(str)
    eligible["_duplicate_key"] = signature.where(title.ne(""), eligible["listing_id"])
    eligible = eligible.sort_values(["_day", "listing_id"], kind="stable")
    unique = eligible.drop_duplicates("listing_id").drop_duplicates("_duplicate_key")
    return unique, {
        "csv_rows": len(frame),
        "ineligible_rows": int((~mask).sum()),
        "excluded_quality_flag_rows": int((mask & flags).sum()),
        "same_listing_or_title_and_attributes_rows_removed": len(eligible) - len(unique),
        "rows_after_filters": len(unique),
    }


def _split_dates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days = frame.groupby("_day", sort=True).size()
    if len(days) < 3:
        raise ValueError("Cần ít nhất ba ngày đăng khác nhau để chia tập theo thời gian.")
    cumulative = days.cumsum().to_numpy()
    first = min(int(np.searchsorted(cumulative, 0.70 * len(frame))), len(days) - 3)
    second = int(np.clip(np.searchsorted(cumulative, 0.85 * len(frame)), first + 1, len(days) - 2))
    train = frame.loc[frame["_day"].le(days.index[first])]
    validation = frame.loc[frame["_day"].gt(days.index[first]) & frame["_day"].le(days.index[second])]
    test = frame.loc[frame["_day"].gt(days.index[second])]
    if len(train) < MIN_TRAIN or min(len(validation), len(test)) < MIN_EVALUATION:
        raise ValueError(f"Số mẫu train, validation, test lần lượt là {len(train)}, {len(validation)}, {len(test)}.")
    return train, validation, test


def _preprocessor(features: tuple[str, ...]) -> ColumnTransformer:
    numeric = [feature for feature in features if feature in NUMERIC_FEATURES]
    categorical = [feature for feature in features if feature not in NUMERIC_FEATURES]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="(thiếu)")),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=15)),
        ]), categorical),
    ])


def _candidate_models(include_random_forest: bool) -> dict:
    models = {
        "median_baseline": DummyRegressor(strategy="median"),
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(max_depth=16, min_samples_leaf=25, random_state=42),
    }
    if include_random_forest:
        models["random_forest"] = RandomForestRegressor(
            n_estimators=40, max_depth=18, min_samples_leaf=15, n_jobs=-1, random_state=42,
        )
    return models


def _predict_vnd(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, pipeline.predict(frame) * TARGET_SCALE)


def _metrics(actual, predicted) -> dict:
    return {
        "rows": len(actual),
        "mae_vnd": float(mean_absolute_error(actual, predicted)),
        "rmse_vnd": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def _date_range(frame: pd.DataFrame) -> list[str]:
    return [frame["_day"].min().date().isoformat(), frame["_day"].max().date().isoformat()]


def _reference_values(train: pd.DataFrame, features: tuple[str, ...]) -> dict:
    references = {}
    for feature in features:
        values = train[feature].dropna()
        if values.empty:
            continue
        if feature in NUMERIC_FEATURES:
            references[feature] = float(values.median())
        else:
            references[feature] = str(values.mode().iloc[0])
    return references


def _train_group(group: str, frame: pd.DataFrame, include_random_forest: bool) -> tuple[dict, dict, pd.DataFrame]:
    train, validation, test = _split_dates(frame)
    features = FEATURES[group]
    x_train = _prepare_features(train, features)
    x_validation = _prepare_features(validation, features)
    x_test = _prepare_features(test, features)
    y_train = train["price_vnd"] / TARGET_SCALE
    validation_metrics = {}
    test_metrics = {}
    models = {}
    for name, regressor in _candidate_models(include_random_forest).items():
        pipeline = Pipeline([("features", _preprocessor(features)), ("regressor", regressor)])
        pipeline.fit(x_train, y_train)
        validation_metrics[name] = _metrics(validation["price_vnd"], _predict_vnd(pipeline, x_validation))
        test_metrics[name] = _metrics(test["price_vnd"], _predict_vnd(pipeline, x_test))
        models[name] = pipeline
    candidates = [name for name in models if name != "median_baseline"]
    chosen = min(candidates, key=lambda name: validation_metrics[name]["mae_vnd"])
    if validation_metrics[chosen]["mae_vnd"] >= validation_metrics["median_baseline"]["mae_vnd"]:
        chosen = "median_baseline"
    model = models[chosen]
    errors_validation = np.abs(validation["price_vnd"].to_numpy() - _predict_vnd(model, x_validation))
    radius = float(np.quantile(errors_validation, 0.90, method="higher"))
    actual_test = test["price_vnd"].to_numpy()
    predicted_test = _predict_vnd(model, x_test)
    coverage = float(np.mean(np.abs(actual_test - predicted_test) <= radius))
    predictions = pd.DataFrame({
        "listing_id": test["listing_id"],
        "property_group": group,
        "published_day": test["_day"].dt.strftime("%Y-%m-%d"),
        "district_name": test["district_name"],
        "area_m2": test["area_m2"],
        "area_group": pd.cut(test["area_m2"], bins=AREA_BINS, labels=AREA_LABELS),
        "actual_asking_price_vnd": actual_test,
        "predicted_asking_price_vnd": predicted_test,
        "absolute_error_vnd": np.abs(actual_test - predicted_test),
    }).reset_index(drop=True)
    stats = {
        "selected_model": chosen,
        "features": list(features),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "date_ranges": {label: _date_range(part) for label, part in zip(
            ("train", "validation", "test"), (train, validation, test)
        )},
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_error_p90_vnd": radius,
        "empirical_test_interval_coverage": coverage,
    }
    artifact = {
        "model": model,
        "model_name": chosen,
        "features": features,
        "interval_radius_vnd": radius,
        "reference_values": _reference_values(x_train, features),
        "known_districts": sorted(x_train["district_name"].dropna().unique().tolist()),
        "area_p01_p99_m2": x_train["area_m2"].quantile([0.01, 0.99]).tolist(),
    }
    return artifact, stats, predictions


def _subgroup_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for field in ("district_name", "area_group"):
        for (group, subgroup), rows in predictions.groupby(["property_group", field], observed=True):
            enough = len(rows) >= SUBGROUP_MIN
            records.append({
                "property_group": group,
                "breakdown": field,
                "subgroup": subgroup,
                "rows": len(rows),
                "status": "reported" if enough else "too_few_rows",
                **(_metrics(rows["actual_asking_price_vnd"], rows["predicted_asking_price_vnd"]) if enough
                   else {"mae_vnd": None, "rmse_vnd": None, "r2": None}),
            })
    return pd.DataFrame(records)


def train_models(source: Path, output_dir: Path, *, include_random_forest: bool = False) -> dict:
    """Train group models and save a reproducible evaluation of later listings."""
    data, accounting = _read_candidates(source)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "format_version": FORMAT_VERSION,
        "scikit_learn_version": sklearn.__version__,
        "source_csv": source.name,
        "source_sha256": digest,
        "target": "price_vnd",
        "price_type": "asking",
        "input_accounting": accounting,
        "excluded_quality_flags": sorted(EXCLUDED_FLAGS),
        "evaluation_method": "Chronological train/validation/test by complete listing days, approximately 70/15/15. Model selected on validation MAE only.",
        "deduplication": "Earliest matching listing ID or normalized title, type, structured location, rounded area and bedrooms. Relistings with different descriptions or titles may still overlap.",
        "interval_method": "Point estimate ± 90th percentile of absolute validation errors, clipped at zero. Empirical uncertainty on this data, without a future coverage guarantee.",
        "subgroup_min_test_rows": SUBGROUP_MIN,
        "groups": {},
        "skipped_groups": {},
    }
    models = {}
    holdouts = []
    for group in FEATURES:
        group_frame = data.loc[data["property_group"].eq(group)]
        try:
            artifact, stats, predictions = _train_group(group, group_frame, include_random_forest)
        except ValueError as error:
            # Report insufficient dates or group size distinctly from training failures.
            try:
                _split_dates(group_frame)
            except ValueError:
                report["skipped_groups"][group] = str(error)
                continue
            raise
        models[group] = artifact
        report["groups"][group] = stats
        holdouts.append(predictions)
    if not models:
        raise ValueError("Không có nhóm bất động sản đủ dữ liệu để huấn luyện và kiểm tra.")
    predictions = pd.concat(holdouts, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model-build-", dir=output_dir) as tmp:
        stage = Path(tmp)
        joblib.dump({"format_version": FORMAT_VERSION, "source_sha256": digest, "models": models}, stage / MODEL_FILE, compress=3)
        (stage / REPORT_FILE).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _subgroup_errors(predictions).to_csv(stage / "subgroup_errors.csv", index=False, encoding="utf-8-sig")
        predictions.to_csv(stage / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
        for filename in (MODEL_FILE, "subgroup_errors.csv", "holdout_predictions.csv", REPORT_FILE):
            (stage / filename).replace(output_dir / filename)
    return report


def load_artifacts(directory: Path) -> dict:
    """Load only a locally generated, trusted joblib model bundle."""
    directory = Path(directory)
    report = json.loads((directory / REPORT_FILE).read_text(encoding="utf-8"))
    if report.get("format_version") != FORMAT_VERSION or report.get("scikit_learn_version") != sklearn.__version__:
        raise ValueError("Phiên bản mô hình hoặc scikit-learn không khớp. Hãy chạy lại scripts/train_models.py.")
    bundle = joblib.load(directory / MODEL_FILE)
    if bundle.get("format_version") != FORMAT_VERSION or bundle.get("source_sha256") != report.get("source_sha256"):
        raise ValueError("Mô hình và báo cáo không khớp. Hãy tạo lại cùng một lượt huấn luyện.")
    return {"models": bundle["models"], "report": report}


def predict_listing(artifacts: dict, property_group: str, attributes: dict) -> dict:
    """Estimate an asking price and compare individual inputs with training reference values."""
    entry = artifacts["models"].get(property_group)
    if entry is None:
        raise ValueError("Chưa có mô hình đã huấn luyện cho nhóm bất động sản này.")
    features = entry["features"]
    row = _prepare_features(pd.DataFrame([{feature: attributes.get(feature) for feature in features}]), features)
    area = row.at[0, "area_m2"]
    if not pd.notna(area) or area <= 0:
        raise ValueError("Diện tích phải là số lớn hơn 0 m².")
    for feature in set(features) & NUMERIC_FEATURES:
        if pd.notna(row.at[0, feature]) and row.at[0, feature] < 0:
            raise ValueError(f"{feature} phải lớn hơn hoặc bằng 0.")
    price = float(_predict_vnd(entry["model"], row)[0])
    radius = entry["interval_radius_vnd"]
    warnings = []
    low_area, high_area = entry["area_p01_p99_m2"]
    if not low_area <= area <= high_area:
        warnings.append("Diện tích nằm ngoài khoảng chứa 98% dữ liệu huấn luyện.")
    if row.at[0, "district_name"] not in entry["known_districts"]:
        warnings.append("Quận/huyện chưa xuất hiện trong tập huấn luyện của nhóm này.")
    if entry["model_name"] == "median_baseline":
        warnings.append("Mô hình trung vị được chọn vì sai số trên tập validation thấp hơn hoặc bằng các mô hình còn lại.")
    group_report = artifacts.get("report", {}).get("groups", {}).get(property_group, {})
    test_r2 = group_report.get("test_metrics", {}).get(entry["model_name"], {}).get("r2")
    if test_r2 is not None and test_r2 < 0:
        warnings.append("R² trên tập kiểm tra âm. Ước tính của nhóm này có độ tin cậy thấp.")
    if price == 0:
        warnings.append("Mô hình cho kết quả sát 0. Hãy kiểm tra lại đầu vào và vùng dữ liệu hỗ trợ.")
    effects = []
    for feature, reference in entry["reference_values"].items():
        if pd.isna(row.at[0, feature]) or row.at[0, feature] == reference:
            continue
        changed = row.copy()
        changed.at[0, feature] = reference
        difference = price - float(_predict_vnd(entry["model"], changed)[0])
        if abs(difference) >= 1:
            effects.append({"feature": feature, "reference_value": reference, "price_difference_vnd": round(difference)})
    effects.sort(key=lambda item: abs(item["price_difference_vnd"]), reverse=True)
    return {
        "predicted_price_vnd": round(price),
        "predicted_price_per_m2_vnd": round(price / area),
        "interval_vnd": [round(max(0, price - radius)), round(price + radius)],
        "model_name": entry["model_name"],
        "effects": effects[:5],
        "warnings": warnings,
    }
