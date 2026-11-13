from __future__ import annotations

import os
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from advisor.data import DEFAULT_CSV, GROUP_LABELS as TYPES, load_listings
from advisor.knowledge import development_insights, load_facts
from advisor.modeling import FEATURES, load_artifacts, predict_listing
from advisor.search import SearchCriteria, compare_listings, search_listings, solve_csp


ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("HANOI_DATA_CSV", str(DEFAULT_CSV))).expanduser()
MODEL_DIR = Path(os.getenv("HANOI_MODEL_DIR", str(ROOT / "models"))).expanduser()
LABELS = {
    "name": "Tin đăng", "property_group": "Loại hình", "district_name": "Quận / huyện",
    "ward_name": "Phường / xã", "street_name": "Đường", "project_name": "Dự án",
    "price_vnd": "Giá chào bán (đ)", "area_m2": "Diện tích (m²)",
    "price_per_m2_vnd": "Giá chào bán (đ/m²)", "floor_count": "Số tầng",
    "bedroom_count": "Phòng ngủ", "bathroom_count": "Phòng tắm",
    "frontage_m": "Mặt tiền (m)", "road_width_m": "Đường trước nhà (m)",
    "house_direction": "Hướng nhà", "balcony_direction": "Hướng ban công",
    "published_at": "Ngày đăng", "score": "Điểm phù hợp",
}
MODEL_NAMES = {
    "median_baseline": "Trung vị tập huấn luyện",
    "linear_regression": "Linear Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}
PAGES = {
    "Tổng quan": ("01 / DỮ LIỆU", "Tổng quan thị trường tin rao", "Khảo sát độ phủ, thời gian và giá chào bán trong bộ dữ liệu Hà Nội."),
    "Ước tính giá": ("02 / MÔ HÌNH", "Ước tính giá chào bán", "Nhập đặc điểm bất động sản và xem sai số trên những tin đăng mới hơn."),
    "Tìm bất động sản": ("03 / TÌM KIẾM", "Tìm bất động sản", "Đặt điều kiện bắt buộc, sau đó xếp hạng theo các ưu tiên của bạn."),
    "Phát triển đô thị": ("04 / THÔNG TIN KHU VỰC", "Hạ tầng và quy hoạch", "Xem thông tin có nguồn, trạng thái công bố và lý do liên quan đến khu vực."),
    "So sánh và giải thích": ("05 / SO SÁNH", "Đặt các lựa chọn cạnh nhau", "So sánh giá, diện tích và những tiêu chí ảnh hưởng tới thứ hạng."),
    "Thử nghiệm CSP": ("06 / THUẬT TOÁN", "Tìm tổ hợp trong ngân sách", "Theo dõi cách thuật toán quay lui chọn nhiều bất động sản theo ràng buộc."),
}

CSS = """
<style>
[data-testid="stAppViewContainer"] {background:#f4f7f6;}
[data-testid="stHeader"] {background:rgba(244,247,246,.94);}
[data-testid="stSidebar"] {background:#edf4f1;border-right:1px solid #dce8e2;}
[data-testid="stSidebar"] .stRadio label {margin:.12rem 0;}
.block-container {max-width:1240px;padding-top:2.2rem;}
.hero {background:linear-gradient(110deg,#102c2a 0%,#1e5147 100%);padding:clamp(1.25rem,3.2vw,2.7rem);
       border-radius:20px;color:white;margin-bottom:1.25rem;border:1px solid #34655b;}
.hero-kicker {font-size:.72rem;font-weight:700;letter-spacing:.16em;color:#b9e7d5;margin-bottom:.55rem;}
.hero h1 {color:#fff;font-size:clamp(1.7rem,3.6vw,2.9rem);line-height:1.13;padding:0;margin:0;}
.hero p {color:#e3f1eb;font-size:1rem;line-height:1.55;margin:.7rem 0 0;max-width:57rem;}
[data-testid="stMetric"] {background:white;border:1px solid #dce8e2;border-radius:14px;padding:.85rem 1rem;}
[data-testid="stMetricValue"] {font-size:clamp(1.15rem,2.8vw,2rem);}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#dce8e2;}
</style>
"""


def vi_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def amount(value: object, divisor: int = 1_000_000_000, unit: str = "tỷ đ") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) / divisor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".") + f" {unit}"


def label(name: str) -> str:
    return LABELS.get(name, name)


def text(value: object, fallback: str = "Chưa có dữ liệu") -> str:
    if value is None or pd.isna(value):
        return fallback
    return str(value).strip() or fallback


def categories(values: pd.Series) -> list[str]:
    return sorted({str(v).strip() for v in values.dropna() if str(v).strip()}, key=str.casefold)


def area(value: object) -> str:
    return f"{float(value):g} m²" if value is not None and not pd.isna(value) else "—"


@st.cache_resource(show_spinner="Đang nạp mô hình…")
def cached_models(folder: str, snapshot: tuple) -> dict:
    return load_artifacts(Path(folder))


def model_bundle() -> dict | None:
    if not MODEL_DIR.is_dir():
        st.info("Chưa có mô hình đã huấn luyện. Chạy `python scripts/train_models.py` theo README để tạo kết quả.")
        return None
    try:
        snapshot = tuple(sorted(
            (p.name, p.stat().st_mtime_ns) for p in MODEL_DIR.iterdir() if p.is_file()
        ))
        return cached_models(str(MODEL_DIR), snapshot)
    except (OSError, ValueError, RuntimeError, ImportError, EOFError) as exc:
        st.error(f"Không nạp được mô hình. {exc}")
        return None


def hero(page: str) -> None:
    kicker, title, subtitle = PAGES[page]
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">{kicker}</div>'
        f"<h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def dashboard(frame: pd.DataFrame) -> None:
    st.caption("Đơn vị quan sát là tin đăng. Nhiều tin có thể cùng mô tả một bất động sản.")
    filters = st.columns(2)
    groups = categories(frame["property_group"])
    districts = categories(frame["district_name"])
    group = filters[0].selectbox("Loại hình", [None, *groups], format_func=lambda x: TYPES.get(x, x) if x else "Tất cả")
    district = filters[1].selectbox("Quận / huyện theo dữ liệu nguồn", [None, *districts], format_func=lambda x: x or "Tất cả")
    view = frame
    if group:
        view = view[view["property_group"].eq(group)]
    if district:
        view = view[view["district_name"].eq(district)]
    if view.empty:
        st.info("Không có tin trong phạm vi đang chọn.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Tin rao", vi_int(len(view)))
    b.metric("Quận / huyện", vi_int(len(categories(view["district_name"]))))
    c.metric("Giá chào bán trung vị", amount(view["price_vnd"].median()))
    d.metric("Diện tích trung vị", area(view["area_m2"].median()))
    dates = pd.to_datetime(view["published_at"], errors="coerce", utc=True, format="mixed").dropna()
    if not dates.empty:
        st.caption(f"Khoảng ngày đăng có ghi nhận từ {dates.min():%d/%m/%Y} đến {dates.max():%d/%m/%Y}.")
    col_chart, col_detail = st.columns([3, 2])
    with col_chart:
        st.subheader("Khu vực có nhiều tin nhất")
        counts = view["district_name"].replace("", pd.NA).dropna().value_counts().head(12)
        if not counts.empty:
            counts = counts.iloc[::-1]
            chart = go.Figure(go.Bar(
                x=counts.values, y=counts.index, orientation="h", marker_color="#257b64",
                hovertemplate="%{y}<br>%{x:,} tin<extra></extra>",
            ))
            chart.update_layout(height=355, margin=dict(l=0, r=10, t=10, b=15), xaxis_title="Số tin đăng")
            st.plotly_chart(chart, config={"displayModeBar": False})
        else:
            st.caption("Các tin trong phạm vi này thiếu quận / huyện.")
    with col_detail:
        st.subheader("Dữ liệu theo loại hình")
        overview = view.groupby("property_group").agg(
            count=("property_group", "size"), median=("price_vnd", "median")
        ).reset_index()
        overview["property_group"] = overview["property_group"].map(TYPES).fillna(overview["property_group"])
        overview["median"] = overview["median"].map(amount)
        st.dataframe(overview.rename(columns={"property_group": "Loại hình", "count": "Số tin", "median": "Trung vị"}), hide_index=True)
        if not group:
            st.caption("Chọn một loại hình để xem giá trung vị của nhóm đó.")
    with st.expander("Nguồn dữ liệu và độ đầy đủ"):
        keys = ("district_name", "ward_name", "project_name", "bedroom_count", "bathroom_count", "published_at")
        coverage = [
            {"Trường": label(key), "Tỷ lệ có dữ liệu (%)": round(100 * view[key].notna().mul(view[key].ne("")).mean(), 1)}
            for key in keys
        ]
        st.dataframe(pd.DataFrame(coverage), hide_index=True)
        st.caption("Tên quận, huyện và phường được dùng theo bộ dữ liệu tại thời điểm thu thập.")
        if "source_url" in view:
            for url in categories(view["source_url"]):
                if url.startswith("https://"):
                    st.link_button("Mở trang dữ liệu gốc", url)


def evaluation(report: dict, group: str) -> None:
    group_report = report.get("groups", {}).get(group, {})
    metrics = group_report.get("test_metrics", {})
    if not metrics:
        return
    st.subheader("Kết quả trên tập kiểm tra")
    ranges = group_report.get("date_ranges", {})
    test_range = ranges.get("test")
    if test_range:
        st.caption(f"Tập kiểm tra từ {test_range[0]} đến {test_range[-1]} gồm {vi_int(group_report.get('test_rows', 0))} tin.")
    selected = group_report.get("selected_model")
    if selected:
        st.caption(f"Mô hình được chọn theo MAE trên tập xác thực · {MODEL_NAMES.get(selected, selected)}.")
    st.caption("MAE là sai số tuyệt đối trung bình. RMSE nhạy hơn với các dự đoán sai lệch lớn. R² so sánh khả năng giải thích biến động giá.")
    rows = []
    for model, item in metrics.items():
        r2 = item.get("r2")
        rows.append({
            "Mô hình": MODEL_NAMES.get(model, model),
            "MAE (triệu đ)": None if item.get("mae_vnd") is None else round(item["mae_vnd"] / 1e6, 1),
            "RMSE (triệu đ)": None if item.get("rmse_vnd") is None else round(item["rmse_vnd"] / 1e6, 1),
            "R²": None if r2 is None else round(r2, 3),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    with st.expander("Biểu đồ sai số và đánh giá theo nhóm"):
        show_holdout(group)


@st.cache_data(show_spinner=False)
def read_evaluation_csv(path: Path, changed_at: int) -> pd.DataFrame:
    return pd.read_csv(path)


def show_holdout(group: str) -> None:
    predictions_path = MODEL_DIR / "holdout_predictions.csv"
    breakdown_path = MODEL_DIR / "subgroup_errors.csv"
    try:
        if predictions_path.is_file():
            pred = read_evaluation_csv(predictions_path, predictions_path.stat().st_mtime_ns)
            pred = pred.loc[pred["property_group"].eq(group)]
            if not pred.empty:
                st.caption("Mỗi chấm là một tin trong mẫu tập kiểm tra. Giá ước tính càng gần giá chào bán thì sai số càng nhỏ.")
                display = pred.sample(min(1800, len(pred)), random_state=42)
                actual = display["actual_asking_price_vnd"] / 1e9
                predicted = display["predicted_asking_price_vnd"] / 1e9
                lower = min(actual.min(), predicted.min())
                upper = max(actual.max(), predicted.max())
                chart = go.Figure()
                chart.add_trace(go.Scatter(
                    x=[lower, upper], y=[lower, upper], mode="lines", name="Sai số bằng 0",
                    line={"dash": "dash", "color": "#657974"}, hoverinfo="skip",
                ))
                chart.add_trace(go.Scatter(
                    x=actual, y=predicted, mode="markers", name="Tin kiểm tra",
                    marker={"color": "#257b64", "size": 6, "opacity": 0.45},
                    hovertemplate="Chào bán %{x:.2f} tỷ đ<br>Ước tính %{y:.2f} tỷ đ<extra></extra>",
                ))
                chart.update_layout(
                    height=360, margin=dict(l=0, r=10, t=5, b=10),
                    xaxis_title="Giá chào bán (tỷ đ)", yaxis_title="Giá ước tính (tỷ đ)",
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(chart, config={"displayModeBar": False})
        if breakdown_path.is_file():
            report = read_evaluation_csv(breakdown_path, breakdown_path.stat().st_mtime_ns)
            report = report.loc[report["property_group"].eq(group)].copy()
            if not report.empty:
                report["breakdown"] = report["breakdown"].map(
                    {"district_name": "Quận / huyện", "area_group": "Nhóm diện tích"}
                ).fillna(report["breakdown"])
                report["status"] = report["status"].map(
                    lambda value: "Có báo cáo" if value == "reported" else "Mẫu hạn chế"
                )
                for field in ("mae_vnd", "rmse_vnd"):
                    report[field] = (report[field] / 1e6).round(1)
                report["r2"] = report["r2"].round(3)
                st.dataframe(report.drop(columns="property_group").rename(columns={
                    "breakdown": "Phân nhóm", "subgroup": "Nhóm", "rows": "Số tin", "status": "Trạng thái",
                    "mae_vnd": "MAE (triệu đ)", "rmse_vnd": "RMSE (triệu đ)", "r2": "R²",
                }), hide_index=True)
    except (OSError, ValueError, KeyError):
        st.caption("Chưa đọc được bảng sai số chi tiết. Hãy tạo lại kết quả huấn luyện.")


def estimator(frame: pd.DataFrame) -> None:
    bundle = model_bundle()
    if not bundle:
        return
    trained_groups = [g for g in TYPES if g in bundle.get("models", {}) and g in FEATURES]
    if not trained_groups:
        st.info("Chưa có nhóm bất động sản nào đủ dữ liệu để ước tính.")
        return
    group = st.selectbox("Loại bất động sản", trained_groups, format_func=lambda x: TYPES[x], key="estimate_group")
    pool = frame[frame["property_group"].eq(group)]
    attributes: dict = {}
    with st.form(f"estimate_{group}"):
        columns = st.columns(3)
        for i, field in enumerate(FEATURES[group]):
            panel = columns[i % len(columns)]
            if field in {"district_name", "ward_name", "project_name", "house_direction", "balcony_direction"}:
                opts = categories(pool[field]) if field in pool else []
                attributes[field] = panel.selectbox(label(field), [None, *opts], format_func=lambda x: x or "Không rõ")
            elif field in {"floor_count", "bedroom_count", "bathroom_count"}:
                attributes[field] = panel.number_input(label(field), min_value=0, value=None, step=1, placeholder="Không rõ")
            else:
                attributes[field] = panel.number_input(
                    label(field), min_value=0.1, value=75.0 if field == "area_m2" else None, step=0.5,
                    placeholder="Không rõ" if field != "area_m2" else "Nhập diện tích",
                )
        submitted = st.form_submit_button("Ước tính giá", type="primary")
    if submitted:
        try:
            result = predict_listing(bundle, group, attributes)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Kiểm tra thông số đã nhập. {exc}")
        else:
            col_a, col_b = st.columns(2)
            col_a.metric("Giá chào bán ước tính", amount(result["predicted_price_vnd"]))
            col_b.metric("Đơn giá ước tính", amount(result["predicted_price_per_m2_vnd"], 1_000_000, "triệu đ/m²"))
            st.caption(f"Mô hình được chọn • {MODEL_NAMES.get(result['model_name'], result['model_name'])}")
            interval = result.get("interval_vnd")
            if interval:
                st.write(f"**Khoảng tham khảo từ sai số mô hình**  {amount(interval[0])} – {amount(interval[1])}")
            for warning in result.get("warnings", []):
                st.warning(warning)
            if result.get("effects"):
                with st.expander("Đặc trưng ảnh hưởng thế nào trong mô hình?"):
                    st.caption("So sánh khi lần lượt đổi một đặc trưng về giá trị tham chiếu, giữ các đặc trưng khác cố định.")
                    for effect in result["effects"]:
                        change = effect["price_difference_vnd"]
                        st.write(f"**{label(effect['feature'])}** · Tham chiếu {text(effect['reference_value'])} · {amount(change, 1_000_000, 'triệu đ')}")
    evaluation(bundle.get("report", {}), group)


def filters(frame: pd.DataFrame) -> tuple[bool, SearchCriteria]:
    with st.form("property_filters"):
        st.markdown("**Điều kiện bắt buộc**")
        row = st.columns(3)
        price = row[0].number_input("Giá tối đa (tỷ đ)", min_value=0.01, value=10.0, step=0.5)
        minimum_area = row[1].number_input("Diện tích tối thiểu (m²)", min_value=0.0, value=45.0, step=5.0)
        bedrooms = row[2].number_input("Phòng ngủ tối thiểu", min_value=0, value=2, step=1, help="Chọn 0 để bỏ qua điều kiện số phòng ngủ.")
        row = st.columns(2)
        districts = row[0].multiselect("Quận / huyện", categories(frame["district_name"]), placeholder="Tất cả khu vực")
        groups = row[1].multiselect("Loại bất động sản", categories(frame["property_group"]), format_func=lambda x: TYPES.get(x, x), placeholder="Tất cả loại hình")
        st.markdown("**Ưu tiên để xếp hạng**")
        row = st.columns(3)
        preferred_price = row[0].number_input("Giá mong muốn (tỷ đ)", min_value=0.01, value=None, step=0.5, placeholder="Bỏ qua")
        preferred_area = row[1].number_input("Diện tích mong muốn (m²)", min_value=1.0, value=None, step=5.0, placeholder="Bỏ qua")
        preferred_districts = row[2].multiselect("Khu vực ưu tiên", categories(frame["district_name"]), placeholder="Bỏ qua")
        row = st.columns(3)
        weights = [
            cell.slider(name, 0, 100, value) / 100 for cell, name, value in
            zip(row, ("Trọng số giá", "Trọng số diện tích", "Trọng số khu vực"), (45, 35, 20))
        ]
        exclude = st.checkbox("Bỏ qua các tin có cờ cần rà soát giá hoặc diện tích", value=True)
        submit = st.form_submit_button("Tìm và xếp hạng", type="primary")
    return submit, SearchCriteria(
        max_price_vnd=round(price * 1e9), min_area_m2=minimum_area or None,
        min_bedrooms=bedrooms or None, districts=tuple(districts), property_groups=tuple(groups),
        preferred_price_vnd=None if preferred_price is None else round(preferred_price * 1e9),
        preferred_area_m2=preferred_area, preferred_districts=tuple(preferred_districts),
        price_weight=weights[0], area_weight=weights[1], district_weight=weights[2],
        exclude_quality_outliers=exclude,
    )


def planning_context(district: str) -> None:
    if not district:
        st.caption("Tin thiếu khu vực để đối chiếu thông tin quy hoạch.")
        return
    try:
        facts = development_insights(district)
    except (OSError, ValueError):
        st.caption("Chưa đọc được kho thông tin phát triển.")
        return
    if facts:
        for fact in facts[:2]:
            st.write(f"**{fact['title']}** · {fact['status']}")
            st.caption(fact["explanation"])
            st.markdown(f"[Nguồn · {fact['source_name']}]({fact['source_url']})")
        st.caption("Liên hệ theo khu vực. Tác động tới từng bất động sản cần đối chiếu vị trí và hồ sơ chi tiết.")
    else:
        st.caption("Kho thông tin hiện chưa có hồ sơ đã đối chiếu cho khu vực này.")


def results_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    cols = ("name", "property_group", "district_name", "price_vnd", "area_m2", "bedroom_count", "score")
    table = table.reindex(columns=cols)
    table["property_group"] = table["property_group"].map(TYPES).fillna(table["property_group"])
    table["price_vnd"] = table["price_vnd"].map(amount)
    return table.rename(columns={**LABELS, "price_vnd": "Giá chào bán"})


def search(frame: pd.DataFrame) -> None:
    submitted, criteria = filters(frame)
    if submitted:
        st.session_state["search_results"] = []
        try:
            with st.spinner("Đang áp dụng ràng buộc và xếp hạng…"):
                st.session_state["search_results"] = search_listings(frame, criteria, limit=30)
        except ValueError as exc:
            st.error(str(exc))
            return
    rows = st.session_state.get("search_results")
    if rows is None:
        st.caption("Kết quả sẽ hiện sau khi bạn bấm Tìm và xếp hạng.")
        return
    if not rows:
        st.info("Không có tin đáp ứng tất cả điều kiện. Bạn có thể tăng ngân sách hoặc nới điều kiện diện tích, phòng ngủ.")
        return
    st.subheader(f"{len(rows)} tin phù hợp nhất")
    st.caption("Điểm 0–100 thể hiện mức độ phù hợp với tiêu chí đã nhập.")
    st.dataframe(results_table(rows), hide_index=True)
    st.subheader("Vì sao các tin đầu được đề xuất?")
    for i, row in enumerate(rows[:5], start=1):
        name = text(row.get("name"), "Tin đăng không có tiêu đề")
        with st.expander(f"{i}. {name} · {float(row['score']):.1f} điểm", expanded=i == 1):
            st.write(f"**{amount(row['price_vnd'])}** · {area(row['area_m2'])} · {text(row.get('district_name'))}")
            for reason in row.get("reasons", []):
                st.write(f"- {reason}")
            if row.get("missing_fields"):
                st.caption("Trường còn thiếu · " + ", ".join(label(v) for v in row["missing_fields"]))
            highlights = text(row.get("dac_diem_noi_bat"), "")
            if highlights:
                st.caption(f"Tiêu đề tin có nhắc · {highlights}")
            planning_context(text(row.get("district_name"), ""))
    st.caption("Dùng trang So sánh và giải thích để đặt tối đa 4 tin trong danh sách này cạnh nhau.")


def development(frame: pd.DataFrame | None) -> None:
    st.caption("Thông tin phát triển được hiển thị theo phạm vi và trạng thái ghi trong hồ sơ nguồn.")
    try:
        facts = load_facts()
        districts = set(categories(frame["district_name"])) if frame is not None and not frame.empty else set()
        for fact in facts:
            raw = fact.get("districts", [])
            districts.update([raw] if isinstance(raw, str) else raw)
        district = st.selectbox("Khu vực quan tâm", [None, *sorted(districts)], format_func=lambda x: x or "Toàn Hà Nội")
        relevant = development_insights(district) if district else facts
    except (OSError, ValueError) as exc:
        st.error(f"Không đọc được kho thông tin phát triển. {exc}")
        return
    if not relevant:
        st.info("Chưa có thông tin phù hợp trong kho dữ liệu hiện tại.")
        return
    for fact in relevant:
        with st.container(border=True):
            st.subheader(fact["title"])
            st.caption(f"{fact['status']} · Công bố {fact['published_at']}")
            st.write(fact["description"])
            st.caption("Khu vực có trong hồ sơ · " + ", ".join(fact["districts"]))
            if "explanation" in fact:
                st.write(fact["explanation"])
                with st.expander("Quy tắc liên hệ với khu vực"):
                    st.write(fact["matched_rule"])
            st.markdown(f"[Mở nguồn · {fact['source_name']}]({fact['source_url']})")


def comparison(frame: pd.DataFrame) -> None:
    ranked = st.session_state.get("search_results", [])
    if not ranked:
        st.info("Hãy tìm và xếp hạng bất động sản trước để tạo danh sách so sánh.")
        return
    ranked_by_id = {row["listing_id"]: row for row in ranked}
    chosen = st.multiselect(
        "Chọn tối đa 4 tin từ kết quả tìm kiếm",
        list(ranked_by_id),
        default=list(ranked_by_id)[:2],
        max_selections=4,
        format_func=lambda key: text(ranked_by_id[key].get("name"), "Tin chưa có tiêu đề"),
    )
    if not chosen:
        st.caption("Chọn tin để xem bảng so sánh.")
        return
    selected = compare_listings(frame.loc[frame["listing_id"].isin(chosen)], chosen)
    columns = [
        "property_group", "district_name", "ward_name", "project_name", "price_vnd",
        "area_m2", "price_per_m2_vnd", "bedroom_count", "bathroom_count", "floor_count", "published_at",
    ]
    values = {}
    for i, row in enumerate(selected, 1):
        st.caption(f"Tin {i} · {escape(text(row.get('name'), 'Tin chưa có tiêu đề'))}")
        row_data = {col: text(row.get(col), "—") for col in columns}
        row_data["property_group"] = TYPES.get(row.get("property_group"), row_data["property_group"])
        row_data["price_vnd"] = amount(row.get("price_vnd"))
        row_data["price_per_m2_vnd"] = amount(row.get("price_per_m2_vnd"), 1_000_000, "triệu đ/m²")
        row_data["area_m2"] = area(row.get("area_m2"))
        values[f"Tin {i}"] = {label(k): v for k, v in row_data.items()}
    st.dataframe(pd.DataFrame(values))
    st.subheader("Điểm xếp hạng theo nhu cầu")
    for i, key in enumerate(chosen, 1):
        row = ranked_by_id[key]
        with st.expander(f"Tin {i} · {row['score']:.1f}/100 điểm"):
            for reason in row.get("reasons", []):
                st.write(f"- {reason}")
            if row.get("missing_fields"):
                st.caption("Trường còn thiếu · " + ", ".join(label(v) for v in row["missing_fields"]))


def csp() -> None:
    rows = st.session_state.get("search_results", [])
    if not rows:
        st.info("Hãy chạy trang Tìm bất động sản để tạo tập ứng viên cho bài toán CSP.")
        return
    st.write("Chọn một số tin khác nhau sao cho tổng giá chào bán nằm trong ngân sách. Mỗi vị trí chọn là một biến, với các tin sau lọc làm miền giá trị.")
    with st.form("csp_form"):
        a, b, c = st.columns(3)
        count = a.number_input("Số bất động sản cần chọn", min_value=1, max_value=min(5, len(rows)), value=min(2, len(rows)), step=1)
        budget = b.number_input("Tổng ngân sách (tỷ đ)", min_value=0.01, value=15.0, step=0.5)
        limit = c.number_input("Số ứng viên thử", min_value=1, max_value=len(rows), value=min(16, len(rows)), step=1)
        submit = st.form_submit_button("Chạy Backtracking", type="primary")
    if not submit:
        return
    try:
        result = solve_csp(rows, count=count, total_budget_vnd=round(budget * 1e9), candidate_limit=limit)
    except ValueError as exc:
        st.error(str(exc))
        return
    if result["status"] == "solved":
        st.success(f"Tìm được tổ hợp {len(result['selected'])} tin với tổng giá {amount(result['total_price_vnd'])}.")
        st.dataframe(results_table(result["selected"]), hide_index=True)
    elif result["status"] == "node_limit":
        st.warning("Đã đạt giới hạn duyệt. Thử giảm số ứng viên hoặc thay đổi ràng buộc.")
    else:
        st.info("Không tìm được tổ hợp trong tập ứng viên đã thử.")
    a, b, c = st.columns(3)
    a.metric("Ứng viên", result["candidate_count"])
    b.metric("Nút đã duyệt", result["nodes"])
    c.metric("Lượt quay lui", result["backtracks"])
    if result.get("truncated"):
        st.caption("Chỉ xét tập ứng viên giới hạn trong danh sách xếp hạng.")
    if result.get("trace"):
        with st.expander("Các bước thử, loại và quay lui", expanded=True):
            for step in result["trace"]:
                st.text(step)


def main() -> None:
    st.set_page_config(page_title="Hanoi Real Estate Advisor", page_icon="🏙️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Hanoi Real Estate Advisor")
        st.caption("Đồ án Nhập môn Trí tuệ nhân tạo · Hà Nội")
        page = st.radio("Chức năng", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption("Dữ liệu lịch sử từ tin rao. Giá hiển thị và mô hình cùng dùng giá chào bán.")
    hero(page)
    frame = None
    if DATA_PATH.is_file():
        try:
            with st.spinner("Đang đọc dữ liệu Hà Nội…"):
                frame = load_listings(DATA_PATH)
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Không đọc được CSV đã xử lý. {exc}")
    if page == "Phát triển đô thị":
        development(frame)
        return
    if frame is None or frame.empty:
        st.info("Hãy tạo bộ CSV Hà Nội bằng pipeline trong README rồi mở lại ứng dụng.")
        st.code(str(DATA_PATH))
        return
    if page == "Tổng quan":
        dashboard(frame)
    elif page == "Ước tính giá":
        estimator(frame)
    elif page == "Tìm bất động sản":
        search(frame)
    elif page == "So sánh và giải thích":
        comparison(frame)
    else:
        csp()


if __name__ == "__main__":
    main()
