from __future__ import annotations

import base64
import os
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from advisor.data import DEFAULT_CSV, GROUP_LABELS, load_listings
from advisor.infrastructure import rank_listings
from advisor.listing_dedup import deduplicate_listings
from advisor.modeling import FEATURES, load_artifacts, predict_listing
from advisor.search import SearchCriteria


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
DATA_PATH = Path(os.getenv("HANOI_DATA_CSV", str(DEFAULT_CSV)))
MODEL_DIR = Path(os.getenv("HANOI_MODEL_DIR", str(ROOT / "models")))
TABS = ("Trang chủ", "Khám phá", "Ước tính giá", "Đã lưu & so sánh")
PRIORITIES = {"mobility": "Đi lại thuận tiện", "development": "Theo dõi phát triển đô thị"}
SORTS = {"needs": "Phù hợp nhu cầu", "infrastructure": "Phù hợp hạ tầng"}
FEATURE_LABELS = {
    "district_name": "Khu vực", "ward_name": "Phường / xã", "project_name": "Dự án",
    "area_m2": "Diện tích (m²)", "bedroom_count": "Phòng ngủ", "bathroom_count": "Phòng tắm",
    "floor_count": "Số tầng", "frontage_m": "Mặt tiền (m)", "road_width_m": "Đường trước nhà (m)",
    "house_direction": "Hướng nhà", "balcony_direction": "Hướng ban công",
}
PAGE_SIZE = 9
RESULT_LIMIT = 72
DISPLAY_MIN_PRICE_PER_M2 = 15_000_000


def value(raw: object, fallback: str = "Chưa có thông tin") -> str:
    if raw is None or pd.isna(raw):
        return fallback
    return str(raw).strip() or fallback


def vi_number(raw: float, decimals: int = 0) -> str:
    return f"{raw:,.{decimals}f}".replace(",", "~").replace(".", ",").replace("~", ".")


def money(raw: object, *, unit_price: bool = False) -> str:
    if raw is None or pd.isna(raw):
        return "Chưa có giá"
    divisor = 1e6 if unit_price else 1e9
    suffix = "triệu ₫/m²" if unit_price else "tỷ ₫"
    return f"{vi_number(float(raw) / divisor, 2)} {suffix}"


def measurement(raw: object, suffix: str) -> str:
    if raw is None or pd.isna(raw):
        return "—"
    number = float(raw)
    return f"{vi_number(number, 0 if number.is_integer() else 1)} {suffix}"


def options(series: pd.Series) -> list[str]:
    return sorted({str(v).strip() for v in series.dropna() if str(v).strip()}, key=str.casefold)


@st.cache_data(show_spinner=False)
def visual_assets(css_stamp: int, image_stamp: int) -> tuple[str, str]:
    css_file = ASSETS / "storefront.css"
    photo = ASSETS / "hanoi-hero.jpg"
    css = css_file.read_text(encoding="utf-8") if css_stamp else ""
    photo_url = "data:image/jpeg;base64," + base64.b64encode(photo.read_bytes()).decode() if image_stamp else ""
    return css, photo_url


def stamp(path: Path) -> int:
    return path.stat().st_mtime_ns if path.is_file() else 0


def apply_visuals() -> None:
    css, photo = visual_assets(stamp(ASSETS / "storefront.css"), stamp(ASSETS / "hanoi-hero.jpg"))
    background = f':root{{--estate-hero-image:url("{photo}");}}' if photo else ""
    st.markdown(f"<style>{background}{css}</style>", unsafe_allow_html=True)


def go(tab: str, *, clear_detail: bool = True) -> None:
    st.session_state["navigation"] = tab
    if clear_detail:
        st.session_state.pop("open_listing", None)


def close_detail() -> None:
    st.session_state.pop("open_listing", None)


def open_detail(row: dict) -> None:
    st.session_state["open_listing"] = row


def saved_items() -> dict:
    return st.session_state.setdefault("saved_listings", {})


def toggle_save(row: dict) -> None:
    key = row["listing_id"]
    saved = saved_items()
    if key in saved:
        del saved[key]
    else:
        saved[key] = row


def form_criteria(frame: pd.DataFrame, prefix: str, *, compact: bool) -> tuple[SearchCriteria, str, bool]:
    with st.form(f"{prefix}_search"):
        first = st.columns(3)
        district = first[0].selectbox("Khu vực", [None, *options(frame["district_name"])],
                                      format_func=lambda x: x or "Toàn Hà Nội", key=f"{prefix}_district")
        group = first[1].selectbox("Loại bất động sản", [None, *GROUP_LABELS],
                                    format_func=lambda x: GROUP_LABELS.get(x, "Tất cả loại hình"), key=f"{prefix}_group")
        budget = first[2].number_input("Ngân sách tối đa (tỷ ₫)", min_value=0.1, value=8.0, step=0.5, key=f"{prefix}_budget")
        second = st.columns(3)
        min_area = second[0].number_input("Diện tích từ (m²)", min_value=0, value=50, step=5, key=f"{prefix}_area")
        bedrooms = second[1].selectbox("Phòng ngủ tối thiểu", range(7),
                                       format_func=lambda x: f"{x} phòng" if x else "Linh hoạt", key=f"{prefix}_beds")
        priority = second[2].selectbox("Ưu tiên khu vực", PRIORITIES, format_func=PRIORITIES.get, key=f"{prefix}_priority")
        preferred_area = None
        if not compact:
            preferred_area = st.number_input("Diện tích mong muốn (m²)", min_value=1.0, value=None, step=5.0,
                                            placeholder="Linh hoạt", key=f"{prefix}_target_area")
        submitted = st.form_submit_button("Xem gợi ý phù hợp" if compact else "Áp dụng bộ lọc",
                                          type="primary", use_container_width=compact)
    criteria = SearchCriteria(
        max_price_vnd=round(budget * 1e9), districts=(district,) if district else (),
        property_groups=(group,) if group else (), min_area_m2=min_area or None,
        min_bedrooms=bedrooms or None, preferred_area_m2=preferred_area,
    )
    return criteria, priority, submitted


def set_search(criteria: SearchCriteria, priority: str) -> None:
    st.session_state["last_criteria"] = criteria
    st.session_state["last_priority"] = priority
    st.session_state["results_page"] = 1


def quick_to_explore(criteria: SearchCriteria, priority: str) -> None:
    set_search(criteria, priority)
    st.session_state["preload_explore"] = (criteria, priority)
    go("Khám phá")


def preload_explore() -> None:
    entry = st.session_state.pop("preload_explore", None)
    if not entry:
        return
    criteria, priority = entry
    st.session_state.update({
        "explore_district": next(iter(criteria.districts), None),
        "explore_group": next(iter(criteria.property_groups), None),
        "explore_budget": (criteria.max_price_vnd or 8_000_000_000) / 1e9,
        "explore_area": int(criteria.min_area_m2 or 0),
        "explore_beds": criteria.min_bedrooms or 0,
        "explore_priority": priority,
        "explore_target_area": criteria.preferred_area_m2,
    })


@st.cache_resource(show_spinner=False, max_entries=2)
def search_catalog(dataset_path: str, dataset_stamp: int) -> pd.DataFrame:
    """Reuse the same representative for a group across search filters and sort orders."""
    frame = load_listings(dataset_path)
    # Very low implied prices can come from unit prices entered into the total-price field.
    eligible = frame.loc[frame["price_per_m2_vnd"].ge(DISPLAY_MIN_PRICE_PER_M2)]
    return deduplicate_listings(eligible)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_rank(dataset_path: str, dataset_stamp: int, criteria: SearchCriteria, priority: str, order: str) -> list[dict]:
    return rank_listings(
        search_catalog(dataset_path, dataset_stamp), criteria, priority=priority, sort_by=order, limit=RESULT_LIMIT
    )


def ranked(criteria: SearchCriteria, priority: str, order: str) -> list[dict]:
    with st.spinner("Đang tìm lựa chọn phù hợp…"):
        return cached_rank(str(DATA_PATH), stamp(DATA_PATH), criteria, priority, order)


def section(kicker: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="estate-section-label">{escape(kicker)}</div>'
        f'<h2 class="estate-section-title">{escape(title)}</h2>'
        + (f'<p class="estate-section-lead">{escape(subtitle)}</p>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def title_for(row: dict) -> str:
    group = GROUP_LABELS.get(row.get("property_group"), "Bất động sản")
    project = value(row.get("project_name"), "")
    if project:
        return project
    title = value(row.get("name"), "")
    if title:
        return title if len(title) <= 82 else title[:80].rsplit(" ", 1)[0] + "…"
    return f"{group} · {value(row.get('district_name'), 'Hà Nội')}"


def assessment(row: dict) -> dict:
    return row.get("infrastructure") or {}


def infrastructure_reason(row: dict) -> str:
    info = assessment(row)
    return value(info.get("reason"), "Khu vực này chưa có hồ sơ hạ tầng phù hợp trong dữ liệu.")


def score_markup(row: dict) -> str:
    score = assessment(row).get("score")
    infra = f"{vi_number(score)}/100" if score is not None else "Chưa đủ dữ liệu"
    return (
        '<div class="estate-card-scores">'
        f'<div class="estate-score estate-score-needs"><span>Phù hợp nhu cầu</span><strong>{vi_number(row["score"])}/100</strong></div>'
        f'<div class="estate-score estate-score-infra"><span>Phù hợp hạ tầng</span><strong>{infra}</strong></div>'
        '</div>'
    )


def card(row: dict, index: int, prefix: str) -> None:
    group = GROUP_LABELS.get(row.get("property_group"), "Bất động sản")
    district = value(row.get("district_name"), "Hà Nội")
    ward = value(row.get("ward_name"), "")
    address = ", ".join(part for part in (ward, district) if part)
    meta = " · ".join(x for x in (
        measurement(row.get("area_m2"), "m²"),
        measurement(row.get("bedroom_count"), "phòng ngủ") if value(row.get("bedroom_count"), "") else "",
    ) if x)
    key = row["listing_id"]
    with st.container(border=True):
        st.markdown(
            '<article class="estate-card">'
            f'<div class="estate-card-top"><span class="estate-card-category">{escape(group)}</span></div>'
            f'<div class="estate-card-title">{escape(title_for(row))}</div>'
            f'<div class="estate-card-meta">{escape(address)}</div>'
            f'<div class="estate-card-price">{escape(money(row.get("price_vnd")))}</div>'
            f'<div class="estate-card-meta">{escape(meta)} · giá chào bán</div>'
            f'{score_markup(row)}'
            f'<div class="estate-card-reason">{escape(infrastructure_reason(row))}</div></article>',
            unsafe_allow_html=True,
        )
        details, save = st.columns([1.6, 1])
        details.button("Xem chi tiết", key=f"{prefix}_detail_{index}_{key}", on_click=open_detail, args=(row,),
                       use_container_width=True)
        save.button("Bỏ lưu" if key in saved_items() else "Lưu tin", key=f"{prefix}_save_{index}_{key}",
                    on_click=toggle_save, args=(row,), use_container_width=True)


def grid(rows: list[dict], prefix: str) -> None:
    for start in range(0, len(rows), 3):
        for offset, (column, row) in enumerate(zip(st.columns(3), rows[start:start + 3])):
            with column:
                card(row, start + offset, prefix)


def order_control(key: str) -> str:
    return st.radio("Sắp xếp theo", SORTS, format_func=SORTS.get, horizontal=True, key=key)


def homepage(frame: pd.DataFrame) -> None:
    st.markdown(
        '<div class="estate-hero"><div class="estate-hero-content">'
        '<div class="estate-kicker">HANOI REAL ESTATE ADVISOR</div>'
        '<h1>Tìm nơi bạn muốn sống ở Hà Nội.</h1>'
        '<p class="estate-lead">Chọn căn phù hợp ngân sách và hiểu thêm về khu vực bạn sắp gọi là nhà.</p>'
        '</div></div>', unsafe_allow_html=True,
    )
    section("TƯ VẤN NHANH", "Bạn đang tìm một nơi như thế nào?")
    criteria, priority, submitted = form_criteria(frame, "home", compact=True)
    if submitted:
        st.session_state["home_query"] = (criteria, priority)
    query = st.session_state.get("home_query")
    if query:
        criteria, priority = query
        section("DÀNH CHO BẠN", "Những lựa chọn nổi bật", f"Ưu tiên khu vực · {PRIORITIES[priority]}")
        order = order_control("home_order")
        rows = ranked(criteria, priority, order)
        if rows:
            grid(rows[:6], "home")
            st.button("Xem thêm trong Khám phá", on_click=quick_to_explore, args=(criteria, priority))
        else:
            st.info("Chưa có tin phù hợp. Thử mở rộng khu vực hoặc điều chỉnh ngân sách.")
    else:
        section("KHÁM PHÁ HÀ NỘI", "Xem tin theo khu vực và loại hình")
        st.button("Mở danh mục bất động sản", on_click=go, args=("Khám phá",))


def explore(frame: pd.DataFrame) -> None:
    preload_explore()
    section("KHÁM PHÁ", "Tìm bất động sản tại Hà Nội", "Chọn điều kiện cần có và ưu tiên của bạn.")
    criteria, priority, submitted = form_criteria(frame, "explore", compact=False)
    if submitted or "last_criteria" not in st.session_state:
        set_search(criteria, priority)
    criteria = st.session_state["last_criteria"]
    priority = st.session_state["last_priority"]
    order = order_control("explore_order")
    if order != st.session_state.get("last_explore_order"):
        st.session_state["last_explore_order"] = order
        st.session_state["results_page"] = 1
    rows = ranked(criteria, priority, order)
    if not rows:
        st.info("Chưa có tin phù hợp. Thử tăng ngân sách hoặc giảm diện tích tối thiểu.")
        return
    pages = (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE
    current = min(st.session_state.get("results_page", 1), pages)
    st.caption(f"Hiển thị {min(PAGE_SIZE, len(rows) - (current - 1) * PAGE_SIZE)} tin · Ưu tiên {PRIORITIES[priority].lower()}")
    grid(rows[(current - 1) * PAGE_SIZE:current * PAGE_SIZE], f"explore_{current}")
    if pages > 1:
        def page_move(page: int) -> None:
            st.session_state["results_page"] = page

        previous, status, following = st.columns([1, 2, 1])
        previous.button("Trang trước", on_click=page_move, args=(current - 1,), disabled=current <= 1)
        status.caption(f"Trang {current} / {pages} · Tối đa {RESULT_LIMIT} kết quả")
        following.button("Trang sau", on_click=page_move, args=(current + 1,), disabled=current >= pages)


def detail(row: dict) -> None:
    st.button("← Quay lại", on_click=close_detail)
    section("CHI TIẾT TIN", title_for(row), value(row.get("name"), ""))
    st.caption(GROUP_LABELS.get(row.get("property_group"), "Bất động sản"))
    columns = st.columns(3)
    columns[0].metric("Giá chào bán", money(row.get("price_vnd")))
    columns[1].metric("Diện tích theo dữ liệu", measurement(row.get("area_m2"), "m²"))
    columns[2].metric("Giá chào bán theo m²", money(row.get("price_per_m2_vnd"), unit_price=True))
    st.markdown(score_markup(row), unsafe_allow_html=True)
    st.button("Bỏ lưu tin" if row["listing_id"] in saved_items() else "Lưu để so sánh",
              on_click=toggle_save, args=(row,), type="primary")
    with st.container(border=True):
        st.markdown("#### Vì sao phù hợp với nhu cầu của bạn?")
        for reason in row.get("reasons", []):
            st.write("•", reason)
        col_a, col_b = st.columns(2)
        col_a.write(f"**Phòng ngủ** · {measurement(row.get('bedroom_count'), 'phòng')}")
        col_b.write(f"**Phòng tắm** · {measurement(row.get('bathroom_count'), 'phòng')}")
        col_a.write(f"**Đường** · {value(row.get('street_name'))}")
        col_b.write(f"**Phường / xã** · {value(row.get('ward_name'))}")
    with st.container(border=True):
        st.markdown("#### Hạ tầng và quy hoạch quanh khu vực")
        st.write(infrastructure_reason(row))
        for fact in assessment(row).get("facts", []):
            st.markdown(f"**{escape(value(fact.get('title')))}**")
            st.caption(f"{value(fact.get('status'))} · {value(fact.get('published_at'))}")
            st.write(value(fact.get("description")))
            if fact.get("source_url"):
                st.markdown(f"[Xem nguồn · {escape(value(fact.get('source_name')))}]({fact['source_url']})")
            if fact.get("scope_url"):
                st.markdown(f"[Xem hồ sơ phạm vi dự án]({fact['scope_url']})")
        st.caption("Điểm hạ tầng dựa trên hồ sơ khớp khu vực và ưu tiên khi tìm. Vị trí căn và phạm vi ảnh hưởng cần đối chiếu trên bản đồ quy hoạch chi tiết.")
    with st.expander("Thông tin tin rao"):
        st.write("Ngày đăng", value(value(row.get("published_at"), "")[:10]))
        st.write("Nguồn bộ dữ liệu", value(row.get("source")))
        count = row.get("related_listing_count", 1)
        if count > 1:
            st.caption(f"Đã gộp {count} tin đăng tương tự và hiển thị bản có thông tin đầy đủ hơn.")


def saved_page() -> None:
    section("ĐÃ LƯU", "Danh sách bạn đang cân nhắc", "Chọn tối đa bốn tin để so sánh.")
    saved = saved_items()
    if not saved:
        st.info("Bạn chưa lưu tin nào. Hãy mở Khám phá và chọn Lưu tin.")
        st.button("Khám phá bất động sản", on_click=go, args=("Khám phá",))
        return
    selected = st.multiselect("Tin để so sánh", list(saved), default=list(saved)[:2],
                              max_selections=4, format_func=lambda k: title_for(saved[k]))
    if selected:
        table = {}
        for i, key in enumerate(selected, 1):
            row = saved[key]
            infra = assessment(row).get("score")
            table[f"Tin {i}"] = {
                "Bất động sản": title_for(row), "Khu vực": value(row.get("district_name")),
                "Giá chào bán": money(row.get("price_vnd")), "Diện tích": measurement(row.get("area_m2"), "m²"),
                "Phòng ngủ": measurement(row.get("bedroom_count"), "phòng"),
                "Phù hợp nhu cầu": f"{vi_number(row['score'])}/100",
                "Phù hợp hạ tầng": f"{vi_number(infra)}/100" if infra is not None else "Chưa đủ dữ liệu",
            }
        st.dataframe(pd.DataFrame(table), use_container_width=True)
        st.caption("Các điểm phản ánh ưu tiên tại thời điểm lưu từng tin.")
    grid(list(saved.values()), "saved")


@st.cache_resource(show_spinner=False)
def model_snapshot(path: str, report_stamp: int, artifact_stamp: int) -> dict:
    return load_artifacts(Path(path))


def estimate(frame: pd.DataFrame) -> None:
    section("THAM KHẢO GIÁ", "Ước tính giá chào bán", "Nhập các đặc điểm bạn biết để có một mức giá tham khảo.")
    try:
        bundle = model_snapshot(str(MODEL_DIR), stamp(MODEL_DIR / "evaluation.json"), stamp(MODEL_DIR / "price_models.joblib"))
    except (OSError, ValueError, RuntimeError, ImportError, EOFError):
        st.info("Tính năng ước tính giá hiện chưa sẵn sàng. Bạn vẫn có thể xem giá tin rao trong Khám phá.")
        return
    groups = [key for key in GROUP_LABELS if key in bundle.get("models", {})]
    if not groups:
        st.info("Chưa có mức giá tham khảo cho các loại bất động sản hiện tại.")
        return
    group = st.selectbox("Loại bất động sản", groups, format_func=GROUP_LABELS.get)
    pool = frame[frame["property_group"].eq(group)]
    attrs = {}
    with st.form(f"consumer_estimate_{group}"):
        columns = st.columns(3)
        for i, field in enumerate(FEATURES[group]):
            column = columns[i % 3]
            label = FEATURE_LABELS[field]
            if field in {"district_name", "ward_name", "project_name", "house_direction", "balcony_direction"}:
                attrs[field] = column.selectbox(label, [None, *options(pool[field])], format_func=lambda x: x or "Chưa rõ")
            elif field.endswith("_count"):
                attrs[field] = column.number_input(label, min_value=0, value=None, step=1)
            else:
                attrs[field] = column.number_input(label, min_value=0.1, value=75.0 if field == "area_m2" else None)
        submitted = st.form_submit_button("Xem giá tham khảo", type="primary")
    if submitted:
        try:
            result = predict_listing(bundle, group, attrs)
        except (ValueError, TypeError, KeyError):
            st.warning("Kiểm tra lại diện tích và các thông tin đã nhập.")
            return
        col_a, col_b = st.columns(2)
        col_a.metric("Giá chào bán ước tính", money(result["predicted_price_vnd"]))
        col_b.metric("Đơn giá ước tính", money(result["predicted_price_per_m2_vnd"], unit_price=True))
        interval = result.get("interval_vnd")
        if interval:
            st.caption(f"Khoảng tham khảo · {money(interval[0])} đến {money(interval[1])}")
        if result.get("warnings"):
            st.warning("Mức giá tham khảo có độ chắc chắn hạn chế với loại hình hoặc thông tin bạn đã nhập.")
    st.caption("Ước tính dựa trên giá chào bán trong dữ liệu tin rao năm 2025.")


def main() -> None:
    st.set_page_config(page_title="Hanoi Real Estate Advisor", page_icon="🏙️", layout="wide")
    apply_visuals()
    st.markdown('<div class="estate-brand"><span>HANOI</span> REAL ESTATE ADVISOR</div>', unsafe_allow_html=True)
    page = st.radio("Điều hướng", TABS, horizontal=True, key="navigation", label_visibility="collapsed",
                    on_change=close_detail)
    try:
        frame = load_listings(DATA_PATH)
    except (OSError, ValueError):
        st.info("Danh mục bất động sản hiện chưa sẵn sàng. Vui lòng thử lại sau.")
        return
    if "open_listing" in st.session_state:
        detail(st.session_state["open_listing"])
    elif page == "Trang chủ":
        homepage(frame)
    elif page == "Khám phá":
        explore(frame)
    elif page == "Ước tính giá":
        estimate(frame)
    else:
        saved_page()
    footer = "Hanoi Real Estate Advisor · Giá chào bán từ dữ liệu tin rao năm 2025"
    if (ASSETS / "hanoi-hero.jpg").is_file():
        footer += (
            '<br><a href="https://commons.wikimedia.org/wiki/File:Sunset_over_Hanoi_After_the_Rain.jpg">'
            'Ảnh Hà Nội · Christopher Crouzet</a> · '
            '<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>'
        )
    st.markdown(f'<footer class="estate-footer">{footer}</footer>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
