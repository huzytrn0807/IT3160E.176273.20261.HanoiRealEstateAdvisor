"""District-level infrastructure facts and explicit inference rules."""
from functools import lru_cache
import json

from .data import ROOT, district_key

FACTS_PATH = ROOT / "data" / "knowledge" / "development_facts.json"
RULES = {
    "operational_at_source_date": (
        "IF khu vực được nguồn nêu tên AND hạ tầng đã khai thác tại thời điểm nguồn THEN ghi nhận hạ tầng đã vận hành.",
        "Có cơ sở để kiểm tra kết nối giao thông ở khu vực này. Cần địa chỉ hoặc tọa độ để tính khoảng cách của từng căn.",
    ),
    "works_reported_at_source_date": (
        "IF khu vực được nguồn nêu tên AND nguồn có ghi nhận thi công THEN ghi nhận mốc thi công theo ngày nguồn.",
        "Thông tin áp dụng ở cấp khu vực. Chưa đủ dữ liệu để xác định căn cụ thể thuộc phạm vi thu hồi đất.",
    ),
    "alignment_announced": (
        "IF khu vực được nguồn nêu tên AND nguồn công bố phương án tuyến THEN hiển thị mốc công bố phương án.",
        "Công bố phương án là một mốc hồ sơ. Cần văn bản và bản đồ chi tiết để đối chiếu ảnh hưởng đến một địa chỉ.",
    ),
}


@lru_cache(maxsize=1)
def load_facts() -> list[dict]:
    """Read the small, manually maintained knowledge base."""
    return json.loads(FACTS_PATH.read_text(encoding="utf-8"))


def development_insights(district: str) -> list[dict]:
    key = district_key(district) if district else ""
    if not key:
        return []
    matches = []
    for fact in load_facts():
        if key not in {district_key(name) for name in fact["districts"]}:
            continue
        rule, explanation = RULES[fact["status_code"]]
        matches.append({**fact, "matched_rule": rule, "explanation": explanation})
    return sorted(matches, key=lambda item: item["published_at"], reverse=True)
