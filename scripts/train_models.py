"""Train models for Hanoi asking prices and export chronological evaluation reports."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advisor.modeling import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, nargs="?", default=ROOT / "data/processed/hanoi_properties_final.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--random-forest", action="store_true", help="Huấn luyện thêm Random Forest.")
    args = parser.parse_args()
    try:
        report = train_models(args.input_csv, args.output_dir, include_random_forest=args.random_forest)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Lỗi huấn luyện: {error}\n")
    for group, result in report["groups"].items():
        chosen = result["selected_model"]
        print(f"{group}: {chosen}, MAE kiểm tra {result['test_metrics'][chosen]['mae_vnd']:,.0f} VND")
    for group, reason in report["skipped_groups"].items():
        print(f"Bỏ qua {group}: {reason}")
    print(f"Đã lưu mô hình và báo cáo tại {args.output_dir}")


if __name__ == "__main__":
    main()
