# Hanoi Real Estate Advisor

Ứng dụng tư vấn bất động sản Hà Nội. Trang chủ có form tư vấn nhanh. Trang Khám phá cho phép lọc, đổi cách sắp xếp, mở chi tiết tin và lưu để so sánh trong phiên. Trang Ước tính giá cung cấp giá chào bán tham khảo. Dữ liệu gồm căn hộ chung cư, nhà tập thể, nhà, biệt thự hoặc liền kề và shophouse.

Kết quả hiển thị hai điểm riêng. **Phù hợp nhu cầu** phản ánh ngân sách và diện tích. **Phù hợp hạ tầng** dùng hồ sơ có nguồn của khu vực, ưu tiên mốc khai thác khi người dùng quan tâm đi lại và ưu tiên hồ sơ phát triển khi người dùng muốn theo dõi quy hoạch. Khu vực thiếu hồ sơ hiển thị “Chưa đủ dữ liệu”. Hai cách sắp xếp đều áp dụng sau khi lọc theo điều kiện bắt buộc.

Trang gợi ý giữ ngưỡng rà soát riêng cho bản demo. Các tin có giá quy đổi dưới 15 triệu đồng/m² được giữ trong CSV nhưng không đưa vào danh sách gợi ý, vì có thể bị nhập nhầm đơn giá vào cột giá tổng. Ngưỡng này phục vụ sàng lọc dữ liệu nguồn, không dùng để định giá.

Trước khi xếp hạng, web gom các tin nghi đăng lại khi cùng loại hình, khu vực, giá và diện tích, tiêu đề đủ giống và các thông tin có ở cả hai tin nhất quán. Mỗi nhóm dùng một bản ghi đại diện, ưu tiên tin có số phòng ngủ, rồi xét độ đầy đủ và ngày đăng. Trang chi tiết cho biết số tin đã được gom. Các bản ghi nguồn vẫn có thể đối chiếu trong CSV.

## Chạy trên Windows

Mở Terminal trong thư mục có `app.py`. Cần Python 3.11 trở lên. Cài các thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sau khi clone, tải `vietnam_real_estate.csv` từ [Kaggle](https://www.kaggle.com/datasets/yashdev01/vietnam-real-estates), đặt vào `data/raw/` và chạy các lệnh sau để tạo CSV đã xử lý, báo cáo và mô hình. Nếu đã có CSV đã xử lý và mô hình tương thích, có thể chuyển thẳng sang bước mở ứng dụng.

```powershell
.\.venv\Scripts\python.exe scripts\build_hanoi_dataset.py data\raw\vietnam_real_estate.csv
.\.venv\Scripts\python.exe scripts\profile_hanoi_dataset.py data\processed\hanoi_properties_final.csv
.\.venv\Scripts\python.exe scripts\train_models.py --random-forest
```

Mở ứng dụng và chạy kiểm thử bằng các lệnh sau (dùng Terminal riêng cho mỗi lệnh):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest -q
```

Khi môi trường dùng phiên bản scikit-learn khác bản huấn luyện, chạy lại lệnh train trong môi trường đó. Chỉ dùng tệp mô hình do nhóm tạo từ pipeline này.

Để thuyết trình đồ án, mở giao diện riêng có biểu đồ dữ liệu, đánh giá mô hình và minh họa thuật toán CSP.

```powershell
.\.venv\Scripts\python.exe -m streamlit run course_demo.py --server.port 8502
```

## Bàn giao theo phân công

| Thành viên | Tệp chính | Kết quả |
|---|---|---|
| TV1 · Dữ liệu | `scripts/build_hanoi_dataset.py`, `scripts/profile_hanoi_dataset.py`, `docs/data_dictionary.md`, `docs/sources.md` | CSV UTF-8 BOM dấu phẩy, báo cáo số dòng và lý do lọc, thống kê độ phủ và phân bố theo loại hình. |
| TV2 · Mô hình giá | `advisor/modeling.py`, `scripts/train_models.py` | So sánh hồi quy tuyến tính, cây quyết định, Random Forest khi bật, và baseline trung vị; MAE, RMSE, R² trên tập kiểm tra theo thời gian. |
| TV3 · Search/CSP | `advisor/search.py`, `advisor/knowledge.py`, `data/knowledge/development_facts.json` | Lọc bắt buộc, xếp hạng có giải thích, so sánh, quay lui với nhật ký và suy luận IF–THEN ở cấp khu vực. |
| TV4 · Tích hợp | `advisor/data.py`, `docs/integration.md` | Hàm nạp CSV dùng chung, kiểm tra schema, cache theo lần cập nhật tệp. |
| TV5 · Giao diện | `app.py`, `course_demo.py` | Giao diện người mua nhà và bản trình bày kỹ thuật riêng cho nhóm. |

## Kiểm tra dữ liệu trước khi bàn giao

- `data/processed/hanoi_properties_final.csv` chứa các cột tương tự bản Hai Bà Trưng và các đặc trưng bổ sung. Khi mở bằng Excel, dùng **Data → From Text/CSV**, chọn **65001 UTF-8** và dấu phẩy. Giữ ô trống khi nguồn thiếu dữ liệu.
- `quality_report.json` ghi số dòng và lý do lọc. `eda_summary.json` ghi độ phủ đặc trưng, số mẫu theo tháng và phân bố giá, diện tích theo loại hình.
- Các cờ trong CSV cho phép rà soát giá hoặc diện tích bất thường. Bộ lọc phân biệt bán/thuê theo tín hiệu văn bản, vì nguồn không có cột mục đích giao dịch.
- Dữ liệu ghi nhận giai đoạn 2025 trong nguồn. Các chỉ số mô hình đo sai số đối với tin đăng của tập kiểm tra; kết quả không phản ánh giá giao dịch hay mức giá thị trường tại ngày chạy.
- Dữ liệu tin đăng không có tọa độ. Phần hạ tầng ghép theo nhãn khu vực và hiển thị nguồn, thời điểm và quy tắc áp dụng. Muốn tính đường đi, kiểm tra ranh quy hoạch hoặc xác định giải tỏa theo căn cần thêm dữ liệu vị trí và hồ sơ chính thức.

Xem thêm [nguồn và phạm vi](docs/sources.md), [từ điển dữ liệu](docs/data_dictionary.md) và [tích hợp](docs/integration.md).
