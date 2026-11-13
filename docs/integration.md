# Kết nối các module

TV1 xuất `data/processed/hanoi_properties_final.csv`. TV4 dùng `advisor.data.load_listings(path)` để kiểm tra và trả về DataFrame. Các trường số dùng VND, m² hoặc m, trường thiếu nạp thành giá trị thiếu của pandas. Chuẩn hóa so khớp địa danh bằng `district_key` khi cần. Lỗi tệp thiếu, thiếu cột, sai số, nhóm ngoài phạm vi và mã trùng được báo rõ.

TV2 gọi `scripts/train_models.py` và đặt kết quả ở `models/`. `load_artifacts` đọc bản mô hình tại chỗ. `predict_listing` nhận nhóm bất động sản cùng dữ liệu đầu vào, trả giá tổng, giá/m², khoảng sai số tham khảo và giải thích. Báo cáo đánh giá và sai số phân nhóm nằm cạnh tệp mô hình.

TV3 gọi `search_listings` với bảng dữ liệu và `SearchCriteria`. Kết quả gồm dữ liệu tin, điểm, các thành phần điểm và lý do. Dùng `compare_listings` để truy các tin đã chọn. `solve_csp` dùng một miền giới hạn từ tập ứng viên, trả phương án và nhật ký chọn thử, loại và quay lui.

`development_insights(district)` trả facts đã khớp địa danh từ kho có nguồn. Danh sách rỗng có nghĩa kho hiện chưa có bản ghi phù hợp. Mỗi kết quả dùng ngày và mức xác minh của nguồn, chỉ áp dụng cho phạm vi được nguồn nêu.

TV5 chạy `streamlit run app.py` để mở giao diện người mua. `advisor.infrastructure.rank_listings` áp dụng các điều kiện tìm kiếm trước khi sắp xếp theo điểm nhu cầu hoặc điểm hạ tầng. Điểm hạ tầng lấy hồ sơ khớp khu vực phù hợp nhất với ưu tiên đang chọn, giữ nguyên điểm nhu cầu và xếp tin chưa đủ hồ sơ xuống cuối khi chọn thứ tự hạ tầng.

Khi thuyết trình, chạy `streamlit run course_demo.py --server.port 8502` để xem Dashboard, đánh giá mô hình và thử CSP. Khi demo giao diện người mua, thử tư vấn nhanh, đổi thứ tự, xem nguồn ở trang chi tiết và lưu các tin để so sánh.
