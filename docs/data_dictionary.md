# Từ điển dữ liệu tin rao bán Hà Nội

Một dòng biểu diễn một tin có tín hiệu rao bán và có giá, diện tích hợp lệ về mặt định dạng. Địa danh là nhãn tại nguồn tin, có thể dùng địa giới cũ. Ô trống biểu diễn thiếu dữ liệu.

| Cột | Kiểu và đơn vị | Có thể trống | Ý nghĩa và quy tắc |
|---|---|---|---|
| `listing_id` | Chuỗi SHA-256 | Không | Dấu vân tay của các đặc điểm và nội dung tin đã chuẩn hóa. Không phải mã căn hoặc mã giao dịch. |
| `name` | Chuỗi UTF-8 | Có | Tiêu đề tin đã chuẩn hóa khoảng trắng và che các mẫu thông tin liên hệ nhận diện được. Có thể chứa giá trong văn bản. Khi trường văn bản nguồn mở đầu bằng `=`, `+`, `-` hoặc `@`, CSV thêm dấu nháy đơn để Excel đọc như văn bản. |
| `property_type_name` | Chuỗi | Không | Nhãn loại hình trong CSV nguồn. |
| `property_group` | Nhãn | Không | `apartment`, `collective_housing` (nhà tập thể), `house`, `villa_townhouse` hoặc `shophouse`. Với tin chung cư hoặc nhà, quy tắc đổi nhóm khi tiêu đề xác định đối tượng bán là nhà tập thể. |
| `province_name` | Chuỗi | Không | Chuẩn hóa về Hà Nội từ nhãn tỉnh/thành phố nguồn. |
| `district_name` | Chuỗi | Có | Nhãn quận, huyện hoặc thị xã tại nguồn. Ứng dụng chuẩn hóa cách so khớp; giữ nguyên địa danh nguồn để đối chiếu. |
| `ward_name` | Chuỗi | Có | Phường, xã tại nguồn, không nội suy từ địa chỉ. |
| `street_name` | Chuỗi | Có | Tên đường tại nguồn. |
| `project_name` | Chuỗi | Có | Dự án được gắn trong tin. |
| `price_vnd` | Số nguyên dương, VND | Không | Tổng giá chào bán trong trường `price`. Đây là biến mục tiêu của mô hình. |
| `area_m2` | Số thực dương, m² | Không | Diện tích trong trường `area`; nguồn không tách nhất quán diện tích thông thủy và diện tích sử dụng. |
| `price_per_m2_vnd` | Số thực, VND/m² | Không | Tổng giá chia diện tích, làm tròn 2 chữ số. Dùng phân tích và hiển thị. |
| `floor_count` | Số nguyên dương | Có | Số tầng do nguồn ghi; ý nghĩa với căn hộ có thể khác số tầng của một nhà xây. |
| `bedroom_count` | Số nguyên không âm | Có | Số phòng ngủ. Giá trị 0 được giữ nếu nguồn ghi 0; ô trống biểu diễn chưa biết. |
| `bathroom_count` | Số nguyên không âm | Có | Số phòng tắm theo nguồn. |
| `frontage_m`, `depth_m`, `road_width_m` | Số thực dương, m | Có | Mặt tiền, chiều sâu, độ rộng đường theo nguồn. |
| `house_direction`, `balcony_direction` | Chuỗi | Có | Hướng nhà và hướng ban công theo nguồn. |
| `published_at` | Chuỗi thời gian ISO 8601 | Có | Ngày đăng từ nguồn. Giữ múi giờ nếu nguồn có ghi; không tự gán nếu thiếu. |
| `price_type` | Nhãn | Không | `asking`, giá chào trên tin. |
| `listing_purpose` | Nhãn | Không | `sale` trên các dòng qua bộ lọc. |
| `purpose_evidence` | Nhãn | Không | `title_sale_signal` hoặc `description_sale_signal` cho biết nơi tìm thấy dấu hiệu rao bán. |
| `source`, `source_url`, `data_origin` | Chuỗi | Không | Nguồn dữ liệu và liên kết trang bộ dữ liệu; không phải URL của từng tin. |
| `dac_diem_noi_bat` | Chuỗi | Có | Nhãn nhận diện từ tiêu đề, ví dụ có nhắc chỗ đỗ ô tô. Đây là nội dung tin tự mô tả, chưa được đối chiếu. |
| `quality_flags` | Chuỗi phân cách bằng `|` | Có | Tín hiệu cần xem lại như giá/m² hoặc diện tích ngoài ngưỡng, thiếu địa danh/ngày đăng, hoặc chỉ có tín hiệu bán từ mô tả. |
| `source_record` | Số nguyên dương | Không | Số thứ tự bản ghi CSV đầu vào tính từ 1 sau tiêu đề, kể cả khi nội dung trải nhiều dòng văn bản. |

Mô hình nhận danh sách đặc trưng có sẵn khi ước tính, như diện tích, địa danh, dự án và phòng ngủ. Loại `price_per_m2_vnd`, `name`, mã tin, thông tin nguồn và cột kiểm tra khỏi đầu vào để giảm rò rỉ giá hoặc học nhầm mã định danh. Dùng `published_at` để chia dữ liệu theo thời gian.
