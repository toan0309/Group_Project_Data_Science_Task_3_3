# Báo Cáo Đồ Án: Phân Loại Tiền Tệ Bằng Machine Learning

**Nhóm thực hiện:** Nhóm sinh viên năm 3 - Môn Khoa học Dữ liệu

Mục tiêu của đồ án: Xây dựng một mô hình Machine Learning có khả năng tự động nhận diện và phân loại tiền tệ của 5 quốc gia từ hình ảnh đầu vào.

Danh sách 5 loại tiền tệ cần phân loại:
- 🇨🇦 Canada
- 🇯🇵 Nhật Bản (Japan)
- 🇰🇷 Hàn Quốc (Korea)
- 🇺🇸 Hoa Kỳ (USA)
- 🇻🇳 Việt Nam (Vietnam)

---

## 🛠️ Bước 1: Cài đặt môi trường

Để chạy mã nguồn, môi trường cần được cài đặt các thư viện cơ bản phục vụ cho xử lý ảnh và Machine Learning. Mở Command Prompt (CMD) hoặc Terminal và chạy lệnh sau:

```bash
pip install scikit-learn scikit-image numpy matplotlib seaborn
```

*Mục đích của các thư viện:*
- `numpy`: Xử lý tính toán ma trận và mảng dữ liệu.
- `matplotlib` & `seaborn`: Trực quan hóa dữ liệu và vẽ biểu đồ đánh giá.
- `scikit-image`: Đọc, thay đổi kích thước và trích xuất đặc trưng hình ảnh.
- `scikit-learn`: Cung cấp các thuật toán Machine Learning (SVM) và công cụ đánh giá mô hình.

---

## 📂 Bước 2: Cấu trúc thư mục

Để dữ liệu được đọc chính xác, tập tin mã nguồn và tập dữ liệu hình ảnh được sắp xếp theo cấu trúc sau:

```text
Group_Project_Data_Science_Task_3_3/
│
├── Main_CurrencyClassification.ipynb   <-- File mã nguồn chính
├── README.md                           <-- File tài liệu mô tả (hiện hành)
│
└── images/                             <-- Thư mục chứa tập dữ liệu hình ảnh
    ├── Canada/                         <-- Dữ liệu ảnh tiền Canada
    ├── Japan/                          <-- Dữ liệu ảnh tiền Nhật Bản
    ├── Korea/                          <-- Dữ liệu ảnh tiền Hàn Quốc
    ├── USA/                            <-- Dữ liệu ảnh tiền Hoa Kỳ
    └── Vietnam/                        <-- Dữ liệu ảnh tiền Việt Nam
```

---

## 🚀 Bước 3: Hướng dẫn thực thi

1. Khởi động môi trường **Jupyter Notebook**.
2. Mở tệp tin `Main_CurrencyClassification.ipynb`.
3. Nhấn **Run** (hoặc tổ hợp phím `Shift + Enter`) để thực thi mã từ ô (cell) trên cùng xuống dưới cùng theo tuần tự.

---

## 🧠 Giải thích thuật toán (Mô tả trực quan)

Đồ án sử dụng phương pháp trích xuất đặc trưng **HOG** kết hợp với bộ phân lớp **SVM**. 

### 1. HOG (Histogram of Oriented Gradients) - Trích xuất đặc trưng hình học
Máy tính nhận dạng hình ảnh dưới dạng ma trận điểm ảnh. Để nhận diện được các điểm khác biệt cốt lõi giữa các tờ tiền, thuật toán HOG được áp dụng.

Có thể hình dung HOG như một công cụ chuyên ghi nhận **đường viền, góc cạnh, và hoa văn** của tờ tiền:
- Thuật toán chia hình ảnh ra thành nhiều ô vuông nhỏ.
- Tại mỗi ô, nó tính toán hướng phân bố của các đường nét (đường sọc ngang, dọc, chéo, v.v.).
- Tập hợp kết quả từ tất cả các ô này tạo thành một danh sách các con số (được gọi là *vector đặc trưng*).
👉 **Kết quả:** Quá trình này giúp thu gọn thông tin hình ảnh phức tạp thành một danh sách các đường nét đặc trưng cốt lõi để đưa vào mô hình học.

### 2. SVM (Support Vector Machine) - Mô hình phân lớp
Sau khi HOG tạo ra các vector đặc trưng, dữ liệu này được đưa vào mô hình **SVM** để học cách phân loại.

Có thể hình dung bài toán này như việc chia các loại trái cây (tượng trưng cho các loại tiền) trên một mặt bàn:
Nhiệm vụ của **SVM** là tìm ra một **đường ranh giới** tối ưu nhất để tách biệt hoàn toàn các loại trái cây này với nhau.
- Khi có một điểm dữ liệu mới (một bức ảnh tiền chưa từng thấy), SVM sẽ kiểm tra xem điểm đó nằm ở khu vực nào so với đường ranh giới để đưa ra kết quả phân loại.
- Đối với dữ liệu phức tạp không thể phân tách bằng đường thẳng, mô hình sử dụng kỹ thuật *Kernel RBF* để ánh xạ dữ liệu lên không gian nhiều chiều hơn, từ đó dễ dàng tìm ra mặt phẳng phân cách.

---

## 📖 Giải thích chi tiết luồng thực thi của Mã nguồn

Mã nguồn trong tệp `Main_CurrencyClassification.ipynb` được tổ chức thành 6 khối xử lý logic chính:

### Khối 1 (Ô 1): Khai báo và Thiết lập ban đầu
- **Nhập thư viện:** Nạp các module cần thiết từ thư viện `scikit-learn` (cho mô hình SVM, chuẩn hóa dữ liệu, chia tập huấn luyện) và `scikit-image` (cho thuật toán HOG, xử lý ảnh).
- **Cấu hình tham số:** Định nghĩa danh sách các nhãn phân loại (`CLASS_NAMES` bao gồm 5 quốc gia) và thiết lập kích thước chuẩn (`IMG_SIZE = 96x96`) để đảm bảo tính đồng nhất cho toàn bộ hình ảnh đầu vào.

### Khối 2 (Ô 2): Tiền xử lý dữ liệu và Trích xuất HOG
- **Tiền xử lý:** Duyệt qua các thư mục chứa hình ảnh đầu vào. Mỗi hình ảnh khi được đọc sẽ trải qua các bước: chuyển đổi kênh màu (đảm bảo ảnh RGB hoặc ảnh xám), chuẩn hóa giá trị pixel về dải [0, 1] và thu phóng (resize) ép buộc về đúng kích thước 96x96.
- **Trích xuất đặc trưng:** Hình ảnh sau khi chuẩn hóa được đưa qua hàm phân tích của HOG. Hàm này trả về một vector các giá trị số nguyên đại diện cho thông tin hình học, cạnh và đường nét của tờ tiền, hoàn toàn độc lập với màu sắc. Toàn bộ các vector này được lưu trữ thành tập dữ liệu `X_raw`, và nhãn tương ứng được lưu ở `y_raw`.

### Khối 3 (Ô 3): Trực quan hóa kết quả trích xuất đặc trưng
- Mục đích của khối này là kiểm chứng mức độ hiệu quả của thuật toán HOG. Bằng cách vẽ trực tiếp hình ảnh gốc đặt cạnh một bản đồ hiển thị hướng gradient của HOG, người nghiên cứu có thể quan sát trực quan những đặc điểm nào của tờ tiền (chữ số, đường gân, quốc huy) đã được thuật toán nhận diện và thu giữ lại.

### Khối 4 (Ô 4): Chuẩn hóa và Huấn luyện mô hình phân lớp SVM
- **Chuẩn hóa thang đo (Standardization):** Do các giá trị trong vector HOG có thể dao động trong những biên độ khác nhau, phương pháp `StandardScaler` được áp dụng để căn chỉnh lại tất cả các giá trị về cùng một thang đo. Bước này giúp hạn chế tình trạng sai số tỷ lệ và đảm bảo mô hình SVM hội tụ nhanh, chính xác hơn.
- **Chia tập dữ liệu:** Sử dụng kỹ thuật `train_test_split` để phân tách tập dữ liệu thành 2 phần: 80% dành cho việc huấn luyện mô hình và 20% được tách biệt hoàn toàn để sử dụng cho mục đích kiểm thử khách quan sau này. Tham số `stratify` được khai báo nhằm duy trì tỷ lệ các loại tiền tệ phân bố đồng đều giữa 2 tập.
- **Huấn luyện (Training):** Khởi tạo bộ phân lớp SVC (Support Vector Classifier) với cấu hình kernel RBF. Hàm `fit()` được gọi để tiến hành quá trình học thuật toán dựa trên dữ liệu huấn luyện. Kết thúc khối này, một bản báo cáo mức độ chính xác (Accuracy) tổng quan trên tập kiểm thử được in ra.

### Khối 5 (Ô 5): Đánh giá chi tiết bằng Ma trận nhầm lẫn (Confusion Matrix)
- **Mục đích:** Chỉ số độ chính xác (Accuracy) tổng quan không phản ánh được việc mô hình đang gặp khó khăn ở loại tiền tệ cụ thể nào. Ma trận nhầm lẫn (Confusion Matrix) được sử dụng để giải quyết giới hạn này.
- **Cơ cấu phân tích:** Trục dọc của ma trận biểu thị nhãn thực tế (True Label) của hình ảnh, trục ngang biểu thị nhãn do mô hình dự đoán (Predicted Label).
- **Quy tắc phân tích:**
  - Đường chéo chính của ma trận (từ góc trên cùng bên trái xuống góc dưới cùng bên phải) tập trung số lượng hình ảnh được mô hình dự đoán **chính xác** hoàn toàn.
  - Các ô nằm ngoài đường chéo chính biểu thị số lượng hình ảnh bị **dự đoán sai** (nhầm lẫn loại tiền này sang loại tiền khác). 
  - **Phân tích chiều sâu:** Ví dụ, nếu ô giao cắt giữa "Hàn Quốc" (Thực tế) và "Nhật Bản" (Dự đoán) có con số cao bất thường, điều này khẳng định mô hình đang không tìm thấy các đặc điểm khác biệt giữa hai loại tiền tệ này. Phát hiện này là tiền đề cốt lõi để đưa ra các biện pháp nâng cấp hệ thống (thu thập thêm dữ liệu hoặc tinh chỉnh lại tham số lưới HOG).

### Khối 6 (Ô 6): Dự đoán trên dữ liệu thực tế và Tổng kết
- Mô hình đã được huấn luyện hoàn chỉnh được tái sử dụng để dự đoán lại toàn bộ hình ảnh trong cơ sở dữ liệu.
- Thay vì chỉ xuất ra kết quả phân loại cuối cùng, hàm `predict_proba()` được gọi để tính toán mức độ tự tin (xác suất phần trăm) của mô hình đối với quyết định đó.
- Kết quả được minh họa trực quan bằng các biểu đồ cột biểu thị xác suất của 5 quốc gia đi kèm cảnh báo `[OK]` (đúng) hoặc `[WRONG]` (sai). Các thông số tổng quát về độ tin cậy được tính toán ở cuối tiến trình.