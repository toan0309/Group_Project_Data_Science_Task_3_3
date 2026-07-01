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
Máy tính nhận dạng hình ảnh dưới dạng ma trận điểm ảnh. Để mô hình nhận diện được điểm khác biệt giữa các tờ tiền, chúng ta sử dụng thuật toán HOG.

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

## 📖 Giải thích luồng thực thi của Mã nguồn

Mã nguồn trong `Main_CurrencyClassification.ipynb` được tổ chức thành 6 bước logic chính:

*   **Ô 1: Khởi tạo.** Import các thư viện cần thiết và thiết lập các biến cấu hình cơ bản (tên lớp, kích thước ảnh chuẩn).
*   **Ô 2: Xử lý ảnh và trích xuất HOG.** 
    *   Chương trình duyệt qua các thư mục con trong `images/` để đọc toàn bộ dữ liệu.
    *   Chuẩn hóa tất cả các ảnh về kích thước đồng nhất (96x96 pixel).
    *   Áp dụng thuật toán HOG để chuyển đổi mỗi bức ảnh thành một vector đặc trưng.
*   **Ô 3: Trực quan hóa đặc trưng.** Hiển thị hình ảnh gốc song song với hình ảnh hiển thị hướng gradient của HOG để đánh giá tính hiệu quả của việc trích xuất.
*   **Ô 4: Huấn luyện mô hình SVM.**
    *   Tập dữ liệu được chia làm 2 phần: 80% dùng để huấn luyện (train) và 20% dùng để kiểm thử (test).
    *   Sử dụng hàm `svm.fit()` để mô hình học cách phân loại trên tập huấn luyện.
*   **Ô 5: Đánh giá bằng Ma trận nhầm lẫn (Confusion Matrix).** Sử dụng kết quả trên tập kiểm thử để vẽ ma trận, nhằm đánh giá chi tiết tỷ lệ dự đoán đúng/sai cho từng cặp nhãn.
*   **Ô 6: Dự đoán thực tế.** Trực quan hóa kết quả phân loại trên tập dữ liệu gốc, hiển thị xác suất dự đoán (%) của mô hình đối với mỗi hình ảnh.