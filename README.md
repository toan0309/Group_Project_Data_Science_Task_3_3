# 💵 Nhận Diện Tiền Tệ Dành Cho Người Mới Bắt Đầu (Newbie-friendly)

Chào bạn! Đây là hướng dẫn cực kỳ chi tiết và đơn giản cho đồ án **"Phân loại tiền tệ các nước bằng Machine Learning"**. Đồ án này được thiết kế sao cho dù bạn mới học code hay mới tiếp xúc với Khoa học Dữ liệu cũng có thể hiểu và chạy được.

Mục tiêu của chúng ta là: **Đưa cho máy tính một bức ảnh tờ tiền, máy tính sẽ đoán được đó là tiền của nước nào.**

Chúng ta sẽ phân loại 5 loại tiền:
- 🇨🇦 Canada
- 🇯🇵 Nhật Bản (Japan)
- 🇰🇷 Hàn Quốc (Korea)
- 🇺🇸 Hoa Kỳ (USA)
- 🇻🇳 Việt Nam (Vietnam)

---

## 🛠️ Bước 1: Chuẩn bị "đồ nghề" (Cài đặt thư viện)

Để code chạy được, máy tính của bạn cần được "dạy" một số kỹ năng cơ bản (thông qua các thư viện). Bạn hãy làm theo các bước sau:

1. Mở **Command Prompt (CMD)** hoặc **Terminal**.
2. Copy và dán dòng lệnh này vào rồi ấn Enter:

```bash
pip install scikit-learn scikit-image numpy matplotlib seaborn
```

*Giải thích nhanh các "đồ nghề":*
- `numpy`: Máy tính siêu phàm, chuyên tính toán các con số.
- `matplotlib` & `seaborn`: Bút dạ màu, chuyên dùng để vẽ biểu đồ cho đẹp.
- `scikit-image`: Kính lúp, giúp đọc và xử lý hình ảnh.
- `scikit-learn`: Bộ não AI, chứa các thuật toán học máy (Machine Learning).

---

## 📂 Bước 2: Bố trí thư mục

Bạn cần sắp xếp các file y hệt như cấu trúc dưới đây để code có thể tìm thấy ảnh:

```text
Group_Project_Data_Science_Task_3_3/
│
├── Main_CurrencyClassification.ipynb   <-- File chứa code chính (bạn sẽ chạy file này)
├── README.md                           <-- Là file bạn đang đọc đây
│
└── images/                             <-- Thư mục chứa toàn bộ ảnh tiền
    ├── Canada/                         <-- Bạn bỏ ảnh tiền Canada vào đây
    ├── Japan/                          <-- Bạn bỏ ảnh tiền Nhật Bản vào đây
    ├── Korea/                          <-- ... tương tự
    ├── USA/
    └── Vietnam/
```
*(Nếu chưa có ảnh, bạn hãy tải vài tấm ảnh tiền trên mạng, chia vào đúng các thư mục này nhé)*

---

## 🚀 Bước 3: Chạy thử chương trình

1. Mở phần mềm **Jupyter Notebook**.
2. Tìm và mở file có tên `Main_CurrencyClassification.ipynb`.
3. Trong Jupyter, bạn sẽ thấy code được chia thành từng ô (cell) hình chữ nhật.
4. Bạn hãy click chuột vào ô đầu tiên, rồi nhấn nút **Run** (hoặc tổ hợp phím `Shift + Enter`).
5. Cứ thế chạy lần lượt từ ô trên cùng xuống ô dưới cùng. Đừng nhảy cóc nhé!

---

## 🧠 Giải thích thuật toán (Bằng ví dụ đời thường)

Để máy tính nhận ra tờ tiền, chúng ta dùng một "bộ đôi hoàn hảo": **HOG** (để nhìn) và **SVM** (để suy nghĩ).

### 1. HOG (Histogram of Oriented Gradients) - "Chuyên gia tìm đường viền"
Máy tính không nhìn thấy hình ảnh như con người, nó chỉ thấy một ma trận các con số. Làm sao để nó biết tờ 10.000 VNĐ khác tờ 1 USD?

**HOG** sinh ra để giải quyết việc này. Bạn cứ tưởng tượng HOG là một người cầm bút chì, đi đồ lại toàn bộ **đường viền, góc cạnh, và hoa văn** của tờ tiền.
- Nó chia tờ tiền ra thành hàng trăm ô vuông nhỏ li ti.
- Ở mỗi ô, nó xem xét: "Chỗ này có cái sọc ngang nào không? Có sọc dọc nào không? Sọc chéo thì sao?".
- Cuối cùng, nó tổng hợp lại thành một danh sách rất dài các con số (gọi là *vector đặc trưng*).
👉 **Kết quả:** Thay vì nhìn cả tấm ảnh màu mè phức tạp, máy tính giờ chỉ cần nhìn vào danh sách các đường viền đặc trưng do HOG tạo ra.

### 2. SVM (Support Vector Machine) - "Kẻ vẽ ranh giới"
Sau khi HOG đã tạo ra danh sách đặc trưng, chúng ta giao danh sách đó cho **SVM** để học cách phân loại.

Hãy tưởng tượng bạn có một cái bàn. Bạn rải lên đó rất nhiều quả cam (tượng trưng cho tiền VNĐ) và quả chanh (tượng trưng cho tiền USD).
Nhiệm vụ của **SVM** là: **Kẻ một đường thẳng trên mặt bàn sao cho cam nằm hết một bên, chanh nằm hết một bên.**
- Khi có một quả mới ném lên bàn (một tờ tiền mới), SVM chỉ cần nhìn xem quả đó rơi vào bên nào của đường thẳng là biết ngay nó là cam hay chanh.
- Nếu các quả xen kẽ nhau quá phức tạp không thể kẻ đường thẳng, SVM có một tuyệt chiêu (gọi là *Kernel RBF*): Nó "tung" các quả đó lên không trung (tạo không gian 3D), rồi nhét một tấm bìa vào giữa để tách chúng ra!

---

## 📖 Đọc hiểu Code (Đơn giản hóa)

Trong file `Main_CurrencyClassification.ipynb`, code được chia thành 6 bước (6 ô):

*   **Ô 1: Chuẩn bị đồ nghề.** Chúng ta gọi (import) các thư viện đã cài ở Bước 1 vào để dùng.
*   **Ô 2: Dùng HOG để "đọc" ảnh.** 
    *   Chương trình sẽ tự động mở thư mục `images/`, vào từng thư mục con để đọc từng tấm ảnh.
    *   Mọi tấm ảnh dù to dù nhỏ đều bị ép về chung một kích thước (96x96 pixel).
    *   Sau đó HOG sẽ biến ảnh thành một dãy số (đặc trưng).
*   **Ô 3: Chụp X-Quang.** Ô này chỉ dùng để vẽ hình minh họa cho bạn xem HOG thực chất đã "nhìn" thấy những nét vẽ gì trên tờ tiền.
*   **Ô 4: Dạy SVM học (Huấn luyện).**
    *   Chia bài: Lấy 80% số ảnh ra để "dạy" cho SVM học cách kẻ ranh giới. Cất 20% ảnh còn lại đi để lát nữa "kiểm tra bài cũ" xem SVM học có giỏi không.
    *   Lệnh `svm.fit(...)` chính là lúc máy tính đang cặm cụi học bài.
*   **Ô 5: Chấm điểm (Ma trận nhầm lẫn).** Vẽ ra một bảng (gọi là Confusion Matrix) để xem mô hình SVM đoán trúng bao nhiêu câu, và hay đoán nhầm loại tiền nào với loại tiền nào nhất.
*   **Ô 6: Biểu diễn thực tế.** Đưa ảnh thật vào, SVM sẽ đoán xem nó giống tiền nước nào nhất (ra được % chắc chắn). In ra kết quả OK (đoán trúng) hoặc WRONG (đoán sai).

---

Chúc bạn có một đồ án thành công và đạt điểm cao nhé! Đừng ngại thử nghiệm bằng cách bỏ thêm ảnh tiền mới vào thư mục để thử thách mô hình.