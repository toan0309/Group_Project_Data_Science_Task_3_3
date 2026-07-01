# Phân Loại Tiền Tệ — HOG + SVM

Dự án môn Khoa học Dữ liệu — Nhóm sinh viên năm 3.  
Mô hình tự động nhận diện tiền tệ của 5 quốc gia từ ảnh chụp, sử dụng **HOG** để trích xuất đặc trưng và **SVM** để phân loại.

---

## Giới thiệu

Chương trình nhận vào một ảnh tiền tệ và dự đoán thuộc quốc gia nào trong 5 nước:

| Nhãn | Quốc gia |
|------|----------|
| Canada | 🇨🇦 Canada |
| Japan | 🇯🇵 Nhật Bản |
| Korea | 🇰🇷 Hàn Quốc |
| USA | 🇺🇸 Hoa Kỳ |
| Vietnam | 🇻🇳 Việt Nam |

---

## Cấu trúc thư mục

```
Group_Project_Data_Science_Task_3_3/
│
├── Main_CurrencyClassification.ipynb   # File code chính
├── currency_feature_dataset.csv        # File dữ liệu
├── README.md                           # File hướng dẫn này
│
└── images/
    ├── Canada/
    ├── Japan/
    ├── Korea/
    ├── USA/
    └── Vietnam/
```

---

## Cài đặt

```bash
pip install scikit-learn scikit-image numpy matplotlib seaborn
```

---

## Cách chạy

1. Đặt ảnh tiền vào đúng thư mục con trong `images/`.
2. Mở `Main_CurrencyClassification.ipynb` bằng Jupyter Notebook.
3. Chạy tuần tự từng cell từ trên xuống dưới (`Shift + Enter`).

---

## Hướng dẫn code

Notebook gồm 6 cell chính:

### Cell 1 — Khai báo thư viện

```python
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['Canada', 'Japan', 'Korea', 'USA', 'Vietnam']
IMG_SIZE    = (96, 96)
```

- Dùng `scikit-image` để xử lý ảnh và trích xuất HOG.
- Dùng `scikit-learn` cho SVM, chuẩn hóa và đánh giá mô hình.
- `IMG_SIZE = (96, 96)`: tất cả ảnh được resize về 96×96 pixel để đồng nhất đầu vào.

---

### Cell 2 — Đọc ảnh và trích xuất HOG

```python
def extract_hog_features(img_path, img_size=(96, 96)):
    img = plt.imread(img_path)
    img = transform.resize(img, img_size)   # Resize về 96x96
    gray = color.rgb2gray(img)              # Chuyển sang ảnh xám
    features = hog(gray,
                   orientations=9,          # 9 hướng gradient
                   pixels_per_cell=(8, 8),  # Ô 8x8 pixel
                   cells_per_block=(2, 2))  # Khối 2x2 ô
    return features, img
```

- **Resize**: đưa ảnh về cùng kích thước để HOG cho ra vector cùng độ dài.
- **Chuyển xám**: HOG chỉ cần ảnh xám (1 kênh), không cần màu.
- **HOG**: phân tích hướng của các cạnh trong ảnh → tạo ra vector đặc trưng số.

---

### Cell 3 — Minh họa HOG

Hiển thị ảnh gốc và ảnh HOG tương ứng của mỗi loại tiền — giúp nhìn thấy trực quan những gì HOG "thấy" trong ảnh (các đường nét, cạnh của tờ tiền).

---

### Cell 4 — Chuẩn hóa và huấn luyện SVM

```python
# Chuẩn hóa đặc trưng
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Chia train (80%) / test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# Huấn luyện SVM
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm.fit(X_train, y_train)
```

- **StandardScaler**: chuẩn hóa đặc trưng về cùng thang đo, giúp SVM hoạt động tốt hơn.
- **train_test_split**: chia 80% ảnh để train, 20% để test; `stratify=y_raw` đảm bảo mỗi lớp được phân chia đều.
- **SVC(kernel='rbf')**: SVM dùng nhân RBF — phù hợp khi đặc trưng không phân tách tuyến tính.
- `C=10`: tham số kiểm soát độ chặt của phân lớp; `probability=True` để lấy được xác suất đầu ra.

---

### Cell 5 — Confusion Matrix

Hiển thị ma trận nhầm lẫn để thấy rõ mô hình hay nhầm cặp lớp nào với nhau.

---

### Cell 6 — Dự đoán từng ảnh và tổng kết

```python
probs     = svm.predict_proba(features_scaled)[0]  # Xác suất 5 lớp
pred_idx  = int(np.argmax(probs))                   # Lấy lớp cao nhất
predicted = CLASS_NAMES[pred_idx]
```

- Với mỗi ảnh, SVM trả về 5 xác suất (tổng = 1).
- Lấy lớp có xác suất cao nhất làm kết quả dự đoán.
- Hiển thị ảnh + biểu đồ cột xác suất, nhãn `[OK]` / `[WRONG]`.
- Cuối cùng in tổng kết độ chính xác.

---

## Hướng dẫn thuật toán

### 1. Tổng quan pipeline

```
[Ảnh đầu vào]
      │
      ▼
[Resize 96×96 → Chuyển xám]
      │
      ▼
[Trích đặc trưng HOG]  →  Vector số (vd: 1764 con số)
      │
      ▼
[Chuẩn hóa StandardScaler]
      │
      ├─── 80% ──→ [Huấn luyện SVM]
      └─── 20% ──→ [Đánh giá SVM]
                        │
                        ▼
               [Kết quả: Canada/Japan/...]
```

---

### 2. HOG là gì?

**HOG (Histogram of Oriented Gradients)** — Biểu đồ hướng gradient.

Ý tưởng đơn giản: **hình dạng và đường nét của tờ tiền khác nhau giữa các quốc gia** → HOG phát hiện các đường nét đó.

**Cách hoạt động (3 bước):**

**Bước 1**: Chia ảnh thành các ô nhỏ 8×8 pixel.

```
┌────┬────┬────┬────┐
│ ô  │ ô  │ ô  │ ô  │
├────┼────┼────┼────┤
│ ô  │ ô  │ ô  │ ô  │
└────┴────┴────┴────┘
     Ảnh 96×96 → 12×12 = 144 ô
```

**Bước 2**: Trong mỗi ô, tính hướng của các cạnh (gradient) và thống kê vào 9 hướng (0°, 20°, 40°, ..., 160°).

```
Cạnh ngang → đóng góp vào hướng 0°
Cạnh chéo  → đóng góp vào hướng 45°
Cạnh đứng  → đóng góp vào hướng 90°
```

**Bước 3**: Ghép tất cả các cột histogram lại → **vector đặc trưng** đại diện cho ảnh.

**Kết quả**: Mỗi ảnh 96×96 → vector ~1764 số → SVM dùng vector này để phân loại.

---

### 3. SVM là gì?

**SVM (Support Vector Machine)** — Máy vector hỗ trợ.

Ý tưởng: Tìm một **đường ranh giới** phân tách tốt nhất giữa các lớp trong không gian đặc trưng.

```
Lớp A: ●●●  |  ○○○ :Lớp B
             |
         Đường ranh giới (hyperplane)
         → Khoảng cách tới 2 lớp là lớn nhất
```

**Kernel RBF**: Khi các lớp không phân tách được bằng đường thẳng, SVM dùng kernel RBF để "chiếu" dữ liệu lên không gian chiều cao hơn, ở đó chúng có thể phân tách được.

**Trong dự án này**:
- **Đầu vào**: vector HOG ~1764 chiều của mỗi ảnh.
- **Đầu ra**: 1 trong 5 nhãn (Canada, Japan, Korea, USA, Vietnam) kèm theo xác suất.

---

### 4. Tại sao HOG + SVM phù hợp?

| Tiêu chí | HOG + SVM | CNN phức tạp (MobileNetV2) |
|----------|-----------|--------------------------|
| Dễ hiểu | ✅ Rất dễ | ❌ Khó |
| Dataset nhỏ | ✅ Hoạt động tốt | ⚠️ Dễ overfit |
| Thời gian train | ✅ Vài giây | ⚠️ Vài phút – giờ |
| Cần GPU | ✅ Không | ⚠️ Nên có |
| Độ chính xác | ✅ Tốt với ảnh rõ | ✅ Cao hơn nếu nhiều data |

HOG + SVM là lựa chọn **cổ điển và hiệu quả** cho bài toán phân loại ảnh với dataset nhỏ — rất phù hợp cho đồ án sinh viên.

---

## Nhóm thực hiện

> *Báo cáo môn Khoa học Dữ liệu — Nhóm sinh viên năm 3*