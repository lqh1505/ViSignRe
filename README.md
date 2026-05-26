# ViSignRe

Hệ thống **nhận diện ngôn ngữ ký hiệu Việt Nam (VSL)** từ video: trích keypoint tay (MediaPipe) → BiLSTM (TFLite) → ghép từ khóa → (tùy chọn) Groq làm câu tiếng Việt.

**Phiên bản:** 

---

## Mục lục

1. [Từ vựng](#từ-vựng-hỗ-trợ)
2. [Cài đặt](#cài-đặt)
3. [Hướng dẫn sử dụng đầy đủ](#hướng-dẫn-sử-dụng-đầy-đủ) — từ quay data → train → main
4. [Cấu hình `config.py`](#cấu-hình-configpy)
5. [Cấu trúc dự án](#cấu-trúc-dự-án)
6. [Xử lý sự cố](#xử-lý-sự-cố)

---

## Từ vựng hỗ trợ

| Nhãn (tên thư mục / model) | Hiển thị trên UI |
|----------------------------|------------------|
|            `21`            |        21        |
|            `Blank`         |      (trống)     |
|            `Gap`           |        Gặp       |
|            `Ha_Noi`        |        Hà Nội    |
|            `Hung`          |        Hưng      |
|            `Song`          |        Sống      |
|            `Ten`           |        Tên       |
|            `Toi`           |        Tôi       |
|            `Tuoi`          |        Tuổi      |
|            `Vui`           |        Vui       |
|            `Xin_chao`      |        Xin chào  |

Tên thư mục dataset **phải trùng** nhãn (ví dụ `data/dataset_words/Gap/`, không dấu).

---

## Cài đặt

Mọi lệnh chạy từ **thư mục gốc** dự án (`ViSignRe/`).

```bash
cd ViSignRe
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
# Cài đặt thư viện để chạy ứng dụng (bắt buộc)
pip install -r requirements.txt

# (Tùy chọn) NẾU BẠN MUỐN TỰ HUẤN LUYỆN LẠI MÔ HÌNH, hãy cài thêm:
pip install -r requirements_train.txt
```

**Linux / macOS:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Groq (tùy chọn — bước làm câu cuối)

```bash
copy .env.example .env    # Windows
# cp .env.example .env   # Linux/macOS
```

Sửa `.env`:

```env
GROQ_API_KEY=gsk_your_key_here
```

Lấy key tại: https://console.groq.com/keys

---

## Hướng dẫn sử dụng đầy đủ

Luồng chuẩn từ đầu đến cuối:

```
Quay keypoint (.npy)  →  Train (.keras)  →  Export TFLite  →  main.py (nhận diện)
```

### Tổng quan các bước

| Bước |       Việc cần làm           |             Lệnh / file               |
|------|------------------------------|---------------------------------------|
|   1  | Thu thập dữ liệu (webcam)    | `src/pipelines/collect_data.py`       |
|   2  | *(Tùy chọn)* Xem quỹ đạo tay | `src/ui/quy_dao.py`                   |
|   3  | Huấn luyện model             | `src/pipelines/train.py`              |
|   4  | Export sang TFLite           | `scripts/export_tflite.py` hoặc Colab |
|   5  | Cấu hình video / model       | `config.py`                           |
|   6  | Chạy nhận diện               | `main.py`                             |

---

### Bước 1 — Quay / thu thập dữ liệu (`collect_data.py`)

Mỗi **từ/ký hiệu** là một thư mục chứa file `.npy` (chuỗi 45 frame × 126 số).

#### 1.1. Chỉnh cấu hình thu thập

Mở `src/pipelines/collect_data.py`, sửa phần đầu file:

```python
WORD_TO_RECORD = "Gap"   # Tên lớp — trùng ACTIONS trong config.py
NUM_SAMPLES = 30         # Số lần bạn múa thật (quay tay)
TARGET_SAMPLES = 120     # Tổng file .npy sau khi tự augment
```

Lặp lại cho **từng từ** cần thêm/bổ sung (đổi `WORD_TO_RECORD` rồi chạy lại script).

#### 1.2. Chuẩn bị khi quay

- Dùng **webcam**, ánh sáng đủ, nền đơn giản.
- Đứng sao cho **vai – hông** nhìn thấy (để vẽ **action zone** — đường ngang xanh).
- Tay thực hiện ký hiệu **phía trên** đường zone (tay dưới line = bị bỏ qua).
- Mỗi lần SPACE: chỉ múa **một** ký hiệu, tốc độ tự nhiên (~1.5–2 giây cho 45 frame).

#### 1.3. Chạy thu thập

```bash
# Chạy từ thư mục gốc ViSignRe (khuyến nghị)
python src/pipelines/collect_data.py
```

Có thể bấm Run trong VS Code trên file `collect_data.py` — script tự `chdir` về root dự án.

**Phím trong cửa sổ "Thu thap Data":**

| Phím | Chức năng |
|------|-----------|
| `SPACE` | Bắt đầu mẫu mới (đếm ngược `PREP_TIME` 2 giây → tự ghi 45 frame) |
| `Q` | Thoát sớm |

**Luồng một mẫu:**

1. Nhấn `SPACE` → màn hình hiện "Chuẩn bị...".
2. Sau 2 giây → "ĐANG GHI 1/45" … "45/45".
3. Script chuẩn hóa chuỗi (`normalize_sequence`) và lưu `data/dataset_words/<WORD>/sample_N.npy`.
4. Nếu quá nhiều frame không có tay → bỏ mẫu, thử lại.

Sau khi quay đủ `NUM_SAMPLES`, script **tự augment** (scale, noise, lật tay) đến `TARGET_SAMPLES` file.

#### 1.4. Cấu trúc dataset sau khi quay

```
data/dataset_words/
├── 21/
│   ├── sample_0.npy
│   └── ...
├── Blank/
├── Gap/
├── Ha_Noi/
├── Hung/
├── Song/
├── Ten/
├── Toi/
├── Tuoi/
├── Vui/
└── Xin_chao/
```

Mỗi `sample_*.npy`: shape `(45, 126)`.

**Gợi ý:** Mỗi lớp nên có **đủ mẫu** (ví dụ ~120 file/lớp) trước khi train. Lớp `Blank` là “không có cử chỉ” / tay ngoài vùng.

---

### Bước 2 — (Tùy chọn) Trực quan hóa quỹ đạo

Kiểm tra chất lượng động tác đã lưu:

```bash
python src/ui/quy_dao.py
```

Kết quả: `reports/quy_dao_khong_gian.png` (quỹ đạo đầu ngón trỏ).

---

### Bước 3 — Huấn luyện (`train.py`)

Đảm bảo đủ thư mục lớp dưới `data/dataset_words/` (tên khớp bảng từ vựng).

```bash
python src/pipelines/train.py
```

**Script sẽ:**

1. Load toàn bộ `.npy`, chuẩn hóa, lọc mẫu quá ít frame có tay.
2. Chia train / val / test; augment thêm trên tập train.
3. Train BiLSTM (tối đa 300 epoch, dừng sớm nếu val không cải thiện).
4. Lưu model tốt nhất: **`models/ViSignRe.keras`**
5. In classification report; lưu:
   - `reports/confusion_matrix.png`
   - `reports/learning_curve.png`

**Tham số train** (đầu `train.py`): `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `DATA_PATH`, `MODEL_SAVE_PATH`.

---

### Bước 4 — Export TFLite (bắt buộc trước khi chạy `main.py`)

`main.py` chạy **`models/ViSignRe.tflite`**, không dùng `.keras` trực tiếp.

#### Cách 1 — Script local (ưu tiên)

```bash
python scripts/export_tflite.py
```

Script ép **CPU** (`CUDA_VISIBLE_DEVICES=-1`) và bật `SELECT_TF_OPS` cho LSTM.

#### Cách 2 — Google Colab (nếu máy local lỗi)

1. Upload `models/ViSignRe.keras` lên Colab.
2. Chạy:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

print("Thiết bị:", tf.config.list_physical_devices())
model = tf.keras.models.load_model("ViSignRe.keras", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.experimental_enable_resource_variables = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]

with open("ViSignRe.tflite", "wb") as f:
    f.write(converter.convert())
```

3. Tải `ViSignRe.tflite` về → đặt vào `models/ViSignRe.tflite`.

---

### Bước 5 — Cấu hình trước khi nhận diện

Mở `config.py`:

| Biến | Ý nghĩa | Gợi ý |
|------|---------|--------|
| `MODEL_PATH` | File TFLite | `models/ViSignRe.tflite` |
| `VIDEO_PATH` | Video test hoặc webcam | `data/test_video.mp4` hoặc `'0'` |
| `OUTPUT_TXT` | File kết quả | `result.txt` |
| `FONT_PATH` | Font tiếng Việt | `arial.ttf` (Windows thường có sẵn) |

Sau khi train từ khóa mới, nếu thêm/bớt lớp phải cập nhật **`ACTIONS`**, **`LABEL_DISPLAY`**, **`THRESHOLDS`** trong `config.py` cho khớp dataset.

---

### Bước 6 — Chạy nhận diện (`main.py`)

```bash
python main.py
```

**Điều kiện:** có `models/ViSignRe.tflite` và video/webcam mở được theo `VIDEO_PATH`.

**Phím:**

| Phím | Chức năng |
|------|-----------|
| `Q` | Thoát |
| `F` | Bật/tắt lật gương video |

**Cách hoạt động trên UI:**

1. Video chạy; đường **action zone** (xanh/vàng) theo pose.
2. Tay trong vùng → model dự đoán từ; thanh confidence bên phải.
3. Kết thúc một cử chỉ (idle đủ frame) → từ được **ghép vào câu** (console in `>> Detected: ...`).
4. Thoát (`Q`) → in metrics FPS; nếu có Groq → làm câu hoàn chỉnh; ghi **`result.txt`**.

**Ví dụ console:**

```
>> Detected: Xin_chao | Translated: 'Xin chào'
>> Detected: Toi | Translated: 'Tôi'
...
[AI] Refined Sentence: Xin chào, tôi tên là Hưng, ...
[RESULT] Final Translation: ...
```

---

## Cấu hình `config.py`

Ngoài bảng bước 5, các hằng số quan trọng:

| Nhóm | Tham số | Mô tả ngắn |
|------|---------|------------|
| Model input | `WINDOW_SIZE=45`, `KP_SIZE=126` | Khớp collect & train |
| Action zone | `ACTION_ZONE_FALLBACK`, `ACTION_ZONE_SMOOTH` | Vùng tay hợp lệ |
| Nhận diện | `IDLE_THRESHOLD`, `VOTE_WINDOW`, `THRESHOLDS` | State machine & ngưỡng từng lớp |
| Hiệu năng | `PREDICT_EVERY=3` | Inference mỗi 3 frame |

---

## Cấu trúc dự án

```
ViSignRe/
├── main.py                      # Bước 6: nhận diện + UI
├── config.py                    # Cấu hình chung
├── requirements.txt
├── .env.example
├── data/
│   ├── dataset_words/           # Bước 1: dataset .npy
│   └── test_video.mp4           # Video mẫu cho main
├── models/
│   ├── ViSignRe.keras           # Bước 3: sau train
│   └── ViSignRe.tflite          # Bước 4: dùng cho main
├── reports/                     # Biểu đồ train / quy_dao
├── scripts/
│   └── export_tflite.py         # Bước 4
└── src/
    ├── core/                    # MediaPipe, gesture recognizer
    ├── pipelines/
    │   ├── collect_data.py      # Bước 1
    │   └── train.py             # Bước 3
    ├── processors/              # Video, Groq
    ├── ui/
    │   ├── renderers.py
    │   └── quy_dao.py           # Bước 2
    └── utils/
        └── sequence_utils.py    # Chuẩn hóa chuỗi 45 frame
```

---

## Luồng xử lý (runtime)

```
Video / webcam
    → MediaPipe Pose (action zone) + Hands (keypoint)
    → Cửa sổ 45 frame × 126
    → TFLite BiLSTM (11 lớp)
    → GestureRecognizer (vote, idle, chống lặp từ)
    → UI tiếng Việt
    → [Groq] câu hoàn chỉnh
    → [VieNeu-TTS-v2] Phát âm thanh Offline (Giọng Nam/Nữ) → result.txt
```

---

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|-------------|------------|
| `No module named 'src'` | Chạy từ root: `python src/pipelines/collect_data.py`, hoặc dùng bản script đã có bootstrap path |
| Webcam không mở (collect) | Đổi index `VideoCapture(0)` → `1`; kiểm tra quyền camera |
| Mẫu bị SKIP liên tục | Tay giữ **trên** line xanh; đứng đủ người trong khung |
| Train báo `Path not found` | Tạo `data/dataset_words/<TenLop>/` đúng tên |
| Train ít sample | Quay thêm hoặc giảm `min_active` trong `train.py` |
| Export TFLite lỗi | Dùng `scripts/export_tflite.py` hoặc Colab (CPU + SELECT_TF_OPS) |
| `Model not found` khi main | Đặt `ViSignRe.tflite` vào `models/`, khớp `MODEL_PATH` |
| `Cannot open video` | Sửa `VIDEO_PATH`; thử `'0'` |
| Groq disabled | Thêm `GROQ_API_KEY` vào `.env` |
| Chữ Việt lỗi | Đổi `FONT_PATH` sang font có dấu (`.ttf`) |
| OpenCV xung đột | Chỉ cài **một** trong `opencv-python` / `opencv-contrib-python` |

---

## Giới hạn

- Chỉ **11 từ** đã huấn luyện; không phải VSL đầy đủ.
- Độ chính xác phụ thuộc ánh sáng, góc quay, đồng nhất cách múa lúc quay và lúc test.
- Phù hợp đồ án / demo; chưa tối ưu production.

---

## Phiên bản
- **v1.0.0**: Phát hành phiên bản đầu tiên. Tích hợp AI Biên dịch (Groq Llama-3) và Hệ thống đọc giọng nói Offline (VieNeu-TTS-v2).

---

## Giấy phép
Dự án được phát triển dưới dạng mã nguồn mở phục vụ mục đích nghiên cứu và học tập. Cảm ơn sự hỗ trợ từ các thư viện mã nguồn mở:
- MediaPipe (Google)
- Llama-3 (Meta / Groq)
- VieNeu-TTS-v2 (Phạm Nguyễn Ngọc Bảo)