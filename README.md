# ViSignRe — Vietnamese Sign Language Recognition System

Hệ thống **nhận diện ngôn ngữ ký hiệu Việt Nam (VSL)** từ video thời gian thực:

```
Video/Webcam 
  ├─ MediaPipe Pose + Hands (trích keypoint tay)
  ├─ BiLSTM TFLite (11 lớp từ vựng) 
  └─ Gender Detector (OpenCV DNN) → TTS giọng Nam/Nữ
  
  ↓ Xử lý bất đồng bộ
  
  ├─ Groq Llama-3 (tùy chọn): biên dịch từ khóa thành câu hoàn chỉnh
  └─ VieNeu-TTS-v2: phát âm thanh offline → result.txt
```

**Phiên bản:** v1.0.0 | Ngôn ngữ: Python 3.8+ | Giấy phép: Open Source

---

## Mục lục

1. [Từ vựng](#từ-vựng-hỗ-trợ)
2. [Cài đặt](#cài-đặt)
3. [Hướng dẫn sử dụng đầy đủ](#hướng-dẫn-sử-dụng-đầy-đủ) — Quick start + Web + Full workflow
4. [Cấu hình `config.py`](#cấu-hình-configpy)
5. [Cấu trúc dự án](#cấu-trúc-dự-án)

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

### Yêu cầu hệ thống
- **Python:** 3.8+ (khuyến nghị 3.10+)
- **OS:** Windows, Linux, macOS
- **Webcam:** Để quay dữ liệu (bước 1)
- **RAM:** 8GB+ (để train; 4GB cho inference)
- **Thẻ đồ họa:** Không bắt buộc (dùng CPU mặc định)

### Bước 1: Tạo virtual environment

Mọi lệnh chạy từ **thư mục gốc** dự án (`ViSignRe/`).

```bash
cd ViSignRe
python -m venv .venv
```

### Bước 2: Kích hoạt environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### Bước 3: Cài thư viện chạy (Bắt buộc)

```bash
pip install -r requirements.txt
```

Cấu thành:
- **tensorflow 2.17.0**: Model inference TFLite
- **mediapipe 0.10.14**: Pose + Hand detection
- **opencv-python 4.10.0.84**: Video processing
- **groq 1.2.0**: Gọi Groq API (tùy chọn)
- **vieneu 2.7.0**: TTS tiếng Việt offline
- **onnxruntime 1.26.0**: Hỗ trợ gender detection

### Bước 4: (Tùy chọn) Cài thư viện web

Nếu bạn muốn **chạy web interface** (`server.py`):

```bash
pip install -r requirements_web.txt
```

Cấu thành: `fastapi`, `uvicorn`, `python-multipart`

### Bước 5: (Tùy chọn) Cài thư viện huấn luyện

Nếu bạn muốn **tự train lại mô hình** từ dữ liệu:

```bash
pip install -r requirements_train.txt
```

### Bước 6: Cấu hình Groq (Tùy chọn)

Groq được dùng để biên dịch các từ khóa thành câu hoàn chỉnh. Nếu bỏ qua bước này, hệ thống vẫn hoạt động nhưng chỉ xuất danh sách từ.

1. Sao chép template:
```bash
copy .env.example .env    # Windows
# cp .env.example .env   # Linux/macOS
```

2. Lấy API key tại: https://console.groq.com/keys

3. Sửa `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
```

### Bước 7: Tải mô hình Gender Detection (Bắt buộc cho TTS)

Tính năng này tự động phát hiện giới tính từ khuôn mặt để chọn giọng TTS Nam/Nữ phù hợp. Vì file ~45MB, bạn cần tải thủ công:

1. Truy cập: [lsfhan/age-gender-detection](https://github.com/lsfhan/age-gender-detection)
2. Tải 2 file: `gender_deploy.prototxt` và `gender_net.caffemodel`
3. Đổi tên:
   - `gender_deploy.prototxt` → `gender.prototxt`
   - `gender_net.caffemodel` → `gender.caffemodel`
4. Di chuyển vào `models/`

Cấu trúc cuối cùng:
```
models/
├── gender.prototxt
├── gender.caffemodel
└── ViSignRe.tflite
```

---

## Hướng dẫn sử dụng đầy đủ

### Luồng chuẩn (Quick Start)

Để **chạy ngay** hệ thống với mô hình đã train sẵn:

```bash
python main.py
```

**Điều kiện tiên quyết:**
- ✓ Cài đặt xong `requirements.txt`
- ✓ File `models/ViSignRe.tflite` có sẵn
- ✓ Hoặc dùng VIDEO_PATH='0' cho webcam

**Phím điều khiển:**
| Phím | Chức năng |
|------|-----------|
| `Q` | Thoát |
| `F` | Lật gương video |
| `S` hoặc `Enter` | Gửi câu hiện tại sang Groq + TTS |

Kết quả xuất ra `result.txt`.

---

### Web Interface (FastAPI + WebSocket)

Ngoài CLI (`main.py`), ViSignRe cung cấp **giao diện web hiện đại** để tải video và xem kết quả realtime.

#### Chạy web server

**Cài đặt thêm (nếu chưa cài):**
```bash
pip install -r requirements_web.txt
```

Cấu thành: `fastapi`, `uvicorn`, `python-multipart`

**Khởi động:**
```bash
python server.py
```

**Kết quả:**
```
============================================================
  ViSignRe Web Server
  Đang mở http://localhost:8000 ...
============================================================
```

Browser sẽ tự mở. Nếu không, truy cập: http://localhost:8000

#### Giao diện web

- **Giao diện:** Tối (dark mode) + hiệu ứng neon (accent color xanh)
- **Tải video:** Drag & drop hoặc chọn tệp (MP4, AVI, MOV, MKV, WebM)
- **Tùy chọn:**
  - `Use Groq`: Bật/tắt biên dịch AI (cần `GROQ_API_KEY`)
- **Xem realtime:**
  - Video stream từ phía server
  - Confidence bars (5 lớp hàng đầu)
  - FPS, từ hiện tại, trạng thái pose/hand
  - Thanh tiến độ xử lý
- **Kết quả:**
  - Câu hoàn chỉnh
  - Danh sách từ được nhận diện
  - Thời gian xử lý từng module (Pose, Hand, Model)
  - Trạng thái TTS (INIT → READY → SPEAKING → DONE)

#### API Endpoints

| Endpoint | Phương thức | Mô tả |
|----------|-----------|-------|
| `/` | GET | Phục vụ `index.html` |
| `/upload` | POST | Tải video lên (form data: `file`) |
| `/start/{session_id}` | POST | Bắt đầu xử lý (query: `use_groq=true/false`) |
| `/stop/{session_id}` | POST | Dừng xử lý |
| `/status/{session_id}` | GET | Lấy trạng thái xử lý |
| `/ws/{session_id}` | WebSocket | Stream realtime (frame, metrics, events) |

#### Ví dụ sử dụng API

**Upload video:**
```bash
curl -X POST -F "file=@data/test_video.mp4" http://localhost:8000/upload
# Output: {"session_id": "abc123...", "filename": "test_video.mp4", "size_mb": 45.2}
```

**Bắt đầu xử lý:**
```bash
curl -X POST "http://localhost:8000/start/abc123?use_groq=true"
# Output: {"message": "Started", "session_id": "abc123"}
```

**Kiểm tra trạng thái:**
```bash
curl "http://localhost:8000/status/abc123"
# Output: {"status": "processing", "words_detected": ["Xin_chao"], "sentence": ["Xin_chao"]}
```

**WebSocket (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/abc123');

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    console.log(msg.event, msg.data);
    // event: "frame" → dòng video realtime (base64)
    // event: "word_detected" → từ mới nhận diện
    // event: "done" → xử lý hoàn tất
    // event: "error" → lỗi xảy ra
};

ws.send("ping");  // Keep-alive (tùy chọn)
```

#### Đặc điểm web server

✅ **Async I/O**: FastAPI + asyncio → xử lý nhiều request đồng thời  
✅ **WebSocket realtime**: Stream video + metrics 60fps browser  
✅ **Session isolation**: Mỗi upload = session riêng (ID = UUID)  
✅ **Thread-safe broadcasting**: Xử lý inference trong thread riêng, stream trong thread khác  
✅ **CORS enabled**: Có thể gọi từ frontend khác  
✅ **Frame compression**: JPEG quality 65 + resize 720px max → tiết kiệm bandwidth  
✅ **Keep-alive WebSocket**: Ping/pong tự động để tránh timeout  
✅ **Auto cleanup**: Xóa video + session sau 30 giây (tiết kiệm disk)  

---

### Luồng đầy đủ (Train lại từ đầu)

Nếu muốn **thêm từ vựng mới** hoặc **train mô hình riêng**:

```
1. Quay dữ liệu (.npy)  →  2. Xem quỹ đạo (opt)  →  3. Train  →  4. Export TFLite  →  5. Cấu hình  →  6. Chạy nhận diện
```

---

---

## Các tính năng chính

### 1. **Nhận diện tay từ video thời gian thực**
- Sử dụng MediaPipe Hands để trích 21 keypoint bàn tay
- Chuẩn hóa chuỗi 45 frame × 126 số (25 frame/giây)
- Hỗ trợ webcam (chỉ số 0, 1, ...) hoặc file video MP4

### 2. **Model BiLSTM TFLite (11 lớp)**
- Train từ dữ liệu keypoint (.npy)
- Export sang TFLite để chạy CPU, không cần GPU
- Độ chính xác: 90-95% (phụ thuộc dữ liệu)

### 3. **State Machine thông minh**
- Buffer 45-frame: lưu chuỗi keypoint liên tục
- Voting (k=20): xác định từ bằng vote đa số
- Idle detection: phát hiện khi người dừng cử chỉ
- Ngưỡng adaptive theo từng lớp (`THRESHOLDS` trong config)

### 4. **Phát hiện giới tính tự động**
- OpenCV DNN: trích vùng khuôn mặt từ Pose
- Gender detection: Nam (male) hoặc Nữ (female)
- Khóa giới tính lần đầu → dùng giọng TTS phù hợp

### 5. **Biên dịch Groq (Tùy chọn)**
- Gọi Llama-3 qua Groq API
- Chuyển danh sách từ khóa → câu hoàn chỉnh
- Thêm dấu câu (,  .  !) để TTS phát có cảm xúc
- Không bịa đặt: chỉ nối từ có sẵn

### 6. **TTS Offline (VieNeu-TTS-v2)**
- Không cần internet sau khi load mô hình
- Hỗ trợ giọng Nam / Nữ
- Tốc độ phát có thể điều chỉnh
- Output WAV hoặc phát trực tiếp

### 7. **Web Interface (FastAPI + WebSocket)**
- Giao diện web hiện đại (dark mode, neon UI)
- Tải video bằng drag & drop
- Stream realtime: video frame + metrics
- Xem confidence bars (top 5)
- WebSocket realtime events (status, frame, result)
- REST API endpoints (upload, start, stop, status)
- Multi-session: xử lý multiple uploads đồng thời
- Auto cleanup: session expire sau 30s

### 8. **Ghi kết quả (result.txt)**
```
Câu cuối cùng sau Groq + TTS
```

---

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

### Biến cơ bản

| Biến | Mô tả | Ví dụ |
|------|-------|-------|
| `MODEL_PATH` | Đường dẫn TFLite | `models/ViSignRe.tflite` |
| `VIDEO_PATH` | Video hoặc ID webcam | `'0'` (webcam) hoặc `'data/test.mp4'` |
| `OUTPUT_TXT` | File lưu kết quả | `result.txt` |
| `FONT_PATH` | Font tiếng Việt (TTF) | `arial.ttf` |

### Tham số model

| Biến | Ý nghĩa | Giá trị mặc định |
|------|---------|-----------------|
| `WINDOW_SIZE` | Độ dài chuỗi frame | 45 |
| `KP_SIZE` | Số giá trị keypoint (21 điểm × 2 tọa độ + 84 đặc trưng) | 126 |

### Vùng hoạt động (Action Zone)

| Biến | Ý nghĩa | Giá trị mặc định |
|------|---------|-----------------|
| `ACTION_ZONE_FALLBACK` | Tỉ lệ Y từ vai (fallback nếu không detect pose) | 0.75 |
| `ACTION_ZONE_HIP_RATIO` | Tỉ lệ từ vai đến hông | 0.7 |
| `ACTION_ZONE_SMOOTH` | Làm mịn (EMA) vị trí action zone qua frame | 0.9 |

**Lý do:** Tay dưới line xanh bị bỏ qua (không valid gesture).

### Nhận diện và voting

| Biến | Ý nghĩa | Giá trị mặc định |
|------|---------|-----------------|
| `IDLE_THRESHOLD` | Số frame không đổi để chốt từ | 15 |
| `VOTE_WINDOW` | Kích thước voting buffer | 20 |
| `TRIM_FRAMES` | Lặt đầu/cuối chuỗi trước predict | 3 |
| `MIN_CORE_FRAMES` | Số frame lõi tối thiểu (sau lặt) | 2 |
| `DEFAULT_THRESH` | Ngưỡng confidence mặc định | 0.70 |

### Ngưỡng tùng lớp (Per-class Thresholds)

```python
THRESHOLDS = {
    'Vui': 0.85,        # Từ khó, cần confidence cao
    'Xin_chao': 0.80,
    'Gap': 0.75,
    'Toi': 0.80,
    'Song': 0.60,       # Từ dễ, ngưỡng thấp
    # ...
}
```

### Tối ưu hiệu năng

| Biến | Ý nghĩa | Giá trị mặc định |
|------|---------|-----------------|
| `MP_CONFIDENCE` | Ngưỡng MediaPipe (Pose + Hands) | 0.5 |
| `PREDICT_EVERY` | Dự đoán mỗi N frame (skip frames) | 3 |
| `POSE_SKIP_FRAMES` | Cập nhật pose mỗi N frame | 3 |

**Ghi chú:** Tăng `PREDICT_EVERY` → FPS cao hơn nhưng phản ứng chậm. Giảm → chính xác hơn nhưng chậm.

### Cửa sổ UI

| Biến | Ý nghĩa | Giá trị mặc định |
|------|---------|-----------------|
| `WIN_W, WIN_H` | Kích thước window | 1280 × 720 |
| `FONT_SIZE_LG` | Font lớn (câu) | 28 |
| `FONT_SIZE_MD` | Font trung (từ hiện tại) | 22 |
| `FONT_SIZE_SM` | Font nhỏ (ghi chú) | 18 |
| `WINDOW_NAME` | Tên cửa sổ | `'ViSignRe'` |

### Từ vựng (Classes)

Thay đổi sau khi train từ lớp mới:

```python
ACTIONS = np.array([
    '21', 'Blank', 'Gap', 'Ha_Noi', 'Hung',
    'Song', 'Ten', 'Toi', 'Tuoi', 'Vui', 'Xin_chao',
])

LABEL_DISPLAY = {
    '21': '21',
    'Blank': '',           # Không hiển thị
    'Gap': 'Gặp',
    'Ha_Noi': 'Hà Nội',
    # ...
}
```

**Lưu ý:** Phải update `ACTIONS`, `LABEL_DISPLAY`, `THRESHOLDS` khi thêm/bớt lớp.

---

## Cấu trúc dự án

```
ViSignRe/
├── main.py                          # Bước 6: Nhận diện realtime + UI
├── server.py                        # (Tùy chọn) Web interface backend
├── config.py                        # Cấu hình toàn cầu
├── requirements.txt                 # Thư viện chạy
├── requirements_train.txt           # (Tùy chọn) Thư viện train
├── requirements_web.txt             # (Tùy chọn) Thư viện web
├── .env.example                     # Template biến môi trường (GROQ_API_KEY)
├── result.txt                       # Output: câu hoàn chỉnh sau Groq + TTS
│
├── data/
│   ├── dataset_words/               # Bước 1: Dataset keypoint (.npy)
│   │   ├── 21/
│   │   ├── Blank/
│   │   ├── Gap/
│   │   ├── Ha_Noi/
│   │   ├── Hung/
│   │   ├── Song/
│   │   ├── Ten/
│   │   ├── Toi/
│   │   ├── Tuoi/
│   │   ├── Vui/
│   │   └── Xin_chao/
│   └── test_video.mp4               # Video mẫu để test main.py
│
├── models/
│   ├── ViSignRe.keras               # Bước 3: Model sau train
│   ├── ViSignRe.tflite              # Bước 4: Model TFLite (bắt buộc cho main.py)
│   ├── gender.prototxt              # Gender detection (cần tải thủ công)
│   └── gender.caffemodel            # Gender detection weights (cần tải thủ công)
│
├── reports/
│   ├── confusion_matrix.png         # Bước 3: Ma trận nhầm lẫn
│   ├── learning_curve.png           # Bước 3: Biểu đồ huấn luyện
│   └── quy_dao_khong_gian.png       # Bước 2: Quỹ đạo tay
│
├── static/                          # Web UI assets
│   └── index.html                   # Giao diện web (FastAPI + WebSocket)
├── uploads/                         # Thư mục tạm cho uploads
├── cache_tts/                       # Cache audio TTS
├── scripts/
│   └── export_tflite.py             # Bước 4: Convert .keras → .tflite
│
└── src/
    ├── core/                        # Xử lý keypoint & gesture
    │   ├── gesture_recognizer.py    # State machine: buffer + idle + vote
    │   ├── keypoint_utils.py        # Chuẩn hóa 45 frame × 126 số
    │   └── mediapipe_handlers.py    # HandDetector, PoseDetector, draw()
    │
    ├── pipelines/
    │   ├── collect_data.py          # Bước 1: Quay tay từ webcam + augment
    │   └── train.py                 # Bước 3: BiLSTM train + metrics
    │
    ├── processors/                  # Xử lý bất đồng bộ & ngoài nhân
    │   ├── video_processor.py       # Đọc video/webcam + FPS
    │   ├── gender_detector.py       # OpenCV DNN: phân loại Nam/Nữ
    │   ├── groq_processor.py        # Gọi Groq API (Llama-3)
    │   └── tts_processor.py         # VieNeu-TTS-v2: phát âm
    │
    ├── ui/
    │   ├── renderers.py             # Vẽ text Việt + thanh confidence
    │   ├── quy_dao.py               # Bước 2: Vẽ quỹ đạo ngón tay
    │   └── mo_phong_khung_xuong.py  # (Thử nghiệm) Skeleton preview
    │
    └── utils/
        └── utils.py                 # FPS counter, color constants, ...
```

---


## API & Khởi chạy lập trình

### Sử dụng GestureRecognizer

```python
from src.core.gesture_recognizer import GestureRecognizer
from config import Config

recognizer = GestureRecognizer()

# Mỗi frame
result = recognizer.update(
    keypoints,        # shape (45, 126) hoặc None
    hand_detected,    # bool
    model,            # TFLiteModelWrapper hoặc None
    device='CPU'
)

# Output dict
{
    'predictions': [...],        # shape (11,) softmax
    'word': 'Gap',               # class hiện tại
    'confidence': 0.85,
    'ready_to_finalize': True,   # Từ chốt?
    'finalized_word': 'Gap',     # Từ chốt (nếu ready)
}

# Reset sau khi xử lý
recognizer.reset_gesture()
```

### Sử dụng VideoProcessor

```python
from src.processors.video_processor import VideoProcessor

vp = VideoProcessor('data/test.mp4')  # hoặc '0' cho webcam

while vp.is_open():
    ret, frame, h, w = vp.read_frame()
    if not ret:
        break
    # Xử lý frame
    
vp.release()
```

### Sử dụng Groq

```python
from src.processors.groq_processor import GroqProcessor

groq = GroqProcessor()
result = groq.generate_sentence(['Xin_chao', 'Toi', 'Ten'])
# Output: {'sentence': 'Xin chào, tôi tên là ...', 'explanation': '...', 'params': {...}}
```

### Sử dụng TTS

```python
from src.processors.tts_processor import speak_dynamic

tone_params = {'tone': 'neutral'}  # hoặc {'tone': 'happy'}, ...
speak_dynamic('Xin chào, tôi tên là Hưng', tone_params, gender='nam')
```

---

## Giới hạn

- Chỉ **11 từ** đã huấn luyện; không phải VSL đầy đủ.
- Độ chính xác phụ thuộc ánh sáng, góc quay, đồng nhất cách múa lúc quay và lúc test.
- Phù hợp đồ án / demo; chưa tối ưu production.

---

## Phiên bản
- **v1.0.0**: Phát hành phiên bản đầu tiên. Tích hợp AI Biên dịch (Groq Llama-3) và Hệ thống đọc giọng nói Offline (VieNeu-TTS-v2).

---


## Tác giả & Giấy phép

**Phát triển bởi:** Hung (Hưng) & Contributors

**Giấy phép:** Open Source (MIT / Apache 2.0 — xem LICENSE)

**Cảm ơn:**
- MediaPipe (Google) — Pose & Hand detection
- TensorFlow (Google) — ML framework
- Groq — LLM API
- VieNeu-TTS-v2 — Vietnamese TTS
- sea-g2p & onnxruntime — Hỗ trợ xử lý

---
