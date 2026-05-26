"""
ViSignRe FastAPI Web Server
Provides HTTP endpoints and WebSocket streaming for the sign language recognition UI.
"""

import os
import sys
import uuid
import asyncio
import threading
import json
import logging
import time
import base64
import shutil

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI(title="ViSignRe Web Interface", version="1.0.0")

# Main event loop — captured at startup, used by background threads for send_text
_main_loop: asyncio.AbstractEventLoop | None = None

@app.on_event("startup")
async def _capture_loop():
    global _main_loop
    _main_loop = asyncio.get_running_loop()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────
active_sessions: dict = {}   # session_id → state dict
ws_clients: dict = {}        # session_id → list[WebSocket]


def _broadcast_sync(session_id: str, message: dict):
    """
    Send a JSON message to all WebSocket clients of a session.
    Safe to call from any background thread.
    """
    if _main_loop is None or _main_loop.is_closed():
        return

    session_ws = ws_clients.get(session_id, [])
    payload = json.dumps(message)

    for ws in list(session_ws):
        asyncio.run_coroutine_threadsafe(
            ws.send_text(payload), _main_loop
        )


# ─────────────────────────────────────────────────────────────────────────────
# Processing worker (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────
def _processing_worker(session_id: str, video_path: str, use_groq: bool):
    state = active_sessions[session_id]
    state.update({
        'status': 'initializing',
        'sentence': [],
        'words_detected': [],
        'current_word': '',
        'tts_status': 'INIT',
        'fps': 0.0,
    })

    def emit(event: str, data: dict):
        _broadcast_sync(session_id, {"event": event, "data": data})

    try:
        emit("status", {"message": "Đang khởi tạo hệ thống...", "phase": "init"})

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
        import warnings
        warnings.filterwarnings('ignore')

        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)

        import cv2
        import numpy as np
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

        _root = os.path.abspath(os.path.dirname(__file__))
        if _root not in sys.path:
            sys.path.insert(0, _root)

        from config import Config
        from src.core.gesture_recognizer import GestureRecognizer
        from src.core.keypoint_utils import extract_keypoints
        from src.core.mediapipe_handlers import HandDetector, PoseDetector
        from src.processors.video_processor import VideoProcessor
        from src.processors.gender_detector import GenderDetector

        emit("status", {"message": "Đang tải mô hình AI...", "phase": "loading_model"})

        class _DummyTensor:
            def __init__(self, d): self.data = d
            def numpy(self): return self.data

        class _TFLiteWrapper:
            def __init__(self, path):
                self.interp = tf.lite.Interpreter(model_path=path)
                self.interp.allocate_tensors()
                self.inp = self.interp.get_input_details()
                self.out = self.interp.get_output_details()

            def __call__(self, x, training=False):
                if hasattr(x, 'numpy'): x = x.numpy()
                self.interp.set_tensor(self.inp[0]['index'], x.astype(np.float32))
                self.interp.invoke()
                return _DummyTensor(self.interp.get_tensor(self.out[0]['index']))

        model = _TFLiteWrapper(Config.MODEL_PATH)
        model(np.zeros((1, Config.WINDOW_SIZE, Config.KP_SIZE), dtype=np.float32))

        emit("status", {"message": "Đang khởi tạo MediaPipe...", "phase": "loading_mediapipe"})

        hand_detector = HandDetector()
        pose_detector = PoseDetector()
        gender_detector = GenderDetector()
        gesture_recognizer = GestureRecognizer()
        video_proc = VideoProcessor(video_path)
        total_frames = video_proc.get_frame_count()

        emit("status", {"message": "Bắt đầu phân tích video...", "phase": "processing"})
        state['status'] = 'processing'

        # ── Groq (optional) ───────────────────────────────────────────────────
        groq_proc = None
        if use_groq:
            try:
                from src.processors.groq_processor import GroqProcessor
                groq_proc = GroqProcessor()
                state['tts_status'] = 'READY'
                emit("tts_status", {"status": "READY"})
            except Exception as e:
                logging.warning("Groq disabled: %s", e)
                state['tts_status'] = 'DISABLED'
                emit("tts_status", {"status": "DISABLED"})

        # TTS luôn được load nền (không phụ thuộc Groq)
        try:
            from src.processors.tts_processor import speak_dynamic as _speak_dynamic
            _tts_available = True
        except Exception as e:
            logging.warning("TTS unavailable: %s", e)
            _tts_available = False

        sentence = []
        action_zone_y = Config.ACTION_ZONE_FALLBACK
        pose_detected = False
        current_gender = None
        _pose_skip = 0
        _predict_counter = 0

        t_pose = t_hand = t_model = 0.0
        fps_times = []
        last_tick = time.perf_counter()

        # Encode params — quality 65 đủ rõ, tiết kiệm ~30% so với 72
        ENCODE_PARAMS  = [cv2.IMWRITE_JPEG_QUALITY, 65]
        # Stream tối đa 720px chiều rộng để giảm payload
        STREAM_MAX_W   = 720
        STREAM_FPS_CAP = 20          # browser không cần hơn 20fps để hiển thị mượt
        STREAM_INTERVAL = 1.0 / STREAM_FPS_CAP

        import numpy as _np  # import 1 lần, không import trong loop

        # ── Shared state giữa inference thread và stream thread ───────────────
        _stream_lock   = threading.Lock()
        _latest_frame  = {"buf": None, "meta": {}}   # frame mới nhất sẵn sàng gửi
        _inference_done = threading.Event()

        # ── Stream thread — chỉ lo encode + gửi, không chạm inference ─────────
        def _stream_worker():
            last_sent = 0.0
            while not _inference_done.is_set() or _latest_frame["buf"] is not None:
                now = time.perf_counter()
                if now - last_sent < STREAM_INTERVAL:
                    time.sleep(0.002)
                    continue

                with _stream_lock:
                    buf  = _latest_frame["buf"]
                    meta = _latest_frame["meta"].copy()
                    _latest_frame["buf"] = None   # consume

                if buf is None:
                    time.sleep(0.002)
                    continue

                frame_b64 = base64.b64encode(buf).decode('utf-8')
                emit("frame", {"image": frame_b64, **meta})
                last_sent = now

        stream_thread = threading.Thread(target=_stream_worker, daemon=True)
        stream_thread.start()

        # ── Inference loop — chạy hết tốc độ, không sleep ─────────────────────
        while video_proc.is_open() and not state.get('stop', False):
            ret, frame, h, w = video_proc.read_frame()
            if not ret:
                break

            now = time.perf_counter()
            dt = now - last_tick
            last_tick = now
            fps_times.append(dt)
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = 1.0 / (sum(fps_times) / len(fps_times) + 1e-9)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Pose ──────────────────────────────────────────────────────────
            t0 = time.perf_counter()
            _pose_skip += 1
            if _pose_skip >= Config.POSE_SKIP_FRAMES:
                _pose_skip = 0
                pose_results = pose_detector.process(image_rgb)
                pose_detected = False
                if pose_results.pose_landmarks:
                    zone_y = pose_detector.get_action_zone_y(pose_results.pose_landmarks)
                    if zone_y is not None:
                        action_zone_y = (Config.ACTION_ZONE_SMOOTH * action_zone_y
                                         + (1 - Config.ACTION_ZONE_SMOOTH) * zone_y)
                        pose_detected = True
                    if current_gender is None:
                        try:
                            lm = pose_results.pose_landmarks.landmark
                            face_x = [lm[i].x for i in Config.FACE_REGION_INDICES]
                            face_y = [lm[i].y for i in Config.FACE_REGION_INDICES]
                            x1, x2 = int(min(face_x)*w), int(max(face_x)*w)
                            y1, y2 = int(min(face_y)*h), int(max(face_y)*h)
                            m = int((x2-x1)*Config.FACE_ROI_MARGIN_RATIO)
                            face_crop = frame[max(0,y1-m):min(h,y2+m), max(0,x1-m):min(w,x2+m)]
                            if face_crop.size > 0:
                                current_gender = gender_detector.detect(face_crop)
                        except Exception:
                            pass
            t_pose += time.perf_counter() - t0

            # ── Hands ─────────────────────────────────────────────────────────
            t0 = time.perf_counter()
            hand_results = hand_detector.process(image_rgb)
            kp, hand_detected = extract_keypoints(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness or [],
                action_zone_y,
            )
            t_hand += time.perf_counter() - t0

            if hand_results.multi_hand_landmarks:
                for hand_lm in hand_results.multi_hand_landmarks:
                    wrist = hand_lm.landmark[0]
                    color = (0, 220, 80) if wrist.y < action_zone_y else (50, 50, 230)
                    hand_detector.draw(frame, hand_lm, color=color)

            # ── Model inference ───────────────────────────────────────────────
            t0 = time.perf_counter()
            _predict_counter += 1
            run_model = None
            if _predict_counter >= Config.PREDICT_EVERY:
                _predict_counter = 0
                run_model = model

            result = gesture_recognizer.update(kp, hand_detected, run_model, device='CPU')
            t_model += time.perf_counter() - t0

            predictions = result['predictions']
            word        = result['word']
            confidence  = result['confidence']
            word_display = Config.display(word)

            if result['ready_to_finalize']:
                finalized = result['finalized_word']
                if finalized:
                    sentence.append(finalized)
                    state['words_detected'] = list(sentence)
                    emit("word_detected", {
                        "word": finalized,
                        "word_display": Config.display(finalized),
                        "sentence": Config.display_sentence(sentence),
                        "sentence_raw": list(sentence),
                    })
                gesture_recognizer.reset_gesture()

            # ── Draw overlays ─────────────────────────────────────────────────
            fc = max(video_proc.frame_count, 1)
            y_limit = int(h * action_zone_y)
            zone_color = (0, 220, 80) if pose_detected else (0, 220, 220)
            cv2.line(frame, (0, y_limit), (w, y_limit), zone_color, 2)

            top_idx = _np.argsort(predictions)[::-1][:5]
            bar_x, bar_y, bar_max_w = w - 280, 70, 240
            for rank, idx in enumerate(top_idx):
                conf = float(predictions[idx])
                lbl = Config.ACTIONS[idx]
                bar_w = int(conf * bar_max_w)
                y = bar_y + rank * 30
                color = (0, 200, 80) if rank == 0 else (100, 100, 100)
                cv2.rectangle(frame, (bar_x, y), (bar_x+bar_max_w, y+20), (30,30,30), -1)
                cv2.rectangle(frame, (bar_x, y), (bar_x+bar_w, y+20), color, -1)
                cv2.putText(frame, f"{Config.display(lbl)}: {conf*100:.1f}%",
                            (bar_x+3, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230,230,230), 1)

            if word != 'Blank' and hand_detected:
                cv2.putText(frame, f"{word_display} ({confidence*100:.1f}%)",
                            (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,80), 2)
            else:
                cv2.putText(frame, "Standby", (16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140,140,140), 2)

            cv2.putText(frame, f"FPS: {fps:.1f}", (16, h-16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            progress_pct = min(100, int(fc / total_frames * 100)) if total_frames > 0 else 0

            # ── Đẩy frame mới nhất vào shared buffer (stream thread sẽ lấy) ───
            # Resize nhỏ lại trước khi encode để giảm payload WebSocket
            if w > STREAM_MAX_W:
                scale = STREAM_MAX_W / w
                stream_frame = cv2.resize(frame, (STREAM_MAX_W, int(h * scale)),
                                          interpolation=cv2.INTER_LINEAR)
            else:
                stream_frame = frame

            _, buf = cv2.imencode('.jpg', stream_frame, ENCODE_PARAMS)

            with _stream_lock:
                _latest_frame["buf"] = buf.tobytes()
                _latest_frame["meta"] = {
                    "fps":           round(fps, 1),
                    "word":          word_display if hand_detected and word != 'Blank' else '',
                    "confidence":    round(confidence * 100, 1),
                    "hand_detected": hand_detected,
                    "pose_detected": pose_detected,
                    "is_gesturing":  gesture_recognizer.is_gesturing,
                    "progress":      progress_pct,
                    "frame_count":   fc,
                    "total_frames":  total_frames,
                    "pose_ms":       round((t_pose / fc) * 1000, 2),
                    "hand_ms":       round((t_hand / fc) * 1000, 2),
                    "model_ms":      round((t_model / fc) * 1000, 2),
                    "gender":        current_gender or "unknown",
                }

        _inference_done.set()
        stream_thread.join(timeout=2.0)

        # ── Video finished ────────────────────────────────────────────────────
        state['status'] = 'done'
        final_sentence = Config.display_sentence(sentence) if sentence else ''
        tts_gender = 'female'  # default; overridden by Groq params if available

        if groq_proc and sentence:
            try:
                emit("tts_status", {"status": "PROCESSING"})
                enhanced = groq_proc.generate_sentence(sentence)
                final_sentence = enhanced.get('sentence', final_sentence)
                # Lấy gender từ Groq params để chọn giọng TTS
                params = enhanced.get('params', {})
                raw_gender = params.get('gender', 'nu')
                tts_gender = 'male' if raw_gender == 'nam' else 'female'
                emit("tts_status", {"status": "DONE"})
            except Exception as e:
                logging.error("Groq enhance failed: %s", e)

        fc = video_proc.frame_count or 1
        emit("done", {
            "sentence":       final_sentence,
            "sentence_raw":   list(sentence),
            "words_detected": [Config.display(w) for w in sentence],
            "total_frames":   fc,
            "pose_ms":        round((t_pose / fc) * 1000, 2),
            "hand_ms":        round((t_hand / fc) * 1000, 2),
            "model_ms":       round((t_model / fc) * 1000, 2),
        })

        # ── TTS: đọc câu cuối cùng ───────────────────────────────────────────
        if _tts_available and final_sentence:
            try:
                emit("tts_status", {"status": "SPEAKING"})
                # Chạy TTS trong thread riêng để không block emit done
                def _tts_thread():
                    try:
                        _speak_dynamic(final_sentence, gender=tts_gender)
                    except Exception as e:
                        logging.error("TTS speak failed: %s", e)
                    finally:
                        emit("tts_status", {"status": "DONE"})
                threading.Thread(target=_tts_thread, daemon=True).start()
            except Exception as e:
                logging.error("TTS launch failed: %s", e)

        try:
            with open("result.txt", 'w', encoding='utf-8') as f:
                f.write(final_sentence + '\n')
        except Exception:
            pass

        video_proc.release()
        hand_detector.close()
        pose_detector.close()

    except Exception as e:
        logging.error("Processing worker error: %s", e, exc_info=True)
        emit("error", {"message": str(e)})
        state['status'] = 'error'
    finally:
        # Delete video file after processing
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logging.info("Deleted video: %s", video_path)
        except Exception as e:
            logging.warning("Failed to delete video: %s", e)

        # Cleanup session state after a delay so the client can still poll /status
        def _deferred_cleanup():
            time.sleep(30)
            active_sessions.pop(session_id, None)
            ws_clients.pop(session_id, None)
        threading.Thread(target=_deferred_cleanup, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
        raise HTTPException(400, "Unsupported file type")

    session_id = str(uuid.uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")

    with open(dest_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    active_sessions[session_id] = {
        'status': 'uploaded',
        'video_path': dest_path,
        'sentence': [],
        'words_detected': [],
    }

    file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    logging.info("Uploaded: %s (%.1f MB) → session %s", file.filename, file_size_mb, session_id[:8])

    return JSONResponse({
        "session_id": session_id,
        "filename": file.filename,
        "size_mb": round(file_size_mb, 2),
    })


@app.post("/start/{session_id}")
async def start_processing(session_id: str, use_groq: bool = False):
    if session_id not in active_sessions:
        raise HTTPException(404, "Session not found")

    session = active_sessions[session_id]
    if session.get('status') == 'processing':
        raise HTTPException(409, "Already processing")

    video_path = session['video_path']
    session['stop'] = False

    thread = threading.Thread(
        target=_processing_worker,
        args=(session_id, video_path, use_groq),
        daemon=True,
    )
    thread.start()
    session['thread'] = thread

    return JSONResponse({"message": "Started", "session_id": session_id})


@app.post("/stop/{session_id}")
async def stop_processing(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(404, "Session not found")
    active_sessions[session_id]['stop'] = True
    return JSONResponse({"message": "Stop signal sent"})


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(404, "Session not found")
    s = active_sessions[session_id]
    return JSONResponse({
        "status": s.get('status'),
        "words_detected": s.get('words_detected', []),
        "sentence": s.get('sentence', []),
    })


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket  — stays open for the entire session lifetime
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in ws_clients:
        ws_clients[session_id] = []
    ws_clients[session_id].append(websocket)

    logging.info("WebSocket connected: session %s", session_id[:8])

    try:
        # Keep-alive: send a ping every 20 s and wait for any client message.
        # We do NOT use wait_for with a short timeout — that was what killed the
        # connection.  Instead we rely on WebSocketDisconnect being raised when
        # the browser tab closes.
        while True:
            try:
                # Wait up to 20 s for a message from the client (ping or anything)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=20.0)
                # Client sent something — just ignore it (keep-alive pong etc.)
            except asyncio.TimeoutError:
                # No message from client — send a server-side ping to keep TCP alive
                try:
                    await websocket.send_text(json.dumps({"event": "ping"}))
                except Exception:
                    break   # Socket is gone
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.debug("WebSocket error: %s", e)
    finally:
        if session_id in ws_clients:
            ws_clients[session_id] = [
                ws for ws in ws_clients[session_id] if ws != websocket
            ]
        logging.info("WebSocket disconnected: session %s", session_id[:8])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    URL = "http://localhost:8000"

    print("\n" + "=" * 60)
    print("  ViSignRe Web Server")
    print(f"  Đang mở {URL} ...")
    print("=" * 60 + "\n")

    # Mở browser sau 1.2 giây để server kịp khởi động
    def _open_browser():
        import time as _t
        _t.sleep(1.2)
        webbrowser.open(URL)

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")