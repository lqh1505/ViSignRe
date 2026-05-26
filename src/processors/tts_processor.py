"""Offline text-to-speech via VieNeu-TTS."""

import os
import logging
import uuid
import threading
from playsound import playsound

logging.getLogger("vieneu").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models", "vieneu_cache")
AUDIO_TMP_DIR = os.path.join(os.getcwd(), "models", "vieneu_cache", "tmp_audio")

tts_engine = None
voice_female = None
voice_male = None
_model_ready = threading.Event()

def _cleanup_old_audio():
    if os.path.exists(AUDIO_TMP_DIR):
        try:
            files = os.listdir(AUDIO_TMP_DIR)
            deleted_count = 0
            for f in files:
                if f.startswith("output_") and f.endswith(".wav"):
                    try:
                        os.remove(os.path.join(AUDIO_TMP_DIR, f))
                        deleted_count += 1
                    except OSError:
                        pass
            if deleted_count > 0:
                print(f"[INFO] [TTS] Cleaned up {deleted_count} temporary audio files.")
        except Exception as e:
            print(f"[WARNING] [TTS] Cleanup failed: {e}")

def _load_model_bg():
    global tts_engine, voice_female, voice_male
    try:
        os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
        _cleanup_old_audio()

        print("[INFO] [TTS] Loading VieNeu model...")
        from vieneu import Vieneu

        tts_engine = Vieneu(mode="turbo")
        voice_female = tts_engine.get_preset_voice("Bích Ngọc (Nữ - Miền Bắc)")
        voice_male = tts_engine.get_preset_voice("Phạm Tuyên (Nam - Miền Bắc)")
        
        print("[INFO] [TTS] Model ready.")
        _model_ready.set()
    except Exception as e:
        print(f"[ERROR] [TTS] Initialization failed: {e}")

threading.Thread(target=_load_model_bg, daemon=True).start()

def speak_dynamic(text: str, emotion_params: dict = None, gender: str = "female"):
    if not _model_ready.is_set():
        logging.warning("[TTS] Waiting for model to load...")
        _model_ready.wait()

    if tts_engine is None:
        logging.error("[TTS] Engine not initialized.")
        return

    try:
        selected_voice = voice_male if gender.lower() == "male" else voice_female
        logging.info("[TTS] Speaking (%s)", gender.upper())

        audio = tts_engine.infer(text=text, voice=selected_voice)
        
        os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
        filename = os.path.join(AUDIO_TMP_DIR, f"output_{uuid.uuid4().hex[:8]}.wav")
        tts_engine.save(audio, filename)
        
        playsound(filename)

        try:
            if os.path.exists(filename):
                os.remove(filename)
        except OSError:
            pass
            
    except Exception as e:
        logging.error("[TTS] Playback failed: %s", e)