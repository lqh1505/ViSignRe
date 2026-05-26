"""ViSignRe runtime: video capture, recognition, and optional LLM/TTS post-processing."""

import os
import sys
import warnings
import threading

class SuppressStderr:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fd = os.dup(2)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self.save_fd, 2)
        os.close(self.null_fd)
        os.close(self.save_fd)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

with SuppressStderr():
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    import cv2
    import numpy as np
    from collections import Counter
    import time as _time
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)

    from config import Config
    from src.core.gesture_recognizer import GestureRecognizer
    from src.core.keypoint_utils import extract_keypoints
    from src.core.mediapipe_handlers import HandDetector, PoseDetector
    from src.processors.video_processor import VideoProcessor
    from src.processors.groq_processor import GroqProcessor
    from src.processors.gender_detector import GenderDetector
    from src.ui.renderers import VietnameseRenderer, draw_confidence_bars
    from src.utils.utils import FPSCounter, C_GREEN, C_RED, C_YELLOW, C_GRAY, C_WHITE
    from src.processors.tts_processor import speak_dynamic

dev_info = {'tf_version': tf.__version__}


class DummyTensor:
    def __init__(self, data):
        self.data = data

    def numpy(self):
        return self.data


class TFLiteModelWrapper:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, input_data, training=False):
        if hasattr(input_data, 'numpy'):
            input_data = input_data.numpy()
        input_data = input_data.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return DummyTensor(output_data)

    def predict(self, input_data, verbose=0):
        return self.__call__(input_data)


def main():
    print("\n" + "=" * 60)
    print(" ViSignRe | Initializing")
    print("=" * 60 + "\n")

    logging.info("TensorFlow %s", dev_info['tf_version'])
    logging.info("Execution mode: TFLite CPU")

    try:
        Config.validate()
    except FileNotFoundError as e:
        logging.error("Configuration error: %s", e)
        sys.exit(1)

    try:
        tflite_path = Config.MODEL_PATH.replace('.keras', '.tflite').replace('.h5', '.tflite')
        if not os.path.exists(tflite_path):
            tflite_path = 'models/ViSignRe.tflite'

        logging.info("Loading model: %s", tflite_path)

        with SuppressStderr():
            model = TFLiteModelWrapper(tflite_path)
            dummy = np.zeros((1, Config.WINDOW_SIZE, Config.KP_SIZE), dtype=np.float32)
            _ = model(dummy, training=False)

        logging.info("Model ready")
    except Exception as e:
        logging.error("Model initialization failed: %s", e)
        sys.exit(1)

    try:
        with SuppressStderr():
            hand_detector = HandDetector()
            pose_detector = PoseDetector()
            gender_detector = GenderDetector()
            current_gender = None

            dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            hand_detector.process(dummy_img)
            pose_detector.process(dummy_img)

        video_processor = VideoProcessor(Config.VIDEO_PATH)
        renderer = VietnameseRenderer(Config.FONT_PATH, {
            'lg': Config.FONT_SIZE_LG,
            'md': Config.FONT_SIZE_MD,
            'sm': Config.FONT_SIZE_SM,
        })
        gesture_recognizer = GestureRecognizer()
        fps_counter = FPSCounter()

        groq_processor = None
        try:
            groq_processor = GroqProcessor()
            logging.info("Groq service: connected")
        except ValueError as e:
            logging.warning("Groq service: disabled (%s)", e)

    except Exception as e:
        logging.error("Component initialization failed: %s", e)
        sys.exit(1)

    cv2.namedWindow('ViSignRe', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ViSignRe', Config.WIN_W, Config.WIN_H)

    sentence = []
    action_zone_y = Config.ACTION_ZONE_FALLBACK
    pose_detected = False
    flip_video = Config.FLIP_VIDEO

    _t_pose = _t_hand = _t_model = 0.0
    _pose_skip = 0

    print("\n[INFO] Stream started")
    print("[INFO] Keys: Q = quit, F = flip video\n")

    try:
        while video_processor.is_open():
            ret, frame, h, w = video_processor.read_frame()
            if not ret:
                break

            if flip_video:
                frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fps = fps_counter.tick()

            _t0 = _time.perf_counter()
            _pose_skip += 1
            if _pose_skip >= 3:
                _pose_skip = 0
                pose_results = pose_detector.process(image_rgb)
                pose_detected = False

                if pose_results.pose_landmarks:
                    zone_y_new = pose_detector.get_action_zone_y(pose_results.pose_landmarks)
                    if zone_y_new is not None:
                        action_zone_y = (
                            Config.ACTION_ZONE_SMOOTH * action_zone_y
                            + (1 - Config.ACTION_ZONE_SMOOTH) * zone_y_new
                        )
                        pose_detected = True

                    if current_gender is None:
                        try:
                            lm = pose_results.pose_landmarks.landmark
                            face_x = [lm[i].x for i in range(11)]
                            face_y = [lm[i].y for i in range(11)]

                            x_min, x_max = int(min(face_x) * w), int(max(face_x) * w)
                            y_min, y_max = int(min(face_y) * h), int(max(face_y) * h)

                            margin_x = int((x_max - x_min) * 0.2)
                            margin_y = int((y_max - y_min) * 0.2)

                            fx1, fx2 = max(0, x_min - margin_x), min(w, x_max + margin_x)
                            fy1, fy2 = max(0, y_min - margin_y), min(h, y_max + margin_y)

                            face_crop = frame[fy1:fy2, fx1:fx2]
                            if face_crop.size > 0:
                                current_gender = gender_detector.detect(face_crop)
                                logging.info("Gender locked: %s", current_gender.upper())
                        except Exception as e:
                            logging.debug("Face crop skipped: %s", e)

            _t_pose += _time.perf_counter() - _t0
            _t0 = _time.perf_counter()
            hand_results = hand_detector.process(image_rgb)
            kp, hand_detected = extract_keypoints(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness or [],
                action_zone_y,
            )
            _t_hand += _time.perf_counter() - _t0

            if hand_results.multi_hand_landmarks:
                for hand_lm in hand_results.multi_hand_landmarks:
                    wrist = hand_lm.landmark[0]
                    color = C_GREEN if wrist.y < action_zone_y else C_RED
                    hand_detector.draw(frame, hand_lm, color=color)

            _t0 = _time.perf_counter()
            result = gesture_recognizer.update(
                kp,
                hand_detected,
                model if video_processor.should_predict() else None,
                device='CPU',
            )
            _t_model += _time.perf_counter() - _t0

            predictions = result['predictions']
            word = result['word']
            confidence = result['confidence']
            word_display = Config.display(word)

            if result['ready_to_finalize']:
                finalized_word = result['finalized_word']
                if finalized_word:
                    sentence.append(finalized_word)
                    print(
                        f"[DETECT] {finalized_word} -> {Config.display(finalized_word)}"
                    )
                gesture_recognizer.reset_gesture()

            if len(predictions) > 0:
                frame = draw_confidence_bars(frame, predictions, Config.ACTIONS)

            y_limit = int(h * action_zone_y)
            zone_color = C_GREEN if pose_detected else C_YELLOW
            cv2.line(frame, (0, y_limit), (w, y_limit), zone_color, 2)

            texts_to_draw = []
            if not hand_detected or word == 'Blank':
                texts_to_draw.append(("Standby", (16, 16), C_GRAY, 'md'))
            else:
                texts_to_draw.append(
                    (f"{word_display} ({confidence * 100:.1f}%)", (16, 16), C_GREEN, 'md')
                )

            if gesture_recognizer.is_gesturing and gesture_recognizer.cycle_buffer:
                current_top = Counter(
                    [w for w, c in gesture_recognizer.cycle_buffer]
                ).most_common(1)[0][0]
                texts_to_draw.append(
                    (f"Observing: {Config.display(current_top)}", (16, 48), C_YELLOW, 'sm')
                )

            sentence_display = Config.display_sentence(sentence) if sentence else '...'
            texts_to_draw.append((sentence_display, (16, h - 52), C_WHITE, 'lg'))

            frame = renderer.put_text_multi(frame, texts_to_draw)
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (w - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_WHITE, 2,
            )
            cv2.rectangle(frame, (0, h - 65), (w, h), (20, 20, 20), -1)

            cv2.imshow('ViSignRe', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('f'):
                flip_video = not flip_video

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    except Exception as e:
        logging.error("Runtime error: %s", e, exc_info=True)

    finally:
        fc = video_processor.frame_count or 1
        total_ms = (_t_pose + _t_hand + _t_model) / fc * 1000
        print("\n" + "=" * 30 + " PERFORMANCE " + "=" * 30)
        print(f" Pose:   {(_t_pose / fc) * 1000:.2f} ms/frame")
        print(f" Hands:  {(_t_hand / fc) * 1000:.2f} ms/frame")
        print(f" Model:  {(_t_model / fc) * 1000:.2f} ms/frame")
        print(f" Total:  {total_ms:.2f} ms/frame")
        print(f" Max FPS: {1000 / max(total_ms, 1):.1f}")
        print("=" * 81 + "\n")

        video_processor.release()
        with SuppressStderr():
            hand_detector.close()
            pose_detector.close()
        cv2.destroyAllWindows()

        result_text = (
            Config.display_sentence(sentence) if sentence else "No output generated"
        )

        if groq_processor and sentence:
            def process_ai_and_speak():
                final_result = result_text
                try:
                    final_enhanced = groq_processor.generate_sentence(sentence)
                    final_result = final_enhanced.get('sentence', result_text)
                    explanation = final_enhanced.get('explanation')
                    tone_params = final_enhanced.get('params', {})
                    final_gender = tone_params.get('gender', current_gender or "nam")

                    logging.info("LLM output: %s", final_result)
                    if explanation:
                        logging.info("LLM note: %s", explanation)
                    logging.info("TTS gender: %s", final_gender.upper())

                    speak_dynamic(final_result, tone_params, gender=final_gender)
                except Exception as e:
                    logging.error("Post-processing failed: %s", e)
                finally:
                    print(f"\n[RESULT] {final_result}\n")
                    video_processor.save_result(final_result)

            ai_thread = threading.Thread(target=process_ai_and_speak)
            ai_thread.start()
            ai_thread.join()
        else:
            print(f"[RESULT] {result_text}\n")
            video_processor.save_result(result_text)


if __name__ == '__main__':
    main()
