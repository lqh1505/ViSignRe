"""Gesture recognition state machine with voting and idle detection."""

from collections import deque, Counter
import numpy as np
import logging
from config import Config


class GestureRecognizer:
    def __init__(self):
        self.window = deque(maxlen=Config.WINDOW_SIZE)
        self.cycle_buffer = []
        self.is_gesturing = False
        self.idle_frames = 0
        self.last_word = None

        self._cached_predictions = np.zeros(len(Config.ACTIONS))
        self._cached_word = 'Blank'
        self._cached_confidence = 0.0
        self._cache_ready = False

    def update(self, keypoint: np.ndarray, hand_detected: bool, model,
               predictions_out=None, device: str = 'CPU') -> dict:
        self.window.append(keypoint)

        result = {
            'predictions': np.zeros(len(Config.ACTIONS)),
            'word': 'Blank',
            'confidence': 0.0,
            'ready_to_finalize': False,
            'finalized_word': None,
        }

        if len(self.window) != Config.WINDOW_SIZE:
            return result

        if model is not None:
            import tensorflow as tf
            x = np.expand_dims(list(self.window), axis=0).astype(np.float32)
            with tf.device(f'/{device}:0'):
                preds = model(x, training=False).numpy()[0]

            if predictions_out is not None:
                predictions_out.append(preds)

            idx = int(np.argmax(preds))
            word = Config.ACTIONS[idx]
            confidence = float(preds[idx])

            self._cached_predictions = preds
            self._cached_word = word
            self._cached_confidence = confidence
            self._cache_ready = True
        else:
            if not self._cache_ready:
                return result
            preds = self._cached_predictions
            word = self._cached_word
            confidence = self._cached_confidence

        result['predictions'] = preds
        result['word'] = word
        result['confidence'] = confidence

        req_thresh = Config.THRESHOLDS.get(word, Config.DEFAULT_THRESH)

        if not hand_detected or word == 'Blank' or confidence < req_thresh:
            self.idle_frames += 1
        else:
            self.idle_frames = 0
            if not self.is_gesturing:
                self.is_gesturing = True
                self.cycle_buffer.clear()
            self.cycle_buffer.append((word, confidence))

        if self.idle_frames >= Config.IDLE_THRESHOLD and self.is_gesturing:
            result['ready_to_finalize'] = True
            finalized = self._finalize_gesture()
            if finalized:
                result['finalized_word'] = finalized

        return result

    def _finalize_gesture(self) -> str:
        if not self.cycle_buffer:
            return None

        if len(self.cycle_buffer) > Config.TRIM_FRAMES * 2:
            core_frames = self.cycle_buffer[Config.TRIM_FRAMES:-Config.TRIM_FRAMES]
        else:
            core_frames = self.cycle_buffer

        if not core_frames:
            logging.warning("Rejected: empty buffer after trim")
            return None

        vote_frames = core_frames[-Config.VOTE_WINDOW:]
        word_scores = {}
        word_counts = {}
        for w, c in vote_frames:
            word_scores[w] = word_scores.get(w, 0) + c
            word_counts[w] = word_counts.get(w, 0) + 1

        best_word = max(word_scores, key=word_scores.get)
        frame_count = word_counts[best_word]

        if frame_count < Config.MIN_CORE_FRAMES:
            logging.warning(
                "Rejected: %s (%d/%d frames)",
                best_word, frame_count, Config.MIN_CORE_FRAMES,
            )
            return None

        if best_word == self.last_word:
            logging.warning("Rejected: duplicate word '%s'", best_word)
            return None

        confs = [c for w, c in vote_frames if w == best_word]
        avg_conf = np.mean(confs)
        logging.info(
            "Finalized: %-12s | conf %.1f%% | frames %d",
            best_word, avg_conf * 100, frame_count,
        )

        self.last_word = best_word
        return best_word

    def reset_gesture(self):
        self.is_gesturing = False
        self.cycle_buffer.clear()
        self.idle_frames = 0

    def get_buffer_stats(self) -> dict:
        return {
            'buffer_size': len(self.cycle_buffer),
            'idle_frames': self.idle_frames,
            'is_gesturing': self.is_gesturing,
            'last_word': self.last_word,
        }
