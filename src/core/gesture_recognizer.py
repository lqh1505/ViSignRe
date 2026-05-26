"""Gesture recognition state machine with voting and idle detection."""

from collections import deque, Counter
from typing import Optional, Dict, List, Tuple
import numpy as np
import logging
from config import Config


class GestureRecognizer:
    """
    State machine for recognizing hand gestures from keypoints.
    
    Manages the lifecycle of gesture recognition:
    - Buffers incoming keypoint predictions
    - Detects idle periods to finalize gestures
    - Applies voting and confidence thresholding
    - Prevents duplicate word repetition
    
    Attributes:
        window: Sliding window of recent keypoints (WINDOW_SIZE frames)
        cycle_buffer: List of (word, confidence) tuples during active gesture
        is_gesturing: Whether currently in gesture recognition state
        idle_frames: Count of consecutive frames below threshold
        last_word: Previously finalized word (for duplicate prevention)
    """

    def __init__(self) -> None:
        """Initialize gesture recognizer with empty buffers and idle state."""
        self.window: deque = deque(maxlen=Config.WINDOW_SIZE)
        self.cycle_buffer: List[Tuple[str, float]] = []
        self.is_gesturing: bool = False
        self.idle_frames: int = 0
        self.last_word: Optional[str] = None

        self._cached_predictions: np.ndarray = np.zeros(len(Config.ACTIONS))
        self._cached_word: str = 'Blank'
        self._cached_confidence: float = 0.0
        self._cache_ready: bool = False

    def update(
        self,
        keypoint: np.ndarray,
        hand_detected: bool,
        model: Optional[object] = None,
        predictions_out: Optional[List] = None,
        device: str = 'CPU'
    ) -> Dict:
        """
        Process a single keypoint and return recognition result.

        Args:
            keypoint: Normalized hand keypoint array (shape: KP_SIZE,)
            hand_detected: Whether valid hand was detected in current frame
            model: TFLite model for inference (if None, uses cached predictions)
            predictions_out: Optional list to accumulate prediction arrays
            device: Device placement hint ('CPU' or 'GPU')

        Returns:
            Dictionary with keys:
            - predictions: Model output probabilities (len=ACTIONS)
            - word: Predicted word label
            - confidence: Confidence score [0, 1]
            - ready_to_finalize: True if gesture should be finalized
            - finalized_word: Word to add to sentence (if ready_to_finalize=True)
        """
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

    def _finalize_gesture(self) -> Optional[str]:
        """
        Apply voting and validation to finalize a gesture into a word.

        Steps:
        1. Trim edges (TRIM_FRAMES from each side)
        2. Vote on word using last VOTE_WINDOW frames
        3. Validate: minimum frame count, non-duplicate, confidence check

        Returns:
            Word label if validation passes, else None
        """
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
        word_scores: Dict[str, float] = {}
        word_counts: Dict[str, int] = {}
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

    def reset_gesture(self) -> None:
        """Reset gesture state and buffers."""
        self.is_gesturing = False
        self.cycle_buffer.clear()
        self.idle_frames = 0

    def get_buffer_stats(self) -> Dict:
        """Return current internal state for debugging."""
        return {
            'buffer_size': len(self.cycle_buffer),
            'idle_frames': self.idle_frames,
            'is_gesturing': self.is_gesturing,
            'last_word': self.last_word,
        }
