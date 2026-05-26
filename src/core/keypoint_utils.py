"""Hand keypoint extraction relative to the wrist."""

from typing import Tuple
import numpy as np
from config import Config

_KP_PER_HAND = Config.KP_SIZE // 2


def extract_keypoints(
    multi_hand_landmarks,
    multi_handedness,
    action_zone_y: float
) -> Tuple[np.ndarray, bool]:
    """
    Extract and normalize hand keypoints relative to wrist position.

    Keypoint layout:
    - Left hand (indices 0-62): 21 landmarks × 3 (x, y, z)
    - Right hand (indices 63-125): 21 landmarks × 3 (x, y, z)
    All coordinates are relative to wrist position.
    Hands with wrist below action_zone_y are filtered out.

    Args:
        multi_hand_landmarks: List of MediaPipe hand landmark objects
        multi_handedness: List of hand classification (Left/Right) from MediaPipe
        action_zone_y: Y threshold for valid hand region [0, 1]

    Returns:
        Tuple of:
        - kp: Flattened keypoint array (shape: KP_SIZE=126,)
        - hand_detected: True if at least one valid hand in frame
    """
    kp = np.zeros(Config.KP_SIZE)
    hand_detected = False

    if not multi_hand_landmarks:
        return kp, hand_detected

    hand_map = {}
    for hand_lm, hand_cls in zip(multi_hand_landmarks, multi_handedness):
        label = hand_cls.classification[0].label
        hand_map[label] = hand_lm

    slot = {'Left': 0, 'Right': 1}
    n_landmarks = _KP_PER_HAND // 3

    for label, hand_lm in hand_map.items():
        wrist = hand_lm.landmark[0]
        if wrist.y >= action_zone_y:
            continue

        hand_detected = True
        start = slot[label] * _KP_PER_HAND
        kp[start:start + _KP_PER_HAND] = np.array([
            [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
            for lm in hand_lm.landmark[:n_landmarks]
        ]).flatten()

    return kp, hand_detected
