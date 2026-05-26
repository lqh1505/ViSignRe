"""Hand keypoint extraction relative to the wrist."""

import numpy as np
from config import Config

_KP_PER_HAND = Config.KP_SIZE // 2


def extract_keypoints(multi_hand_landmarks, multi_handedness,
                      action_zone_y: float) -> tuple:
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
