"""Record sign-language keypoint sequences and augment the dataset."""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import cv2
import mediapipe as mp
import numpy as np
import time
import glob
import random
from src.utils.sequence_utils import normalize_sequence, any_hand_active

WORD_TO_RECORD = "21"
NUM_SAMPLES = 30
TARGET_SAMPLES = 120
SEQUENCE_LENGTH = 45
KP_SIZE = 126

DATA_PATH = os.path.join("data", "dataset_words", WORD_TO_RECORD)
os.makedirs(DATA_PATH, exist_ok=True)

ACTION_ZONE_Y = 0.75
PREP_TIME = 2.0
SCALE_RANGE = (0.9, 1.1)
NOISE_LEVEL = 0.005
ZERO_THRESHOLD = 1e-6
MAX_ZERO_RATIO = 0.80

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands_sol = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
pose_sol = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0,
)


def draw_text(frame, text, pos, color=(255, 255, 255), scale=1.0, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_zone_line(frame, action_zone_y, color=(0, 200, 80)):
    h, w = frame.shape[:2]
    y = int(h * action_zone_y)
    cv2.line(frame, (0, y), (w, y), color, 2)
    draw_text(frame, f"Action zone {action_zone_y:.2f}", (8, y - 6), color, 0.5, 1)


def next_available_index(data_path: str) -> int:
    idx = 0
    while os.path.exists(os.path.join(data_path, f"sample_{idx}.npy")):
        idx += 1
    return idx


def update_action_zone(action_zone_y: float, pose_results) -> float:
    if not pose_results.pose_landmarks:
        return action_zone_y
    lm = pose_results.pose_landmarks.landmark
    PL = mp_pose.PoseLandmark
    pts = [PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER, PL.LEFT_HIP, PL.RIGHT_HIP]
    if all(lm[p].visibility > 0.5 for p in pts):
        shoulder_y = (lm[PL.LEFT_SHOULDER].y + lm[PL.RIGHT_SHOULDER].y) / 2
        hip_y = (lm[PL.LEFT_HIP].y + lm[PL.RIGHT_HIP].y) / 2
        target = shoulder_y + (hip_y - shoulder_y) * 0.7
        return 0.9 * action_zone_y + 0.1 * target
    return action_zone_y


def extract_keypoints(hand_results, action_zone_y: float) -> tuple:
    kp = np.zeros(KP_SIZE)
    hand_detected = False
    if not hand_results.multi_hand_landmarks:
        return kp, hand_detected

    hand_map = {
        hand_cls.classification[0].label: hand_lm
        for hand_lm, hand_cls in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness or [],
        )
    }
    slot = {'Left': 0, 'Right': 1}

    for label, hand_lm in hand_map.items():
        wrist = hand_lm.landmark[0]
        if wrist.y >= action_zone_y:
            continue
        hand_detected = True
        start = slot[label] * 63
        kp[start : start + 63] = np.array([
            [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
            for lm in hand_lm.landmark
        ]).flatten()
    return kp, hand_detected


def augment_scale_noise(data: np.ndarray) -> np.ndarray:
    aug = np.copy(data)
    for i in range(len(aug)):
        if not any_hand_active(aug[i]):
            continue
        for start in (0, 63):
            slot = aug[i, start:start + 63]
            if np.linalg.norm(slot) > ZERO_THRESHOLD:
                slot *= np.random.uniform(*SCALE_RANGE)
                slot += np.random.normal(0, NOISE_LEVEL, slot.shape)
                aug[i, start:start + 63] = slot
    return aug


def augment_flip(data: np.ndarray) -> np.ndarray:
    aug = np.copy(data)
    for i in range(len(aug)):
        if not any_hand_active(aug[i]):
            continue
        left, right = aug[i, 0:63].copy(), aug[i, 63:126].copy()
        left[0::3] *= -1
        right[0::3] *= -1
        aug[i, 0:63], aug[i, 63:126] = right, left
    return aug


def quality_check(frames: list) -> float:
    empty_count = sum(1 for kp in frames if not any_hand_active(kp))
    return empty_count / len(frames)


def run_collection():
    file_idx = next_available_index(DATA_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not available (index 0)")
        sys.exit(1)

    cv2.namedWindow('ViSignRe Collection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ViSignRe Collection', 1024, 768)

    action_zone_y = ACTION_ZONE_Y
    sample = 0
    state = 'idle'
    prep_start = 0.0
    frames_buffer = []

    print("\n" + "=" * 55)
    print(f" Class: {WORD_TO_RECORD} | Target: {NUM_SAMPLES} recordings")
    print(f" Output: {DATA_PATH}")
    print("=" * 55)
    print("[INFO] SPACE = start sample | Q = quit\n")

    while sample < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Webcam disconnected")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = pose_sol.process(rgb)
        action_zone_y = update_action_zone(action_zone_y, pose_results)
        hand_results = hands_sol.process(rgb)
        kp, hand_detected = extract_keypoints(hand_results, action_zone_y)

        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                wrist = hand_lm.landmark[0]
                color = (0, 220, 80) if wrist.y < action_zone_y else (0, 80, 220)
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        draw_zone_line(frame, action_zone_y)

        if state == 'idle':
            draw_text(
                frame,
                f"[{WORD_TO_RECORD}] Sample {sample + 1}/{NUM_SAMPLES} - press SPACE",
                (16, 32), (0, 255, 255), 0.7,
            )
            draw_text(
                frame,
                "Keep hands inside the green zone",
                (16, 64), (200, 200, 200), 0.55,
            )
        elif state == 'prep':
            left = max(0.0, PREP_TIME - (time.time() - prep_start))
            draw_text(frame, f"Prepare... {left:.1f}s", (16, 32), (0, 200, 255), 0.9)
        elif state == 'recording':
            frames_buffer.append(kp.copy())
            draw_text(
                frame,
                f"Recording {len(frames_buffer)}/{SEQUENCE_LENGTH}",
                (16, 32), (0, 80, 255), 0.9,
            )
            if len(frames_buffer) >= SEQUENCE_LENGTH:
                ratio = quality_check(frames_buffer)
                if ratio > MAX_ZERO_RATIO:
                    print(
                        f"[SKIP] Sample {sample + 1}: too many empty frames "
                        f"({ratio * 100:.0f}%)"
                    )
                    state = 'idle'
                    frames_buffer = []
                    continue

                seq = normalize_sequence(
                    np.array(frames_buffer, dtype=np.float32), SEQUENCE_LENGTH
                )
                out_path = os.path.join(DATA_PATH, f"sample_{file_idx}.npy")
                np.save(out_path, seq)
                print(f"[OK] Saved {out_path}")
                file_idx += 1
                sample += 1
                state = 'idle'
                frames_buffer = []

        cv2.imshow('ViSignRe Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] Collection stopped")
            break
        if key == ord(' ') and state == 'idle':
            prep_start = time.time()
            state = 'prep'
        elif state == 'prep' and (time.time() - prep_start) >= PREP_TIME:
            frames_buffer = []
            state = 'recording'

    cap.release()
    cv2.destroyAllWindows()
    hands_sol.close()
    pose_sol.close()
    return sample


def run_augmentation():
    existing_files = glob.glob(os.path.join(DATA_PATH, "*.npy"))
    if not existing_files:
        print("[WARN] No .npy files found; skipping augmentation")
        return

    print(f"\n[INFO] Augmenting {len(existing_files)} -> {TARGET_SAMPLES} samples")
    aug_idx = next_available_index(DATA_PATH)

    while len(existing_files) < TARGET_SAMPLES:
        source_file = random.choice(existing_files)
        data = np.load(source_file)
        if random.random() < 0.5:
            new_data = augment_scale_noise(data)
        else:
            new_data = augment_scale_noise(augment_flip(data))

        np.save(os.path.join(DATA_PATH, f"sample_{aug_idx}.npy"), new_data)
        aug_idx += 1
        existing_files = glob.glob(os.path.join(DATA_PATH, "*.npy"))

    print(f"[OK] Total samples in {DATA_PATH}: {len(existing_files)}")


if __name__ == '__main__':
    recorded = run_collection()
    if recorded > 0:
        run_augmentation()
    else:
        print("[WARN] No samples recorded; augmentation skipped")
