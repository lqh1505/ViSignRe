"""MediaPipe hand and pose detection wrappers."""

import mediapipe as mp
import logging
from config import Config


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=Config.MP_CONFIDENCE,
            min_tracking_confidence=Config.MP_CONFIDENCE,
            max_num_hands=2,
        )
        logging.info("HandDetector ready")

    def process(self, rgb_image):
        return self.hands.process(rgb_image)

    def draw(self, frame, hand_landmarks, color=(0, 255, 0)):
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
            self.mp_draw.DrawingSpec(color=color, thickness=2),
        )
        return frame

    def close(self):
        if self.hands:
            self.hands.close()
            logging.info("HandDetector closed")


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=Config.MP_CONFIDENCE,
            min_tracking_confidence=Config.MP_CONFIDENCE,
            model_complexity=0,
        )
        logging.info("PoseDetector ready")

    def process(self, rgb_image):
        return self.pose.process(rgb_image)

    def get_action_zone_y(self, pose_landmarks) -> float:
        lm = pose_landmarks.landmark
        PL = self.mp_pose.PoseLandmark

        pts = [PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER, PL.LEFT_HIP, PL.RIGHT_HIP]
        if not all(lm[p].visibility > Config.MP_CONFIDENCE for p in pts):
            return None

        shoulder_y = (lm[PL.LEFT_SHOULDER].y + lm[PL.RIGHT_SHOULDER].y) / 2
        hip_y = (lm[PL.LEFT_HIP].y + lm[PL.RIGHT_HIP].y) / 2
        return shoulder_y + (hip_y - shoulder_y) * Config.ACTION_ZONE_HIP_RATIO

    def close(self):
        if self.pose:
            self.pose.close()
            logging.info("PoseDetector closed")
