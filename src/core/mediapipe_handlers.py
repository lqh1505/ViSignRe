"""MediaPipe hand and pose detection wrappers."""

from typing import Optional, Tuple
import mediapipe as mp
import numpy as np
import logging
from config import Config


class HandDetector:
    """Wrapper for MediaPipe hand detection and drawing."""

    def __init__(self) -> None:
        """Initialize hand detector with MediaPipe Hands model."""
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=Config.MP_CONFIDENCE,
            min_tracking_confidence=Config.MP_CONFIDENCE,
            max_num_hands=2,
        )
        logging.info("HandDetector ready")

    def process(self, rgb_image: np.ndarray):
        """
        Process image and detect hand landmarks.

        Args:
            rgb_image: RGB image array (numpy)

        Returns:
            MediaPipe hands result object with multi_hand_landmarks
        """
        return self.hands.process(rgb_image)

    def draw(self, frame: np.ndarray, hand_landmarks, color: Tuple = (0, 255, 0)) -> np.ndarray:
        """
        Draw hand landmarks and connections on frame.

        Args:
            frame: Image to draw on (modified in-place)
            hand_landmarks: MediaPipe hand landmarks
            color: RGB color tuple

        Returns:
            Modified frame
        """
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
            self.mp_draw.DrawingSpec(color=color, thickness=2),
        )
        return frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.hands:
            self.hands.close()
            logging.info("HandDetector closed")


class PoseDetector:
    """Wrapper for MediaPipe pose detection to determine action zone."""

    def __init__(self) -> None:
        """Initialize pose detector with MediaPipe Pose model."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=Config.MP_CONFIDENCE,
            min_tracking_confidence=Config.MP_CONFIDENCE,
            model_complexity=0,
        )
        logging.info("PoseDetector ready")

    def process(self, rgb_image: np.ndarray):
        """
        Process image and detect pose landmarks.

        Args:
            rgb_image: RGB image array (numpy)

        Returns:
            MediaPipe pose result object with pose_landmarks
        """
        return self.pose.process(rgb_image)

    def get_action_zone_y(self, pose_landmarks) -> Optional[float]:
        """
        Calculate Y coordinate for action zone (hand validity region).

        Action zone is defined as: shoulder_y + (hip_y - shoulder_y) * ACTION_ZONE_HIP_RATIO
        Hands above this line are valid; below are ignored.

        Args:
            pose_landmarks: MediaPipe pose landmarks

        Returns:
            Y coordinate [0, 1] if valid, else None
        """
        lm = pose_landmarks.landmark
        PL = self.mp_pose.PoseLandmark

        pts = [PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER, PL.LEFT_HIP, PL.RIGHT_HIP]
        if not all(lm[p].visibility > Config.MP_CONFIDENCE for p in pts):
            return None

        shoulder_y = (lm[PL.LEFT_SHOULDER].y + lm[PL.RIGHT_SHOULDER].y) / 2
        hip_y = (lm[PL.LEFT_HIP].y + lm[PL.RIGHT_HIP].y) / 2
        return shoulder_y + (hip_y - shoulder_y) * Config.ACTION_ZONE_HIP_RATIO

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.pose:
            self.pose.close()
            logging.info("PoseDetector closed")
