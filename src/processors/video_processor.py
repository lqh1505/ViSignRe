"""Video capture and result file I/O."""

import cv2
import logging
from config import Config


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_source = int(video_path) if isinstance(video_path, str) and video_path.isdigit() else video_path
        self.cap = cv2.VideoCapture(self.video_source)
        self.frame_count = 0
        self._predict_counter = 0
        self._open()

    def _open(self):
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        logging.info("Video source: %s", self.video_path)

    def read_frame(self) -> tuple:
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            h, w = frame.shape[:2]
            return ret, frame, h, w
        return ret, None, 0, 0

    def should_predict(self) -> bool:
        self._predict_counter += 1
        if self._predict_counter >= Config.PREDICT_EVERY:
            self._predict_counter = 0
            return True
        return False

    def is_open(self) -> bool:
        return self.cap and self.cap.isOpened()

    def release(self):
        if self.cap:
            self.cap.release()
            logging.info("Frames processed: %d", self.frame_count)

    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0.0

    def get_frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else 0

    def save_result(self, text: str, output_path: str = None) -> bool:
        output_path = output_path or Config.OUTPUT_TXT
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text + '\n')
            logging.info("Result saved: %s", output_path)
            return True
        except IOError as e:
            logging.error("Failed to save result: %s", e)
            return False
