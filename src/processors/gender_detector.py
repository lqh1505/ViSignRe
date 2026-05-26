"""Face-based gender estimation for TTS voice selection."""

import cv2
import logging
import os


class GenderDetector:
    def __init__(
        self,
        prototxt_path="models/gender.prototxt",
        model_path="models/gender.caffemodel",
    ):
        self.net = None
        if os.path.exists(prototxt_path) and os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                logging.info("Gender model loaded")
            except Exception as e:
                logging.error("Gender model load failed: %s", e)
        else:
            logging.warning("Gender model not found; defaulting to male voice")

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.GENDER_LIST = ['male', 'female']

    def detect(self, face_img):
        if self.net is None or face_img is None or face_img.size == 0:
            return "male"

        try:
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False,
            )
            self.net.setInput(blob)
            preds = self.net.forward()
            return self.GENDER_LIST[preds[0].argmax()]
        except Exception as e:
            logging.error("Gender detection failed: %s", e)
            return "male"
