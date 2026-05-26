"""Vietnamese text rendering and confidence bar overlay."""

import cv2
import numpy as np
import logging
from PIL import ImageFont, ImageDraw, Image


class VietnameseRenderer:
    def __init__(self, font_path: str, sizes: dict):
        self.fonts = {}
        for name, size in sizes.items():
            try:
                self.fonts[name] = ImageFont.truetype(font_path, size)
            except Exception as e:
                logging.warning("Font load failed (%s): %s", font_path, e)
                self.fonts[name] = ImageFont.load_default()

    def put_text(self, frame: np.ndarray, text: str, pos: tuple,
                 color: tuple, font: str = 'md') -> np.ndarray:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_obj = self.fonts.get(font, list(self.fonts.values())[0])
        rgb_color = (color[2], color[1], color[0])
        draw.text(pos, text, font=font_obj, fill=rgb_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def put_text_multi(self, frame: np.ndarray, items: list) -> np.ndarray:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        for text, pos, color, font in items:
            font_obj = self.fonts.get(font, list(self.fonts.values())[0])
            rgb_color = (color[2], color[1], color[0])
            draw.text(pos, text, font=font_obj, fill=rgb_color)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_confidence_bars(frame: np.ndarray, predictions: np.ndarray,
                         actions: np.ndarray, top_n: int = 5) -> np.ndarray:
    top_idx = np.argsort(predictions)[::-1][:top_n]
    bar_x = frame.shape[1] - 320
    bar_y = 80
    bar_max_w = 280

    for rank, idx in enumerate(top_idx):
        conf = predictions[idx]
        label = actions[idx]
        color = (0, 255, 0) if rank == 0 else (180, 180, 180)
        bar_w = int(conf * bar_max_w)
        y = bar_y + rank * 36

        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_w, y + 22), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + 22), color, -1)
        cv2.putText(
            frame, f"{label}: {conf * 100:.1f}%",
            (bar_x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )

    return frame
