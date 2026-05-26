"""Application configuration for ViSignRe."""

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class Config:
    MODEL_PATH = 'models/ViSignRe.tflite'
    VIDEO_PATH = 'data/test_video.mp4'
    OUTPUT_TXT = 'result.txt'
    FONT_PATH = 'arial.ttf'

    WIN_W, WIN_H = 1280, 720
    WINDOW_SIZE = 45
    KP_SIZE = 126

    ACTION_ZONE_FALLBACK = 0.75
    ACTION_ZONE_HIP_RATIO = 0.7
    ACTION_ZONE_SMOOTH = 0.9

    IDLE_THRESHOLD = 15
    VOTE_WINDOW = 20
    TRIM_FRAMES = 3
    MIN_CORE_FRAMES = 2
    DEFAULT_THRESH = 0.70

    MP_CONFIDENCE = 0.5
    FPS_EPSILON = 1e-9
    PREDICT_EVERY = 3

    THRESHOLDS = {
        'Vui': 0.85,
        'Xin_chao': 0.80,
        'Gap': 0.75,
        'Toi': 0.80,
        'Song': 0.60,
        'Ha_Noi': 0.70,
        'Tuoi': 0.75,
        'Ten': 0.70,
        'Hung': 0.70,
        '21': 0.80,
        'Blank': 0.50,
    }

    ACTIONS = np.array([
        '21', 'Blank', 'Gap', 'Ha_Noi', 'Hung',
        'Song', 'Ten', 'Toi', 'Tuoi', 'Vui', 'Xin_chao',
    ])

    LABEL_DISPLAY = {
        '21': '21',
        'Blank': '',
        'Gap': 'Gặp',
        'Ha_Noi': 'Hà Nội',
        'Hung': 'Hưng',
        'Song': 'Sống',
        'Ten': 'Tên',
        'Toi': 'Tôi',
        'Tuoi': 'Tuổi',
        'Vui': 'Vui',
        'Xin_chao': 'Xin chào',
    }

    FONT_SIZE_LG = 28
    FONT_SIZE_MD = 22
    FONT_SIZE_SM = 18
    FLIP_VIDEO = False

    POSE_SKIP_FRAMES = 3
    WINDOW_NAME = 'ViSignRe'
    GESTURE_WINDOW_INDEX = 0
    HAND_WRIST_INDEX = 0
    FACE_REGION_INDICES = list(range(11))
    FACE_ROI_MARGIN_RATIO = 1.0

    @classmethod
    def display(cls, label: str) -> str:
        return cls.LABEL_DISPLAY.get(label, label)

    @classmethod
    def display_sentence(cls, word_list: list) -> str:
        return ' '.join(
            cls.display(w) for w in word_list if cls.LABEL_DISPLAY.get(w, w)
        )

    @classmethod
    def get_llm_constraint(cls) -> str:
        dict_str = ", ".join(f"{k}={v}" for k, v in cls.LABEL_DISPLAY.items() if v)
        return (
            "NGỮ CẢNH: Đây là danh sách các từ vựng được dịch thô từ hệ thống Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL). "
            "Người khiếm thính thường giao tiếp bằng cách ghép các từ khóa chính, lược bỏ giới từ, từ nối và mạo từ.\n\n"
            "NHIỆM VỤ: Hãy đóng vai một biên dịch viên chuyên nghiệp. Nhiệm vụ của bạn là sắp xếp và thêm các từ nối "
            "cần thiết để biến chuỗi từ khóa này thành một câu giao tiếp tiếng Việt hoàn chỉnh, trơn tru, tự nhiên và chuẩn ngữ pháp.\n\n"
            "QUY TẮC BẮT BUỘC:\n"
            f"1. Bảng từ điển tham chiếu: {dict_str}\n"
            "2. Tuyệt đối KHÔNG làm sai lệch, KHÔNG tự ý thay thế, và KHÔNG bỏ sót các từ khóa được cung cấp ở đầu vào.\n"
            "3. Được phép (và nên) thêm các hư từ, từ nối, lượng từ (ví dụ: là, ở, tại, năm nay, và, đang, thì, các, những...) để liên kết câu trôi chảy.\n"
            "4. Tự động nhận diện và viết hoa chữ cái đầu câu, cùng các danh từ riêng (tên người, địa danh) dựa theo ngữ cảnh.\n"
            "5. Tuyệt đối KHÔNG tự suy diễn, bịa đặt thêm thông tin, sự kiện hoặc chi tiết không tồn tại trong tập từ khóa đầu vào.\n"
        )

    LLM_PROMPT_ENHANCE = (
        "{constraint}\n"
        "ĐẦU VÀO HIỆN TẠI: {words}\n"
        "Chỉ trả về đúng một câu kết quả, không giải thích thêm."
    )

    LLM_PROMPT_GENERATE = (
        "{constraint}\n"
        "NHIỆM VỤ THÊM: Tự động thêm dấu phẩy (,), dấu chấm (.), hoặc chấm than (!) hợp lý để hệ thống AI Voice đọc có cảm xúc.\n"
        "ĐẦU VÀO CẦN XỬ LÝ: {words}\n\n"
        "Trả lời NGHIÊM NGẶT bằng MỘT CHUỖI JSON HỢP LỆ (không kèm giải thích, không bọc trong markdown ```json). "
        "Sử dụng đúng cấu trúc sau:\n"
        "{{\n"
        '  "sentence": "<câu tiếng Việt hoàn chỉnh kèm dấu câu>",\n'
        '  "explanation": "<lý do chỉnh sửa hoặc null>",\n'
        '  "params": {{\n'
        '    "gender": "<nam hoặc nu>"\n'
        "  }}\n"
        "}}"
    )

    @classmethod
    def validate(cls) -> bool:
        if not os.path.exists(cls.MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {cls.MODEL_PATH}")
        if not os.path.exists(cls.VIDEO_PATH) and cls.VIDEO_PATH != '0':
            raise FileNotFoundError(f"Video not found: {cls.VIDEO_PATH}")
        return True
