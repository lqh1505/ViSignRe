import cv2
import numpy as np
import os
import time

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),         # Ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),    # Ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ngón áp út
    (13, 17), (17, 18), (18, 19), (19, 20), # Ngón út
    (0, 17), (5, 9), (9, 13), (13, 17)      # Nối lòng bàn tay (Palm)
]

def play_npy_centered(npy_path):
    print(f"Đang phát lại file: {npy_path}")
    data = np.load(npy_path) # Kích thước (45, 126)
    width, height = 800, 600
    
    # --- THUẬT TOÁN TỰ ĐỘNG CANH GIỮA VÀ PHÓNG TO ---
    # 1. Gộp tất cả 45 frame lại để tìm "khung chứa" lớn nhất của toàn bộ động tác
    all_points = data.reshape(-1, 3)
    valid_points = all_points[np.sum(np.abs(all_points), axis=1) > 0]
    
    if len(valid_points) == 0:
        print("Dữ liệu trống, không có tay!")
        return
        
    # 2. Tìm tọa độ giới hạn (Nhỏ nhất và Lớn nhất)
    min_x, max_x = np.min(valid_points[:, 0]), np.max(valid_points[:, 0])
    min_y, max_y = np.min(valid_points[:, 1]), np.max(valid_points[:, 1])
    
    # 3. Tính tọa độ TÂM của toàn bộ động tác
    seq_cx = (min_x + max_x) / 2
    seq_cy = (min_y + max_y) / 2
    
    # 4. Tính Tỷ lệ phóng to (Sao cho khung xương vừa bằng 60% màn hình)
    range_x = max_x - min_x if max_x - min_x > 0 else 1
    range_y = max_y - min_y if max_y - min_y > 0 else 1
    scale = min(width / range_x, height / range_y) * 0.6
    # --------------------------------------------------

    for frame_idx in range(45):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        def draw_hand(hand_data, color_point, color_line, offset_x=0):
            points = hand_data.reshape(21, 3)
            if np.sum(np.abs(points)) == 0:
                return
            
            pixel_points = []
            for i in range(21):
                # CỘNG THÊM offset_x VÀO TRỤC X ĐỂ DỜI TAY RA HAI BÊN
                x_pixel = int((points[i, 0] - seq_cx) * scale + width / 2 + offset_x)
                y_pixel = int((points[i, 1] - seq_cy) * scale + height / 2)
                pixel_points.append((x_pixel, y_pixel))
                
            for connection in HAND_CONNECTIONS:
                p1, p2 = connection
                cv2.line(img, pixel_points[p1], pixel_points[p2], color_line, thickness=2)
                
            for p in pixel_points:
                cv2.circle(img, p, radius=5, color=color_point, thickness=-1)

        draw_hand(data[frame_idx, :63], color_point=(0, 165, 255), color_line=(200, 200, 200), offset_x=-100)
        draw_hand(data[frame_idx, 63:], color_point=(255, 0, 0), color_line=(255, 200, 200), offset_x=100)
        
        cv2.putText(img, f"Frame: {frame_idx+1}/45 | An 'Q' de thoat", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Trinh phat lai Data (OpenCV)", img)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    time.sleep(0.5)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_file = "data/dataset_words/Tuoi/sample_1.npy"
    if os.path.exists(test_file):
        for _ in range(3):
            play_npy_centered(test_file)
    else:
        print(f"Lỗi: Không tìm thấy file {test_file}")