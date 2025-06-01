import mediapipe as mp
import cv2
import random
import math
import pygame as pg
import importlib
import numpy as np
from rhythm import FaceTracker, RhythmBall

song_files = {
    "1": "sounds.lagu1",
    "2": "sounds.lagu2"
}

def pilih_lagu_ui():
    pilihan = None
    # Koordinat tombol (x1, y1, x2, y2)
    tombol1 = (70, 110, 430, 160)
    tombol2 = (70, 170, 430, 220)
    clicked = [None]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if tombol1[0] <= x <= tombol1[2] and tombol1[1] <= y <= tombol1[3]:
                clicked[0] = "1"
            elif tombol2[0] <= x <= tombol2[2] and tombol2[1] <= y <= tombol2[3]:
                clicked[0] = "2"

    cv2.namedWindow("Menu Lagu")
    cv2.setMouseCallback("Menu Lagu", mouse_callback)

    while pilihan not in song_files:
        menu = 255 * np.ones((300, 500, 3), dtype=np.uint8)
        cv2.putText(menu, "Pilih Lagu:", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
        # Tombol 1
        cv2.rectangle(menu, (tombol1[0], tombol1[1]), (tombol1[2], tombol1[3]), (0,0,255), -1)
        cv2.putText(menu, "OST Jumbo " , (tombol1[0]+10, tombol1[1]+35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # Tombol 2
        cv2.rectangle(menu, (tombol2[0], tombol2[1]), (tombol2[2], tombol2[3]), (255,0,0), -1)
        cv2.putText(menu, "Lagu 2" , (tombol2[0]+10, tombol2[1]+35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(menu, "Klik salah satu tombol", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,0), 2)
        cv2.imshow("Menu Lagu", menu)
        if clicked[0] in song_files:
            pilihan = clicked[0]
        if cv2.waitKey(50) & 0xFF == 27:  # ESC untuk keluar
            break

    cv2.destroyWindow("Menu Lagu")
    return song_files.get(pilihan, "1")

pg.mixer.init()

if __name__ == "__main__":
    song_module = pilih_lagu_ui()
    notes = importlib.import_module(song_module).notes
    cap = cv2.VideoCapture(0)
    tracker = FaceTracker()
    max_balls = len(notes)  # Jumlah bola sesuai jumlah not pada lagu

    # Inisialisasi bola-bola dengan nomor
    frame_ready = False
    balls = []
    margin = 50  # margin dari pinggir (pixel)
    min_distance = 100  # jarak minimum antar bola (pixel)
    while not frame_ready:
        ret, frame = cap.read()
        if ret:
            frame_height, frame_width = frame.shape[:2]
            for i in range(max_balls):
                attempt = 0
                while True:
                    new_ball = RhythmBall(frame_width, frame_height, margin)
                    # Cek jarak ke bola-bola sebelumnya
                    if all(
                        math.hypot(new_ball.x - b.x, new_ball.y - b.y) >= min_distance
                        for b in balls
                    ):
                        balls.append(new_ball)
                        break
                    attempt += 1
                    if attempt > 100:  # Hindari infinite loop
                        balls.append(new_ball)
                        break
            frame_ready = True

    current_ball = 0  # indeks bola yang sedang aktif
    circle_img = cv2.imread("assets/circle.png", cv2.IMREAD_UNCHANGED)
    cursor_img = cv2.imread("assets/cursor.png", cv2.IMREAD_UNCHANGED)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        face_pos = tracker.get_face_position(frame)
        cursor = tracker.calculate_cursor_movement(frame.shape[1], frame.shape[0]) if face_pos else None

        # Gambar hanya bola yang sedang aktif
        if current_ball < max_balls:
            ball = balls[current_ball]
            if ball.active:
                if ball.is_hit(cursor):
                    ball.active = False
                    pg.mixer.Sound(notes[current_ball]).play()
                    current_ball += 1  # lanjut ke bola berikutnya
                else:
                    if circle_img is not None:
                        targete_size = ball.radius * 2
                        resized_circle = cv2.resize(circle_img, (targete_size, targete_size), interpolation=cv2.INTER_AREA)
                        ch,cw = resized_circle.shape[:2]
                        y1, y2 = ball.y - ch // 2, ball.y + ch // 2
                        x1, x2 = ball.x - cw // 2, ball.x + cw // 2
                        if y1 >= 0 and x1 >= 0 and y2 < frame.shape[0] and x2 < frame.shape[1]:
                            alpha_s = resized_circle[:, :, 3] / 255.0
                            alpha_l = 1.0 - alpha_s
                            for c in range(0, 3):
                                frame[y1:y2, x1:x2, c] = (alpha_s * resized_circle[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
                            # Hitung posisi tengah untuk teks
                            text = str(current_ball + 1)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            thickness = 2
                            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                            text_x = ball.x - text_width // 2
                            text_y = ball.y + text_height // 2
                            cv2.putText(
                                frame, text, (text_x, text_y),
                                font, font_scale, (255, 255, 255), thickness
                            )
                    else:
                        cv2.circle(frame, (ball.x, ball.y), ball.radius, (255, 0, 0), -1)
                        cv2.putText(
                            frame, str(current_ball + 1), (ball.x - 10, ball.y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                        )

        # Gambar kursor 
        if cursor:
            ch, cw = cursor_img.shape[:2]
            x, y = cursor
            y1, y2 = y - ch // 2, y + ch // 2
            x1, x2 = x - cw // 2, x + cw // 2
        # Pastikan tidak keluar frame
            if y1 >= 0 and x1 >= 0 and y2 < frame.shape[0] and x2 < frame.shape[1]:
                alpha_s = cursor_img[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    frame[y1:y2, x1:x2, c] = (alpha_s * cursor_img[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])

        # Selesai jika semua bola sudah tidak aktif
        if current_ball >= max_balls:
            cv2.putText(frame, "Selesai!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

        cv2.imshow("Face Rhythm Game", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()