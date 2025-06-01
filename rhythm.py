import mediapipe as mp
import cv2
import random
import math

class FaceTracker:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.2)
        self.face_position = None

    def get_face_position(self, frame):
        # Convert BGR to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                self.face_position = (int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h))
        return self.face_position

    def calculate_cursor_movement(self, frame_width, frame_height):
        if self.face_position:
            x, y, w, h = self.face_position
            cursor_x = x + w // 2
            cursor_y = y + h // 2
            cursor_x = max(0, min(cursor_x, frame_width))
            cursor_y = max(0, min(cursor_y, frame_height))
            return cursor_x, cursor_y
        return None

class RhythmBall:
    def __init__(self, frame_width, frame_height, margin=50):
        self.radius = 30
        self.x = random.randint(self.radius + margin, frame_width - self.radius - margin)
        self.y = random.randint(self.radius + margin, frame_height - self.radius - margin)
        self.active = True

    def is_hit(self, cursor):
        if cursor is None:
            return False
        cx, cy = cursor
        distance = math.hypot(self.x - cx, self.y - cy)
        return distance < self.radius + 10  # 10 = radius kursor
    