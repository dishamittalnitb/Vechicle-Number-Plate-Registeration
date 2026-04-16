# stream_backend.py
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import defaultdict, deque
import re
import multiprocessing as mp

def start_inference(frame_queue):

    MODEL_PATH = r"runs\detect\runs\train\license_plate\weights\best.pt"
    CONF_THRESH = 0.4
    WINDOW_SIZE = 30

    model  = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=True)

    plate_history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

    def preprocess_plate(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.convertScaleAbs(gray, alpha=1.5, beta=30)

    def read_plate_text(img):
        processed = preprocess_plate(img)
        results = reader.readtext(processed, detail=0)
        if results:
            return "".join(results).upper().replace(" ", "")
        return None

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=CONF_THRESH, verbose=False)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                text = read_plate_text(plate_crop)

                if text:
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # 🔥 PUSH FRAME TO QUEUE (NO DISK)
        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()