import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr
from collections import defaultdict, deque
import pandas as pd
import os
import datetime

# ── CONFIG ──────────────────────────────────────────────────────
MODEL_PATH  = r"runs\detect\runs\train\license_plate\weights\best.pt"
VIDEO_PATH  = 0
OUTPUT_PATH = "output_video.mp4"
CONF_THRESH = 0.4
WINDOW_SIZE = 30
BLUR_THRESH = 120
OCR_INTERVAL = 8
FRAME_SKIP = 2
# ────────────────────────────────────────────────────────────────

# Load models
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30, embedder="mobilenet")
reader = easyocr.Reader(['en'], gpu=False)

plate_history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
best_frames = {}

# ── UTILS ───────────────────────────────────────────────────────
def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    enhanced = cv2.fastNlMeansDenoising(enhanced, h=10)
    return enhanced

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype="float32")

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def perspective_correction(img):
    h, w = img.shape[:2]

    if w < 50 or h < 20:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    contour = max(contours, key=cv2.contourArea)

    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 4:
        return img

    pts = approx.reshape(4, 2).astype("float32")
    rect = order_points(pts)

    dst = np.array([
        [0, 0],
        [200, 0],
        [200, 80],
        [0, 80]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (200, 80))

def read_plate_text(plate_img):
    plate_img = perspective_correction(plate_img)
    processed = preprocess_plate(plate_img)

    results = reader.readtext(processed, detail=0, paragraph=False)

    if results:
        return "".join(results).upper().replace(" ", "").replace("-", "")
    return None

def get_majority_vote(history_deque):
    if not history_deque:
        return None
    return max(set(history_deque), key=history_deque.count)

# ── VIDEO LOOP ───────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_PATH,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (640, 480))

logged_plates = set()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔥 Skip frames
    if frame_count % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (640, 480))

    # YOLO detection
    results = model(frame)[0]

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf < CONF_THRESH:
                continue

            detections.append(([x1, y1, x2-x1, y2-y1], conf, 'plate'))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        l, t = max(0, l), max(0, t)
        r, b = min(frame.shape[1], r), min(frame.shape[0], b)

        plate_crop = frame[t:b, l:r]

        if plate_crop.size == 0:
            continue

        score = blur_score(plate_crop)

        if score < BLUR_THRESH:
            continue

        # 🔥 Best frame selection
        if track_id not in best_frames or score > best_frames[track_id][1]:
            best_frames[track_id] = (plate_crop, score)

        # 🔥 OCR only periodically
        if frame_count % OCR_INTERVAL == 0 and track_id in best_frames:
            best_crop = best_frames[track_id][0]

            plate_text = read_plate_text(best_crop)

            if plate_text:
                plate_history[track_id].append(plate_text)

        stable_text = get_majority_vote(plate_history[track_id])

        # Draw box
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

        if stable_text:
            cv2.putText(frame, stable_text, (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Log once
            if stable_text not in logged_plates:
                os.makedirs("crops", exist_ok=True)
                crop_filename = f"crops/{stable_text}_{track_id}.jpg"
                cv2.imwrite(crop_filename, best_frames[track_id][0])

                log_data = {
                    "Plate Number": stable_text,
                    "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Gate": "Main Gate",
                    "Direction": "Entry",
                    "Image Path": crop_filename
                }

                df = pd.DataFrame([log_data])
                df.to_csv("logs.csv", mode='a', index=False,
                          header=not os.path.exists("logs.csv"))

                logged_plates.add(stable_text)

        # Zoom preview
        zoomed = cv2.resize(plate_crop, (200, 80))
        frame[10:90, 10:210] = zoomed

    out.write(frame)
    cv2.imwrite("latest_frame.jpg", frame) 
    cv2.imshow("ANPR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Done! Output saved to:", OUTPUT_PATH)