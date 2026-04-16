import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict, deque
import pandas as pd
import os
import datetime

if __name__ == '__main__':

    # ── CONFIG ──────────────────────────────────────────────────────
    MODEL_PATH  = r"runs\detect\runs\train\license_plate\weights\best.pt"
    VIDEO_PATH  = 0   # put your video here
    OUTPUT_PATH = "output_video.mp4"
    CONF_THRESH = 0.4
    WINDOW_SIZE = 30
    # ────────────────────────────────────────────────────────────────

    # Load models
    model  = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=True)

    # Majority voting buffer
    plate_history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

    def preprocess_plate(img):
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        enhanced = cv2.fastNlMeansDenoising(enhanced, h=10)
        return enhanced

        # ── CCI DICTIONARIES [5] ─────────────────────────────────────────
    # # Rules for characters misread as numbers, and numbers misread as letters
    # CHAR_TO_NUM = {'O': '0', 'L': '1', 'I': '1', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}
    # NUM_TO_CHAR = {'0': 'O', '1': 'L', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
    # # ──────────────────────────────────────────────────────────────────

    # def apply_cci(text):
    #     """
    #     Applies the Check Character Index (CCI) to correct OCR errors based on 
    #     the Indian standard format: LL NN LLL NNNN (e.g., MH12ABC1234) [2, 6].
    #     """
    #     # Standard Indian plate without spaces usually has 9 or 10 characters.
    #     # Assuming a 10-character plate for this strict positional mapping:
    #     if len(text) == 10:
    #         corrected = ""
    #         for i, char in enumerate(text):
    #             if i in [1, 2, 5, 6]:  # Expected Letter positions
    #                 corrected += NUM_TO_CHAR.get(char, char)
    #             elif i in [3,4,7,8,9,10]:  # Expected Number positions
    #                 corrected += CHAR_TO_NUM.get(char, char)
    #             else:
    #                 corrected += char
    #         return corrected
    #     return text


    # def is_valid_plate(text):
    #     """Validates the plate using Regex for the Indian Standard [6]."""
    #     # Matches LL NN L/LL/LLL NNNN (e.g., MH12AB1234)
    #     pattern = r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$'
    #     return bool(re.match(pattern, text))

    def read_plate_text(plate_img):
        processed = preprocess_plate(plate_img)
        results = reader.readtext(processed, detail=0, paragraph=False)
        
        if results:
            raw_text = "".join(results).upper().replace(" ", "").replace("-", "")
            # corrected_text = apply_cci(raw_text)
            # if is_valid_plate(corrected_text):
            #     return corrected_text
            return raw_text
        return None

    def get_majority_vote(history_deque):
        if not history_deque:
            return None
        counter = defaultdict(int)
        for r in history_deque:
            counter[r] += 1
        return max(counter, key=counter.get)

    # ── VIDEO LOOP ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    logged_plates = set() 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=CONF_THRESH, verbose=False)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else 0

                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                plate_text = read_plate_text(plate_crop)
                if plate_text:
                    plate_history[track_id].append(plate_text)

                stable_text = get_majority_vote(plate_history[track_id])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw plate text
                # Draw plate text
                if stable_text:
                    cv2.putText(frame, stable_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # ✅ LOG ONLY ONCE PER VEHICLE
                    if stable_text not in logged_plates:

                        # 1. Save cropped plate image
                        os.makedirs("crops", exist_ok=True)
                        crop_filename = f"crops/{stable_text}_{track_id}.jpg"
                        cv2.imwrite(crop_filename, plate_crop)

                        # 2. Save to CSV
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

                # Zoomed plate preview top-left
                if plate_crop.size > 0:
                    zoomed = cv2.resize(plate_crop, (200, 80))
                    frame[10:90, 10:210] = zoomed


        out.write(frame)
        cv2.imwrite("latest_frame.jpg", frame)
        cv2.imshow("License Plate Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done! Output saved to:", OUTPUT_PATH)