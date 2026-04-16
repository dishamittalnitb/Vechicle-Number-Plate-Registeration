from ultralytics import YOLO
import shutil
import os

if __name__ == '__main__':
    # Load YOLOv8 small pretrained model
    model = YOLO("yolov8s.pt")

    # Fine-tune on license plate dataset
    results = model.train(
        data=r"C:\Users\Disha Mittal\Downloads\Vehicle-Regestration-main\Vehicle-Regestration-main\License-Plate-Recognition-4\data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        workers=2,      # reduced from 4 to avoid multiprocessing issues on Windows
        device=0,
        project="runs/train",
        name="license_plate",
        exist_ok=True
    )

    # Save best model
    os.makedirs("saved_models", exist_ok=True)
    shutil.copy(
        "runs/train/license_plate/weights/best.pt",
        "saved_models/license_plate_best.pt"
    )

    print("\n Training complete!")
    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print("Model saved to: saved_models/license_plate_best.pt")


    # This file contains the code to train the YOLOv8 model on the license plate recognition dataset.
    # It loads the pretrained YOLOv8 small model, fine-tunes it on the dataset for 30 epochs, and saves the best model weights to a local directory. 
    # The training results, including the best mAP50 score, are printed at the end.