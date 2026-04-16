## 📌 Project Overview

This project implements a complete end-to-end **ANPR pipeline** that:
1. Takes a video stream or image as input
2. Detects license plates using a fine-tuned **YOLOv8s** model
3. Extracts plate text using **EasyOCR** or **Claude Vision API**
4. Applies **CCI (Check Character Index)** to correct common OCR errors
5. Validates plate format against known regional patterns
6. Outputs annotated video/image with detected plate numbers

Real-world use cases include highway surveillance, parking automation, toll systems, and traffic enforcement cameras.

---

## 🏗️ System Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Input          │────▶│  YOLOv8s Detection   │────▶│  Plate Crop         │
│  Video / Image  │     │  Fine-tuned on        │     │  + Preprocessing    │
└─────────────────┘     │  License Plate Data   │     │  (3x upscale,       │
                        └──────────────────────┘     │   OTSU threshold,   │
                                                      │   denoising)        │
                                                      └─────────┬───────────┘
                                                                │
                                              ┌─────────────────▼──────────────────┐
                                              │  OCR Engine                        │
                                              │  Option A: EasyOCR (offline)       │
                                              │  Option B: Claude Vision (API)     │
                                              └─────────────────┬──────────────────┘
                                                                │
                                              ┌─────────────────▼──────────────────┐
                                              │  Post Processing                   │
                                              │  • CCI Character Correction        │
                                              │  • Format Validation               │
                                              │  • Majority Voting (video)         │
                                              └─────────────────┬──────────────────┘
                                                                │
                                              ┌─────────────────▼──────────────────┐
                                              │  Output                            │
                                              │  Annotated Video / Image           │
                                              │  + Plate Text                      │
                                              └────────────────────────────────────┘
```

---

## 📊 Model Performance

The YOLOv8s model was fine-tuned for **30 epochs** on the Roboflow License Plate dataset.

| Metric          | Value     |
|-----------------|-----------|
| **mAP50**       | **0.986** |
| **mAP50-95**    | 0.708     |
| Precision       | 0.981     |
| Recall          | 0.965     |
| Inference Speed | 5.9ms/image |
| Model Size      | 22.5 MB   |

Training hardware: **NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)**

### Training Progress

| Epoch | mAP50 | box_loss | cls_loss |
|-------|-------|----------|----------|
| 1     | 0.962 | 1.326    | 1.112    |
| 5     | 0.966 | 1.216    | 0.675    |
| 10    | 0.972 | 1.176    | 0.632    |
| 20    | 0.981 | 1.089    | 0.598    |
| 30    | **0.986** | 1.043 | 0.571  |

---

## 🧠 Key Technical Components

### 1. YOLOv8s Fine-Tuning
- Base model: `yolov8s.pt` (pretrained on COCO)
- Transfer learning on license plate specific dataset
- 2647 training images, 2046 validation images
- Batch size: 8 (optimized for 4GB VRAM)

### 2. Image Preprocessing Pipeline
```
Original Crop → 3x Upscale → Grayscale → Contrast Enhancement
             → OTSU Thresholding → Denoising → OCR Input
```

### 3. CCI (Check Character Index)
Corrects common OCR confusion based on expected character type at each position:

| OCR Reads | Corrected To | Rule Applied |
|-----------|-------------|--------------|
| O         | 0           | Digit position → char_to_num |
| I         | 1           | Digit position → char_to_num |
| S         | 5           | Digit position → char_to_num |
| 0         | O           | Letter position → num_to_char |
| 1         | I           | Letter position → num_to_char |

### 4. Majority Voting (Video Mode)
- Maintains a **sliding window of 30 frames** per tracked vehicle
- Most frequently detected plate text wins
- Eliminates flickering / unstable readings between frames

### 5. Multi-Country Format Validation

| Country | Format Pattern | Example |
|---------|---------------|---------|
| 🇮🇳 India (Standard) | LL NN LLL NNNN | MH12AB1234 |
| 🇮🇳 India (BH Series) | NN LL NNNN L | 22BH6517A |
| 🇬🇧 UK | LL NN LLL | AB12CDE |
| 🇻🇳 Vietnam | NN L NNNNN | 51A65474 |
| 🇦🇺 Australia | LLL NNN | BE33TA |

---

## 🗂️ Project Structure

```
Vehicle_Regestration/
│
├── 📂 saved_models/
│   └── license_plate_best.pt     # Fine-tuned YOLOv8s weights (mAP50: 0.986)
│
├── 📂 runs/                      # Training logs & metrics
│   └── detect/runs/train/
│       └── license_plate/
│           ├── results.csv       # Epoch-wise metrics
│           └── weights/
│
├── 📂 results/                   # Sample output images
│
├── 🐍 download_dataset.py        # Roboflow dataset downloader
├── 🐍 train.py                   # YOLOv8s fine-tuning script
├── 🐍 inference.py               # Real-time video inference
├── 🐍 test_image.py              # Single image testing + Claude Vision OCR
├── 🐍 test_batch.py              # Batch testing with summary statistics
│
├── 📄 requirements.txt
└── 📄 README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU recommended (CUDA 11.8+)
- 4GB+ VRAM for training

### Step 1 — Clone & Setup Environment
```bash
git clone https://github.com/YOUR_USERNAME/Vehicle_Regestration.git
cd Vehicle_Regestration

python -m venv License_Plate_Recognition_env

# Windows:
.\License_Plate_Recognition_env\Scripts\activate
# Mac/Linux:
source License_Plate_Recognition_env/bin/activate
```

### Step 2 — Install Dependencies
```bash
# With NVIDIA GPU (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# All other packages:
pip install -r requirements.txt
```

### Step 3 — Verify GPU Detection
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected output:
# True
# NVIDIA GeForce RTX XXXX
```

### Step 4 — Download Dataset (optional — only needed for retraining)
Get a free API key from [roboflow.com](https://roboflow.com), paste it into `download_dataset.py`, then:
```bash
python download_dataset.py
```
Dataset: [License Plate Recognition — Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)

### Step 5 — Train (optional — pretrained weights included in repo)
```bash
python train.py
# ~25 minutes on RTX 3050, 30 epochs
```

---

## 🚀 Running the System

### Real-time Video Inference
```bash
python inference.py
# Set VIDEO_PATH = "your_video.mp4" or VIDEO_PATH = 0 for webcam
# Press Q to quit
```

### Single Image Test
```bash
python test_image.py
# Set image path inside the script
```

### Batch Test on Dataset Images
```bash
python test_batch.py
# Loops through all test images one by one
# Press any key to advance, Q to quit early
# Prints full summary at the end
```

---

## 🔍 OCR Engine Options

### Option A — EasyOCR (Default, Fully Offline)
- No API key required
- Runs on GPU for fast inference
- Best for clear, high-resolution plates

### Option B — Claude Vision API (Higher Accuracy)
- Uses Anthropic's Claude Haiku vision model
- Significantly better on blurry, low-res, or unusual plates
- Requires free API key from [console.anthropic.com](https://console.anthropic.com)
- Set `API_KEY = "your-key"` in `test_batch.py`

---

## 📦 Requirements

```
ultralytics       # YOLOv8 detection
easyocr           # OCR text extraction
opencv-python     # Image processing & visualization
numpy             # Array operations
roboflow          # Dataset download & management
requests          # Claude Vision API calls
torch             # Deep learning backend (install separately with CUDA)
```

---

## 🔮 Future Improvements

- [ ] Fine-tune on India-specific plate dataset for higher regional accuracy
- [ ] Train a custom OCR model on license plate fonts
- [ ] Deploy as a web app (Flask / Streamlit)
- [ ] Add database logging of detected plates with timestamps
- [ ] Support for two-line plate formats
- [ ] Speed estimation from video frame differencing

---

## 🛠️ Built With

| Technology | Role |
|------------|------|
| [YOLOv8s — Ultralytics](https://github.com/ultralytics/ultralytics) | License plate detection |
| [EasyOCR — JaidedAI](https://github.com/JaidedAI/EasyOCR) | Offline OCR |
| [Claude Vision — Anthropic](https://console.anthropic.com) | High-accuracy OCR |
| [OpenCV](https://opencv.org) | Image processing & visualization |
| [PyTorch + CUDA](https://pytorch.org) | GPU-accelerated deep learning |
| [Roboflow](https://roboflow.com) | Dataset management |# Vehicle-Regestration
#   V e c h i c l e - N u m b e r - P l a t e - R e g i s t e r a t i o n 
 
 
