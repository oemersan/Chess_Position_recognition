# Real-World Chess Position Recognition Using Deep Learning

This project implements a complete end-to-end computer vision pipeline that reads a real chess position from a single image and converts it into:

* A valid FEN (Forsyth–Edwards Notation) string
* A best move suggestion using Stockfish
* An evaluation score for the position

The system is designed to work on real-world images taken from mobile phones under perspective distortion, lighting variation, and background noise.

---

## 1. Pipeline Overview

The system follows the steps below:

1.  **Piece detection** using object detection models
2.  **Board corner detection**
3.  **Perspective correction** using homography
4.  **Board orientation detection** using OCR
5.  **Mapping detected pieces** to the 8×8 grid
6.  **FEN string generation**
7.  **Stockfish analysis**

The goal is to reconstruct the full chess state from a single image and evaluate it automatically.

---

## 2. Models Used

### 2.1 Piece Detection
We trained and compared the following models for detecting 12 chess piece classes:
* YOLOv11-nano (baseline)
* YOLOv11-small
* RT-DETR-L (transformer-based detector)

**Best performing configuration:**
> **YOLOv11s (50 epochs)**
> mAP50 = 0.891
> mAP50-95 = 0.664

YOLOv11s achieved the best trade-off between accuracy and computational cost.

### 2.2 Corner Detection
Two approaches were tested:
* ResNet-18 regression (predicting 4 corner coordinates)
* YOLO-Pose keypoint detection

**Best result:**
> **ResNet-18** (geometric ordering: TL, TR, BL, BR)
> Normalized Average Euclidean Error = 0.017

Geometric ordering performed significantly better than semantic ordering.

### 2.3 Board Orientation Detection
Board orientation is determined using:
* Tesseract OCR
* Grayscale preprocessing
* Threshold + inversion fallback
* 4× image upscaling
* RANSAC filtering

**Orientation accuracy on the test set:** 94%

### 2.4 End-to-End Performance
FEN Accuracy is defined as: 
`FEN Accuracy = (Correct Squares / 64) × 100%`

**Results:**
* **Average FEN Accuracy:** 88.98%
* **90th percentile FEN Accuracy:** 98.44%

This shows that many images are reconstructed nearly perfectly.

---

## 3. Repository Structure

```text
.
│   dataset_link.txt
│   README.md
│   structure.txt
│
├── corner_detection/
│   ├── corner_dataset.py
│   ├── corner_detector_linear.py
│   ├── train.ipynb
│   ├── test.ipynb
│   └── yolo_pose_train.py
│
├── main/
│   ├── board_detector.py
│   ├── corner_detector_linear.py
│   ├── piece_detector.py
│   ├── orientation_detector.py
│   ├── fen_generator.py
│   └── main_v3.py
│
└── piece_detection_models/
    ├── yolov11nano50epoch_baseline.ipynb
    ├── yolov11s50Epoch.ipynb
    ├── yolov11s_50Epoch_augmented.ipynb
    ├── RT_DETR_30Epoch.ipynb
    ├── RT_DETR_50Epoch.ipynb
    ├── RT_DETR_50epoch_half_data.ipynb
    └── RT_DETR_50Epoc_half_data_320x320.ipynb
