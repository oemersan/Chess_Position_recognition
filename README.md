# Real-World Chess Position Recognition Using Deep Learning

This project implements a complete end-to-end computer vision pipeline that reads a real chess position from a single image and converts it into:

- A valid FEN (Forsyth-Edwards Notation) string
- A best move suggestion using Stockfish
- An evaluation score for the position

The system is designed to work on real-world images taken from mobile phones under perspective distortion, lighting variation, and background noise.

---

## 1. Pipeline Overview

The system follows the steps below:

1. Piece detection using object detection models
2. Board corner detection
3. Perspective correction using homography
4. Board orientation detection using OCR
5. Mapping detected pieces to the 8x8 grid
6. FEN string generation
7. Stockfish analysis

The goal is to reconstruct the full chess state from a single image and evaluate it automatically.

---

## 2. Models Used

### 2.1 Piece Detection

We trained and compared the following models for detecting 12 chess piece classes:

- YOLOv11-nano (baseline)
- YOLOv11-small
- RT-DETR-L (transformer-based detector)

Best performing configuration:

```
YOLOv11s (50 epochs)
mAP50 = 0.891
mAP50-95 = 0.664
```

YOLOv11s achieved the best trade-off between accuracy and computational cost.

### 2.2 Corner Detection

Two approaches were tested:

- ResNet-18 regression (predicting 4 corner coordinates)
- YOLO-Pose keypoint detection

Best result:

```
ResNet-18 (geometric ordering: TL, TR, BL, BR)
Normalized Average Euclidean Error = 0.017
```

Geometric ordering performed significantly better than semantic ordering.

### 2.3 Board Orientation Detection

Board orientation is determined using:

- Tesseract OCR
- Grayscale preprocessing
- Threshold + inversion fallback
- 4x image upscaling
- RANSAC filtering

Orientation accuracy on the test set:

```
94%
```

### 2.4 End-to-End Performance

FEN Accuracy is defined as:

```
FEN Accuracy = (Correct Squares / 64) x 100%
```

Results:

| Metric | Value |
|---|---|
| Average FEN Accuracy | 88.98% |
| 90th Percentile FEN Accuracy | 98.44% |

This shows that many images are reconstructed nearly perfectly.

---

## 3. Repository Structure

```
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
```

---

## 4. Installation

### 4.1 Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 4.2 Create Environment

```bash
conda create -n chess_dl python=3.10
conda activate chess_dl
```

### 4.3 Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

- PyTorch
- Ultralytics
- OpenCV
- Tesseract OCR
- numpy
- matplotlib
- python-chess
- Stockfish

---

## 5. Running the Full Pipeline

From the `main/` directory:

```bash
python main_v3.py --image path_to_image.jpg
```

Output:

- Warped board image
- Predicted piece bounding boxes
- Generated FEN string
- Best move (SAN and UCI)
- Evaluation score

---

## 6. Experiments

### 6.1 Piece Detection Experiments

We evaluated:

- YOLOv11 vs RT-DETR
- 30 vs 50 epochs
- Full dataset vs half dataset
- 640 vs 320 input resolution
- Augmentation ablations

Key findings:

- YOLOv11s provides the best balance of accuracy and efficiency.
- Strong augmentation slightly reduced performance.
- Lower resolution significantly degrades detection performance.
- RT-DETR-L is heavier but does not clearly outperform YOLOv11s.

### 6.2 Corner Detection Experiments

- ResNet-18 regression performed better than YOLO-Pose.
- Geometric corner ordering is more stable than semantic ordering.
- Two-pass refinement improved YOLO-Pose but not ResNet-18.

### 6.3 OCR Experiments

Preprocessing steps such as grayscale conversion, inversion fallback, and 4x upscaling improved OCR reliability significantly.

---

## 7. Limitations

The system may fail in the following scenarios:

- Pieces are heavily occluded.
- Board letters are not visible (OCR fails).
- The board is partially outside the image.
- Extreme perspective distortion.
- Very small or blurry pieces.

---

## 8. Future Work

- Collect more real-world images under harder conditions.
- Replace OCR-based orientation with a learned classifier.
- Add chess-rule consistency validation to reject illegal boards.
- Perform multi-seed statistical evaluation.
- Explore real-time or mobile deployment.

---

## 9. Authors

**Ömer Faruk San**
Artificial Intelligence and Data Engineering
Istanbul Technical University

**Mustafa Kerem Bulut**
Artificial Intelligence and Data Engineering
Istanbul Technical University
