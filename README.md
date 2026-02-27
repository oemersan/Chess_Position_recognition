# Real-World Chess Position Recognition Using Deep Learning

End-to-end computer vision pipeline that reads a real-world chessboard photo (often taken with a phone) and outputs:

- valid **FEN** string  
- best move suggestion (via Stockfish)  
- position evaluation score  

Handles perspective distortion, uneven lighting, background clutter, etc.

## Pipeline Steps

1. Chess piece detection (object detection)  
2. Board corner detection  
3. Perspective correction (homography)  
4. Board orientation detection (OCR)  
5. Piece → 8×8 grid mapping  
6. FEN generation  
7. Stockfish analysis  

## Models & Results

### Piece Detection

**Best model:** YOLOv11s (50 epochs)  
- mAP@50    = 0.891  
- mAP@50:95 = 0.664  

→ Best accuracy ↔ speed trade-off compared to YOLOv11n and RT-DETR-L

### Corner Detection

**Best approach:** ResNet-18 regression with **geometric ordering** (TL → TR → BL → BR)

- Normalized Average Euclidean Error = **0.017**

Geometric ordering was clearly more stable than semantic labeling.

### Board Orientation Detection

Tesseract OCR + aggressive preprocessing:

- Grayscale  
- Thresholding + inversion fallback  
- 4× upscaling  
- RANSAC line filtering  

→ **94%** correct orientation on test set

### End-to-End FEN Reconstruction Accuracy

Definition: `(number of correctly predicted squares / 64) × 100%`

- **Average** FEN accuracy: **88.98%**  
- **90th percentile**: **98.44%**  

→ Many real-world photos are reconstructed almost perfectly.

## Repository Structure

```text
.
├── dataset_link.txt
├── README.md
├── structure.txt
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
    ├── RT_DETR_30Epoch.ipynb '''
    ├── RT_DETR_50Epoch.ipynb
    ├── RT_DETR_50epoch_half_data.ipynb
    └── RT_DETR_50Epoc_half_data_320x320.ipynb
