from ultralytics import YOLO

### nano
# model = YOLO("yolo11n-pose.yaml")
# # model = YOLO("yolo11n-pose.pt")
# model = YOLO("runs/pose/train8/weights/best.pt")

### medium
model = YOLO("yolo11m-pose.yaml")
model = YOLO("yolo11m-pose.pt")

results = model.train(data="test_dataset.yaml", epochs=80, imgsz=640, batch=16, name="medium_train1")

