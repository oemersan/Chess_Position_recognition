from ultralytics import YOLO, RTDETR

class PieceDetectorYOLOV11:
    def __init__(self, model_path="models/piece_yolo_best.pt"):
        self.model = YOLO(model_path)

    def __call__(self, image):
        result = self.model.predict(image, conf=0.6, verbose=False)[0]

        outputs = []
        white_king_seen = False
        black_king_seen = False
        for box in result.boxes:
            if int(box.cls) == 1:
                if black_king_seen:
                    continue
                black_king_seen = True
            elif int(box.cls) == 7:
                if white_king_seen:
                    continue
                white_king_seen = True
            x1 = int(box.xyxy[0][0])
            y1 = int(box.xyxy[0][1])
            x2 = int(box.xyxy[0][2])
            y2 = int(box.xyxy[0][3])
            outputs.append([x1, y1, x2, y2, int(box.cls), float(box.conf)])

        outputs = sorted(outputs, key=lambda x: x[5], reverse=False)
        return outputs
    
class PieceDetectorRTDETR:
    def __init__(self, model_path="models/rtdetr.pt"):
        self.model = RTDETR(model_path)

    def __call__(self, image):
        result = self.model.predict(image, conf=0.6, verbose=False)[0]

        outputs = []
        white_king_seen = False
        black_king_seen = False
        for box in result.boxes:
            if int(box.cls) == 1:
                if black_king_seen:
                    continue
                black_king_seen = True
            elif int(box.cls) == 7:
                if white_king_seen:
                    continue
                white_king_seen = True
            x1 = int(box.xyxy[0][0])
            y1 = int(box.xyxy[0][1])
            x2 = int(box.xyxy[0][2])
            y2 = int(box.xyxy[0][3])
            outputs.append([x1, y1, x2, y2, int(box.cls), float(box.conf)])

        outputs = sorted(outputs, key=lambda x: x[5], reverse=False)
        return outputs
