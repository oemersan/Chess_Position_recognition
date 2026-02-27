import os
import cv2
from ultralytics import YOLO
import numpy as np
import math
from PIL import Image
from torchvision import transforms
import torch

from corner_detector_linear import CornerDetectorLinear

class BoardDetector:
    def __init__(self, model_path="models/medium_best.pt", use_resnet=False):
        self.use_resnet = use_resnet
        if use_resnet:
            self.model = CornerDetectorLinear()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        else:
            self.model = YOLO(model_path)

    def detect_corners(self, image):
        cpy_img = image.copy()
        if isinstance(cpy_img, Image.Image):
            pil_image = cpy_img.convert("RGB")
            rgb = np.array(pil_image)
            cv2_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        elif isinstance(cpy_img, np.ndarray):
            cv2_image = cpy_img
            if cv2_image.ndim == 2:
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
            elif cv2_image.ndim == 3 and cv2_image.shape[2] == 4:
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2BGR)
            elif cv2_image.ndim == 3 and cv2_image.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Unsupported ndarray shape: {cv2_image.shape}")

            pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

        if self.use_resnet:
            return self.detect_corners_resnet(pil_image)
        else:
            return self.detect_corners_yolo(cv2_image)
    
    def detect_corners_yolo(self, image):
        results = self.model(image, verbose=False)
        corners = []
        for result in results:
            xy = result.keypoints.xy
            xyn = result.keypoints.xyn
            kpts = result.keypoints.data

            height, width = image.shape[0], image.shape[1]

            if xyn.shape[0] < 1:
                return None
            
            for i, kpt in enumerate(xyn[0]):
                x, y = int(kpt[0] * width), int(kpt[1] * height)
                corners.append([x, y])
        return corners if len(corners) == 4 else None
    
    def detect_corners_resnet(self, image):
        input_image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_image)
        outputs = outputs.squeeze().numpy().reshape(-1, 2)

        height, width = image.size[1], image.size[0]
        corners = []
        for output in outputs:
            x, y = int(output[0] * width), int(output[1] * height)
            corners.append([x, y])
        return corners if len(corners) == 4 else None

    def warp_board(self, image, corners_xy, out_size=640):
        src = np.array(corners_xy, dtype=np.float32)

        if self.use_resnet:
            dst = np.array([
                [0, 0],
                [out_size - 1, 0],
                [0, out_size - 1],
                [out_size - 1, out_size - 1]
            ], dtype=np.float32)
        else:
            dst = np.array([
                [0, 0],
                [out_size - 1, 0],
                [out_size - 1, out_size - 1],
                [0, out_size - 1]
            ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, H, (out_size, out_size))
        return warped, H
    
    def wide_corners(self, corners, expansion=40, expansion_dynamic=0.2, dynamic_expansion=False):
        corners_copy = [list(corner) for corner in corners]

        center_x = sum(c[0] for c in corners_copy) / 4
        center_y = sum(c[1] for c in corners_copy) / 4

        for corner in corners_copy:
            if dynamic_expansion:
                distance = math.hypot(corner[0] - center_x, corner[1] - center_y)
                expansion = distance * expansion_dynamic

            angle = math.atan2(corner[1] - center_y, corner[0] - center_x)
            corner[0] += int(expansion * math.cos(angle))
            corner[1] += int(expansion * math.sin(angle))

        return corners_copy

    def warped_points_to_original(self, H, warped_points):
        H_inv = np.linalg.inv(H)
        original_points = []
        for point in warped_points:
            wp = np.array([point[0], point[1], 1]).reshape(3, 1)
            op = np.dot(H_inv, wp)
            op /= op[2, 0]
            original_points.append((int(op[0, 0]), int(op[1, 0])))
        return original_points
    
    def calculate_average_corners(self, corners_list1, corners_list2):
        if len(corners_list1) != len(corners_list2):
            raise ValueError("Both corner lists must have the same number of corners.")
        corners_list1 = [tuple(c) for c in corners_list1]
        corners_list2 = [tuple(c) for c in corners_list2]
        averaged_corners = []
        for c1 in corners_list1:
            min_distance = float('inf')
            closest_c2 = None
            for c2 in corners_list2:
                distance = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_c2 = c2
            if closest_c2 is not None:
                avg_x = int((c1[0] + closest_c2[0]) / 2)
                avg_y = int((c1[1] + closest_c2[1]) / 2)
                averaged_corners.append((avg_x, avg_y))
                corners_list2.remove(closest_c2)
        return averaged_corners

    def __call__(self, image):
        corners = self.detect_corners(image)
        if corners is None:
            return None, None, None
        
        warped_board, H1 = self.warp_board(image, corners)

        wide_corners = self.wide_corners(corners, expansion_dynamic=0.2, dynamic_expansion=True)
        wide_warped_board, H2 = self.warp_board(image, wide_corners)

        if self.use_resnet:
            return corners, warped_board, H1, wide_warped_board

        corners_on_wide_warped = self.detect_corners(wide_warped_board)

        if corners_on_wide_warped is not None:
            wide_corners_in_original = self.warped_points_to_original(H2, corners_on_wide_warped)
            corners = self.calculate_average_corners(corners, wide_corners_in_original)
            warped_board, H1 = self.warp_board(image, corners)

        return corners, warped_board, H1, wide_warped_board