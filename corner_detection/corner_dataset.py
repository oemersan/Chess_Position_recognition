import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np


class CornersDatasetOrdered(Dataset):
    def __init__(self, img_dir, corners_dir, transform=None):
        self.img_dir = img_dir
        self.corners_dir = corners_dir
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(img_dir, ext)))

        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_corners(self, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(self.corners_dir, base_name + ".txt")

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Corner file not found: {txt_path}")

        coords = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                x_str, y_str = line.split(",")
                x = float(x_str)
                y = float(y_str)
                coords.extend([x, y])

        if len(coords) != 8:
            raise ValueError(
                f"Expected 4 points (8 values) in {txt_path}, got {len(coords)} values"
            )

        return torch.tensor(coords, dtype=torch.float32)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        target = self._load_corners(img_path)

        img = self.transform(img)

        return img, target


class CornersDatasetTopToBottom(Dataset):
    def __init__(self, img_dir, corners_dir, transform=None):
        self.img_dir = img_dir
        self.corners_dir = corners_dir
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(img_dir, ext)))

        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_corners(self, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(self.corners_dir, base_name + ".txt")

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Corner file not found: {txt_path}")

        coords = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                x_str, y_str = line.split(",")
                x = float(x_str)
                y = float(y_str)
                coords.extend([x, y])

        if len(coords) != 8:
            raise ValueError(
                f"Expected 4 points (8 values) in {txt_path}, got {len(coords)} values"
            )
        
        coords_np = np.array(coords).reshape(4, 2)

        sorted_indices = np.argsort(coords_np[:, 1])
        coords_np = coords_np[sorted_indices]

        top_two = coords_np[:2]
        bottom_two = coords_np[2:]

        if top_two[0, 0] < top_two[1, 0]:
            top_left, top_right = top_two
        else:
            top_right, top_left = top_two

        if bottom_two[0, 0] < bottom_two[1, 0]:
            bottom_left, bottom_right = bottom_two
        else:
            bottom_right, bottom_left = bottom_two
        coords_np = np.array([top_left, top_right, bottom_left, bottom_right]).reshape(-1)

        return torch.tensor(coords_np, dtype=torch.float32)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        target = self._load_corners(img_path)

        img = self.transform(img)

        return img, target