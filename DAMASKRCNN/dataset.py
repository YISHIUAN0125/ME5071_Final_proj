import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob
import random
from utils.augmentation import fourier_augmentation

class CustomDataset(Dataset):
    def __init__(self, root_dir, subset='train', transforms=None, target_img_paths=None, fourier_prob=0.0):
        """
        root_dir: e.g., 'data/domain_a'
        target_img_paths: list of paths to Domain B images (用於 Fourier Augmentation)
        """
        self.img_dir = os.path.join(root_dir, subset, 'images')
        self.lbl_dir = os.path.join(root_dir, subset, 'labels')
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.transforms = transforms
        
        # Fourier 參數
        self.target_img_paths = target_img_paths
        self.fourier_prob = fourier_prob

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # --- Fourier Augmentation (僅針對 Source Domain) ---
        if self.target_img_paths and random.random() < self.fourier_prob:
            trg_path = random.choice(self.target_img_paths)
            img_trg = cv2.imread(trg_path)
            if img_trg is not None:
                img_trg = cv2.cvtColor(img_trg, cv2.COLOR_BGR2RGB)
                img = fourier_augmentation(img, img_trg, beta=0.01)

        # 讀取 YOLO Label
        lbl_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        lbl_path = os.path.join(self.lbl_dir, lbl_name)

        boxes = []
        masks = []
        labels = []

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                poly_norm = np.array(parts[1:]).reshape(-1, 2)
                
                # 反歸一化
                poly = poly_norm.copy()
                poly[:, 0] *= w
                poly[:, 1] *= h
                poly = poly.astype(np.int32)

                # 生成 Binary Mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 1)
                masks.append(mask)

                # 生成 Bbox
                x1, y1 = np.min(poly[:, 0]), np.min(poly[:, 1])
                x2, y2 = np.max(poly[:, 0]), np.max(poly[:, 1])
                boxes.append([x1, y1, x2, y2])
                
                # 類別 (Mask R-CNN 背景是 0，所以 cabbage 設為 1)
                labels.append(1) 

        # 轉 Tensor
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8) if masks else torch.zeros((0, h, w), dtype=torch.uint8)
        target["image_id"] = torch.tensor([idx])
        
        if len(boxes) > 0:
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
            target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
        else:
             target["area"] = torch.zeros((0,), dtype=torch.float32)
             target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        # 圖片轉 Tensor (0-1)
        img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))
