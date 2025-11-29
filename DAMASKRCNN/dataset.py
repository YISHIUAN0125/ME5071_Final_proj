import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob
import random
from utils.augmentation import fourier_augmentation

class CustomDataset(Dataset):
    def __init__(self, root_dir, subset='train', transforms=None, target_img_paths=None, fourier_prob=0.0, beta=0.001):
        """
        root_dir: e.g., 'data/domain_a'
        subset: 'train', 'valid', or 'test'
        """
        self.img_dir = os.path.join(root_dir, subset, 'images')
        self.lbl_dir = os.path.join(root_dir, subset, 'labels')
        
        #supported image type
        extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        self.img_paths = []
        for ext in extensions:
            self.img_paths.extend(glob.glob(os.path.join(self.img_dir, ext)))
        self.img_paths = sorted(list(set(self.img_paths)))
        
        self.transforms = transforms

        # TODO abandoned
        self.target_img_paths = target_img_paths
        self.fourier_prob = fourier_prob
        self.beta = beta

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Read image
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"無法讀取圖片: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # TODO abandoned
        # if self.target_img_paths and random.random() < self.fourier_prob:
        #     trg_path = random.choice(self.target_img_paths)
        #     img_trg = cv2.imread(trg_path)
        #     if img_trg is not None:
        #         img_trg = cv2.cvtColor(img_trg, cv2.COLOR_BGR2RGB)
        #         # Resize target to match source for FFT
        #         img_trg = cv2.resize(img_trg, (w, h))
        #         # img = fourier_augmentation(img, img_trg, beta=self.beta)
        #         img = img

        # 3. 讀取標籤
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
                # parts[0] is class_id (0 for cabbage)
                # parts[1:] are polygon coords
                poly_norm = np.array(parts[1:]).reshape(-1, 2)
                
                # denormalized
                poly = poly_norm.copy()
                poly[:, 0] *= w
                poly[:, 1] *= h
                poly = poly.astype(np.int32)

                # calculate Bbox
                x1, y1 = np.min(poly[:, 0]), np.min(poly[:, 1])
                x2, y2 = np.max(poly[:, 0]), np.max(poly[:, 1])

                # check valid box
                if x2 <= x1 + 1 or y2 <= y1 + 1:
                    continue

                # generate mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 1)
                masks.append(mask)

                boxes.append([x1, y1, x2, y2])
                labels.append(1) # Cabbage (Mask R-CNN 背景固定為 0)

        # 4. 轉換為 Tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1) # (C, H, W)

        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            # Mask 需要是 (N, H, W)
            target["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
            target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
            
        target["image_id"] = torch.tensor([idx])
        
        # target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        # target["labels"] = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        # target["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8) if masks else torch.zeros((0, h, w), dtype=torch.uint8)
        # target["image_id"] = torch.tensor([idx])
        
        # # 計算 area (用於 COCO 評估)
        # if len(boxes) > 0:
        #     target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        #     target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
        # else:
        #      target["area"] = torch.zeros((0,), dtype=torch.float32)
        #      target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        # 圖片歸一化 (Mask R-CNN 內部 transform 會做標準化，這裡轉成 0-1 即可)
        # img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        
        if self.transforms is not None:
            # v2 transforms 接受 (image, target) 並同時轉換兩者
            img_tensor, target = self.transforms(img_tensor, target)

        # 5. 最後確保圖片是 Float (0-1) 這是模型吃的格式
        # 如果 transforms 裡面沒有 ToDtype(float32, scale=True)，這裡要做
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float() / 255.0
        
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))