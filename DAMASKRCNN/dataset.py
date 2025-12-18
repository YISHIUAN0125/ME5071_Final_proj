import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob
import random
from utils.augmentation import fourier_augmentation

class CustomDataset(Dataset):
    def __init__(self, root_dir, subset='train', target_size=(1000, 1000), transforms=None, target_img_paths=None, fourier_prob=0.0, beta=0.001):
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
        
        self.target_size = target_size
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
        img = cv2.resize(img, self.target_size)

        h, w, _ = img.shape

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
                cls_id = int(parts[0]) # 雖然目前固定為 1，但保留讀取邏輯
                
                # 判斷是 YOLO BBox (5個值) 還是 Polygon (>5個值)
                if len(parts) == 5:
                    # --- 處理 YOLO BBox 格式 (class, xc, yc, w, h) ---
                    # 格式: normalized center_x, center_y, width, height
                    cx, cy, bw, bh = parts[1], parts[2], parts[3], parts[4]
                    
                    # 反正規化
                    cx *= w
                    cy *= h
                    bw *= w
                    bh *= h
                    
                    # 轉換為 x1, y1, x2, y2
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    
                    # 為了 Mask R-CNN，必須生成一個矩形的 Mask
                    # 這裡建立矩形的四個角點用於 fillPoly
                    poly = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ], dtype=np.int32)
                    
                else:
                    # --- 處理 YOLO Segmentation 格式 (class, x1, y1, x2, y2, ...) ---
                    poly_norm = np.array(parts[1:]).reshape(-1, 2)
                    
                    # denormalized
                    poly = poly_norm.copy()
                    poly[:, 0] *= w
                    poly[:, 1] *= h
                    poly = poly.astype(np.int32)
                    
                    # calculate Bbox from polygon
                    x1, y1 = np.min(poly[:, 0]), np.min(poly[:, 1])
                    x2, y2 = np.max(poly[:, 0]), np.max(poly[:, 1])

                # check valid box (過濾無效框)
                if x2 <= x1 + 1 or y2 <= y1 + 1:
                    continue

                # 限制座標在圖片範圍內 (避免 padding 導致越界)
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # generate mask
                mask = np.zeros((h, w), dtype=np.uint8)
                # 注意：fillPoly 需要 list of arrays (int32)
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                masks.append(mask)

                boxes.append([x1, y1, x2, y2])
                labels.append(1) # Cabbage

        # if os.path.exists(lbl_path):
        #     with open(lbl_path, 'r') as f:
        #         lines = f.readlines()
            
        #     for line in lines:
        #         parts = list(map(float, line.strip().split()))
        #         # parts[0] is class_id (0 for cabbage)
        #         # parts[1:] are polygon coords
        #         poly_norm = np.array(parts[1:]).reshape(-1, 2)
                
        #         # denormalized
        #         poly = poly_norm.copy()
        #         poly[:, 0] *= w
        #         poly[:, 1] *= h
        #         poly = poly.astype(np.int32)

        #         # calculate Bbox
        #         x1, y1 = np.min(poly[:, 0]), np.min(poly[:, 1])
        #         x2, y2 = np.max(poly[:, 0]), np.max(poly[:, 1])

        #         # check valid box
        #         if x2 <= x1 + 1 or y2 <= y1 + 1:
        #             continue

        #         # generate mask
        #         mask = np.zeros((h, w), dtype=np.uint8)
        #         cv2.fillPoly(mask, [poly], 1)
        #         masks.append(mask)

        #         boxes.append([x1, y1, x2, y2])
        #         labels.append(1) # Cabbage (Mask R-CNN 背景固定為 0)
        

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