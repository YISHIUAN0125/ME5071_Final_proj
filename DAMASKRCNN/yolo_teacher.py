import torch
import numpy as np
import cv2
from ultralytics import YOLO

class YOLOTeacher:
    def __init__(self, model_path, device='cuda', conf_thres=0.7):
        print(f"Loading YOLO Teacher from {model_path}...")
        self.model = YOLO(model_path)
        self.device = device
        self.conf_thres = conf_thres

    def generate_targets(self, images_tensor):
        """
        輸入: Mask R-CNN 的 images (List[Tensor]), 數值 0~1
        輸出: Mask R-CNN 的 targets (List[Dict])
        """
        targets = []
        
        # 1. 轉換格式: Tensor (0-1, RGB) -> Numpy (0-255, BGR) 給 YOLO 用
        imgs_np = []
        for img_t in images_tensor:
            img_np = img_t.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            imgs_np.append(img_np)

        # 2. YOLO 推論 (No Grad)
        with torch.no_grad():
            # retina_masks=True 讓遮罩更精細
            results = self.model(imgs_np, conf=self.conf_thres, verbose=False, retina_masks=True)

        for i, result in enumerate(results):
            h, w = imgs_np[i].shape[:2]
            
            # 初始化空的
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)

            # 如果有偵測到物件
            if result.boxes is not None and len(result.boxes) > 0:
                # A. Boxes
                boxes = result.boxes.xyxy.cpu() # (N, 4)
                
                # B. Labels
                # 假設只有一類高麗菜。Mask R-CNN 背景是0，高麗菜是1。
                # 如果你的 YOLO 有多類，這裡要做 map 映射。目前全部設為 1。
                labels = torch.ones((len(boxes),), dtype=torch.int64)

                # C. Masks
                if result.masks is not None:
                    raw_masks = result.masks.data.cpu().numpy()
                    
                    # 確保 Mask 尺寸與原圖一致 (YOLO有時會輸出縮小的 mask)
                    final_masks = []
                    for m in raw_masks:
                        if m.shape != (h, w):
                            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        final_masks.append(m)
                    
                    if len(final_masks) > 0:
                        masks = torch.as_tensor(np.array(final_masks), dtype=torch.uint8)

            target_dict = {
                "boxes": boxes.to(self.device),
                "labels": labels.to(self.device),
                "masks": masks.to(self.device),
                "image_id": torch.tensor([i], device=self.device)
            }
            targets.append(target_dict)
            
        return targets