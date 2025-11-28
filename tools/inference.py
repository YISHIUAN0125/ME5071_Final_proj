# --- START OF FILE inference.py ---
import torch
import cv2
import numpy as np
import os
import argparse
import random
from DAMASKRCNN.damaskrcnn import DAMaskRCNN


# 設定顏色 (隨機生成)
def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class Inferencer:
    def __init__(self, weight_path, device='cuda', threshold=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # 1. 初始化模型
        print(f"Loading model from {weight_path}...")
        self.model = DAMaskRCNN(num_classes=2) # 2 classes: Background + Cabbage
        
        # 2. 載入權重
        checkpoint = torch.load(weight_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint) # 兼容純 state_dict
            
        self.model.to(self.device)
        self.model.eval() # 設定為評估模式

    def predict_image(self, img_path, output_path=None):
        # 讀取圖片
        if not os.path.exists(img_path):
            print(f"[Error] File not found: {img_path}")
            return

        original_img = cv2.imread(img_path)
        if original_img is None: return
        
        # 前處理 (0-1 Normalize + Tensor conversion)
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb / 255.0).permute(2, 0, 1).float().to(self.device)
        
        # 推論 (Inference Mode)
        with torch.no_grad():
            # model call with mode='inference' returns detections
            # Input 需要是 List[Tensor]
            predictions = self.model([img_tensor], mode='inference')
            
        prediction = predictions[0] # 取第一張圖的結果
        
        # 後處理與繪圖
        result_img = self.visualize(original_img, prediction)
        
        # 顯示或存檔
        if output_path:
            cv2.imshow('test', result_img)
            cv2.waitKey(10)
            cv2.imwrite(output_path, result_img)
            print(f"Saved result to: {output_path}")
        else:
            # 如果沒有輸出路徑，可以考慮 cv2.imshow (但在 Server 上通常不可用)
            pass

    def visualize(self, img, prediction):
        """ 繪製 BBox 與 Mask """
        vis_img = img.copy()
        
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        masks = prediction['masks'].cpu().numpy() # Shape: (N, 1, H, W)

        # 篩選分數大於閾值的物件
        valid_indices = scores > self.threshold
        
        for i in np.where(valid_indices)[0]:
            box = boxes[i].astype(int)
            score = scores[i]
            mask = masks[i, 0] # 取出 mask map (Float 0~1)
            
            # 隨機顏色
            color = (0, 255, 0) # Green for box
            mask_color = random_color()
            
            # 1. 畫 Bounding Box
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # 標籤文字
            label_text = f"Cabbage: {score:.2f}"
            cv2.putText(vis_img, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 2. 畫 Segmentation Mask (半透明)
            # Mask 閾值化 (通常 mask_rcnn 輸出是 soft mask)
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            if mask_binary.sum() > 0:
                # 建立彩色 Mask
                colored_mask = np.zeros_like(vis_img)
                colored_mask[mask_binary == 1] = mask_color
                
                # 疊加 (addWeighted)
                # alpha=1.0 (原圖), beta=0.5 (Mask透明度)
                mask_indices = mask_binary == 1
                vis_img[mask_indices] = cv2.addWeighted(
                    vis_img[mask_indices], 0.6, 
                    colored_mask[mask_indices], 0.4, 0
                ) # 注意這裡 numpy 的 shape trick，簡化寫法如下：
                
                # 更簡單的疊加寫法
                overlay = vis_img.copy()
                overlay[mask_binary == 1] = mask_color
                cv2.addWeighted(overlay, 0.4, vis_img, 0.6, 0, vis_img)

        return vis_img

def main():
    parser = argparse.ArgumentParser(description="Inference for DA-MaskRCNN")
    parser.add_argument('--weights', type=str, default='./runs/exp1_da_cabbage/best_model.pth', help='Path to trained .pth')
    parser.add_argument('--source', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--output', type=str, default='./inference_results', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    inferencer = Inferencer(args.weights, threshold=args.conf)
    
    # 判斷 source 是單張圖還是資料夾
    if os.path.isfile(args.source):
        filename = os.path.basename(args.source)
        save_path = os.path.join(args.output, f"pred_{filename}")
        inferencer.predict_image(args.source, save_path)
    elif os.path.isdir(args.source):
        # 遍歷資料夾
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        for root, dirs, files in os.walk(args.source):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    img_path = os.path.join(root, file)
                    save_path = os.path.join(args.output, f"pred_{file}")
                    inferencer.predict_image(img_path, save_path)
    else:
        print("Invalid source path.")

if __name__ == '__main__':
    main()