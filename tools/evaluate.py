import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import sys
from DAMASKRCNN.dataset import CustomDataset, collate_fn
from DAMASKRCNN.damaskrcnn import DAMaskRCNN
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint

def evaluate(model, val_loader, device):
    """
    計算 BBox 和 Mask 的 mAP 與 Recall
    """
    model.eval()
    
    # 初始化指標計算器
    # class_metrics=True: 會顯示每個類別的獨立分數
    # iou_type: 同時計算 BBox 和 Segmentation Mask
    metric = MeanAveragePrecision(class_metrics=True, iou_type=['bbox', 'segm'])
    metric.to(device)

    print("開始評估 (這可能需要一點時間)...")
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            # 1.搬移資料到 GPU
            images = [img.to(device) for img in images]
            
            # targets 需要留在 GPU 並且格式要正確
            # CustomDataset 的 target 已經是 dict 包含 boxes, labels, masks
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 2. 模型推論
            # model.eval() 模式下，Mask R-CNN 會回傳 predictions (List[Dict])
            preds = model(images)

            # 3. 更新指標
            # torchmetrics 接受 (preds, targets)
            metric.update(preds, targets)

    # 4. 計算最終結果
    print("正在計算統計數據...")
    result = metric.compute()
    
    return result

def print_results(result):
    """ 漂亮地印出結果 """
    
    # 定義要顯示的指標 Mapping
    # map: mAP (IoU=0.50:0.95)
    # map_50: mAP (IoU=0.50)
    # mar_100: Recall (最多考慮前100個框) -> 這是主要的 Recall 指標
    
    keys_to_print = [
        ('map', 'mAP (IoU=0.5:0.95)'),
        ('map_50', 'mAP (IoU=0.50)    '),
        ('map_75', 'mAP (IoU=0.75)    '),
        ('mar_100', 'Recall (AR@100)   ') 
    ]

    print("\n" + "="*40)
    print(" >> Evaluation Results (COCO Style)")
    print("="*40)

    # 分別印出 BBox 和 Mask 的結果
    # torchmetrics 的結果如果包含 segm，通常 key 會是一樣的，
    # 但如果用 iou_type=['bbox', 'segm']，它可能不會自動分層，
    # 實際上 torchmetrics 會回傳混合結果，或是我們需要分開宣告兩個 metric 物件比較保險。
    # 為了簡單起見，上面的 code 使用混合模式，這裡直接印出總表。
    
    # 為了更清楚區分 Box 和 Mask，通常建議跑兩次 metric 或看詳細 key
    # 但 torchmetrics 預設會輸出整體數值。
    
    # 讓我們檢查 result 的 keys 是否有區分 (通常沒有區分後綴，除非分開跑)
    # 這裡我們簡單印出所有數值
    
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 38)
    
    for key, value in result.items():
        # 只印出純量 (Scalar)，忽略 per_class 陣列
        if value.numel() == 1: 
            print(f"{key:<25} | {value.item():.4f}")
            
    print("-" * 38)
    print("解釋:")
    print(" - map: 綜合準確率 (越接近 1 越好)")
    print(" - map_50: 寬鬆準確率 (IoU>0.5 就算對)")
    print(" - mar_100: 召回率 (Recall)，代表有多少真值被抓出來")
    print("="*40)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--data_root', type=str, default='data/domain_a', help='Path to dataset root')
    parser.add_argument('--backbone', type=str, default='resnet34', help='Backbone name (resnet18/34/50)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 準備資料集
    # 注意：驗證時不要開強增強，也不要做多尺度，保持乾淨
    val_dataset = CustomDataset(
        root_dir=args.data_root, 
        subset='valid',
        transforms=None # 驗證集建議不要做額外增強，dataset 內部會做基本的 ToTensor
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    print(f"Validation Images: {len(val_dataset)}")

    # 2. 載入模型
    # 記得 pretrained=False，因為我們要載入自己的權重
    model = DAMaskRCNN(num_classes=2, backbone_name=args.backbone)
    
    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)

    # 3. 執行評估
    results = evaluate(model, val_loader, device)
    
    # 4. 顯示結果
    print_results(results)

if __name__ == '__main__':
    main()