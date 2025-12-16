import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import sys
import numpy as np

from DAMASKRCNN.dataset import CustomDataset, collate_fn
from DAMASKRCNN.damaskrcnn import DAMaskRCNN

# 引入標準評估工具
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision

def calculate_batch_metrics(preds, targets, conf_threshold=0.5, iou_threshold=0.5):
    """
    手動計算 TP, FP, FN 以及數量，用於計算 Precision, Recall, F1
    """
    tp = 0
    fp = 0
    fn = 0
    
    pred_count = 0
    gt_count = 0

    for pred, target in zip(preds, targets):
        # 1. 根據信心度過濾預測框
        keep = pred['scores'] >= conf_threshold
        pred_boxes = pred['boxes'][keep]
        gt_boxes = target['boxes']

        # 統計數量
        pred_count += len(pred_boxes)
        gt_count += len(gt_boxes)

        # 處理邊界情況
        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue
        
        if len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue

        # 2. 計算 IoU (Intersection over Union)
        # 輸出形狀: (預測框數, 真實框數)
        iou_matrix = torchvision.ops.box_iou(pred_boxes, gt_boxes)

        # 3. 進行匹配 (Greedy Matching)
        matched_gt = set()
        
        # 對每一個預測框，找出重疊最大的真實框
        for i in range(len(pred_boxes)):
            max_iou, max_idx = torch.max(iou_matrix[i], dim=0)
            
            # 如果 IoU 大於門檻，且該真實框還沒被配對過 -> TP
            if max_iou >= iou_threshold and max_idx.item() not in matched_gt:
                tp += 1
                matched_gt.add(max_idx.item())
            else:
                # 否則 -> FP (誤判)
                fp += 1
        
        # 沒被配對到的真實框 -> FN (漏抓)
        fn += len(gt_boxes) - len(matched_gt)

    return tp, fp, fn, pred_count, gt_count

def evaluate(model, val_loader, device, conf_threshold=0.5):
    """
    計算 BBox 和 Mask 的 mAP，並額外計算 Precision, Recall, F1, Count
    """
    model.eval()
    
    # 1. 標準 mAP 計算器 (使用 torchmetrics)
    map_metric = MeanAveragePrecision(class_metrics=False, iou_type=['bbox', 'segm'])
    map_metric.to(device)

    # 2. 自定義統計變數
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred_count = 0
    total_gt_count = 0

    print(f"開始評估 (Confidence Threshold: {conf_threshold})...")
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            # 搬移資料
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 推論 (記得加 mode='inference')
            preds = model(images, mode='inference')

            # --- [關鍵修正] 格式轉換 ---
            # torchmetrics/pycocotools 要求 Mask 必須是 uint8
            # 且 Mask R-CNN 輸出 shape 為 (N, 1, H, W)，通常需要 squeeze 成 (N, H, W)
            
            for p in preds:
                # 預測 Mask: Float(0~1) -> Threshold(>0.5) -> Bool -> Uint8
                p['masks'] = (p['masks'] > 0.5).squeeze(1).to(dtype=torch.uint8)
            
            for t in targets:
                # 真實 Mask: 確保也是 Uint8 (有些 transform 可能會轉成 float)
                t['masks'] = t['masks'].to(dtype=torch.uint8)
            # --------------------------

            # 更新標準 mAP 指標
            map_metric.update(preds, targets)

            # 計算自定義指標
            tp, fp, fn, p_count, g_count = calculate_batch_metrics(
                preds, targets, conf_threshold=conf_threshold, iou_threshold=0.5
            )
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_pred_count += p_count
            total_gt_count += g_count

    print("正在計算統計數據...")
    
    # --- 計算 mAP 結果 ---
    map_results = map_metric.compute()

    # print(map_results.keys())
    
    # --- 計算自定義結果 ---
    epsilon = 1e-7 # 防止除以零
    
    # Precision = TP / (TP + FP)
    precision = total_tp / (total_tp + total_fp + epsilon)
    
    # Recall = TP / (TP + FN)
    recall = total_tp / (total_tp + total_fn + epsilon)
    
    # F1 Score = 2 * (P * R) / (P + R)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # 數量誤差
    count_diff = total_pred_count - total_gt_count
    mae = abs(count_diff) # 平均絕對誤差的概念，這裡是總數差異

    custom_results = {
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1_score,
        "Pred_Count": total_pred_count,
        "GT_Count": total_gt_count,
        "Count_Diff": count_diff
    }

    return map_results, custom_results

def print_results(map_results, custom_results):
    """ 格式化輸出結果 """
    print("\n" + "="*50)
    print(" >> Evaluation Results")
    print("="*50)

    print(f"{'Metric':<30} | {'Value':<10}")
    print("-" * 45)
    
    # 1. 標準 COCO mAP
    # map: IoU=0.50:0.95 的平均
    # map_50: IoU=0.50 的 mAP
    if 'bbox_map' in map_results:
        print(f"{'mAP (IoU=0.5:0.95)':<30} | {map_results['bbox_map'].item():.4f}")
    if 'bbox_map_50' in map_results:
        print(f"{'mAP (IoU=0.50)':<30} | {map_results['bbox_map_50'].item():.4f}")

    print("-" * 45)
    
    # 2. 自定義指標 (基於特定 Conf Threshold)
    print(f"{'Precision':<30} | {custom_results['Precision']:.4f}")
    print(f"{'Recall':<30} | {custom_results['Recall']:.4f}")
    print(f"{'F1 Score':<30} | {custom_results['F1_Score']:.4f}")
    
    print("-" * 45)
    
    # 3. 數量統計
    pred_c = custom_results['Pred_Count']
    gt_c = custom_results['GT_Count']
    diff = custom_results['Count_Diff']
    
    print(f"{'Total Predicted Count':<30} | {pred_c}")
    print(f"{'Total Ground Truth Count':<30} | {gt_c}")
    print(f"{'Difference (Pred - GT)':<30} | {diff}")
    
    # 計算計數準確率 (1 - 誤差率)
    if gt_c > 0:
        counting_acc = 1.0 - (abs(diff) / gt_c)
        print(f"{'Counting Accuracy':<30} | {max(0, counting_acc):.4f}")

    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--data_root', type=str, default='data/domain_a', help='Path to dataset root')
    parser.add_argument('--backbone', type=str, default='resnet34', help='Backbone name (resnet18/34/50)')
    # 新增參數：信心閾值
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for metrics (default: 0.5)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 準備資料集
    # 驗證集使用 valid subset
    val_dataset = CustomDataset(
        root_dir=args.data_root, 
        subset='valid',
        transforms=None 
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
    # 注意：這裡 num_classes=2 (背景+高麗菜)
    model = DAMaskRCNN(num_classes=2, backbone_name=args.backbone)
    
    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device)
    
    # 處理權重載入 (相容包含 model_state_dict 或直接存 dict 的情況)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)

    # 3. 執行評估
    map_res, custom_res = evaluate(model, val_loader, device, conf_threshold=args.conf)
    
    # 4. 顯示結果
    print_results(map_res, custom_res)

if __name__ == '__main__':
    main()