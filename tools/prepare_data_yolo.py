import json
import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 全域設定
TARGET_SIZE = 1000 # 圖片將被 Resize 成 1000x1000

def prepare_dataset_yolo_seg_custom(
    annotation_file,     # 原始 annotation.json 路徑
    source_root_dir,     # 圖片的根目錄 (例如: ./source_dir/cabbages/images)
    output_base_dir,     # 輸出的根目錄 (例如: ./data)
    val_ratio=0.15,      # 驗證集比例
    test_ratio=0.10,     # 測試集比例
    seed=42
):
    # 實際的 YOLO 資料集根目錄
    yolo_root_dir = os.path.join(output_base_dir, 'yolo')
    
    # --- 1. 資料夾結構初始化 ---
    subsets = ['train', 'valid', 'test']
    for sub in subsets:
        # 建立 data/yolo/{train, valid, test}/images
        os.makedirs(os.path.join(yolo_root_dir, sub, 'images'), exist_ok=True)
        # 建立 data/yolo/{train, valid, test}/labels
        os.makedirs(os.path.join(yolo_root_dir, sub, 'labels'), exist_ok=True)
    
    # --- 2. 讀取標註檔 ---
    print(f"正在讀取標註檔: {annotation_file} ...")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # 建立索引
    img_id_to_anns = {}
    for ann in annotations:
        img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    # --- 3. 資料切分 (隨機切分) ---
    print("正在進行隨機切分...")
    img_ids = [img['id'] for img in images]
    
    # 計算 train/val/test 數量
    temp_ratio = val_ratio + test_ratio
    
    train_ids, temp_ids = train_test_split(
        img_ids, test_size=temp_ratio, random_state=seed
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=test_ratio / temp_ratio, random_state=seed
    )
    
    dataset_splits = {
        'train': set(train_ids),
        'valid': set(val_ids),
        'test': set(test_ids)
    }
    
    # --- 4. 轉換與複製 ---
    print(f"開始轉換、Resize ({TARGET_SIZE}x{TARGET_SIZE}) 並複製檔案...")

    # Class ID 映射：cabbage 類別對應到 YOLO Index 0
    yolo_class_index = 0
    
    for img_info in tqdm(images, desc="處理圖片"):
        img_id = img_info['id']
        file_name = img_info['file_name']
        original_width = img_info['width']
        original_height = img_info['height']
        
        # A. 決定圖片屬於哪個子集
        subset_name = next(
            (name for name, ids_set in dataset_splits.items() if img_id in ids_set), 
            None
        )
        if not subset_name: continue
        
        # B. 構造來源圖片路徑 (處理 JSON 中包含子目錄的路徑)
        rel_path = img_info.get('path', file_name)
        if rel_path.startswith('/'): rel_path = rel_path[1:]
        src_img_path = os.path.join(source_root_dir, rel_path)
        
        # C. 構造輸出路徑 (符合 data/yolo/{sub}/images 和 labels 結構)
        file_basename = os.path.basename(file_name)
        file_name_no_ext, _ = os.path.splitext(file_basename)
        
        dst_img_path = os.path.join(yolo_root_dir, subset_name, 'images', file_basename)
        dst_label_path = os.path.join(yolo_root_dir, subset_name, 'labels', file_name_no_ext + '.txt')

        # D. 載入、Resize 圖片
        if not os.path.exists(src_img_path):
            print(f"[警告] 找不到圖片: {src_img_path}")
            continue
            
        try:
            img = cv2.imread(src_img_path)
            if img is None:
                print(f"[警告] 無法讀取圖片: {src_img_path}")
                continue
                
            # Resize 圖片
            img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(dst_img_path, img_resized)
            
        except Exception as e:
            print(f"[錯誤] 處理圖片 {file_name} 失敗: {e}")
            continue

        # E. 轉換標註到 YOLO Seg 格式
        yolo_lines = []
        if img_id in img_id_to_anns:
            for ann in img_id_to_anns[img_id]:
                if 'segmentation' in ann and ann['segmentation']:
                    seg_data = ann['segmentation']
                    if not seg_data or not isinstance(seg_data[0], list) or len(seg_data[0]) < 6:
                        continue 

                    yolo_line = str(yolo_class_index)
                    
                    # 獲取多邊形座標 [x1, y1, x2, y2, ...]
                    polygon = np.array(seg_data[0]).flatten()
                    
                    # 歸一化座標 (除以原始寬高)
                    polygon[::2] /= original_width   # X 座標
                    polygon[1::2] /= original_height # Y 座標
                    
                    # 確保所有座標都在 [0, 1] 範圍內 (防止越界)
                    polygon = np.clip(polygon, 0.0, 1.0) 
                    
                    normalized_coords = ' '.join(f"{coord:.6f}" for coord in polygon)
                    yolo_line += " " + normalized_coords
                    yolo_lines.append(yolo_line)

        # F. 儲存 YOLO 標籤檔案 (只在有標註時儲存)
        if yolo_lines:
            with open(dst_label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines) + '\n')
            
    # --- 5. 生成 data.yaml (YOLOv8 配置檔) ---
    yaml_content = f"""
# YOLOv8 配置檔 - Cabbage Segmentation
path: {os.path.abspath(yolo_root_dir)}
train: train/images
val: valid/images
test: test/images

# 類別數量
nc: 1

# 類別名稱 (0 必須對應 cabbage)
names: ['cabbage']
"""
    yaml_path = os.path.join(output_base_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print("\n=============================================")
    print("YOLO Seg 資料集轉換完成！")
    print(f"訓練資料夾路徑: {os.path.abspath(yolo_root_dir)}")
    print(f"配置文件 data.yaml 已生成於 {yaml_path}")
    print("=============================================")


# ==========================================
# 執行設定 (請根據您的環境修改)
# ==========================================
if __name__ == "__main__":
    
    # 1. 您的原始標註檔
    ANNOTATION_FILE = "./source_dir/cabbages/annotation.json" 
    
    # 2. 圖片的根目錄 (所有圖片的共同父資料夾)
    # 例如: /path/to/source_dir/cabbages/images
    SOURCE_ROOT = "./source_dir/cabbages/images" 
    
    # 3. 輸出的根目錄 (將生成 ./data/yolo 和 ./data/data.yaml)
    OUTPUT_BASE_DIR = "./data"
    
    # 檢查檔案是否存在
    if os.path.exists(ANNOTATION_FILE) and os.path.exists(SOURCE_ROOT):
        prepare_dataset_yolo_seg_custom(
            ANNOTATION_FILE, 
            SOURCE_ROOT, 
            OUTPUT_BASE_DIR
        )
    else:
        print("錯誤：找不到 annotation.json 或 圖片根目錄，請檢查路徑設定。")