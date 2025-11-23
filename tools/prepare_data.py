# prepare dual dataset DA DB
import json
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= 設定區 =================
DOMAIN_A_JSON = "./source_dir/cabbages/annotation.json"
DOMAIN_A_ROOT = "../source_dir/cabbages/images"

DOMAIN_B_JSON = "./source_dir/white_cabbage/annotation_white_cabbage.json"
DOMAIN_B_ROOT = "./source_dir/white_cabbage"

OUTPUT_BASE = "./data"
TARGET_SIZE = 800 # Mask R-CNN 常用輸入尺寸
SEED = 42
# =========================================

def parse_coco_format(json_data):
    """ 解析 COCO 格式 """
    images = json_data['images']
    img_map = {img['id']: img for img in images}
    ann_map = {}
    for ann in json_data['annotations']:
        ann_map.setdefault(ann['image_id'], []).append(ann)
    
    parsed_data = []
    for img_id, img_info in img_map.items():
        file_name = img_info['file_name']
        
        # 提取多邊形
        polygons = []
        if img_id in ann_map:
            for ann in ann_map[img_id]:
                if 'segmentation' in ann and ann['segmentation']:
                    # COCO segmentation 通常是 [[x1, y1, x2, y2...]]
                    for seg in ann['segmentation']:
                        poly = np.array(seg).flatten()
                        polygons.append(poly) # 這裡是絕對座標
        
        parsed_data.append({
            'file_name': file_name,
            'polygons': polygons,
            # COCO 通常有寬高，但為了保險起見，我們稍後讀圖時再確認或覆蓋
            'width': img_info.get('width'),
            'height': img_info.get('height') 
        })
    return parsed_data

def parse_custom_via_format(json_data):
    """ 解析 Custom/VIA 格式 (Key is filename) """
    parsed_data = []
    
    # 遍歷 dict: {"fname.jpg": [regions...]}
    for file_name, regions in json_data.items():
        polygons = []
        for region in regions:
            # 檢查是否有 shape_attributes
            attrs = region.get('shape_attributes', {})
            if attrs.get('name') == 'polygon':
                all_x = attrs.get('all_points_x', [])
                all_y = attrs.get('all_points_y', [])
                
                if not all_x or not all_y:
                    continue
                
                # 組合成 [x1, y1, x2, y2, ...]
                poly = []
                for x, y in zip(all_x, all_y):
                    poly.extend([x, y])
                polygons.append(np.array(poly))
        
        parsed_data.append({
            'file_name': file_name,
            'polygons': polygons,
            'width': None, # VIA 格式通常不包含圖片寬高，需要讀圖獲取
            'height': None
        })
    return parsed_data

def convert_to_yolo(json_file, source_root, output_dir, domain_name):
    print(f"\n>>> 正在處理 {domain_name} ...")
    
    # 1. 讀取與自動偵測格式
    with open(json_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 簡單的格式判斷邏輯
    if 'images' in raw_data and 'annotations' in raw_data:
        print(f"    偵測格式: COCO Standard")
        items = parse_coco_format(raw_data)
    else:
        print(f"    偵測格式: Custom / VIA Style")
        items = parse_custom_via_format(raw_data)

    if not items:
        print("    [警告] 未找到有效資料，跳過。")
        return

    # 2. 建立目錄
    for sub in ['train', 'valid']:
        os.makedirs(os.path.join(output_dir, sub, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, sub, 'labels'), exist_ok=True)

    # 3. 切分資料集
    train_items, val_items = train_test_split(items, test_size=0.1, random_state=SEED)
    splits = {'train': train_items, 'valid': val_items}

    # 4. 處理圖片與標註
    for subset, item_list in splits.items():
        for item in tqdm(item_list, desc=f"Converting {subset}"):
            fname = item['file_name']
            polygons = item['polygons']
            
            # 處理路徑 (有些 json 包含子資料夾，這裡假設 source_root 是圖片所在的根目錄)
            # 如果 fname 包含路徑分隔符，僅取檔名
            base_fname = os.path.basename(fname)
            src_path = os.path.join(source_root, base_fname)
            
            # 如果直接拼湊找不到，嘗試用原始 fname (應對資料夾結構)
            if not os.path.exists(src_path):
                src_path = os.path.join(source_root, fname)

            if not os.path.exists(src_path):
                # print(f"    [Skip] 找不到圖片: {src_path}")
                continue

            # 讀取圖片 (為了 Resize 和 歸一化)
            img = cv2.imread(src_path)
            if img is None: continue
            
            h, w = img.shape[:2]
            
            # Resize
            img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
            dst_img_path = os.path.join(output_dir, subset, 'images', base_fname)
            cv2.imwrite(dst_img_path, img_resized)

            # 產生 YOLO Label
            yolo_lines = []
            for poly in polygons:
                # poly 是絕對座標 [x1, y1, x2, y2...]
                # 歸一化
                poly = poly.astype(float)
                poly[0::2] /= w  # X 除以寬
                poly[1::2] /= h  # Y 除以高
                
                # Clip 確保在 0~1 之間
                poly = np.clip(poly, 0.0, 1.0)
                
                # 格式: <class-index> <x1> <y1> ...
                # 這裡假設只有一類 cabbage，index = 0
                line_str = "0 " + " ".join([f"{coord:.6f}" for coord in poly])
                yolo_lines.append(line_str)

            # 寫入 txt
            txt_name = os.path.splitext(base_fname)[0] + ".txt"
            dst_txt_path = os.path.join(output_dir, subset, 'labels', txt_name)
            with open(dst_txt_path, 'w') as f_txt:
                f_txt.write("\n".join(yolo_lines))

if __name__ == "__main__":
    # 處理 Domain A
    if os.path.exists(DOMAIN_A_JSON):
        convert_to_yolo(DOMAIN_A_JSON, DOMAIN_A_ROOT, os.path.join(OUTPUT_BASE, "domain_a"), "Domain A")
    else:
        print(f"找不到 Domain A 標註檔: {DOMAIN_A_JSON}")

    # 處理 Domain B
    if os.path.exists(DOMAIN_B_JSON):
        convert_to_yolo(DOMAIN_B_JSON, DOMAIN_B_ROOT, os.path.join(OUTPUT_BASE, "domain_b"), "Domain B")
    else:
        print(f"找不到 Domain B 標註檔: {DOMAIN_B_JSON}")

    print("\n所有資料轉換完成！")