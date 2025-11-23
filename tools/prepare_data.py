import json
import os
import shutil
from PIL import Image, ImageOps
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

def letterbox_resize(img_path, target_size=1000):
    """Letterbox resize to target_size x target_size, return new image + scale + padding"""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    new_img.paste(resized, (pad_left, pad_top))

    return new_img, scale, (pad_left, pad_top), (orig_w, orig_h)


def scale_bbox(bbox, scale, pad):
    x, y, w, h = bbox
    pad_l, pad_t = pad
    return [
        x * scale + pad_l,
        y * scale + pad_t,
        w * scale,
        h * scale
    ]


def scale_polygon(poly, scale, pad):
    poly = np.array(poly).reshape(-1, 2)
    poly = poly * scale
    poly[:, 0] += pad[0]
    poly[:, 1] += pad[1]
    return poly.flatten().tolist()


def scale_segmentation(seg, scale, pad):
    if isinstance(seg, dict):  # RLE (目前你的資料沒有)
        return seg
    elif isinstance(seg, list):
        if not seg:
            return seg
        # CVAT 輸出是 [[x1,y1,x2,y2,...]] 單一多邊形
        if isinstance(seg[0], list):
            return [scale_polygon(p, scale, pad) for p in seg]
        else:
            return [scale_polygon(seg, scale, pad)]
    return seg


def prepare_cvat_dataset_for_training(
    annotation_file,
    source_root_dir,      # 你的 images 根目錄，例如 "./source_dir/cabbages/images"
    output_dir="data_1000_letterbox",
    target_size=1000,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    print(f"正在讀取 CVAT 標註檔：{annotation_file}")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = data['images']
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])

    print(f"總共 {len(images)} 張圖片，開始統一 resize 到 {target_size}×{target_size} (letterbox)...")

    processed_images = []
    processed_annotations = []
    temp_image_dir = os.path.join(output_dir, "_temp_images")
    os.makedirs(temp_image_dir, exist_ok=True)

    for img_info in tqdm(images, desc="Resize 圖片與標註"):
        img_id = img_info['id']
        
        # 從 path 提取相對路徑（CVAT 格式）
        rel_path = img_info['path']
        if rel_path.startswith('/'):
            rel_path = rel_path[1:]
        src_path = os.path.join(source_root_dir, rel_path)

        if not os.path.exists(src_path):
            print(f"[警告] 找不到圖片：{src_path}")
            continue

        # Letterbox resize
        new_img, scale, (pad_l, pad_t), (orig_w, orig_h) = letterbox_resize(src_path, target_size)

        # 新檔名（保留原始檔名）
        new_filename = img_info['file_name']

        # 更新 image info
        new_img_info = img_info.copy()
        new_img_info['file_name'] = new_filename
        new_img_info['width'] = target_size
        new_img_info['height'] = target_size
        # 移除 CVAT 特有且不需要的欄位（避免後續框架報錯）
        for key in ['path', 'annotated', 'annotating', 'num_annotations', 'metadata',
                    'deleted', 'milliseconds', 'events', 'regenerate_thumbnail']:
            new_img_info.pop(key, None)

        processed_images.append(new_img_info)

        # 處理該圖的所有標註
        for ann in annotations:
            if ann['image_id'] != img_id:
                continue
            new_ann = ann.copy()

            # 縮放 bbox
            new_ann['bbox'] = scale_bbox(ann['bbox'], scale, (pad_l, pad_t))

            # 縮放 area
            new_ann['area'] = ann['area'] * scale * scale

            # 縮放 segmentation
            if 'segmentation' in ann and ann['segmentation']:
                new_ann['segmentation'] = scale_segmentation(
                    ann['segmentation'], scale, (pad_l, pad_t)
                )

            # 移除不需要的欄位
            for key in ['color', 'metadata', 'isbbox', 'iscrowd']:
                new_ann.pop(key, None)

            processed_annotations.append(new_ann)

        # 儲存 resize 後的圖片到暫存
        new_img.save(os.path.join(temp_image_dir, new_filename))

    print(f"Resize 完成，共處理 {len(processed_images)} 張圖片")

    # 隨機切分
    img_ids = [img['id'] for img in processed_images]
    train_ids, remain_ids = train_test_split(img_ids, test_size=(val_ratio + test_ratio), random_state=seed)
    val_ids, test_ids = train_test_split(remain_ids, test_size=test_ratio/(val_ratio + test_ratio), random_state=seed)

    splits = {
        'train': train_ids,
        'valid': val_ids,
        'test' : test_ids
    }

    # 產生最終資料夾與 JSON
    for split_name, ids_set in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        img_list = [img for img in processed_images if img['id'] in ids_set]
        ann_list = [ann for ann in processed_annotations if ann['image_id'] in ids_set]

        print(f"正在產生 {split_name} ({len(img_list)} 張)...")

        for img_info in img_list:
            src = os.path.join(temp_image_dir, img_info['file_name'])
            dst = os.path.join(split_dir, img_info['file_name'])
            if os.path.exists(src):
                shutil.move(src, dst)

        # 產生標準 COCO JSON
        coco_json = {
            "images": img_list,
            "annotations": ann_list,
            "categories": categories or [{"id": 1, "name": "cabbage"}]
        }

        with open(os.path.join(split_dir, "_annotations.coco.json"), 'w', encoding='utf-8') as f:
            json.dump(coco_json, f, ensure_ascii=False, indent=4)

    # 清理暫存
    shutil.rmtree(temp_image_dir)
    print(f"\n全部完成！資料已輸出至：{output_dir}")
    print("   ├─ train/")
    print("   ├─ valid/")
    print("   └─ test/")
    print("每個資料夾內都有圖片 + _annotations.coco.json，可直接餵給任何框架訓練！")


# ========================================
# 執行區塊（直接改這邊就行）
# ========================================
if __name__ == "__main__":
    # 請依照你的實際路徑修改下面兩行
    ANNOTATION_FILE = "./source_dir/cabbages/annotation.json"
    SOURCE_ROOT     = "./source_dir/cabbages/images"   # 你的原始圖片根目錄

    OUTPUT_DIR = "data"

    prepare_cvat_dataset_for_training(
        annotation_file=ANNOTATION_FILE,
        source_root_dir=SOURCE_ROOT,
        output_dir=OUTPUT_DIR,
        target_size=1000,
        val_ratio=0.2,
        test_ratio=0.1,   # 可改成 0.2 + 0.0 如果不要 test
        seed=42
    )