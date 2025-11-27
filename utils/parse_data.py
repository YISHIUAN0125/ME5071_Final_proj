import json
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

class DatasetBuilder:
    def __init__(self, target_size=(640, 640), val_ratio=0.2, test_ratio=0.1, seed=42):
        self.target_size = target_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.all_items = []  # 儲存圖片資料

    def add_source(self, json_path, source_img_root, prefix=""):
        """
        加入資料來源
        :param prefix: 檔名前綴 (混合資料集時用來防止檔名衝突，單獨資料集可留空)
        """
        if not os.path.exists(json_path):
            print(f"[跳過] 找不到檔案: {json_path}")
            return

        print(f"[讀取中] {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        parsed_items = []
        if isinstance(data, dict) and 'images' in data and 'annotations' in data:
            parsed_items = self._parse_coco(data)
        elif isinstance(data, dict):
            parsed_items = self._parse_custom_via(data)
        
        # 加入來源資訊
        for item in parsed_items:
            item['source_root'] = source_img_root
            item['prefix'] = prefix
            self.all_items.append(item)
        
        print(f"   >>> 加入 {len(parsed_items)} 張圖片 (Prefix: '{prefix}')")

    def export(self, output_dir):
        """ 輸出處理後的資料集 """
        if not self.all_items:
            print(f"[警告] {output_dir} 沒有資料可輸出，跳過。")
            return

        print(f"\n[處理中] 正在輸出至: {output_dir}")
        self._make_dirs(output_dir)
        
        # 分割資料
        splits = self._split_data(self.all_items)
        
        # 寫入檔案
        for subset, items in splits.items():
            if not items: continue
            for item in tqdm(items, desc=f"Exporting {subset}"):
                self._process_and_save(item, output_dir, subset)

        # 產生 YAML
        self._create_yaml(output_dir)
        print(f"[完成] 輸出完畢: {output_dir}\n" + "="*40)

    # ---------------- 內部邏輯 ----------------
    def _make_dirs(self, output_dir):
        for sub in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(output_dir, sub, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, sub, 'labels'), exist_ok=True)

    def _split_data(self, items):
        # 確保有足夠數量進行分割
        if len(items) < 5:
            return {'train': items, 'valid': [], 'test': []}

        train_items, temp = train_test_split(items, test_size=(self.val_ratio + self.test_ratio), random_state=self.seed)
        
        if self.test_ratio == 0:
            val_items, test_items = temp, []
        else:
            # 計算剩下的比例
            relative_test_ratio = self.test_ratio / (self.val_ratio + self.test_ratio)
            val_items, test_items = train_test_split(temp, test_size=relative_test_ratio, random_state=self.seed)
            
        return {'train': train_items, 'valid': val_items, 'test': test_items}

    def _process_and_save(self, item, output_dir, subset):
        rel_path = item['file_name']
        source_root = item['source_root']
        prefix = item['prefix']

        # 找尋圖片路徑的容錯邏輯
        possible_paths = [
            os.path.join(source_root, rel_path),
            os.path.join(source_root, os.path.basename(rel_path))
        ]

        src_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if not src_path: return

        try:
            img = cv2.imread(src_path)
            if img is None: return
            h, w = img.shape[:2]
            
            # Resize
            img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

            # 檔名處理
            basename = os.path.basename(rel_path)
            new_fname = f"{prefix}{basename}"
            new_fname_no_ext = os.path.splitext(new_fname)[0]

            dst_img = os.path.join(output_dir, subset, 'images', new_fname)
            dst_lbl = os.path.join(output_dir, subset, 'labels', new_fname_no_ext + '.txt')

            cv2.imwrite(dst_img, img_resized)

            # 座標轉換
            lines = []
            for poly in item['polygons']:
                poly = poly.astype(float)
                poly[0::2] /= w
                poly[1::2] /= h
                poly = np.clip(poly, 0.0, 1.0)
                coords = ' '.join([f"{c:.6f}" for c in poly])
                lines.append(f"0 {coords}")

            if lines:
                with open(dst_lbl, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines) + '\n')
        except Exception:
            pass

    def _create_yaml(self, output_dir):
        content = {
            'path': os.path.abspath(output_dir),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': {0: 'cabbage'}
        }
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            yaml.dump(content, f, sort_keys=False)

    def _parse_coco(self, data):
        imgs = {i['id']: i for i in data['images']}
        anns = {}
        for a in data['annotations']:
            anns.setdefault(a['image_id'], []).append(a)
        
        res = []
        for img_id, img_info in imgs.items():
            fname = img_info.get('file_name', '')
            path = img_info.get('path', fname)

            if path.startswith('/'):
                path = path[1:]

            polys = []
            if img_id in anns:
                for a in anns[img_id]:
                    if a.get('segmentation'):
                        for s in a['segmentation']:
                            polys.append(np.array(s).flatten())
            res.append({'file_name': path, 'polygons': polys})
        return res

    def _parse_custom_via(self, data):
        res = []
        for k, v in data.items():
            regions = v if isinstance(v, list) else v.get('regions', [])
            fname = k if isinstance(v, list) else v.get('filename', k)
            polys = []
            for r in regions:
                attrs = r.get('shape_attributes', {})
                if attrs.get('name') == 'polygon':
                    ax, ay = attrs.get('all_points_x', []), attrs.get('all_points_y', [])
                    if ax and ay:
                        pts = []
                        for x, y in zip(ax, ay): pts.extend([x, y])
                        polys.append(np.array(pts))
            res.append({'file_name': fname, 'polygons': polys})
        return res

# ================= 主程式：分別產生 A, B, Mixed =================
if __name__ == '__main__':
    
    # 設定路徑常數
    PATH_A_JSON = 'source_dir/cabbages/annotation.json'
    PATH_A_ROOT = 'source_dir/cabbages/images'
    
    PATH_B_JSON = 'source_dir/white_cabbage/annotation_white_cabbage.json'
    PATH_B_ROOT = 'source_dir/white_cabbage'

    # 1. 產生 Domain A 單獨資料集
    builder_a = DatasetBuilder()
    builder_a.add_source(PATH_A_JSON, PATH_A_ROOT, prefix="") # 單獨產生時不加前綴，保持原檔名
    builder_a.export('data/domain_a')

    # 2. 產生 Domain B 單獨資料集
    builder_b = DatasetBuilder()
    builder_b.add_source(PATH_B_JSON, PATH_B_ROOT, prefix="")
    builder_b.export('data/domain_b')

    # 3. 產生 Mixed (A + B) 資料集
    builder_mix = DatasetBuilder()
    
    # 這裡加入 prefix 是為了避免兩邊有檔名重複 (例如都有 001.jpg)
    builder_mix.add_source(PATH_A_JSON, PATH_A_ROOT, prefix="A_") 
    builder_mix.add_source(PATH_B_JSON, PATH_B_ROOT, prefix="B_")
    
    builder_mix.export('data/mixed')

    print("\n全部處理完成！")