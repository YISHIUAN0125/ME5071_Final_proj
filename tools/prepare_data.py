from utils.parse_data import DatasetBuilder

# 設定路徑常數
PATH_A_JSON = 'source_dir/cabbages/annotation.json'
PATH_A_ROOT = 'source_dir/cabbages/images'

PATH_B_JSON = 'source_dir/white_cabbage/annotation_white_cabbage.json'
PATH_B_ROOT = 'source_dir/white_cabbage'
# 1. 產生 Domain A 單獨資料集
builder_a = DatasetBuilder(target_size=(1000, 1000))
builder_a.add_source(PATH_A_JSON, PATH_A_ROOT, prefix="") # 單獨產生時不加前綴，保持原檔名
builder_a.export('data/domain_a')
# 2. 產生 Domain B 單獨資料集
builder_b = DatasetBuilder(target_size=(1000, 1000))
builder_b.add_source(PATH_B_JSON, PATH_B_ROOT, prefix="")
builder_b.export('data/domain_b')
# 3. 產生 Mixed (A + B) 資料集
builder_mix = DatasetBuilder(target_size=(1000, 1000))

# 這裡加入 prefix 是為了避免兩邊有檔名重複 (例如都有 001.jpg)
builder_mix.add_source(PATH_A_JSON, PATH_A_ROOT, prefix="A_") 
builder_mix.add_source(PATH_B_JSON, PATH_B_ROOT, prefix="B_")

builder_mix.export('data/mixed')
print("\n全部處理完成！")