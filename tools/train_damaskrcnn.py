import torch
from torch.utils.data import DataLoader
from DAMASKRCNN.dataset import CustomDataset, collate_fn
from DAMASKRCNN.damaskrcnn import DAMaskRCNN
from DAMASKRCNN.trainer import Trainer
from torchvision.transforms import v2

def get_transforms(split):
    if split == 'train':
        return v2.Compose([
            # 1. 基礎轉 Tensor (0-255 uint8)
            # Dataset 已經輸出 Tensor 了，如果需要確保格式可用 v2.ToImage()
            
            # 2. 幾何變換 (同時作用於 Image, BBox, Mask)
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5), # 視資料集特性決定是否開啟

            v2.RandomShortestSize(
                min_size=(640, 672, 704, 736, 768, 800, 1000), 
                max_size=1000
            ),
            
            # 隨機縮放與裁切 (Scale Jitter) - 對 Mask R-CNN 很有用
            # v2.RandomZoomOut(fill={0:0, 1:0, 2:0}, side_range=(1.0, 1.5), p=0.5),
            # v2.RandomIoUCrop(), # 用於模擬遮擋
            
            # 隨機旋轉 (小心 BBox 會變大)
            # v2.RandomRotation(degrees=15),
            
            # 3. 色彩變換 (只作用於 Image)
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            v2.RandomGrayscale(p=0.1),
            
            # 4. 最終處理
            # 確保型態是 float32 並歸一化到 [0, 1]
            v2.ToDtype(torch.float32, scale=True),
        ])
    else:
        # 驗證集：只做格式轉換
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])

def main():
    # 1. 設定參數
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    BATCH_SIZE = 4
    NUM_WORKERS = 12
    EPOCHS = 200
    
    # [New] 定義增強
    train_transforms = get_transforms('train')
    val_transforms = get_transforms('valid')

    # [New] 傳入 transforms
    ds_source = CustomDataset(
        root_dir='data/domain_b', 
        subset='train', 
        transforms=train_transforms, # <--- 這裡
    )
    
    ds_target = CustomDataset(
        root_dir='data/domain_a', 
        subset='train', 
        transforms=train_transforms # Target Domain 也可以做增強
    )
    
    ds_val = CustomDataset(
        root_dir='data/domain_b', 
        subset='valid', 
        transforms=val_transforms # 驗證集不做增強
    )

    # DataLoaders
    loader_source = DataLoader(ds_source, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    loader_target = DataLoader(ds_target, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=NUM_WORKERS, collate_fn=collate_fn)
                               
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

    print(f"Source Images: {len(ds_source)}, Target Images: {len(ds_target)}")

    # 3. 初始化模型
    model = DAMaskRCNN(num_classes=2, backbone_name='resnet34') # 1 class (cabbage) + background

    # 4. 初始化 Trainer
    trainer = Trainer(
        model=model,
        source_loader=loader_source,
        target_loader=loader_target,
        val_loader=loader_val,
        device=DEVICE,
        save_dir='./runs/exp6_da_cabbage', # 輸出目錄
        lambda_domain=0.01,    # Domain Loss 的權重
        total_epochs=EPOCHS,
        save_interval=20,      # Example: Save at epoch 20, 40, 60, 80, 100
        patience=30,
        lr=1e-3,
        weight_decay=5e-4,
        yolo_path='runs/segment/train/weights/best.pt', # 指定你的 .pt 檔
        lambda_yolo=0.1,  # 建議從 0.1 開始嘗試
        scheduler='cosine',
    )

    # 5. 開始訓練
    # trainer.load_checkpoint('./runs/exp1_da_cabbage/model_latest.pth') # 如果要中斷續練
    trainer.train()

if __name__ == '__main__':
    main()