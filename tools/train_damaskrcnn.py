import torch
from torch.utils.data import DataLoader
from DAMASKRCNN.dataset import CustomDataset, collate_fn
from DAMASKRCNN.damaskrcnn import DAMaskRCNN
from DAMASKRCNN.trainer import Trainer

def main():
    # 1. 設定參數
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    EPOCHS = 20
    
    # 2. 準備資料集
    # Source: Domain A (有標籤)
    ds_source = CustomDataset(root_dir='data/domain_a', subset='train')
    
    # Target: Domain B (假設無標籤，但在 DA 中我們用它的圖片來做對齊)
    ds_target = CustomDataset(root_dir='data/domain_b', subset='train')
    
    # Validation: Domain A Valid (用來監控基本性能)
    ds_val = CustomDataset(root_dir='data/domain_a', subset='valid')

    # DataLoaders
    loader_source = DataLoader(ds_source, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    loader_target = DataLoader(ds_target, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=NUM_WORKERS, collate_fn=collate_fn)
                               
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

    print(f"Source Images: {len(ds_source)}, Target Images: {len(ds_target)}")

    # 3. 初始化模型
    model = DAMaskRCNN(num_classes=2) # 1 class (cabbage) + background

    # 4. 初始化 Trainer
    trainer = Trainer(
        model=model,
        source_loader=loader_source,
        target_loader=loader_target,
        val_loader=loader_val,
        device=DEVICE,
        save_dir='./runs/exp1_da_cabbage', # 輸出目錄
        lambda_domain=0.1,    # Domain Loss 的權重
        total_epochs=EPOCHS
    )

    # 5. 開始訓練
    # trainer.load_checkpoint('./runs/exp1_da_cabbage/model_latest.pth') # 如果要中斷續練
    trainer.train()

if __name__ == '__main__':
    main()