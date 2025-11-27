import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import os
from DAMASKRCNN.dataset import CustomDataset, collate_fn
from DAMASKRCNN.damaskrcnn import DAMaskRCNN

# ================= 設定 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2 # 建議至少 2 (1 Source + 1 Target)
LR = 0.005
NUM_EPOCHS = 10
LAMBDA_DA = 0.1 # Domain Loss 權重
# =======================================

def main():
    # 1. 準備資料路徑
    print("正在準備 Dataset...")
    # Domain B 的圖片路徑 (給 Domain A 做 Fourier Aug 用)
    domain_b_imgs = glob.glob("data/domain_b/train/images/*.jpg")
    
    # Dataset A (Source): 開啟 Fourier Augmentation
    ds_source = CustomDataset(
        "data/domain_a", 
        subset='train', 
        target_img_paths=domain_b_imgs, 
        fourier_prob=0.5 # 50% 機率混合 Domain B 風格
    )
    
    # Dataset B (Target): 用於 Adversarial Training (無 Fourier)
    ds_target = CustomDataset(
        "data/domain_b", 
        subset='train',
        fourier_prob=0.0
    )

    loader_source = DataLoader(ds_source, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    loader_target = DataLoader(ds_target, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)

    # 2. 模型與優化器
    model = DAMaskRCNN().to(DEVICE)
    
    # 分組參數：Discriminator 學習率可以設小一點
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(f"開始訓練... Device: {DEVICE}")

    # 3. 訓練迴圈
    for epoch in range(NUM_EPOCHS):
        model.train()
        iter_target = iter(loader_target)
        
        total_loss_det = 0
        total_loss_da = 0

        # 以 Source 為主迴圈
        for batch_idx, (imgs_s, targets_s) in enumerate(loader_source):
            # A. 獲取 Source Data
            imgs_s = [img.to(DEVICE) for img in imgs_s]
            targets_s = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets_s]

            # B. 獲取 Target Data (如果 Target 用完了就重置)
            try:
                imgs_t, _ = next(iter_target)
            except StopIteration:
                iter_target = iter(loader_target)
                imgs_t, _ = next(iter_target)
            
            imgs_t = [img.to(DEVICE) for img in imgs_t]

            # --- Forward ---
            # Alpha (GRL強度) 可以隨訓練進程增加，這裡先固定
            alpha = 1.0 

            # 1. Source Flow: Detection Loss + Domain Loss (Source Label)
            loss_dict_s = model(imgs_s, targets_s, mode='source', alpha=alpha)
            
            # 2. Target Flow: Domain Loss (Target Label)
            loss_dict_t = model(imgs_t, mode='target', alpha=alpha)

            # --- Loss Aggregation ---
            det_losses = sum(loss for k, loss in loss_dict_s.items() if 'domain' not in k)
            da_loss_s = loss_dict_s['domain_loss']
            da_loss_t = loss_dict_t['domain_loss']
            
            total_da_loss = (da_loss_s + da_loss_t) * 0.5
            
            final_loss = det_losses + (LAMBDA_DA * total_da_loss)

            # --- Backward ---
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            total_loss_det += det_losses.item()
            total_loss_da += total_da_loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{batch_idx}] "
                      f"Det Loss: {det_losses.item():.4f} | DA Loss: {total_da_loss.item():.4f}")

        lr_scheduler.step()
        
        torch.save(model.state_dict(), f"da_maskrcnn_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch+1} 完成。平均 Det Loss: {total_loss_det/len(loader_source):.4f}")

if __name__ == "__main__":
    main()