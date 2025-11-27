import torch
import torch.optim as optim
import os, sys
import time
import math
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter  # 如果有安裝 tensorboard
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model,
        source_loader,
        target_loader,
        val_loader=None,
        optimizer=None,
        scheduler=None,
        device='cuda',
        save_dir='./checkpoints',
        lambda_domain=0.1,  # Domain Loss 的權重
        total_epochs=20,
        print_freq=10
    ):
        self.model = model.to(device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.lambda_domain = lambda_domain
        self.total_epochs = total_epochs
        self.print_freq = print_freq
        
        # 初始化優化器
        if optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        else:
            self.optimizer = optimizer
            
        self.scheduler = scheduler
        
        # 建立存檔目錄
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Tensorboard Writer (可選)
        # self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
        
        self.global_step = 0
        self.current_epoch = 0

    def calculate_alpha(self, current_step, total_steps):
        """ 
        計算 GRL (Gradient Reversal Layer) 的 Alpha 值 
        隨著訓練進行，從 0 逐漸增加到 1
        """
        p = float(current_step) / total_steps
        return 2. / (1. + math.exp(-10 * p)) - 1

    def train(self):
        """ 主訓練迴圈 """
        print(f"Start training for {self.total_epochs} epochs...")
        start_time = time.time()

        for epoch in range(self.current_epoch, self.total_epochs):
            self.current_epoch = epoch
            
            # 1. 訓練一個 Epoch
            self.train_one_epoch()
            
            # 2. 更新 Learning Rate
            if self.scheduler:
                self.scheduler.step()
            
            # 3. 驗證 (如果有提供驗證集)
            if self.val_loader:
                self.validate()
            
            # 4. 存檔
            self.save_checkpoint()

        total_time = time.time() - start_time
        print(f"Training finished in {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m")
        # self.writer.close()

    def train_one_epoch(self):
        self.model.train()
        
        # Target Loader 的 Iterator (因為長度可能跟 Source 不一樣)
        target_iter = iter(self.target_loader)
        len_source = len(self.source_loader)
        
        # 進度條
        pbar = tqdm(enumerate(self.source_loader), total=len_source, desc=f"Epoch {self.current_epoch+1}/{self.total_epochs}")

        for i, (images_s, targets_s) in pbar:
            # --- 1. 準備資料 ---
            images_s = list(img.to(self.device) for img in images_s)
            targets_s = [{k: v.to(self.device) for k, v in t.items()} for t in targets_s]
            
            # 獲取 Target Batch (循環讀取)
            try:
                images_t, _ = next(target_iter) # Target 不需要 Label
            except StopIteration:
                target_iter = iter(self.target_loader)
                images_t, _ = next(target_iter)
            images_t = list(img.to(self.device) for img in images_t)

            # --- 2. 計算 Alpha ---
            # 總步數 = Epochs * len(Loader)
            total_steps = self.total_epochs * len_source
            alpha = self.calculate_alpha(self.global_step, total_steps)

            # --- 3. Forward Pass ---
            self.optimizer.zero_grad()

            # A. Source Domain: 訓練 Detection + Domain Classifier
            loss_dict_s = self.model(
                images_s, targets_s, 
                domain_label=0, # Source Label
                alpha=alpha, 
                mode='source'
            )
            
            # B. Target Domain: 訓練 Domain Classifier (對抗式)
            loss_dict_t = self.model(
                images_t, None, 
                domain_label=1, # Target Label
                alpha=alpha, 
                mode='target'
            )

            # --- 4. Loss 計算 ---
            # Detection Losses (RPN + ROI)
            loss_det = sum(loss for k, loss in loss_dict_s.items() if k != 'domain_loss')
            
            # Domain Loss (Source + Target)
            loss_dom = loss_dict_s['domain_loss'] + loss_dict_t['domain_loss']
            
            # Total Loss
            losses = loss_det + self.lambda_domain * loss_dom

            # --- 5. Backward & Optimize ---
            if not math.isfinite(losses.item()):
                print(f"Loss is {losses.item()}, stopping training")
                print(loss_dict_s)
                sys.exit(1)

            losses.backward()
            self.optimizer.step()

            # --- 6. Logging ---
            self.global_step += 1
            
            # 更新進度條資訊
            curr_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'det_loss': f"{loss_det.item():.3f}",
                'dom_loss': f"{loss_dom.item():.3f}",
                'lr': f"{curr_lr:.5f}"
            })

            # Tensorboard 紀錄
            # if self.global_step % self.print_freq == 0:
            #     self.writer.add_scalar('Loss/Total', losses.item(), self.global_step)
            #     self.writer.add_scalar('Loss/Detection', loss_det.item(), self.global_step)
            #     self.writer.add_scalar('Loss/Domain', loss_dom.item(), self.global_step)
            #     self.writer.add_scalar('Params/Alpha', alpha, self.global_step)
            #     self.writer.add_scalar('Params/LR', curr_lr, self.global_step)

    def validate(self):
        """ 
        簡易驗證 
        注意: Mask R-CNN 在 eval() 模式下只會回傳預測結果，不會回傳 Loss。
        若要計算 Validation Loss，需保持 train() 模式但關閉梯度。
        """
        print("Validating...")
        # 這裡我們使用 'training mode with no_grad' 來獲取 Validation Loss
        # 如果要算 mAP，需要另外寫 evaluator，這裡先計算 Loss 指標
        self.model.train() 
        
        total_val_loss = 0
        steps = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 只算 Source Validation 的 Detection Loss
                loss_dict = self.model(images, targets, mode='source')
                
                # 只取 Detection 相關 Loss
                loss = sum(loss for k, loss in loss_dict.items() if k != 'domain_loss')
                total_val_loss += loss.item()
                steps += 1
        
        avg_loss = total_val_loss / steps
        print(f"Validation Detection Loss: {avg_loss:.4f}")
        # self.writer.add_scalar('Loss/Validation', avg_loss, self.current_epoch)

    def save_checkpoint(self):
        checkpoint_path = os.path.join(self.save_dir, f'model_epoch_{self.current_epoch+1}.pth')
        latest_path = os.path.join(self.save_dir, 'model_latest.pth')
        
        state = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step
        }
        
        torch.save(state, checkpoint_path)
        torch.save(state, latest_path)
        # print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        print(f"Loaded checkpoint from {path} (Epoch {self.current_epoch})")