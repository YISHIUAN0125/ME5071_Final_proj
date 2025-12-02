import torch
import torch.optim as optim
import os
import time
import math
import sys
from tqdm import tqdm
import csv
from .yolo_teacher import YOLOTeacher

class Trainer:
    def __init__(
        self,
        model,
        source_loader,
        target_loader,
        val_loader=None,
        optimizer=None,
        scheduler='step',
        lr=5e-4,
        weight_decay=5e-4,
        device='cuda',
        save_dir='./checkpoints',
        lambda_domain=0.001,
        total_epochs=20,
        print_freq=10,
        patience=5,
        min_delta=0.001,
        save_interval=20,
        yolo_path=None, 
        lambda_yolo=0.1,
        **kwargs
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

        self.yolo_path = yolo_path
        self.lambda_yolo = lambda_yolo
        self.yolo_teacher = None
        if self.yolo_path:
            self.yolo_teacher = YOLOTeacher(model_path=self.yolo_path, device=self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        
        # Early Stopping & Saving parameters
        self.patience = patience
        self.min_delta = min_delta
        self.save_interval = save_interval # [New]
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        if optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            self.optimizer = optimizer
        
        if scheduler == 'cosine':
            eta_min = kwargs.get('eta_min', 0)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs, eta_min=eta_min)
        elif scheduler == 'step':
            step_size = kwargs.get('step_size', 6)
            gamma = kwargs.get('gamma', 0.7)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        os.makedirs(self.save_dir, exist_ok=True)

        # [New] 初始化 Log 檔案
        self.log_path = os.path.join(self.save_dir, 'training_log.csv')
        # 如果檔案不存在，寫入 Header
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_det_loss', 'train_dom_loss', 'val_det_loss', 'lr'])        
        
        self.global_step = 0
        self.current_epoch = 0

    def calculate_alpha(self, current_step, total_steps):
        if total_steps == 0: return 0.0
        p = float(current_step) / total_steps
        return 2. / (1. + math.exp(-10 * p)) - 1

    def train(self):
        print(f"Start training for {self.total_epochs} epochs...")
        
        for epoch in range(self.current_epoch, self.total_epochs):
            self.current_epoch = epoch
            
            # 1. Train one epoch (取得平均 loss)
            avg_det_loss, avg_dom_loss = self.train_one_epoch()
            
            # 2. Update Scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler: self.scheduler.step()
            
            # 3. Validate
            val_loss = 0.0
            if self.val_loader:
                val_loss = self.validate()
                if self.check_early_stopping(val_loss):
                    print(f"\n[Early Stopping] Triggered at epoch {epoch+1}!")
                    break
            
            # 4. [New] 寫入 Log 到 CSV
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, f"{avg_det_loss:.4f}", f"{avg_dom_loss:.4f}", f"{val_loss:.4f}", f"{current_lr:.6f}"])
            
            # 5. Save Checkpoints
            self.save_checkpoint(filename='model_latest.pth')
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(filename=f'model_epoch_{epoch+1}.pth')

    def check_early_stopping(self, val_loss): # TODO fix early stopping
        # If loss improves
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            # Save Best Model immediately
            print(f"  >>> New Best Model Found! (Loss: {val_loss:.4f})")
            self.save_checkpoint(filename='best_model.pth')
            return False
        else:
            self.early_stop_counter += 1
            print(f"  >>> Early Stopping Counter: {self.early_stop_counter}/{self.patience}")
            if self.early_stop_counter >= self.patience:
                return True
            return False

    def train_one_epoch(self):
        self.model.train()
        target_iter = iter(self.target_loader)
        len_source = len(self.source_loader)
        
        pbar = tqdm(enumerate(self.source_loader), total=len_source, desc=f"Epoch {self.current_epoch+1}")

        # [New] 用來累計 Loss
        total_det_loss = 0.0
        total_dom_loss = 0.0
        steps = 0

        for i, (images_s, targets_s) in pbar:
            images_s = list(img.to(self.device) for img in images_s)
            targets_s = [{k: v.to(self.device) for k, v in t.items()} for t in targets_s]
            
            try:
                images_t, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader)
                images_t, _ = next(target_iter)
            images_t = list(img.to(self.device) for img in images_t)

            total_steps = self.total_epochs * len_source
            alpha = self.calculate_alpha(self.global_step, total_steps)

            self.optimizer.zero_grad()

            # ==========================
            # A. Source Domain Training
            # ==========================
            # 計算 Detection Loss + Domain Loss
            loss_dict_s = self.model(images_s, targets_s, domain_label=0, alpha=alpha, mode='source')
            
            loss_det_s = sum(loss for k, loss in loss_dict_s.items() if k != 'domain_loss')
            loss_dom_s = loss_dict_s['domain_loss']

            # ==========================
            # B. Target Domain Training
            # ==========================
            
            # B-1. 純 Domain Adaptation (無標籤)
            # 這裡我們只取 Domain Loss
            loss_dict_t_dom = self.model(images_t, None, domain_label=1, alpha=alpha, mode='target')
            loss_dom_t = loss_dict_t_dom['domain_loss']
            
            # B-2. YOLO Knowledge Distillation (偽標籤)
            loss_det_t = torch.tensor(0.0, device=self.device)
            
            if self.yolo_teacher:
                # 產生 Target Domain 的偽標籤
                pseudo_targets = self.yolo_teacher.generate_targets(images_t)
                
                # 檢查這批圖有沒有抓到東西，如果有才算 Loss
                if any(len(t['boxes']) > 0 for t in pseudo_targets):
                    # 使用 mode='source' 來計算 Detection Loss
                    # 但這裡傳入的是 Target Images 和 Pseudo Targets
                    # 我們只取 detection loss 部分，忽略回傳的 domain loss (因為 B-1 算過了)
                    loss_dict_t_yolo = self.model(images_t, pseudo_targets, domain_label=1, alpha=alpha, mode='source')
                    loss_det_t = sum(loss for k, loss in loss_dict_t_yolo.items() if k != 'domain_loss')

            # ==========================
            # C. Total Loss & Optimization
            # ==========================
            # 總 Loss = Source偵測 + Lambda*Domain對抗 + Lambda_YOLO*Target偽標籤偵測
            losses = loss_det_s + \
                     self.lambda_domain * (loss_dom_s + loss_dom_t) + \
                     self.lambda_yolo * loss_det_t

            if not math.isfinite(losses.item()):
                print(f"Loss is NaN. S: {loss_dict_s}")
                sys.exit(1)

            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # 累計數值
            total_det_loss += loss_det_s.item()
            total_dom_loss += (loss_dom_s + loss_dom_t).item()
            steps += 1
            self.global_step += 1

            pbar.set_postfix({
                'det': f"{loss_det_s.item():.3f}",
                'dom': f"{(loss_dom_s + loss_dom_t).item():.3f}"
            })
            
        # 回傳平均 Loss
        return total_det_loss / steps, total_dom_loss / steps
            

    def validate(self):
        self.model.train() 
        total_val_loss = 0
        steps = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation", leave=False):
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets, mode='source')
                loss = sum(loss for k, loss in loss_dict.items() if k != 'domain_loss')
                
                total_val_loss += loss.item()
                steps += 1
        
        avg_loss = total_val_loss / (steps + 1e-6)
        print(f"  >>> Val Detection Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        state = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        torch.save(state, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path): return
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))