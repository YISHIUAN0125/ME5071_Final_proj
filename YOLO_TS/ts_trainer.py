# Trainer for redifine Loss
# src/trainers.py
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics import YOLO
import torch
from .loss import DistillationLoss 

class DualSupervisionTrainer(SegmentationTrainer):
    def __init__(self, teacher_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        print(f"Loading Teacher: {teacher_path}")
        self.teacher = YOLO(teacher_path).model
        
        # freeze teacher model's weights
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.teacher.eval()
        
        # Distillation loss
        self.distill_loss = DistillationLoss(nc=1, alpha=2.0)

    def criterion(self, preds, batch):
        # get origional YOLO Loss (Student vs Ground Truth)
        if not hasattr(self, 'loss_fn'):
            from ultralytics.utils.loss import v8SegmentationLoss
            self.loss_fn = v8SegmentationLoss(self.model)
            
        loss_yolo, loss_items = self.loss_fn(preds, batch)

        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])

        loss_distill = self.distill_loss(preds, teacher_preds)

        # Total Loss
        total_loss = loss_yolo + loss_distill
        
        return total_loss, loss_items
