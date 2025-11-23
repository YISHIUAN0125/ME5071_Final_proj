# Implemeentation of Loss function for T-S architecture
# Total Loss = L_yolo + L_distillation (MSE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss

class DistillationLoss(nn.Module):
    def __init__(self, nc=1, temprtature=1, alpha=1.0):
        super().__init__()
        self.nc = nc
        self.T = temprtature
        self.alpha = alpha

        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, teacher_preds, student_preds): # TODO check
        """
        student_preds: List[Tensor], 其中 [0] 是 [Batch, 4 + nc + 32, Anchors]
        """
        # output shape: [Batch, Channels, Anchors]
        s_out = student_preds[0]
        t_out = teacher_preds[0]

        # get class
        # start with length 4 with length of nc
        # split_start = 4 (Box x,y,w,h)
        # split_end = 4 + self.nc
        s_logits = s_out[:, 4 : 4 + self.nc, :]  
        t_logits = t_out[:, 4 : 4 + self.nc, :]

        # teacher
        with torch.no_grad():
            t_probs = torch.sigmoid(t_logits / self.T)

        # Student
        s_probs = torch.sigmoid(s_logits / self.T)

        # MSE Loss
        loss = self.loss_fn(s_probs, t_probs)
        
        return self.alpha * loss



