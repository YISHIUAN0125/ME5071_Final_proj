import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .discriminator import Discriminator

class DAMaskRCNN(nn.Module):
    def __init__(self, num_classes=2): 
        # num_classes: 0 (background) + 1 (cabbage) = 2
        super(DAMaskRCNN, self).__init__()
        
        # 載入預訓練模型
        # 注意: 新版 torchvision 建議使用 weights 參數
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.base_model = maskrcnn_resnet50_fpn(weights=weights)
        
        # 修改 Box Head (Class predictor)
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 修改 Mask Head
        in_features_mask = self.base_model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.base_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        # 拆解 Mask R-CNN 元件
        self.backbone = self.base_model.backbone
        self.rpn = self.base_model.rpn
        self.roi_heads = self.base_model.roi_heads
        self.transform = self.base_model.transform 

        # 加入 Domain Discriminator
        # FPN 輸出 256 channels
        self.discriminator = Discriminator(in_channels=256)

    def forward(self, images, targets=None, domain_label=0, alpha=1.0, mode='train'):
        """
        Args:
            images: List[Tensor]
            targets: List[Dict] (Ground Truth)
            domain_label: 0 for source, 1 for target
            alpha: GRL alpha value
            mode: 'source', 'target', 'inference'
        """
        
        # 1. 保留原始圖片尺寸 (為了 Inference 後處理)
        original_image_sizes = [img.shape[-2:] for img in images]
        
        # 2. Mask R-CNN 內部預處理 (Normalize, Resize -> PadBatch)
        images, targets = self.transform(images, targets)
        
        # 3. Backbone 提取特徵 (Output: Dict[str, Tensor])
        features = self.backbone(images.tensors)
        
        # ==========================================
        # Mode: Target Domain Training (Only Domain Loss)
        # ==========================================
        if mode == 'target':
            domain_loss = 0
            for _, feat in features.items():
                pred = self.discriminator(feat, alpha)
                target_dom = torch.full(pred.shape, domain_label, dtype=torch.float, device=pred.device)
                domain_loss += nn.BCEWithLogitsLoss()(pred, target_dom)
            
            # 平均多層 Feature 的 Loss
            return {'domain_loss': domain_loss / len(features)}

        # ==========================================
        # Mode: Source Domain Training (Det + Domain Loss)
        # ==========================================
        if mode == 'source':
            # A. Domain Loss
            domain_loss = 0
            for _, feat in features.items():
                pred = self.discriminator(feat, alpha)
                target_dom = torch.full(pred.shape, domain_label, dtype=torch.float, device=pred.device)
                domain_loss += nn.BCEWithLogitsLoss()(pred, target_dom)
            domain_loss /= len(features)

            # B. Detection Loss (RPN + ROI)
            proposals, proposal_losses = self.rpn(images, features, targets)
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
            
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses['domain_loss'] = domain_loss
            return losses

        # ==========================================
        # Mode: Inference (Prediction)
        # ==========================================
        if mode == 'inference':
            # RPN 生成 Proposals
            proposals, _ = self.rpn(images, features, targets)
            
            # ROI Heads 生成 Detections (座標是 Resize 後的)
            detections, _ = self.roi_heads(features, proposals, images.image_sizes, targets)
            
            # [重要] Post-process: 將座標映射回原始圖片大小
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections