import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from .discriminator import Discriminator

class DAMaskRCNN(nn.Module):
    def __init__(self, num_classes=2): # 0: BG, 1: Cabbage
        super(DAMaskRCNN, self).__init__()
        # load pretrained
        self.base_model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # modify nc
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        in_features_mask = self.base_model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.base_model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        # decoupled based model
        self.backbone = self.base_model.backbone
        self.rpn = self.base_model.rpn
        self.roi_heads = self.base_model.roi_heads
        self.transform = self.base_model.transform # 負責 Normalize 和 Resize

        self.discriminator = Discriminator(in_channels=256)

    def forward(self, images, targets=None, domain_label=0, alpha=1.0, mode='train'):
        """
        mode: 'source' (回傳 detection loss + domain loss), 'target' (只回傳 domain loss), 'inference'
        """
        # Mask R-CNN 的前處理 (Resize, Normalize)
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        
        # 1. 提取特徵 (Backbone)
        features = self.backbone(images.tensors) 
        
        domain_loss = 0
        if mode != 'inference':
            # Domain Loss
            for layer_name, feat in features.items():
                pred = self.discriminator(feat, alpha)
                # domain_label: 0 for source, 1 for target
                target_domain = torch.full(pred.shape, domain_label, dtype=torch.float, device=pred.device)
                domain_loss += nn.BCEWithLogitsLoss()(pred, target_domain)
            domain_loss /= len(features)

        if mode == 'target':
            return {'domain_loss': domain_loss}

        # 2. prediction (RPN + ROI Heads)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        if mode == 'inference':
            return detections # 回傳 boxes, labels, scores, masks

        # 3. total Loss
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses['domain_loss'] = domain_loss
        
        return losses