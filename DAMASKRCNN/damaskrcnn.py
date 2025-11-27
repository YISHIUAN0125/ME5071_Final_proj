import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .discriminator import Discriminator

class DAMaskRCNN(nn.Module):
    def __init__(self, num_classes=2, backbone_name='resnet34'): 
        super(DAMaskRCNN, self).__init__()
        
        # 1. Build Custom Backbone (ResNet-34 + FPN)
        # trainable_layers=3 means we train the top 3 blocks (standard for detection)
        if backbone_name == 'resnet34':
            weights = torchvision.models.ResNet34_Weights.DEFAULT #TODO Disable pretrained weight
            backbone = resnet_fpn_backbone('resnet34', weights=weights, trainable_layers=3)
        else:
            # Fallback to ResNet-50 if needed
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            backbone = resnet_fpn_backbone('resnet50', weights=weights, trainable_layers=3)

        # 2. Manually build Mask R-CNN using the custom backbone
        # This replaces "maskrcnn_resnet50_fpn(pretrained=True)"
        self.base_model = MaskRCNN(backbone, num_classes=num_classes) # We set classes later
        
        # 3. Replace Box Head (Class predictor)
        # Get input features of the box predictor (usually 1024 for ResNet)
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 4. Replace Mask Head
        in_features_mask = self.base_model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.base_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        # 5. Decouple components (Same as before)
        self.backbone = self.base_model.backbone
        self.rpn = self.base_model.rpn
        self.roi_heads = self.base_model.roi_heads
        self.transform = self.base_model.transform 

        # 6. Domain Discriminator
        # FPN output is always 256 channels, regardless of ResNet34 or 50
        self.discriminator = Discriminator(in_channels=256)

    def forward(self, images, targets=None, domain_label=0, alpha=1.0, mode='train'):
        # ... (This part stays EXACTLY the same as your previous code) ...
        
        # 1. Keep original sizes
        original_image_sizes = [img.shape[-2:] for img in images]
        
        # 2. Transform
        images, targets = self.transform(images, targets)
        
        # 3. Backbone
        features = self.backbone(images.tensors)
        
        # ... Mode: Target ...
        if mode == 'target':
            domain_loss = 0
            for _, feat in features.items():
                pred = self.discriminator(feat, alpha)
                target_dom = torch.full(pred.shape, domain_label, dtype=torch.float, device=pred.device)
                domain_loss += nn.BCEWithLogitsLoss()(pred, target_dom)
            return {'domain_loss': domain_loss / len(features)}

        # ... Mode: Source ...
        if mode == 'source':
            domain_loss = 0
            for _, feat in features.items():
                pred = self.discriminator(feat, alpha)
                target_dom = torch.full(pred.shape, domain_label, dtype=torch.float, device=pred.device)
                domain_loss += nn.BCEWithLogitsLoss()(pred, target_dom)
            domain_loss /= len(features)

            proposals, proposal_losses = self.rpn(images, features, targets)
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
            
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses['domain_loss'] = domain_loss
            return losses

        # ... Mode: Inference ...
        if mode == 'inference':
            proposals, _ = self.rpn(images, features, targets)
            detections, _ = self.roi_heads(features, proposals, images.image_sizes, targets)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections