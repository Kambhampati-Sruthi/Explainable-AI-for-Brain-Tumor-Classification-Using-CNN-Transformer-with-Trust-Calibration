"""EfficientNet-B0 for 4-class brain-tumor MRI classification.

Architecture (torchvision layout, 256x256 input):
  features[0]    : stem, 3x3 conv stride 2 -> 32 channels
  features[1..7] : seven stages of MBConv blocks (the repeated building block)
  features[8]    : 1x1 conv -> 1280 channels  (last feature map: 8x8 at 256px input; Grad-CAM will use this in Stage 5)
  avgpool        : global average pooling -> 1280 numbers
  classifier     : Dropout(0.2) + Linear(1280 -> num_classes)

MBConv (mobile inverted bottleneck): 1x1 conv expands channels -> depthwise 3x3 or 5x5 conv (one filter per
channel: cheap) -> squeeze-and-excitation (learns per-channel importance) -> 1x1 conv projects back; plus a
skip connection. Activation is SiLU (swish): x * sigmoid(x).
Compound scaling (Tan & Le, 2019, ICML): depth = a^phi, width = b^phi, resolution = c^phi with a*b^2*c^2 ~ 2.
B0 is the phi=0 baseline found by architecture search, ~5.3M parameters with the 1000-class ImageNet head.
"""
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_efficientnet_b0(num_classes=4, pretrained=True, dropout=0.2):
    # IMAGENET1K_V1 = weights trained on ImageNet-1k classification (downloaded and cached on first use).
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features          # 1280
    model.classifier[0].p = dropout                        # dropout probability before the final layer
    model.classifier[1] = nn.Linear(in_features, num_classes)   # NEW randomly initialised head (replaces 1000-way ImageNet head)
    return model


def count_params(model, only_trainable=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_trainable))
