"""U-Net++ with an EfficientNet-B0 encoder, written from scratch.

Node equation (Zhou et al., 2018, DLMIA):
    x^{i,0} = encoder feature at level i
    x^{i,j} = H( [ x^{i,0}, ..., x^{i,j-1}, Up(x^{i+1,j-1}) ] )       for j >= 1
Plain U-Net = only the nodes with j = L-1-i, each fed by [ x^{i,0}, Up(previous diagonal node) ]  (nested=False).
"""
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EffB0Encoder(nn.Module):
    """torchvision EfficientNet-B0 trunk returning 5 feature maps (strides 2, 4, 8, 16, 32).
    features[0] stem; features[1..7] = the seven MBConv stages; features[8] (the 1280-channel head conv) is dropped:
    segmentation does not need it, and dropping it saves parameters and compute."""

    def __init__(self, pretrained=True):
        super().__init__()
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(m.features.children())[:8])    # keep features[0..7]
        self.out_idx = (1, 2, 3, 5, 7)           # last block at each stride: 2, 4, 8, 16, 32
        self.channels = (16, 24, 40, 112, 320)

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_idx:
                outs.append(x)
        return outs


def conv_bn_relu(cin, cout):
    # bias=False because BatchNorm directly after the conv has its own shift parameter
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ConvBlock(nn.Sequential):
    """H(.) in the node equation: two (3x3 conv -> BatchNorm -> ReLU)."""
    def __init__(self, cin, cout):
        super().__init__(conv_bn_relu(cin, cout), conv_bn_relu(cout, cout))


class UNetPlusPlus(nn.Module):
    def __init__(self, pretrained=True, dec_ch=(32, 64, 128, 256), nested=True, num_classes=1):
        super().__init__()
        self.encoder = EffB0Encoder(pretrained)
        enc = list(self.encoder.channels)                  # [16, 24, 40, 112, 320]
        self.L = len(enc)                                  # 5 resolution levels
        assert len(dec_ch) == self.L - 1
        self.nested = nested
        ch = lambda i, j: enc[i] if j == 0 else dec_ch[i]  # channels of node x^{i,j}
        self.nodes = nn.ModuleDict()
        for j in range(1, self.L):
            for i in range(self.L - j):
                if not nested and j != self.L - 1 - i:
                    continue                                # plain U-Net: diagonal nodes only
                skips = range(j) if nested else [0]
                cin = sum(ch(i, k) for k in skips) + ch(i + 1, j - 1)   # skip inputs + upsampled lower-level input
                self.nodes[f"{i}_{j}"] = ConvBlock(cin, dec_ch[i])
        self.final = ConvBlock(dec_ch[0], dec_ch[0] // 2)  # after the last node (stride 2) and a x2 upsample
        self.head = nn.Conv2d(dec_ch[0] // 2, num_classes, 1)   # 1x1 conv -> one tumor LOGIT per pixel

    def decoder_parameters(self):
        return itertools.chain(self.nodes.parameters(), self.final.parameters(), self.head.parameters())

    def forward(self, x):
        size = x.shape[-2:]
        X = {(i, 0): f for i, f in enumerate(self.encoder(x))}
        for j in range(1, self.L):                         # columns left to right ...
            for i in range(self.L - j):                    # ... levels top to bottom
                if not self.nested and j != self.L - 1 - i:
                    continue
                up = F.interpolate(X[(i + 1, j - 1)], size=X[(i, 0)].shape[-2:], mode="nearest")   # Up(.)
                skips = [X[(i, k)] for k in (range(j) if self.nested else [0])]
                X[(i, j)] = self.nodes[f"{i}_{j}"](torch.cat(skips + [up], dim=1))
        out = F.interpolate(X[(0, self.L - 1)], size=size, mode="bilinear", align_corners=False)   # stride 2 -> input size
        return self.head(self.final(out))                  # logits, shape (B, 1, H, W)


from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights


class EffB1Encoder(EffB0Encoder):
    """Same 5 output levels and channel counts as B0 (16, 24, 40, 112, 320): B1 only has more layers per stage
    (depth multiplier 1.1), so it plugs into the same decoder. forward() is inherited."""

    def __init__(self, pretrained=True):
        nn.Module.__init__(self)                                # skip EffB0Encoder.__init__ (it would load B0 weights)
        m = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(m.features.children())[:8])
        self.out_idx = (1, 2, 3, 5, 7)
        self.channels = (16, 24, 40, 112, 320)
