"""Khối Encoder (CNN): trích xuất ảnh thành vector đặc trưng."""

import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    """ResNet-50 (đóng băng) + Linear/BatchNorm/ReLU: ảnh (N, 3, 224, 224) -> (N, embed_dim)."""

    def __init__(self, embed_dim: int, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        in_features = resnet.fc.in_features
        # Bỏ lớp phân loại fc cuối, giữ đến global average pooling.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.freeze = freeze
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(in_features, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU()

    def train(self, mode: bool = True) -> "EncoderCNN":
        super().train(mode)
        # Backbone đóng băng luôn ở eval để không cập nhật thống kê BatchNorm của ResNet.
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze and torch.is_grad_enabled()):
            features = self.backbone(images).flatten(1)  # (N, 2048)
        return self.relu(self.bn(self.fc(features)))  # (N, embed_dim)
