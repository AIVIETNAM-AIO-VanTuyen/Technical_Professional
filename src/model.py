"""Kiến trúc Encoder (CNN) - Decoder (RNN/LSTM) cho bài toán Image Captioning."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

from . import config


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


class DecoderRNN(nn.Module):
    """Embedding -> RNN/LSTM (batch_first) -> Linear ra không gian từ vựng."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
        pad_id: int = config.PAD_ID,
    ):
        super().__init__()
        if not 1 <= num_layers <= 3:
            raise ValueError("num_layers phải nằm trong khoảng 1..3")
        rnn_classes = {"lstm": nn.LSTM, "rnn": nn.RNN}
        if rnn_type not in rnn_classes:
            raise ValueError(f"rnn_type phải là một trong {list(rnn_classes)}")
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        # nn.RNN mặc định nonlinearity="tanh": h_t = tanh(W_xh x_t + b_xh + W_hh h_{t-1} + b_hh)
        self.rnn = rnn_classes[rnn_type](
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """features (N, E), captions (N, L) -> logits (N, L, vocab_size).

        Chuỗi đầu vào là [ảnh, <sos>, w1, ..., w_{L-2}], nên logits[:, t] (t >= 1)
        dự đoán captions[:, t] từ captions[:, t-1] và toàn bộ ngữ cảnh trước đó.
        """
        embeddings = self.dropout(self.embed(captions[:, :-1]))  # (N, L-1, E)
        inputs = torch.cat([features.unsqueeze(1), embeddings], dim=1)  # (N, L, E)
        outputs, _ = self.rnn(inputs)  # (N, L, H)
        return self.fc(self.dropout(outputs))  # (N, L, V)

    def step(self, inputs: torch.Tensor, hidden=None):
        """Một bước thời gian: inputs (N, 1, E) -> logits (N, V), hidden mới."""
        output, hidden = self.rnn(inputs, hidden)
        return self.fc(output.squeeze(1)), hidden


class CNNtoRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        rnn_type: str = config.RNN_TYPE,
        dropout: float = config.DROPOUT,
        pretrained_cnn: bool = True,
        freeze_cnn: bool = config.FREEZE_CNN,
    ):
        super().__init__()
        self.hparams = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "rnn_type": rnn_type,
            "dropout": dropout,
            "freeze_cnn": freeze_cnn,
        }
        self.encoder = EncoderCNN(embed_dim, pretrained=pretrained_cnn, freeze=freeze_cnn)
        self.decoder = DecoderRNN(vocab_size, embed_dim, hidden_dim, num_layers, rnn_type, dropout)

    @classmethod
    def from_hparams(cls, hparams: dict, pretrained_cnn: bool = False) -> "CNNtoRNN":
        return cls(**hparams, pretrained_cnn=pretrained_cnn)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images), captions)

    @torch.no_grad()
    def caption_image(
        self,
        image: torch.Tensor,
        max_length: int = config.MAX_LENGTH,
        sos_id: int = config.SOS_ID,
        eos_id: int = config.EOS_ID,
    ) -> Tuple[List[int], List[float]]:
        """Sinh câu tự hồi quy (greedy) cho 1 ảnh (1, 3, H, W).

        Trả về (token ids không gồm <sos>/<eos>, xác suất của từng từ đã chọn kể cả <eos>).
        """
        self.eval()
        features = self.encoder(image)  # (1, E)
        # Bước 0: vector ảnh làm ngữ cảnh khởi tạo trạng thái ẩn.
        _, hidden = self.decoder.rnn(features.unsqueeze(1))
        token = torch.full((image.size(0), 1), sos_id, dtype=torch.long, device=image.device)
        ids: List[int] = []
        probs: List[float] = []
        for _ in range(max_length + 1):  # +1 để mô hình kịp sinh <eos> sau max_length từ
            logits, hidden = self.decoder.step(self.decoder.embed(token), hidden)
            prob, next_id = logits.softmax(dim=-1).max(dim=-1)
            if next_id.item() == eos_id:
                probs.append(prob.item())
                break
            if len(ids) == max_length:
                break
            probs.append(prob.item())
            ids.append(next_id.item())
            token = next_id.unsqueeze(1)
        return ids, probs


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def load_checkpoint(path, device: Optional[torch.device] = None) -> Tuple[CNNtoRNN, dict]:
    device = device or config.get_device()
    checkpoint = torch.load(path, map_location=device)
    model = CNNtoRNN.from_hparams(checkpoint["hparams"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
