"""Sinh câu mô tả cho một ảnh bất kỳ.

Ví dụ:
    python -m src.inference --image path/to/anh.jpg --language vi
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from . import config
from .dataset import get_transform
from .model_rnn import load_checkpoint
from .tokenizer_utils import decode_ids, load_tokenizer


@dataclass
class CaptionResult:
    caption: str
    confidence: float  # trung bình xác suất của các từ được chọn (0..1)


class CaptionGenerator:
    """Nạp tokenizer + checkpoint một lần, dùng lại cho nhiều ảnh."""

    def __init__(self, language: str = config.DEFAULT_LANGUAGE, checkpoint: Optional[Path] = None,
                 device: Optional[torch.device] = None):
        self.language = language
        self.device = device or config.get_device()
        checkpoint = Path(checkpoint) if checkpoint else config.checkpoint_path(language, "best")
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {checkpoint}. Hãy chạy `python -m src.train --language {language}`."
            )
        self.model, meta = load_checkpoint(checkpoint, self.device)
        self.tokenizer = load_tokenizer(
            config.tokenizer_path(language), meta.get("sequence_length", config.SEQUENCE_LENGTH)
        )
        self.transform = get_transform(train=False)

    def __call__(self, image: Union[Image.Image, str, Path], max_length: int = config.MAX_LENGTH) -> CaptionResult:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        ids, probs = self.model.caption_image(tensor, max_length=max_length)
        return CaptionResult(caption=decode_ids(self.tokenizer, ids), confidence=float(np.mean(probs)) if probs else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh câu mô tả cho một ảnh")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--language", choices=config.SUPPORTED_LANGUAGES, default=config.DEFAULT_LANGUAGE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=config.MAX_LENGTH)
    args = parser.parse_args()

    generator = CaptionGenerator(args.language, args.checkpoint)
    result = generator(args.image, max_length=args.max_length)
    print(f"Mô tả     : {result.caption}")
    print(f"Độ tự tin : {result.confidence:.2%}")


if __name__ == "__main__":
    main()
