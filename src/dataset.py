"""Đọc dữ liệu HuggingFace ở chế độ streaming, chia split và tạo DataLoader."""

import hashlib
import random
from functools import partial
from typing import Iterator, List, Optional, Tuple

import torch
from datasets import concatenate_datasets, load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms

from . import config
from .tokenizer_utils import encode_caption


def get_transform(train: bool) -> transforms.Compose:
    crop = transforms.RandomCrop(config.IMAGE_CROP) if train else transforms.CenterCrop(config.IMAGE_CROP)
    return transforms.Compose(
        [
            transforms.Resize(config.IMAGE_RESIZE),
            crop,
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
        ]
    )


def row_captions(row: dict, caption_columns: List[str]) -> List[str]:
    """Gom tất cả caption không rỗng của một hàng (cột có thể là str hoặc list[str])."""
    captions = []
    for col in caption_columns:
        value = row.get(col)
        values = value if isinstance(value, (list, tuple)) else [value]
        captions.extend(v for v in values if isinstance(v, str) and v.strip())
    return captions


def split_of(row: dict, ds_cfg: dict) -> str:
    """Gán hàng vào train/val/test bằng hash ổn định của khóa ảnh."""
    id_col = ds_cfg["id_column"]
    if id_col:
        key = str(row[id_col])
    else:
        captions = row_captions(row, ds_cfg["caption_columns"])
        key = captions[0] if captions else ""
    bucket = int(hashlib.md5(f"{config.SPLIT_SALT}:{key}".encode("utf-8")).hexdigest(), 16) % 1000 / 1000
    if bucket < config.SPLIT_RATIOS["train"]:
        return "train"
    if bucket < config.SPLIT_RATIOS["train"] + config.SPLIT_RATIOS["val"]:
        return "val"
    return "test"


def _in_split(row: dict, split: str, ds_cfg: dict) -> bool:
    return split_of(row, ds_cfg) == split


def load_streaming_split(language: str, split: str, shuffle: bool = False, with_images: bool = True):
    """Trả về HF IterableDataset (streaming) của một split train/val/test."""
    if split not in config.SPLIT_RATIOS:
        raise ValueError(f"split phải là một trong {list(config.SPLIT_RATIOS)}")
    ds_cfg = config.DATASETS[language]
    parts = [
        load_dataset(
            ds_cfg["hf_name"],
            ds_cfg["hf_config"],
            split=hf_split,
            streaming=True,
            cache_dir=str(config.HF_CACHE_DIR),
        )
        for hf_split in ds_cfg["hf_splits"]
    ]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if not with_images:
        ds = ds.remove_columns([ds_cfg["image_column"]])
    # partial thay vì lambda để dataset pickle được khi DataLoader dùng nhiều worker.
    ds = ds.filter(partial(_in_split, split=split, ds_cfg=ds_cfg))
    if shuffle:
        ds = ds.shuffle(seed=config.SEED, buffer_size=config.HF_SHUFFLE_BUFFER)
    return ds


def iter_split_captions(language: str, split: str = "train") -> Iterator[str]:
    """Duyệt các caption thô của một split (không giải mã ảnh) để xây từ vựng."""
    caption_columns = config.DATASETS[language]["caption_columns"]
    for row in load_streaming_split(language, split, with_images=False):
        yield from row_captions(row, caption_columns)


class CaptionStreamDataset(IterableDataset):
    """Sinh từng cặp (ảnh tensor (3, 224, 224), caption ids (L,)) từ luồng HF.

    Mỗi ảnh có nhiều caption -> mỗi caption là một mẫu huấn luyện. Khi shuffle,
    các cặp đi qua một bộ đệm xáo trộn để trộn caption của các ảnh khác nhau.
    Khi chạy với nhiều worker, HF IterableDataset tự chia shard theo worker.
    """

    def __init__(self, language: str, split: str, tokenizer: Tokenizer, train: Optional[bool] = None):
        super().__init__()
        self.language = language
        self.split = split
        self.tokenizer = tokenizer
        self.train = split == "train" if train is None else train
        self.transform = get_transform(self.train)
        ds_cfg = config.DATASETS[language]
        self.image_column = ds_cfg["image_column"]
        self.caption_columns = ds_cfg["caption_columns"]
        self.hf_ds = load_streaming_split(language, split, shuffle=self.train)
        self.epoch = 0

    @property
    def n_shards(self) -> int:
        return self.hf_ds.n_shards

    def set_epoch(self, epoch: int) -> None:
        """Đổi seed xáo trộn mỗi epoch (gọi trước khi tạo iterator mới)."""
        self.epoch = epoch
        self.hf_ds.set_epoch(epoch)

    def _pairs(self) -> Iterator[Tuple[object, torch.Tensor]]:
        for row in self.hf_ds:
            image = row[self.image_column]
            if image is None:
                continue
            image = image.convert("RGB")
            for caption in row_captions(row, self.caption_columns):
                yield image, encode_caption(self.tokenizer, caption, self.language)

    def _sample(self, pair: Tuple[object, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, ids = pair
        return self.transform(image), ids

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.train:
            for pair in self._pairs():
                yield self._sample(pair)
            return

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(config.SEED + 1000 * self.epoch + worker_id)
        buffer: List[Tuple[object, torch.Tensor]] = []
        for pair in self._pairs():
            if len(buffer) < config.PAIR_SHUFFLE_BUFFER:
                buffer.append(pair)
                continue
            idx = rng.randrange(len(buffer))
            yield self._sample(buffer[idx])
            buffer[idx] = pair
        rng.shuffle(buffer)
        for pair in buffer:
            yield self._sample(pair)


def make_loader(
    dataset: CaptionStreamDataset,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> DataLoader:
    # Không thể có nhiều worker hơn số shard: worker thừa sẽ không nhận dữ liệu.
    num_workers = max(0, min(num_workers, dataset.n_shards))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        # BatchNorm của Encoder cần batch > 1 khi train.
        drop_last=dataset.train,
    )
