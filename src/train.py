"""Huấn luyện mô hình CNN + RNN.

Ví dụ:
    python -m src.train --language vi
    python -m src.train --language en --epochs 20 --batch-size 128
    python -m src.train --language vi --max-steps 20 --max-val-steps 5   # chạy thử nhanh
"""

import argparse
import math
import random
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from . import config
from .dataset import CaptionStreamDataset, iter_split_captions, make_loader
from .model import CNNtoRNN, count_parameters
from .tokenizer_utils import build_tokenizer, decode_ids, load_tokenizer, save_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện Image Captioning CNN + RNN")
    parser.add_argument("--language", choices=config.SUPPORTED_LANGUAGES, default=config.DEFAULT_LANGUAGE)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--rnn-type", choices=["lstm", "rnn"], default=config.RNN_TYPE)
    parser.add_argument("--num-layers", type=int, default=config.NUM_LAYERS)
    parser.add_argument("--max-steps", type=int, default=None, help="Giới hạn số batch mỗi epoch (để chạy thử)")
    parser.add_argument("--max-val-steps", type=int, default=None, help="Giới hạn số batch khi validate")
    parser.add_argument("--rebuild-tokenizer", action="store_true", help="Xây lại từ vựng kể cả khi đã có file")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint 'last'")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_tokenizer(language: str, rebuild: bool):
    path = config.tokenizer_path(language)
    if path.exists() and not rebuild:
        print(f"Nạp tokenizer từ {path}")
        return load_tokenizer(path)
    print(f"Xây dựng từ vựng từ tập train ({config.DATASETS[language]['hf_name']}, streaming)...")
    tokenizer = build_tokenizer(iter_split_captions(language, "train"), language)
    save_tokenizer(tokenizer, path)
    print(f"Đã lưu tokenizer ({tokenizer.get_vocab_size()} từ) vào {path}")
    return tokenizer


def compute_loss(criterion: nn.Module, logits: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
    """Bỏ vị trí 0 (đầu ra từ vector ảnh); logits[:, t] dự đoán captions[:, t]."""
    vocab_size = logits.size(-1)
    return criterion(logits[:, 1:].reshape(-1, vocab_size), captions[:, 1:].reshape(-1))


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, max_steps: Optional[int]) -> float:
    model.train()
    total, steps = 0.0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]", total=max_steps)
    for images, captions in progress:
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        logits = model(images, captions)
        loss = compute_loss(criterion, logits, captions)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        total += loss.item()
        steps += 1
        if steps % config.LOG_EVERY == 0:
            progress.set_postfix(loss=f"{total / steps:.4f}")
        if max_steps and steps >= max_steps:
            break
    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_steps: Optional[int]) -> float:
    model.eval()
    total, steps = 0.0, 0
    for images, captions in tqdm(loader, desc="          [val]", total=max_steps):
        images, captions = images.to(device), captions.to(device)
        total += compute_loss(criterion, model(images, captions), captions).item()
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return total / max(steps, 1)


@torch.no_grad()
def show_sample(model, dataset: CaptionStreamDataset, tokenizer, device) -> None:
    image, caption = next(iter(dataset))
    ids, probs = model.caption_image(image.unsqueeze(0).to(device))
    print(f"  Thực tế : {decode_ids(tokenizer, caption.tolist())}")
    print(f"  Dự đoán : {decode_ids(tokenizer, ids)} (độ tự tin {np.mean(probs):.2f})")


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val: float, history: dict, language: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "hparams": model.hparams,
            "epoch": epoch,
            "best_val_loss": best_val,
            "history": history,
            "language": language,
            "sequence_length": config.SEQUENCE_LENGTH,
        },
        path,
    )


def plot_history(history: dict, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], marker="o", label="train")
    plt.plot(history["val_loss"], marker="o", label="val")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(config.SEED)
    device = config.get_device()
    print(f"Thiết bị: {device} | Ngôn ngữ: {config.LANGUAGE_NAMES[args.language]}")

    tokenizer = get_tokenizer(args.language, args.rebuild_tokenizer)
    train_ds = CaptionStreamDataset(args.language, "train", tokenizer)
    val_ds = CaptionStreamDataset(args.language, "val", tokenizer)
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers)

    model = CNNtoRNN(
        vocab_size=tokenizer.get_vocab_size(),
        rnn_type=args.rnn_type,
        num_layers=args.num_layers,
    ).to(device)
    print(f"Tham số huấn luyện được: {count_parameters(model):,} / tổng {count_parameters(model, False):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_ID)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    start_epoch, best_val = 1, math.inf
    history = {"train_loss": [], "val_loss": []}
    last_path = config.checkpoint_path(args.language, "last")
    best_path = config.checkpoint_path(args.language, "best")
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint["best_val_loss"]
        history = checkpoint["history"]
        print(f"Tiếp tục từ epoch {start_epoch} (best val loss {best_val:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.max_steps)
        val_loss = evaluate(model, val_loader, criterion, device, args.max_val_steps)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        show_sample(model, val_ds, tokenizer, device)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, best_val, history, args.language)
            print(f"  -> Lưu mô hình tốt nhất vào {best_path}")
        save_checkpoint(last_path, model, optimizer, epoch, best_val, history, args.language)
        plot_history(history, config.SAVED_MODELS_DIR / f"loss_curve_{args.language}.png")


if __name__ == "__main__":
    main()
