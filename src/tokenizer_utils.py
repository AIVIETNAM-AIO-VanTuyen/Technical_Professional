"""Chuẩn hóa văn bản, tách từ tiếng Việt, xây dựng từ vựng và vector hóa caption."""

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer

from . import config

# Dấu gạch dưới gốc cũng bị loại bỏ để "_" chỉ còn mang nghĩa nối từ ghép.
_PUNCT_RE = re.compile(r"[^\w\s]|_")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC -> chữ thường -> bỏ dấu câu/ký tự đặc biệt -> gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


@lru_cache(maxsize=None)
def _vi_segmenter(name: str):
    if name == "pyvi":
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize
    if name == "underthesea":
        from underthesea import word_tokenize

        return lambda text: word_tokenize(text, format="text")
    raise ValueError(f"Bộ tách từ không hỗ trợ: {name}")


def preprocess_caption(text: str, language: str) -> str:
    """Chuẩn hóa và (với tiếng Việt) nối từ ghép bằng '_', vd 'con chó' -> 'con_chó'."""
    text = normalize_text(text)
    if language == "vi" and text:
        text = _vi_segmenter(config.VI_SEGMENTER)(text)
        text = _SPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()
    return text


def postprocess_caption(tokens: Sequence[str]) -> str:
    """Nối token thành câu, thay '_' của từ ghép bằng khoảng trắng."""
    text = " ".join(tokens).replace("_", " ")
    text = _SPACE_RE.sub(" ", text).strip()
    return text[:1].upper() + text[1:]


def _configure(tokenizer: Tokenizer, sequence_length: int) -> Tokenizer:
    """Thêm <sos>/<eos>, padding và truncation về độ dài cố định L."""
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.SOS_TOKEN} $A {config.EOS_TOKEN}",
        special_tokens=[(config.SOS_TOKEN, config.SOS_ID), (config.EOS_TOKEN, config.EOS_ID)],
    )
    tokenizer.enable_truncation(max_length=sequence_length)
    tokenizer.enable_padding(pad_id=config.PAD_ID, pad_token=config.PAD_TOKEN, length=sequence_length)
    return tokenizer


def _check_special_ids(tokenizer: Tokenizer) -> None:
    for token, expected in zip(config.SPECIAL_TOKENS, (config.UNK_ID, config.PAD_ID, config.SOS_ID, config.EOS_ID)):
        actual = tokenizer.token_to_id(token)
        if actual != expected:
            raise ValueError(f"Token {token} có ID {actual}, mong đợi {expected}")


def build_tokenizer(
    captions: Iterable[str],
    language: str,
    vocab_size: int = config.VOCAB_SIZE,
    min_frequency: int = config.MIN_FREQUENCY,
    sequence_length: int = config.SEQUENCE_LENGTH,
) -> Tokenizer:
    """Huấn luyện tokenizer WordLevel + Whitespace trên các caption thô."""
    tokenizer = Tokenizer(WordLevel(unk_token=config.UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=config.SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator((preprocess_caption(c, language) for c in captions), trainer=trainer)
    _check_special_ids(tokenizer)
    return _configure(tokenizer, sequence_length)


def save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def load_tokenizer(path: Path, sequence_length: int = config.SEQUENCE_LENGTH) -> Tokenizer:
    if not Path(path).exists():
        raise FileNotFoundError(f"Không tìm thấy tokenizer: {path}. Hãy chạy huấn luyện trước.")
    tokenizer = Tokenizer.from_file(str(path))
    _check_special_ids(tokenizer)
    return _configure(tokenizer, sequence_length)


def encode_caption(tokenizer: Tokenizer, text: str, language: str) -> torch.Tensor:
    """Caption thô -> tensor Token IDs độ dài L, dtype torch.long."""
    ids = tokenizer.encode(preprocess_caption(text, language)).ids
    return torch.tensor(ids, dtype=torch.long)


def decode_ids(tokenizer: Tokenizer, ids: Iterable[int]) -> str:
    """Token IDs -> câu hoàn chỉnh (dừng ở <eos>, bỏ token đặc biệt)."""
    tokens: List[str] = []
    for idx in ids:
        idx = int(idx)
        if idx == config.EOS_ID:
            break
        if idx in (config.PAD_ID, config.SOS_ID):
            continue
        tokens.append(tokenizer.id_to_token(idx) or config.UNK_TOKEN)
    return postprocess_caption(tokens)
