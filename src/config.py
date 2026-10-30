"""Cấu hình toàn cục: đường dẫn, nguồn dữ liệu HuggingFace và siêu tham số."""

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
HF_CACHE_DIR = DATA_DIR / "hf_cache"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"

# ---------------------------------------------------------------------------
# Dữ liệu (HuggingFace Datasets, chế độ streaming)
# ---------------------------------------------------------------------------
# Mỗi ngôn ngữ một bộ dữ liệu. Các split gốc của HF được gộp lại rồi chia
# lại theo tỉ lệ 80/10/10 bằng hash của khóa ảnh (ổn định, không cần tải trước).
#   - caption_columns: các cột chứa câu mô tả (giá trị là str hoặc list[str]).
#   - id_column: cột định danh ảnh để chia split; None -> dùng caption đầu tiên.
DATASETS = {
    "en": {
        "hf_name": "jxie/flickr8k",
        "hf_config": None,
        "hf_splits": ["train", "validation", "test"],
        "image_column": "image",
        "caption_columns": [f"caption_{i}" for i in range(5)],
        "id_column": None,
    },
    "vi": {
        "hf_name": "ai-enthusiasm-community/UIT-ViIC",
        "hf_config": None,
        "hf_splits": ["train", "validation"],
        "image_column": "image",
        "caption_columns": ["caption_vi"],
        "id_column": "image_uid",
    },
}
SUPPORTED_LANGUAGES = tuple(DATASETS)
LANGUAGE_NAMES = {"vi": "Tiếng Việt", "en": "English"}
DEFAULT_LANGUAGE = "vi"

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SALT = "image-captioning-v2"

# Bộ đệm xáo trộn: HF xáo trộn theo hàng (ảnh), dataset của ta xáo trộn tiếp
# theo cặp (ảnh, caption) để các caption của cùng một ảnh không dồn vào 1 batch.
HF_SHUFFLE_BUFFER = 1000
PAIR_SHUFFLE_BUFFER = 500

# ---------------------------------------------------------------------------
# Văn bản & Tokenizer
# ---------------------------------------------------------------------------
VOCAB_SIZE = 5000
MIN_FREQUENCY = 2
SEQUENCE_LENGTH = 25  # độ dài cố định L khi huấn luyện (đã gồm <sos>, <eos>)
MAX_LENGTH = 20  # số từ tối đa khi sinh câu

UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = "<unk>", "<pad>", "<sos>", "<eos>"
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]  # thứ tự = ID
UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

VI_SEGMENTER = "pyvi"  # "pyvi" hoặc "underthesea"

# ---------------------------------------------------------------------------
# Hình ảnh
# ---------------------------------------------------------------------------
IMAGE_RESIZE = (256, 256)
IMAGE_CROP = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Mô hình
# ---------------------------------------------------------------------------
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1  # 1..3
RNN_TYPE = "lstm"  # "lstm" hoặc "rnn" (RNN thuần: h_t = tanh(W_xh x_t + W_hh h_{t-1} + b))
DROPOUT = 0.3
FREEZE_CNN = True

# ---------------------------------------------------------------------------
# Huấn luyện
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
GRAD_CLIP = 5.0
NUM_WORKERS = 2  # bị giới hạn bởi số shard của dataset khi streaming
LOG_EVERY = 50
SEED = 42


def get_device() -> torch.device:
    """Ưu tiên CUDA, sau đó Apple MPS, cuối cùng là CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenizer_path(language: str) -> Path:
    return SAVED_MODELS_DIR / f"tokenizer_{language}.json"


def checkpoint_path(language: str, tag: str = "best") -> Path:
    return SAVED_MODELS_DIR / f"caption_model_{language}_{tag}.pth"
