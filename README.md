# Image Captioning: CNN + RNN (Anh – Việt)

Mô hình sinh câu mô tả ảnh theo kiến trúc Encoder–Decoder: ResNet-50 (đóng băng) → LSTM/RNN.
Dữ liệu đọc trực tiếp từ HuggingFace Datasets ở chế độ **streaming** (không cần tải trước).

| Ngôn ngữ | Dataset HF | Ghi chú |
|---|---|---|
| `en` | `jxie/flickr8k` | 8.000 ảnh × 5 caption |
| `vi` | `ai-enthusiasm-community/UIT-ViIC` | ~3.600 ảnh × 5 caption |

Các split gốc được gộp rồi chia lại 80/10/10 (train/val/test) bằng hash ổn định của khóa ảnh.
Đổi dataset trong `DATASETS` ở [src/config.py](src/config.py).

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Huấn luyện

```bash
python -m src.train --language vi          # tiếng Việt
python -m src.train --language en          # tiếng Anh
python -m src.train --language vi --max-steps 20 --max-val-steps 5 --epochs 1   # chạy thử nhanh
```

Tùy chọn: `--epochs`, `--batch-size`, `--lr`, `--rnn-type lstm|rnn`, `--num-layers 1..3`,
`--num-workers`, `--resume`, `--rebuild-tokenizer`.

Kết quả lưu ở `saved_models/`:
- `tokenizer_{lang}.json`: bộ từ vựng
- `caption_model_{lang}_best.pth`: checkpoint có val loss thấp nhất, `_last.pth` là checkpoint mới nhất
- `loss_curve_{lang}.png`: đồ thị loss

Nếu train trên máy khác, chỉ cần chép 2 file `tokenizer_{lang}.json` và `caption_model_{lang}_best.pth`
vào `saved_models/` của máy chạy web.

## Suy luận & Web

```bash
python -m src.inference --image anh.jpg --language vi
streamlit run app.py
```
