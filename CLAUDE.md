# CLAUDE.md — Hướng dẫn cho AI làm việc với dự án này

Dự án Image Captioning (CNN + RNN), song ngữ Anh – Việt. Nguồn yêu cầu gốc là tài liệu đặc tả v2.0:
https://docs.google.com/document/d/1aA8OCyQmjxKEM2Hl2bq6UrPQS5JuSUaOwetmMDWDhAw/edit
Khi có mâu thuẫn, ưu tiên: yêu cầu mới của người dùng > các quyết định ghi trong file này > tài liệu đặc tả.

Người dùng giao tiếp bằng tiếng Việt (xưng "tớ – cậu"). Comment/docstring/thông báo trong code viết bằng tiếng Việt.

## Git

Remote `origin`: https://github.com/AIVIETNAM-AIO-VanTuyen/Technical_Professional (public), nhánh `main`. Checkpoint, tokenizer và dữ liệu nằm trong `.gitignore`, không commit. Chỉ commit/push khi người dùng yêu cầu. KHÔNG thêm dòng `Co-Authored-By: Claude` hay attribution AI nào vào commit message/PR.

## Cấu trúc (theo Mục 5 của đặc tả, đặt thẳng ở thư mục gốc)

```
data/            cache HF (data/hf_cache), không commit dữ liệu
saved_models/    tokenizer_{lang}.json, caption_model_{lang}_{best|last}.pth, loss_curve_{lang}.png
src/config.py          mọi siêu tham số, đường dẫn, cấu hình dataset theo ngôn ngữ
src/tokenizer_utils.py chuẩn hóa, tách từ VI, tokenizer WordLevel, encode/decode
src/dataset.py         HF streaming -> chia split -> torch IterableDataset -> DataLoader
src/model_cnn.py       EncoderCNN
src/model_rnn.py       DecoderRNN, CNNtoRNN, load_checkpoint
src/train.py           vòng lặp huấn luyện (CLI)
src/inference.py       CaptionGenerator + CLI
app.py                 Streamlit UI
```

Luôn chạy từ thư mục gốc dưới dạng module (`python -m src.train`), vì trong `src/` dùng import tương đối.

## Lệnh

```bash
pip install -r requirements.txt
python -m src.train --language vi|en [--epochs N --batch-size B --lr LR --rnn-type lstm|rnn --num-layers 1..3 --num-workers W --resume --rebuild-tokenizer]
python -m src.train --language vi --epochs 1 --batch-size 8 --max-steps 5 --max-val-steps 2   # smoke test
python -m src.inference --image anh.jpg --language vi
streamlit run app.py
```

Không có test suite. Kiểm tra thay đổi bằng smoke test ở trên. Muốn không ghi đè `saved_models/` thì gán lại
`config.SAVED_MODELS_DIR` trong một script bọc ngoài trước khi gọi `train.main()`.
Người dùng train trên máy khác (GPU). Máy dev là macOS, Python hệ thống là 3.15 alpha và chưa có torch, nên dùng venv Python 3.12 (`/opt/homebrew/bin/python3.12`).

## Các quyết định đã chốt với người dùng (đừng tự đổi)

- **Dữ liệu**: HuggingFace Datasets, `streaming=True`, không tải trước.
  - `en`: `jxie/flickr8k` (cột `image`, `caption_0..4`, các split train/validation/test)
  - `vi`: `ai-enthusiasm-community/UIT-ViIC` (cột `image`, `image_uid`, `caption_vi: list[str]`, các split train/validation)
  - Gộp các split gốc rồi chia lại 80/10/10 bằng `md5(SPLIT_SALT:key)`. Khóa là `id_column`, nếu không có thì dùng caption đầu tiên. Đổi `SPLIT_SALT` sẽ làm thay đổi cách chia.
  - Thêm dataset mới: chỉ cần thêm một mục vào `config.DATASETS` (cột caption có thể là str hoặc list[str]).
- **Hai ngôn ngữ, hai mô hình riêng**, chọn bằng `--language`. Tokenizer và checkpoint được đặt tên theo ngôn ngữ. App chỉ liệt kê những ngôn ngữ đã có `*_best.pth`.
- **Token đặc biệt**: `<unk>=0, <pad>=1, <sos>=2, <eos>=3` (đúng theo đặc tả, `<pad>` KHÔNG phải 0). Thứ tự được bảo đảm bằng thứ tự `SPECIAL_TOKENS` trong trainer và được kiểm tra trong `_check_special_ids`.
- **Văn bản**: NFC → lowercase → bỏ dấu câu và cả `_` gốc → (vi) tách từ bằng pyvi để nối từ ghép bằng `_`. Sau đó dùng pre-tokenizer Whitespace và TemplateProcessing `<sos> $A <eos>`, padding/truncation về `SEQUENCE_LENGTH=25`. Hậu xử lý: thay `_` bằng khoảng trắng và viết hoa chữ đầu câu.
- **Encoder**: ResNet-50 pretrained, đóng băng (backbone luôn ở `eval()`, xem `EncoderCNN.train`), sau đó là `Linear(2048, embed) → BatchNorm1d → ReLU`.
- **Decoder**: mặc định LSTM. Đầu vào là `[vector ảnh, <sos>, w1..w_{L-2}]`, đầu ra là logits `(N, L, V)`. `logits[:, t]` dự đoán `captions[:, t]`. Loss bỏ vị trí 0: `compute_loss` trong train.py dùng `CrossEntropyLoss(ignore_index=PAD_ID)`.
- **Inference**: vector ảnh khởi tạo hidden state, rồi greedy từ `<sos>`, tối đa `MAX_LENGTH=20` từ. "Độ tự tin" = trung bình xác suất của các từ được chọn (gồm cả `<eos>`).
- **UI**: tải ảnh lên → bấm nút "Gửi ảnh để mô tả" → ảnh hiện bên trái, spinner + câu mô tả + độ tự tin bên phải. Mô hình được cache bằng `@st.cache_resource` theo ngôn ngữ. Dùng `width="stretch"`, không dùng `use_container_width` (đã deprecated, yêu cầu streamlit>=1.50).
- **Thiết bị**: CUDA → MPS → CPU (`config.get_device`).
- **Checkpoint** chứa `model_state` (gồm cả trọng số ResNet, khoảng 127MB), `optimizer_state`, `hparams`, `epoch`, `best_val_loss`, `history`, `language`, `sequence_length`. Mô hình được dựng lại bằng `CNNtoRNN.from_hparams(..., pretrained_cnn=False)`, nên máy chạy web không cần tải trọng số ImageNet.

## Lưu ý kỹ thuật / bẫy đã gặp

- HF `IterableDataset` **tự chia shard theo worker** khi được duyệt bên trong worker của DataLoader. Vì vậy KHÔNG gọi thêm `split_dataset_by_node`, nếu không dữ liệu sẽ bị chia hai lần. Số worker bị chặn ở `n_shards` trong `make_loader` (vi: 2 shard, en: 4 shard).
- Hàm truyền vào `.filter()` phải pickle được (dùng `functools.partial` của hàm top-level, không dùng lambda), vì macOS khởi tạo worker bằng spawn.
- Xáo trộn có hai tầng: HF `shuffle(buffer=HF_SHUFFLE_BUFFER)` theo ảnh, và bộ đệm `PAIR_SHUFFLE_BUFFER` theo cặp (ảnh, caption) trong `CaptionStreamDataset`. Phải gọi `set_epoch` trước mỗi epoch.
- `drop_last=True` khi train vì BatchNorm1d lỗi với batch kích thước 1.
- Streaming chậm: mỗi epoch tải lại toàn bộ parquet (khoảng 600MB cho vi, khoảng 1,1GB cho en). Validate cũng quét qua toàn bộ luồng để lọc ra phần 10% của nó. Lấy mẫu val đầu tiên mất khoảng 50 giây.
- pyvi không ghép mọi từ ghép (ghép "đàn_ông", "sân_cỏ" nhưng để nguyên "con chó"). Có thể đổi sang `VI_SEGMENTER="underthesea"` (cần cài thêm underthesea).
- Một lần smoke test bị treo lúc thoát tiến trình sau khi đã lưu checkpoint; chưa tái hiện lại được và chưa rõ nguyên nhân.

## Việc có thể làm tiếp (chưa làm, hỏi người dùng trước)

- Cache dữ liệu ra ổ đĩa, hoặc tính sẵn đặc trưng CNN (vì encoder đóng băng) để train nhanh hơn.
- Beam search, đánh giá BLEU trên tập test.
- Lưu checkpoint không kèm backbone để file nhẹ hơn.
- Kiểm thử tải ảnh trên app thật (mới kiểm tra app render được và báo đúng khi chưa có model).
