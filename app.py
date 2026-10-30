"""Giao diện web Streamlit: tải ảnh lên -> gửi cho mô hình CNN + RNN -> nhận câu mô tả.

Chạy: streamlit run app.py
"""

import streamlit as st
from PIL import Image

from src import config
from src.inference import CaptionGenerator

st.set_page_config(page_title="Image Captioning CNN + RNN", page_icon="🖼️", layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def get_generator(language: str) -> CaptionGenerator:
    return CaptionGenerator(language)


def available_languages():
    return [lang for lang in config.SUPPORTED_LANGUAGES if config.checkpoint_path(lang, "best").exists()]


st.title("🖼️ Hệ thống Mô tả Ảnh Tự động (Image Captioning)")
st.markdown(
    """
Ứng dụng sinh câu mô tả cho bức ảnh bằng kiến trúc **Encoder – Decoder**:

1. **CNN (ResNet-50)** trích xuất ảnh thành một vector đặc trưng.
2. Vector này làm ngữ cảnh khởi tạo cho **RNN/LSTM**.
3. RNN sinh **từng từ một**, lấy từ vừa sinh làm đầu vào cho bước kế tiếp, cho đến khi gặp `<eos>`.
"""
)

with st.sidebar:
    st.header("Cài đặt")
    languages = available_languages()
    if not languages:
        st.error("Chưa có mô hình nào trong `saved_models/`. Hãy huấn luyện trước:\n\n`python -m src.train --language vi`")
        st.stop()
    language = st.selectbox(
        "Ngôn ngữ mô tả",
        languages,
        index=languages.index(config.DEFAULT_LANGUAGE) if config.DEFAULT_LANGUAGE in languages else 0,
        format_func=lambda lang: config.LANGUAGE_NAMES[lang],
    )
    max_length = st.slider("Số từ tối đa", min_value=5, max_value=40, value=config.MAX_LENGTH)

uploaded = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])

left, right = st.columns(2)
if uploaded is None:
    right.info("Hãy tải lên một ảnh định dạng .jpg, .jpeg hoặc .png.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
left.image(image, caption=uploaded.name, width="stretch")

with right:
    if st.button("Gửi ảnh để mô tả", type="primary", width="stretch"):
        try:
            generator = get_generator(language)
        except FileNotFoundError as err:
            st.error(str(err))
            st.stop()
        with st.spinner("Mô hình đang phân tích ảnh..."):
            result = generator(image, max_length=max_length)
        st.subheader("Kết quả")
        st.success(result.caption or "(Mô hình không sinh được câu)")
        st.metric("Độ tự tin", f"{result.confidence:.1%}")
        st.progress(min(max(result.confidence, 0.0), 1.0))
        st.caption("Độ tự tin = trung bình xác suất của từng từ mà mô hình đã chọn.")
    else:
        st.info("Bấm **Gửi ảnh để mô tả** để mô hình sinh câu.")
