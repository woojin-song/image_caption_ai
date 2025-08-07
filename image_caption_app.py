import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
import torch

def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def translate_to_korean(text):
    translator = Translator()
    return translator.translate(text, src='en', dest='ko').text

def generate_caption(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

st.title("🖼️ 이미지 캡셔닝")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    with st.spinner("캡션 생성 중..."):
        processor, model = load_models()
        eng_caption = generate_caption(image, processor, model)
        ko_caption = translate_to_korean(eng_caption)

    st.markdown("### 📝 영어 캡션")
    st.markdown(f"""
    <div style="
        background-color:#f0f4ff; 
        padding:15px; 
        border-radius:10px; 
        font-size:18px; 
        font-weight:600;
        color:#1f2937;
    ">
        {eng_caption}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 한글 캡션")
    st.markdown(f"""
    <div style="
        background-color:#fff4f4; 
        padding:15px; 
        border-radius:10px; 
        font-size:18px; 
        font-weight:600;
        color:#b91c1c;
    ">
        {ko_caption}
    </div>
    """, unsafe_allow_html=True)
