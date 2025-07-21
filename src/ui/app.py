import streamlit as st
import torch
from PIL import Image
import os
from src.model.tintora_ai import TintoraAI
from src.model.preprocess import preprocess_image
from src.model.postprocess import postprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = TintoraAI().to(device)
    model.eval()
    return model

st.title("TintoraAI: Раскраска старых фотографий")
st.write("Загрузите черно-белое или выцветшее фото и получите цветную версию!")

# Конфигуратор
st.sidebar.header("Настройки TintoraAI")
saturation = st.sidebar.slider("Насыщенность", 0.5, 2.0, 1.0, 0.1)
style = st.sidebar.selectbox("Стиль", ["Современный", "Винтаж", "Нейтральный"])

uploaded_file = st.file_uploader("Выберите фото (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Оригинальное фото", use_column_width=True)

    if st.button("Раскрасить"):
        with st.spinner("Обработка изображения..."):
            try:
                input_tensor, original_size = preprocess_image(original_image)
                input_tensor = input_tensor.to(device)
                model = load_model()
                
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                
                colored_image = postprocess_image(output_tensor, original_size, saturation)
                st.image(colored_image, caption="Раскрашенное фото", use_column_width=True)
                
                output_file = "colored_image.jpg"
                colored_image.save(output_file)
                with open(output_file, "rb") as file:
                    st.download_button(
                        label="Скачать раскрашенное фото",
                        data=file,
                        file_name="colored_image.jpg",
                        mime="image/jpeg"
                    )
                os.remove(output_file)
            except Exception as e:
                st.error(f"Ошибка обработки: {str(e)}")
