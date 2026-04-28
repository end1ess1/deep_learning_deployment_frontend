import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

st.set_page_config(page_title="Street Objects Classifier", page_icon="🚗", layout="wide")

st.title("🚗 Street Objects Classifier")
st.markdown("Классификация уличных объектов: велосипед, автомобиль, знак 30, человек, стоп, светофор, грузовик")

API_URL = st.sidebar.text_input(
    "URL бэкенда",
    value="https://deep-learning-deployment-b78b.onrender.com/",
    help="Ссылка на задеплоенный FastAPI"
)

CLASSES_RU = {
    "bicycle": "🚲 Велосипед",
    "car": "🚗 Автомобиль",
    "limit30": "🔵 Знак 30",
    "person": "🚶 Человек",
    "stop": "🛑 Стоп",
    "trafficlight": "🚦 Светофор",
    "truck": "🚚 Грузовик",
}


def get_prediction(image: Image.Image):
    """Отправляет изображение на сервер и возвращает результат предсказания."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    try:
        response = requests.post(
            f"{API_URL.rstrip('/')}/predict",
            files={"file": ("image.jpg", buf, "image/jpeg")},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к серверу. Проверьте URL бэкенда.")
    except requests.exceptions.Timeout:
        st.error("Сервер не ответил за отведённое время (возможно, просыпается — попробуйте ещё раз).")
    except Exception as e:
        st.error(f"Ошибка: {e}")
    return None


def render_distribution(result):
    """Отрисовывает прогноз нейросети с прогресс-барами."""
    st.subheader("Прогноз нейросети:")

    probs = result["probabilities"]
    predicted = result["predicted_class"]

    # Сортировка по убыванию вероятности
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    for cls, val in sorted_items:
        val = float(val)
        label = CLASSES_RU.get(cls, cls.capitalize())
        col_text, col_val = st.columns([2, 1])
        marker = " ✅" if cls == predicted else ""
        col_text.markdown(f"**{label}**{marker}")
        col_val.markdown(f"*{val * 100:.1f}%*")
        st.progress(min(max(val, 0.0), 1.0))


tab1, tab2 = st.tabs(["📁 Загрузить изображение", "✏️ Нарисовать"])

with tab1:
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Загруженное изображение", use_container_width=True)

        if st.button("🔍 Классифицировать", key="btn_upload"):
            with st.spinner("Отправляю на сервер..."):
                result = get_prediction(image)
                if result:
                    with col2:
                        render_distribution(result)

with tab2:
    try:
        from streamlit_drawable_canvas import st_canvas

        st.markdown("Нарисуйте объект на холсте:")

        col1, col2 = st.columns(2)

        with col1:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=8,
                stroke_color="#000000",
                background_color="#ffffff",
                height=300,
                width=300,
                drawing_mode="freedraw",
                key="canvas",
            )

        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data.astype(np.uint8)
            if img_array.sum() > 0:
                image_canvas = Image.fromarray(img_array[:, :, :3])

                if st.button("🔍 Классифицировать рисунок", key="btn_canvas"):
                    with st.spinner("Отправляю на сервер..."):
                        result = get_prediction(image_canvas)
                        if result:
                            with col2:
                                render_distribution(result)

    except ImportError:
        st.info("Для рисования добавьте `streamlit-drawable-canvas` в requirements.txt и перезапустите.")
        st.code("streamlit-drawable-canvas==0.9.3")
