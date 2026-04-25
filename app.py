import streamlit as st
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Street Objects Classifier", page_icon="🚗", layout="centered")

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

tab1, tab2 = st.tabs(["📁 Загрузить изображение", "✏️ Нарисовать"])

with tab1:
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженное изображение", use_container_width=True)

        if st.button("🔍 Классифицировать", key="btn_upload"):
            with st.spinner("Отправляю на сервер..."):
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
                    result = response.json()

                    st.success(f"**Класс:** {CLASSES_RU.get(result['predicted_class'], result['predicted_class'])}")
                    st.metric("Уверенность", f"{result['confidence']:.1%}")

                    st.subheader("Распределение вероятностей")
                    probs = result["probabilities"]
                    labels = [CLASSES_RU.get(k, k) for k in probs.keys()]
                    values = list(probs.values())
                    colors = ["#2ecc71" if k == result["predicted_class"] else "#3498db" for k in probs.keys()]

                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.barh(labels, values, color=colors)
                    ax.set_xlim(0, 1)
                    ax.bar_label(bars, fmt="%.3f", padding=3)
                    ax.set_xlabel("Вероятность")
                    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8)
                    plt.tight_layout()
                    st.pyplot(fig)

                except requests.exceptions.ConnectionError:
                    st.error("Не удалось подключиться к серверу. Проверьте URL бэкенда.")
                except requests.exceptions.Timeout:
                    st.error("Сервер не ответил за 30 секунд (возможно, просыпается — попробуйте ещё раз).")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

with tab2:
    try:
        from streamlit_drawable_canvas import st_canvas

        st.markdown("Нарисуйте объект на холсте:")
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
                        buf = io.BytesIO()
                        image_canvas.save(buf, format="JPEG")
                        buf.seek(0)
                        try:
                            response = requests.post(
                                f"{API_URL.rstrip('/')}/predict",
                                files={"file": ("drawing.jpg", buf, "image/jpeg")},
                                timeout=30,
                            )
                            response.raise_for_status()
                            result = response.json()

                            st.success(f"**Класс:** {CLASSES_RU.get(result['predicted_class'], result['predicted_class'])}")
                            st.metric("Уверенность", f"{result['confidence']:.1%}")

                            st.subheader("Распределение вероятностей")
                            probs = result["probabilities"]
                            labels = [CLASSES_RU.get(k, k) for k in probs.keys()]
                            values = list(probs.values())
                            colors = ["#2ecc71" if k == result["predicted_class"] else "#3498db" for k in probs.keys()]

                            fig, ax = plt.subplots(figsize=(8, 4))
                            bars = ax.barh(labels, values, color=colors)
                            ax.set_xlim(0, 1)
                            ax.bar_label(bars, fmt="%.3f", padding=3)
                            ax.set_xlabel("Вероятность")
                            plt.tight_layout()
                            st.pyplot(fig)

                        except Exception as e:
                            st.error(f"Ошибка: {e}")

    except ImportError:
        st.info("Для рисования добавьте `streamlit-drawable-canvas` в requirements.txt и перезапустите.")
        st.code("streamlit-drawable-canvas==0.9.3")
