import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class_names = [
    "T-shirt/top", 
    "Trouser", 
    "Pullover", 
    "Dress", 
    "Coat",
    "Sandal", 
    "Shirt", 
    "Sneaker", 
    "Bag", 
    "Ankle boot"
]

@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("cnn_fashion_mnist.keras")
    vgg = tf.keras.models.load_model("vgg16_fashion_mnist.keras")
    
    return cnn, vgg

def load_history(path):
    return np.load(path, allow_pickle=True).item()

def preprocess_cnn(image):
    image = image.convert("L").resize((28, 28))

    arr = np.array(image).astype("float32")

    if arr.mean() > 127:
        arr = 255 - arr

    arr = arr / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    return arr

def preprocess_vgg(image):
    image = image.convert("L").resize((96, 96))

    arr = np.array(image).astype("float32")

    if arr.mean() > 127:
        arr = 255 - arr

    arr = arr / 255.0
    arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    arr = np.expand_dims(arr, axis=0)

    return arr

def plot_history(history):
    fig, ax = plt.subplots()
    ax.plot(history["loss"], label="loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax.set_title("Функція втрат")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(history["accuracy"], label="accuracy")
    ax.plot(history["val_accuracy"], label="val_accuracy")
    ax.set_title("Точність моделі")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

st.title("Класифікація Fashion-MNIST")

cnn_model, vgg_model = load_models()

model_choice = st.selectbox(
    "Оберіть модель",
    ["CNN", "VGG16"]
)

uploaded_file = st.file_uploader(
    "Завантажте зображення",
    type=["png", "jpg", "jpeg"]
)

if model_choice == "CNN":
    model = cnn_model
    history = load_history("cnn_history.npy")
else:
    model = vgg_model
    history = load_history("vgg_history.npy")

st.subheader("Графіки навчання")
plot_history(history)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Вхідне зображення")
    st.image(image, width=300)

    if model_choice == "CNN":
        prepared_image = preprocess_cnn(image)
    else:
        prepared_image = preprocess_vgg(image)

    predictions = model.predict(prepared_image)[0]
    predicted_class = np.argmax(predictions)

    st.subheader("Результат класифікації")
    st.write(f"Передбачений клас: **{class_names[predicted_class]}**")

    results = {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

    st.bar_chart(results)

    st.subheader("Ймовірності для кожного класу")
    for class_name, probability in results.items():
        st.write(f"{class_name}: {probability:.4f}")