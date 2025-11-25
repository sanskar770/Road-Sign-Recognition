!pip install streamlit
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("final_epoch_model.h5")

# Class names (0–42)
classes = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing for vehicles over 3.5t",
    "Right-of-way at intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles >3.5t prohibited", "No entry",
    "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End speed + passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing >3.5t"
]

st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")

st.title("🚦 Traffic Sign Recognition App")
st.write("Upload an image of a traffic sign and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=250)

    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)

    st.success(f"### 🛑 Predicted Sign: **{classes[pred_class]}**")