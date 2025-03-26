import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_violence_model.h5')

# Constants (update as per your training setup)
IMG_SIZE = 64
SEQUENCE_LENGTH = 20

# Preprocess video into frames
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frames.append(frame)
            if len(frames) == SEQUENCE_LENGTH:
                break
    finally:
        cap.release()

    # Pad with zeros if fewer frames
    if len(frames) < SEQUENCE_LENGTH:
        padding = [np.zeros((IMG_SIZE, IMG_SIZE, 3))] * (SEQUENCE_LENGTH - len(frames))
        frames.extend(padding)

    return np.expand_dims(np.array(frames), axis=0)  # Shape: (1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)

# Streamlit UI
st.title("Violence Detection in Video")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    st.video(video_file)

    # Process and predict
    st.write("Processing video and making prediction...")
    input_data = preprocess_video(video_path)
    prediction = model.predict(input_data)[0][0]

    if prediction > 0.5:
        st.error(f"Violence Detected! (Confidence: {prediction:.2f})")
    else:
        st.success(f"No Violence Detected (Confidence: {1 - prediction:.2f})")

    # Clean up
    os.remove(video_path)
