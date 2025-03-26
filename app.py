import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model

# Load the trained model
model = load_model('best_violence_model.h5')

# Load a pretrained CNN for feature extraction (InceptionV3)
base_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Constants
IMG_SIZE = 299  # InceptionV3 expects 299x299
SEQUENCE_LENGTH = 30  # Match model expectation
FEATURE_SIZE = 2048  # Feature vector size after CNN

# Preprocess video into feature vectors
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = preprocess_input(frame)  # Normalize for InceptionV3
            frames.append(frame)
            if len(frames) == SEQUENCE_LENGTH:
                break
    finally:
        cap.release()

    # Pad if needed
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    frames_array = np.array(frames)  # Shape: (SEQUENCE_LENGTH, 299, 299, 3)

    # Extract features using InceptionV3
    features = feature_extractor.predict(frames_array)  # Shape: (SEQUENCE_LENGTH, 2048)

    return np.expand_dims(features, axis=0)  # Final shape: (1, SEQUENCE_LENGTH, FEATURE_SIZE)

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

    # Debugging: Display input shape
    st.write(f"Input shape for prediction: {input_data.shape}")  # Should be (1, 30, 2048)

    try:
        prediction = model.predict(input_data)[0][0]
        if prediction > 0.5:
            st.error(f"Violence Detected! (Confidence: {prediction:.2f})")
        else:
            st.success(f"No Violence Detected (Confidence: {1 - prediction:.2f})")
    except ValueError as e:
        st.error(f"Prediction error: {str(e)}")

    # Clean up
    os.remove(video_path)
