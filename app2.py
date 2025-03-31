import streamlit as st
import numpy as np
import cv2
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model

# ------------------------------
# 1. Load Pre-trained CNN for Feature Extraction
# ------------------------------
st.title("🔍 Violence Detection in Videos")

base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Load trained LSTM model
model = load_model('best_violence_model.h5')

# ------------------------------
# 2. Extract Features from Video Frames
# ------------------------------
def extract_features(frames):
    """Extracts CNN features from video frames using InceptionV3."""
    features = []
    for frame in frames:
        frame = cv2.resize(frame, (299, 299))  # Resize for InceptionV3
        frame = preprocess_input(frame)  # Normalize
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        feature = feature_extractor.predict(frame)
        features.append(feature.squeeze())  # Remove extra dimensions
    return np.array(features)  # Shape: (num_frames, 2048)

# ------------------------------
# 3. Predict Violence & Get Timestamps
# ------------------------------
def predict_violence(video_frames, timestamps):
    """Predicts violence and returns timestamps of violent frames."""
    video_features = extract_features(video_frames)  # Shape: (num_frames, 2048)
    
    if video_features.shape[0] < 30:
        padding = np.zeros((30 - video_features.shape[0], 2048))
        video_features = np.vstack((video_features, padding))  # Ensure shape: (30, 2048)

    video_features = np.expand_dims(video_features, axis=0)  # Shape: (1, 30, 2048)
    predictions = model.predict(video_features)[0]  # Get predictions for each frame

    # Identify timestamps where violence is detected
    violent_timestamps = [timestamps[i] for i, prob in enumerate(predictions) if prob > 0.5]
    
    return predictions, violent_timestamps

# ------------------------------
# 4. Streamlit UI - Upload & Process Video
# ------------------------------
uploaded_file = st.file_uploader("📤 Upload a video file...", type=["mp4", "avi", "mov", "mpeg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)  # Display uploaded video
    
    # Extract frames and timestamps
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS

    frame_id = 0
    while len(frames) < 30 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_id / fps  # Convert frame number to seconds
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert frame to RGB
        timestamps.append(timestamp)
        frame_id += 1

    cap.release()
    
    st.write(f"📸 Extracted {len(frames)} frames from video.")

    # Predict violence if frames are available
    if len(frames) > 0:
        predictions, violent_timestamps = predict_violence(frames, timestamps)
        violence_probability = np.max(predictions)  # Get highest probability in video

        st.write(f"🔬 **Highest Violence Probability: {violence_probability:.4f}**")

        if violence_probability > 0.5:
            st.error("⚠️ **Violence detected in the video!**")
            st.write("🕒 **Violence detected at timestamps (seconds):**")
            st.write(violent_timestamps if violent_timestamps else "No timestamps recorded.")
        else:
            st.success("✅ **No violence detected in the video.**")
