import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
import tempfile

# Load pre-trained feature extractor (CNN)
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Load trained LSTM model
model = load_model('best_violence_model.h5')

def extract_features(frames):
    features = []
    for frame in frames:
        frame = cv2.resize(frame, (299, 299))  # Resize for InceptionV3
        frame = preprocess_input(frame)  # Normalize
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        feature = feature_extractor.predict(frame)
        features.append(feature.squeeze())  # Remove extra dimensions
    
    st.write(f"Extracted {len(features)} feature vectors")
    return np.array(features)  # Shape: (num_frames, 2048)

def predict_violence(video_frames):
    video_features = extract_features(video_frames)  # Convert to (num_frames, 2048)
    
    if video_features.shape[0] < 30:
        padding = np.zeros((30 - video_features.shape[0], 2048))
        video_features = np.vstack((video_features, padding))  # Shape: (30, 2048)

    video_features = np.expand_dims(video_features, axis=0)  # Shape: (1, 30, 2048)
    prediction = model.predict(video_features)
    return prediction

st.title("Violence Detection in Videos")
uploaded_file = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mpeg"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < 30 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    
    st.write(f"Extracted {len(frames)} frames from video")
    
    if len(frames) > 0:
        prediction = predict_violence(frames)
        violence_probability = prediction[0][0]
        
        st.write(f"**Violence Probability: {violence_probability:.4f}**")
        
        if violence_probability > 0.5:
            st.error("⚠️ Violence detected in the video!")
        else:
            st.success("✅ No violence detected in the video.")
