import streamlit as st
import numpy as np
import cv2
import smtplib
import os
import tempfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
# 2. Email Alert Function
# ------------------------------
EMAIL_SENDER = "projectmajorvd25@gmail.com"
EMAIL_PASSWORD = "Ap10@w0667"  # Store this securely (use environment variables)
EMAIL_RECEIVER = "projectmajorvd25@gmail.com"  # Recipient email

def send_alert_email(violence_prob):
    """Sends an email alert when violence is detected in a video."""
    subject = "🚨 Violence Detected in Video Alert!"
    body = f"ALERT: A video has been processed, and violence was detected with a probability of {violence_prob:.2%}.\n\nPlease review the uploaded video."

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        st.success("🚀 Alert email sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")

# ------------------------------
# 3. Feature Extraction from Video Frames
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
# 4. Predict Violence in Video
# ------------------------------
def predict_violence(video_frames):
    """Predicts whether the uploaded video contains violence."""
    video_features = extract_features(video_frames)  # Shape: (num_frames, 2048)

    if video_features.shape[0] < 30:
        padding = np.zeros((30 - video_features.shape[0], 2048))
        video_features = np.vstack((video_features, padding))  # Ensure shape: (30, 2048)

    video_features = np.expand_dims(video_features, axis=0)  # Shape: (1, 30, 2048)
    prediction = model.predict(video_features)
    return prediction

# ------------------------------
# 5. Streamlit UI - Upload & Process Video
# ------------------------------
uploaded_file = st.file_uploader("📤 Upload a video file...", type=["mp4", "avi", "mov", "mpeg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)  # Display uploaded video
    
    # Extract frames from video
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < 30 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        frames.append(frame)

    cap.release()
    
    st.write(f"📸 Extracted {len(frames)} frames from video.")

    # Predict violence if frames are available
    if len(frames) > 0:
        prediction = predict_violence(frames)
        violence_probability = prediction[0][0]

        st.write(f"🔬 **Violence Probability: {violence_probability:.4f}**")

        if violence_probability > 0.5:
            st.error("⚠️ **Violence detected in the video!**")
            send_alert_email(violence_probability)  # Send email alert
        else:
            st.success("✅ **No violence detected in the video.**")
