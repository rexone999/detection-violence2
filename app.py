import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

def send_email(recipient_email, subject, body):
    sender_email = "drippletriplesid@gmail.com"  # Replace with your email
    sender_password = "Meatlover45@"  # Replace with your email password

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        # Set up the server
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Use your email provider's SMTP server
        server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
        server.login(sender_email, sender_password)  # Log in to your email account
        server.send_message(msg)  # Send the email
        server.quit()  # Close the connection
        st.success("Email notification sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Streamlit UI
st.title("Violence Detection in Videos")
uploaded_file = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mpeg"])
recipient_email = st.text_input("Enter your email address for notifications:")

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
        
        if violence_probability > 0.3:
            st.error("⚠️ Violence detected in the video!")
            if recipient_email:
                send_email(recipient_email, "Violence Detection Alert", "Violence has been detected in the video you uploaded.")
        else:
            st.success("✅ No violence detected in the video.")
