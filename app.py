import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from utils import detect_faces, crop_face_raw, draw_box  # Import from your utils.py
# Constants (from your main.py and enroll.py)
DATASET_PATH = "data"  # Folder for training images
ENROLL_OUTPUT = "data"  # Folder for enrolled images
MODEL_PATH = "ifw_face_model.yml"
LABEL_MAP_PATH = "ifw_label_map.npy"
IMAGE_SIZE = (200, 200)

# Ensure directories exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(ENROLL_OUTPUT, exist_ok=True)


# Ensure LBPH is available (opencv-contrib-python)
def _ensure_lbph_available():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        st.error(
            "LBPH not available. Install opencv-contrib-python and restart the app."
        )
        st.stop()

# Function to train the model (adapted from main.py)
def train_model():
    _ensure_lbph_available()
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    # Find all directories that contain image files (supports nested datasets)
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    person_dirs = []
    for root, _, files in os.walk(DATASET_PATH):
        image_files = [f for f in files if f.lower().endswith(image_exts)]
        if len(image_files) >= 2:
            person_dirs.append((root, image_files))

    # Sort for stable labels
    person_dirs.sort(key=lambda x: x[0].lower())

    for person_path, image_files in person_dirs:
        # Use relative path from DATASET_PATH as the label (avoids collisions)
        person_label = os.path.relpath(person_path, DATASET_PATH)
        label_map[current_label] = person_label

        for image_name in image_files:
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, IMAGE_SIZE)
            faces.append(gray)
            labels.append(current_label)

        current_label += 1

    if not faces:
        st.error("No valid training data found. Add images to the 'ifw' folder.")
        return False

    faces = np.array(faces)
    labels = np.array(labels)

    model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    model.train(faces, labels)
    model.save(MODEL_PATH)
    np.save(LABEL_MAP_PATH, label_map)

    st.success(f"Training complete! Persons: {len(label_map)}, Images: {len(faces)}")
    return True

# Function to load the trained model
def load_model():
    _ensure_lbph_available()
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        return None, None
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_PATH)
    label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    return model, label_map

# Streamlit App Layout
st.title("Face Detection & Recognition App")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a section:", ["Enrollment", "Training", "Detection & Recognition"])

if page == "Enrollment":
    st.header("Enroll a New Person")
    st.write("Capture face images using your webcam and save them for training.")
    
    # Initialize session state for enrollment
    if "enroll_saved" not in st.session_state:
        st.session_state.enroll_saved = 0
    if "enroll_name" not in st.session_state:
        st.session_state.enroll_name = ""
    if "enroll_count" not in st.session_state:
        st.session_state.enroll_count = 25
    
    name = st.text_input("Enter person name:", st.session_state.enroll_name)
    count = st.slider("Number of images to capture:", 5, 50, st.session_state.enroll_count)
    
    # Update session state on input change
    st.session_state.enroll_name = name
    st.session_state.enroll_count = count
    
    if not name:
        st.error("Please enter a name.")
    else:
        person_dir = os.path.join(ENROLL_OUTPUT, name)
        os.makedirs(person_dir, exist_ok=True)
        
        st.write(f"Progress: {st.session_state.enroll_saved}/{count} images saved.")
        
        if st.session_state.enroll_saved < count:
            img_file = st.camera_input(f"Capture image {st.session_state.enroll_saved + 1}/{count}")
            if st.button("Capture & Save This Image"):
                if img_file is not None:
                    # Convert uploaded image to OpenCV format
                    img_array = np.frombuffer(img_file.getvalue(), np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        st.error("Failed to decode the captured image. Try recapturing.")
                        st.stop()
                    faces = detect_faces(frame)
                    if len(faces) > 0:
                        face = crop_face_raw(frame, faces[0],size=IMAGE_SIZE)  # Use first detected face
                        fname = os.path.join(person_dir, f"{st.session_state.enroll_saved:03d}.jpg")
                        cv2.imwrite(fname, face)
                        st.session_state.enroll_saved += 1
                        st.success(f"Saved image {st.session_state.enroll_saved}/{count}")
                        st.rerun()  # Force re-run to update progress (optional, but helps UI)
                    else:
                        st.warning("No face detected. Try again with better lighting/position.")
                else:
                    st.warning("Please capture an image first.")
        else:
            st.success(f"Enrollment complete! Saved {st.session_state.enroll_saved} images for '{name}' in '{person_dir}'.")
            st.info("Move images from 'data' to 'ifw' for training. Click below to reset for another person.")
            if st.button("Reset Enrollment"):
                st.session_state.enroll_saved = 0
                st.session_state.enroll_name = ""
                st.rerun()

elif page == "Training":
    st.header("Train the Face Recognition Model")
    st.write("Train the model using images from the 'ifw' folder. Ensure enrolled images are moved there.")
    
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            success = train_model()
        if success:
            st.balloons()

elif page == "Detection & Recognition":
    st.header("Detect & Recognize Faces")
    st.write("Capture an image to detect faces and recognize identities (if model is trained).")
    
    model, label_map = load_model()
    if model is None:
        st.warning("No trained model found. Train the model first.")
    else:
        img_file = st.camera_input("Capture an image for detection/recognition")
        if img_file is not None:
            img_array = np.frombuffer(img_file.getvalue(), np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            faces = detect_faces(frame)
            recognized_labels = []
            
            for bbox in faces:
                face = crop_face_raw(frame, bbox, size=IMAGE_SIZE)
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
              #  gray = cv2.resize(gray, IMAGE_SIZE)
                
                # Predict label
                label_id, confidence = model.predict(gray)
                label = label_map.get(label_id, "Unknown") if confidence < 100 else "Unknown"  # Threshold for confidence
                recognized_labels.append(f"{label} ({confidence:.2f})")
                
                # Draw on frame
                frame = draw_box(frame, bbox, f"{label} ({confidence:.2f})")
            
            st.image(frame, channels="BGR", caption="Processed Image")
            st.write(f"Detected faces: {len(faces)}")
            if recognized_labels:
                st.write("Recognized identities:", ", ".join(recognized_labels))
