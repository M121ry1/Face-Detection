import cv2
import numpy as np

# Load Haar Cascade once when OpenCV is valid.
face_cascade = None
if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def detect_faces(frame):
    """
    Detect faces in a frame using Haar Cascade.
    Returns list of bounding boxes (x, y, w, h)
    """
    if face_cascade is None:
        raise RuntimeError(
            "OpenCV is not installed correctly. Reinstall opencv-contrib-python."
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

def crop_face_raw(frame, bbox, size=(200, 200)):
    """
    Crop and resize face for LBPH (no normalization).
    """
    x, y, w, h = bbox
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, size)
    return face

def crop_face_facenet(frame, bbox, size=(160, 160)):
    """
    Crop and preprocess face for CNN (FaceNet).
    """
    x, y, w, h = bbox
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, size)
    face = face.astype("float32")

    # Normalize for FaceNet
    mean, std = face.mean(), face.std()
    face = (face - mean) / (std + 1e-8)

    return face

def draw_box(frame, bbox, label='', color=(0, 255, 0)):
    """
    Draw bounding box and label on frame
    """
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    if label:
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
    return frame
