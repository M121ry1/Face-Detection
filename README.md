# Face Recognition System

Complete face recognition system with enrollment, embedding extraction, and real-time detection/recognition.

## Installation

Install dependencies:
```powershell
pip install -r requirement.txt
pip install deepface scikit-learn imutils
```

## Files Overview

| File | Purpose |
|------|---------|
| `enroll.py` | Capture face images from webcam for enrollment |
| `extract_embeddings.py` | Extract embeddings from enrolled images (DeepFace) |
| `recognize_realtime.py` | Real-time face recognition with identity matching |
| `detect_realtime.py` | Real-time face detection (no recognition) |
| `utils.py` | Utility functions for face detection and drawing |
| `main.py` | LBPH face recognizer (training model) |

## Quick Start

### 1. Enroll a Person
Capture 25 images of a person:
```powershell
python enroll.py --name Alice --count 25
```
Images save to `data/Alice/`

### 2. Enroll More People (Optional)
```powershell
python enroll.py --name Bob --count 25
python enroll.py --name Charlie --count 25
```

### 3. Extract Embeddings
Convert all enrolled images to embeddings:
```powershell
python extract_embeddings.py
```
Saves to `embeddings.pkl`

### 4. Run Real-Time Recognition
Start live face recognition:
```powershell
python recognize_realtime.py
```
- **Green box** = Recognized face
- **Red box** = Unknown face
- **Press Q** to quit

### 5. Just Detect (Optional)
To only detect faces without recognition:
```powershell
python detect_realtime.py
```

## Key Features

- **Enrollment**: Capture multiple images per person from webcam
- **Embedding Extraction**: Uses DeepFace (Facenet model) for robust face embeddings
- **Real-time Recognition**: KNN-based matching with cosine distance
- **Face Detection**: Haar Cascade for fast face detection
- **Confidence Scores**: Shows confidence/distance for each match

## Adjustable Parameters

In `recognize_realtime.py`, line ~20:
```python
threshold = 0.35  # Adjust to tune sensitivity
```
- **Lower** (e.g., 0.25) = Stricter matching, fewer false positives
- **Higher** (e.g., 0.45) = More lenient, higher false positive rate

## Troubleshooting

- **No webcam detected**: Check if another app is using it
- **Poor recognition**: Enroll more images or vary lighting/angles
- **Embeddings.pkl not found**: Run `extract_embeddings.py` first
- **Windows dlib issue**: Use DeepFace instead (already default)

## Training Model (Alternative)

The original `main.py` uses LBPH recognizer. To train it:
```powershell
python main.py
```
Requires `ifw/` folder with person subfolders.
