#!/usr/bin/env python3
"""
Enrollment script: Capture labeled face images from webcam into data/<name>/ folder.
Usage: python enroll.py --name Alice --count 25
"""
import os
import cv2
import argparse
import utils

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def enroll(name: str, count: int, output_dir: str = 'data'):

    """Capture and save face images for a person."""
    person_dir = os.path.join(output_dir, name)
    ensure_dir(person_dir)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('❌ Cannot open webcam')
        return

    saved = 0
    print(f'📷 Starting enrollment for "{name}" — saving {count} images to {person_dir}')
    print('💡 Position your face in the frame. Press Q to quit.')

    while saved < count:
        ret, frame = cap.read()
        if not ret:
            break

        faces = utils.detect_faces(frame)

        for bbox in faces:
            if saved >= count:
                break
            face = utils.crop_face_raw(frame, bbox)
            fname = os.path.join(person_dir, f"{saved:03d}.jpg")
            cv2.imwrite(fname, face)
            saved += 1
            frame = utils.draw_box(frame, bbox, f"Saved {saved}/{count}", (0, 255, 0))
            break

        cv2.putText(frame, f"Saved: {saved}/{count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Enroll Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'✅ Enrollment complete — saved {saved} images for "{name}"')

def parse_args():
    p = argparse.ArgumentParser(description='Enroll a person by capturing face images from webcam.')
    p.add_argument('--name', default='unknown', help='Person name (default folder name)')

    p.add_argument('--count', type=int, default=25, help='Number of images to capture')
    p.add_argument('--output', default='data', help='Output data folder')
    return p.parse_args()

if __name__ == '__main__':
   name = input("Enter person name: ")
enroll(name, 25, 'data')
    
