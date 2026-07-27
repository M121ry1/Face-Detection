#!/usr/bin/env python3
"""
Real-time face detection (no recognition) - just shows detected faces.
Usage: python detect_realtime.py
Press Q to quit.
"""
import cv2
import utils

def detect_only():
    """Real-time face detection from webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('❌ Cannot open webcam')
        return

    print('🎥 Real-time face detection started. Press Q to stop.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = utils.detect_faces(frame)
        
        for i, bbox in enumerate(faces, 1):
            label = f"Face {i}"
            frame = utils.draw_box(frame, bbox, label, (0, 255, 0))

        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('✅ Face detection stopped.')

if __name__ == '__main__':
    detect_only()
