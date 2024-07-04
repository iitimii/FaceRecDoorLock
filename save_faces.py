import cv2
import numpy as np
import os
import pickle

EMBEDDINGS_FILE = 'face_embeddings.pickle'
FACES_FOLDER = 'saved_faces'

def extract_features(face_roi):
    # Simple feature extraction: flatten the image and normalize
    return face_roi.flatten() / 255.0

def load_face_embeddings():
    embeddings = {}
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    embeddings.update(data)
                except EOFError:
                    break
    return embeddings

def save_face_embedding(name, embedding):
    with open(EMBEDDINGS_FILE, 'ab') as f:
        pickle.dump({name: embedding}, f)

def save_face_to_folder(frame, face_roi, name):
    if not os.path.exists(FACES_FOLDER):
        os.makedirs(FACES_FOLDER)
    
    face_folder = os.path.join(FACES_FOLDER, name)
    if not os.path.exists(face_folder):
        os.makedirs(face_folder)
    
    # Save full frame
    frame_filename = os.path.join(face_folder, f"{name}_full.jpg")
    cv2.imwrite(frame_filename, frame)
    
    # Save face ROI
    face_filename = os.path.join(face_folder, f"{name}_face.jpg")
    cv2.imwrite(face_filename, face_roi)

def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if len(faces) > 0:
                name = input("Enter the name for this face: ")
                if name:
                    x, y, w, h = faces[0]  # Use the first detected face
                    face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                    
                    # Save face to folder
                    save_face_to_folder(frame, face_roi, name)
                    
                    # Generate and save embedding
                    face_embedding = extract_features(face_roi)
                    save_face_embedding(name, face_embedding)
                    
                    print(f"Saved face and embedding for {name}")
            else:
                print("No face detected")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
