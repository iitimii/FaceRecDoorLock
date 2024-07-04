import cv2
import telepot
import time
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

start = False
first = False

# File to store face embeddings
EMBEDDINGS_FILE = 'face_embeddings.pickle'

def unlock_door():
    print("Door Unlocked")

def lock_door():
    print("Door Locked")

def send_picture(img):
    global chat_id
    filename = "image.jpg"
    cv2.imwrite(filename, img)
    bot.sendPhoto(chat_id, open(filename, 'rb'))
    bot.sendMessage(chat_id, 'Person at the door\n/allow or /decline')

def handle(msg):
    global telegramText
    global chat_id
    global start
    global first
  
    chat_id = msg['chat']['id']
    telegramText = msg['text']
  
    print('Message received from ' + str(chat_id))

    if not first:
        bot.sendMessage(chat_id, '\nCommands:\n/start\n/stop\n/allow\n/decline\n/open\n/close\n/help')
        first = True

    if telegramText == '/start' and not start:
        bot.sendMessage(chat_id, 'Security camera is activated')
        start = True
    
    if start == True:
        if telegramText == '/open':
            bot.sendMessage(chat_id, 'Unlocking Door')
            unlock_door()
        elif telegramText == '/close':
            bot.sendMessage(chat_id, 'Locking Door')
            lock_door()
        elif telegramText == '/decline':
            bot.sendMessage(chat_id, 'Declining Access')
            lock_door()
        elif telegramText == '/allow':
            bot.sendMessage(chat_id, 'Allowing Access')
            unlock_door()
            time.sleep(10)
            lock_door()
        elif telegramText == '/help':
            bot.sendMessage(chat_id, '\nCommands:\n/start\n/stop\n/allow\n/decline\n/open\n/close\n/help')
        elif telegramText == '/stop':
            bot.sendMessage(chat_id, 'Security camera is deactivated')
            start = False

bot = telepot.Bot('7278572730:AAHGeuGYFRWXdsC2XMV9j5uHmys3USvoubc')
bot.message_loop(handle)  

def find_available_cameras():
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found an available webcam at index {i}")
            available_cameras.append(i)
        cap.release()
    return available_cameras

def load_face_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_face_embedding(name, embedding):
    embeddings = load_face_embeddings()
    embeddings[name] = embedding
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)

def recognize_face(face_embedding, known_embeddings, threshold=0.6):
    for name, known_embedding in known_embeddings.items():
        similarity = cosine_similarity([face_embedding], [known_embedding])[0][0]
        if similarity > threshold:
            return name
    return None

def main():
    doorUnlock = False
    last_pic_time = 0.0
    prevTime = 0
    recognized = False
    
    available_cameras = find_available_cameras()
    
    if len(available_cameras) == 0:
        print("Error: No webcams found.")
        return
    
    if len(available_cameras) >= 2:
        print("Using two cameras")
        face_recognition_cam = cv2.VideoCapture(available_cameras[0])
        image_capture_cam = cv2.VideoCapture(available_cameras[1])
        dual_camera_mode = True
    else:
        print("Using single camera for both face recognition and image capture")
        face_recognition_cam = cv2.VideoCapture(available_cameras[0])
        image_capture_cam = face_recognition_cam
        dual_camera_mode = False
    
    if not face_recognition_cam.isOpened():
        print("Error: Could not open webcam.")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    known_embeddings = load_face_embeddings()
    
    while True:
        ret_face, frame_face = face_recognition_cam.read()
        frame_face = cv2.flip(frame_face, 1)
        if dual_camera_mode:
            ret_capture, frame_capture = image_capture_cam.read()
            frame_capture = cv2.flip(frame_capture, 1)
        else:
            ret_capture, frame_capture = ret_face, frame_face
        
        if not ret_face or not ret_capture:
            print("Error: Failed to capture image.")
            break

        if frame_face is None or frame_face.size == 0 or frame_capture is None or frame_capture.size == 0:
            print("Error: Captured frame is empty.")
            continue

        gray = cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_detected = len(faces) > 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame_face, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = gray[y:y+h, x:x+w]
            face_embedding = face_recognizer.compute(face_roi)[1]
            
            name = recognize_face(face_embedding, known_embeddings)
            if name:
                cv2.putText(frame_face, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                recognized = True
            else:
                cv2.putText(frame_face, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        if recognized:
            print("Face Recognized, Door unlock")
            unlock_door()
            doorUnlock = True
            prevTime = time.time()
        elif face_detected:
            print("Face Detected")
            if start and (time.time() - last_pic_time > 30):
                send_picture(frame_capture)
                last_pic_time = time.time()
                print("Sending Picture")

        cv2.imshow("Face Recognition Camera", frame_face)
        if dual_camera_mode:
            cv2.imshow("Image Capture Camera", frame_capture)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            # Save the face embedding for a new person
            name = input("Enter the name for this face: ")
            if name and len(faces) > 0:
                x, y, w, h = faces[0]  # Use the first detected face
                face_roi = gray[y:y+h, x:x+w]
                face_embedding = face_recognizer.compute(face_roi)[1]
                save_face_embedding(name, face_embedding)
                print(f"Saved face embedding for {name}")

    face_recognition_cam.release()
    if dual_camera_mode:
        image_capture_cam.release()
    cv2.destroyAllWindows()

main()
