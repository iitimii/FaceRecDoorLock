import cv2
import telepot
import time
import numpy as np
from deepface import DeepFace
import os
import threading

start = False
first = False
chat_id = None

def unlock_door():
    print("Door Unlocked")

def lock_door():
    print("Door Locked")

def send_picture(img):
    def send_picture_thread():
        global chat_id
        filename = "image.jpg"
        cv2.imwrite(filename, img)
        bot.sendPhoto(chat_id, open(filename, 'rb'))
        bot.sendMessage(chat_id, 'Person at the door\n/allow or /decline')
        os.remove(filename)

    thread = threading.Thread(target=send_picture_thread)
    thread.start()

def capture_and_send_video(camera):
    def video_thread():
        global chat_id
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

        start_time = time.time()
        duration = 5  # 5 seconds

        while time.time() - start_time < duration:
            ret, frame = camera.read()
            if ret:
                out.write(frame)
            else:
                break

        out.release()

        os.system("ffmpeg -i output.mp4 -vcodec libx264 telegram_video.mp4 -y")

        with open("telegram_video.mp4", "rb") as video_file:
            bot.sendVideo(chat_id, video_file, caption="5-second video of the door")

        os.remove("output.mp4")
        os.remove("telegram_video.mp4")

    thread = threading.Thread(target=video_thread)
    thread.start()

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

def main():
    doorUnlock = False
    last_detection_time = 0.0
    prevTime = 0
    face_detected_start_time = None
    
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
        
        try: 
            faces = DeepFace.extract_faces(frame_face, detector_backend='mediapipe', align=True, enforce_detection=True)
            face_detected = True
        except Exception as e:
            faces = []
            face_detected = False

        if len(faces) > 0:
                for i in range(len(faces)):
                    face_dict = faces[i]["facial_area"]
                    x = int(face_dict["x"])
                    y = int(face_dict["y"])
                    w = int(face_dict["w"])
                    h = int(face_dict["h"])
                    cv2.circle(frame_face, (x+w//2, y+h//2), 5, (0, 0, 255), -1)
        
        recognized = False

        try: 
            dfs = DeepFace.find(frame_face, model_name='Facenet', detector_backend='mediapipe', align=True, enforce_detection=True, db_path='database')
            if len(dfs) > 0:
                for i in range(len(dfs)):
                    df = dfs[i]
                    if len(df) > 0:
                        identity = str(df['identity'].iloc[0])
                        name = identity.split("/")[-1].split(".")[0]
                        bx = int(df['source_x'].iloc[0])
                        by = int(df['source_y'].iloc[0])
                        bw = int(df['source_w'].iloc[0])
                        bh = int(df['source_h'].iloc[0])
                        cv2.rectangle(frame_face, (bx, by), (bx+bw, by+bh), (0, 255, 0))
                        cv2.putText(frame_face, name, (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        print(f'face recognised {name}')
                        recognized = True

        except Exception as e:
            recognized = False

        if recognized:
            print(f"Face Recognized, Door unlock for {name}")
            unlock_door()
            doorUnlock = True
            prevTime = time.time()

        elif face_detected and not recognized:
            print("Unknown Face Detected")
            if face_detected_start_time is None:
                face_detected_start_time = time.time()
            elif time.time() - face_detected_start_time >= 10:
                if start and (time.time() - last_detection_time > 90):
                    send_picture(frame_face)
                    capture_and_send_video(image_capture_cam)
                    last_detection_time = time.time()
                    print("Sending Picture and Video")
                face_detected_start_time = None
        else:
            face_detected_start_time = None

        if doorUnlock and time.time() - prevTime > 10:
            lock_door()
            doorUnlock = False
            print("Door locked back")

        cv2.imshow("Face Recognition Camera", frame_face)
        if dual_camera_mode:
            cv2.imshow("Image Capture Camera", frame_capture)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            name = input("Enter the name for this face: ")
            if name and len(faces) > 0:
                img_name = f"database/{name}.jpg"
                cv2.imwrite(img_name, frame_face)
                print(f"Saved face for {name}")

    face_recognition_cam.release()
    if dual_camera_mode:
        image_capture_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()