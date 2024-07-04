import cv2
import telepot
import time
from deepface import DeepFace

start = False
first = False

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

def main(model_name="Dlib", detector_backend='mediapipe'):
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
    
    while True:
        ret_face, frame_face = face_recognition_cam.read()
        if dual_camera_mode:
            ret_capture, frame_capture = image_capture_cam.read()
        else:
            ret_capture, frame_capture = ret_face, frame_face
        
        if not ret_face or not ret_capture:
            print("Error: Failed to capture image.")
            break

        if frame_face is None or frame_face.size == 0 or frame_capture is None or frame_capture.size == 0:
            print("Error: Captured frame is empty.")
            continue

        recognized = False
        face_detected = False

        try: 
            faces = DeepFace.extract_faces(frame_face, detector_backend=detector_backend, align=True, enforce_detection=True)
            face_detected = True
        except Exception as e:
            faces = []
            face_detected = False
        
        try: 
            dfs = DeepFace.find(frame_face, db_path='dataset', model_name=model_name, detector_backend=detector_backend, align=True, enforce_detection=True, silent=True)

            if len(dfs) > 0:
                for i in range(len(dfs)):
                    df = dfs[i]
                    if len(df) > 0:
                        identity = str(df["identity"].iloc[0])
                        name = identity.split("/")[-1].split(".")[0]
                        bx = int(df['source_x'].iloc[0])
                        by = int(df['source_y'].iloc[0])
                        bw = int(df['source_w'].iloc[0])
                        bh = int(df['source_h'].iloc[0])
                        cv2.rectangle(frame_face, (bx, by), (bx+bw, by+bh), (0, 255, 0))
                        cv2.putText(frame_face, name, (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        recognized = True

        except Exception as e:
            recognized = False
            print('No Face')
        
        if len(faces) > 0:
            for i in range(len(faces)):
                face_dict = faces[i]["facial_area"]
                x = int(face_dict["x"])
                y = int(face_dict["y"])
                w = int(face_dict["w"])
                h = int(face_dict["h"])
                cv2.circle(frame_face, (x+w//2, y+h//2), 5, (0, 0, 255), -1)

        if recognized == True:
            print("Face Recognized, Door unlock")
            unlock_door()
            doorUnlock = True
            prevTime = time.time()

        else:
            if start and (time.time() - last_pic_time > 30) and face_detected:
                send_picture(frame_capture)
                last_pic_time = time.time()

        if doorUnlock == True and time.time() - prevTime > 10:
            lock_door()
            doorUnlock = False
            print("Door locked back")
            
        cv2.imshow("Face Recognition Camera", frame_face)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    face_recognition_cam.release()
    if dual_camera_mode:
        image_capture_cam.release()
    cv2.destroyAllWindows()

main()
