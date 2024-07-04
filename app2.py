import cv2
import telepot
import time
from deepface import DeepFace

class DoorSecuritySystem:
    def __init__(self):
        self.start = False
        self.first = False
        self.chat_id = None
        self.telegramText = None
        self.bot = telepot.Bot('YOUR_TELEGRAM_BOT_TOKEN')

    def unlock_door(self):
        print("Door Unlocked")
        # Implement actual door unlocking mechanism

    def lock_door(self):
        print("Door Locked")
        # Implement actual door locking mechanism

    def send_picture(self, img):
        filename = "image.jpg"
        cv2.imwrite(filename, img)
        self.bot.sendPhoto(self.chat_id, open(filename, 'rb'))
        self.bot.sendMessage(self.chat_id, 'Person at the door\n/allow or /decline')

    def handle_message(self, msg):
        self.chat_id = msg['chat']['id']
        self.telegramText = msg['text']
        
        print(f'Message received from {self.chat_id}')

        if not self.first:
            self.bot.sendMessage(self.chat_id, '\nCommands:\n/start\n/stop\n/allow\n/decline\n/open\n/close\n/help')
            self.first = True

        if self.telegramText == '/start' and not self.start:
            self.bot.sendMessage(self.chat_id, 'Security camera is activated')
            self.start = True

        if self.start:
            if self.telegramText == '/open':
                self.bot.sendMessage(self.chat_id, 'Unlocking Door')
                self.unlock_door()
            elif self.telegramText == '/close':
                self.bot.sendMessage(self.chat_id, 'Locking Door')
                self.lock_door()
            elif self.telegramText == '/decline':
                self.bot.sendMessage(self.chat_id, 'Declining Access')
                self.lock_door()
            elif self.telegramText == '/allow':
                self.bot.sendMessage(self.chat_id, 'Allowing Access')
                self.unlock_door()
                time.sleep(10)
                self.lock_door()
            elif self.telegramText == '/help':
                self.bot.sendMessage(self.chat_id, '\nCommands:\n/start\n/stop\n/allow\n/decline\n/open\n/close\n/help')
            elif self.telegramText == '/stop':
                self.bot.sendMessage(self.chat_id, 'Security camera is deactivated')
                self.start = False

    def find_available_cameras(self):
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Found an available webcam at index {i}")
                available_cameras.append(i)
            cap.release()
        return available_cameras

    def main(self, model_name="Dlib", detector_backend='mediapipe'):
        door_unlocked = False
        last_pic_time = 0.0
        prevTime = 0
        recognized = False
        
        available_cameras = self.find_available_cameras()
        
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
        
        try:
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

                if recognized:
                    print("Face Recognized, Door unlock")
                    self.unlock_door()
                    door_unlocked = True
                    prevTime = time.time()

                else:
                    if self.start and (time.time() - last_pic_time > 30) and face_detected:
                        self.send_picture(frame_capture)
                        last_pic_time = time.time()

                if door_unlocked and time.time() - prevTime > 10:
                    self.lock_door()
                    door_unlocked = False
                    print("Door locked back")
                    
                cv2.imshow("Face Recognition Camera", frame_face)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

        finally:
            face_recognition_cam.release()
            if dual_camera_mode:
                image_capture_cam.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = DoorSecuritySystem()
    system.main()
