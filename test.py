import cv2
import telepot
import time
import numpy as np
from deepface import DeepFace
import os
import threading
import logging
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TELEGRAM_BOT_TOKEN = '7278572730:AAHGeuGYFRWXdsC2XMV9j5uHmys3USvoubc'
DATABASE_PATH = 'database'
DETECTION_COOLDOWN = 30  # seconds
DOOR_LOCK_DELAY = 10  # seconds
VIDEO_DURATION = 5  # seconds

# Global variables
start = False
chat_id = None
face_queue = deque(maxlen=5)  # Store last 5 face detections for smoothing

class DoorControl:
    @staticmethod
    def unlock():
        logger.info("Door Unlocked")

    @staticmethod
    def lock():
        logger.info("Door Locked")

class TelegramBot:
    def __init__(self, token):
        self.bot = telepot.Bot(token)
        self.bot.message_loop(self.handle_message)

    def handle_message(self, msg):
        global start, chat_id
        
        chat_id = msg['chat']['id']
        command = msg['text']
        
        logger.info(f'Message received from {chat_id}: {command}')

        if command == '/start' and not start:
            self.bot.sendMessage(chat_id, 'Security camera is activated')
            start = True
        elif command == '/stop' and start:
            self.bot.sendMessage(chat_id, 'Security camera is deactivated')
            start = False
        elif start:
            self.process_command(command)
        else:
            self.bot.sendMessage(chat_id, 'Please start the security camera first with /start')

    def process_command(self, command):
        commands = {
            '/open': (DoorControl.unlock, 'Unlocking Door'),
            '/close': (DoorControl.lock, 'Locking Door'),
            '/decline': (DoorControl.lock, 'Declining Access'),
            '/allow': (self.allow_access, 'Allowing Access'),
            '/help': (self.send_help, 'Sending help information')
        }
        
        action, message = commands.get(command, (None, 'Unknown command'))
        if action:
            self.bot.sendMessage(chat_id, message)
            action()

    def allow_access(self):
        DoorControl.unlock()
        threading.Timer(DOOR_LOCK_DELAY, DoorControl.lock).start()

    def send_help(self):
        help_text = '\nCommands:\n/start\n/stop\n/allow\n/decline\n/open\n/close\n/help'
        self.bot.sendMessage(chat_id, help_text)

    def send_picture(self, img):
        def send_picture_thread():
            filename = "image.jpg"
            cv2.imwrite(filename, img)
            with open(filename, 'rb') as photo:
                self.bot.sendPhoto(chat_id, photo)
            self.bot.sendMessage(chat_id, 'Person at the door\n/allow or /decline')
            os.remove(filename)

        threading.Thread(target=send_picture_thread).start()

    def send_video(self, video_path):
        def send_video_thread():
            with open(video_path, 'rb') as video_file:
                self.bot.sendVideo(chat_id, video_file, caption="5-second video of the door")
            os.remove(video_path)

        threading.Thread(target=send_video_thread).start()

class VideoCapture:
    def __init__(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            return cv2.flip(frame, 1)
        return None

    def release(self):
        self.cap.release()

def capture_video(camera, duration):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
    
    start_time = time.time()
    while time.time() - start_time < duration:
        frame = camera.read()
        if frame is not None:
            out.write(frame)
    
    out.release()
    os.system("ffmpeg -i output.mp4 -vcodec libx264 telegram_video.mp4 -y")
    os.remove("output.mp4")
    return "telegram_video.mp4"

def find_available_cameras():
    return [i for i in range(10) if cv2.VideoCapture(i).read()[0]]

def process_face(frame, model):
    try:
        faces = DeepFace.extract_faces(frame, detector_backend='mediapipe', align=True, enforce_detection=False)
        if faces:
            face_queue.append(1)
            dfs = DeepFace.find(frame, model_name=model, detector_backend='mediapipe', align=True, enforce_detection=False, db_path=DATABASE_PATH)
            if dfs and len(dfs[0]) > 0:
                df = dfs[0]
                identity = str(df['identity'].iloc[0])
                name = identity.split("/")[-1].split(".")[0]
                return True, name
    except Exception as e:
        logger.error(f"Error in face processing: {e}")
    
    face_queue.append(0)
    return False, None

def main():
    available_cameras = find_available_cameras()
    if not available_cameras:
        logger.error("No cameras found.")
        return

    face_cam = VideoCapture(available_cameras[0])
    image_cam = VideoCapture(available_cameras[1]) if len(available_cameras) > 1 else face_cam

    bot = TelegramBot(TELEGRAM_BOT_TOKEN)
    face_model = DeepFace.build_model("Facenet")

    door_unlocked = False
    last_detection_time = 0
    unlock_time = 0

    while True:
        frame_face = face_cam.read()
        frame_capture = image_cam.read() if image_cam != face_cam else frame_face

        if frame_face is None or frame_capture is None:
            logger.error("Failed to capture frame.")
            continue

        recognized, name = process_face(frame_face, face_model)

        if recognized:
            logger.info(f"Face Recognized: {name}")
            DoorControl.unlock()
            door_unlocked = True
            unlock_time = time.time()
        elif sum(face_queue) >= 3 and not recognized:  # Face detected in 3 out of last 5 frames
            logger.info("Unknown Face Detected")
            if start and (time.time() - last_detection_time > DETECTION_COOLDOWN):
                bot.send_picture(frame_face)
                video_path = capture_video(image_cam, VIDEO_DURATION)
                bot.send_video(video_path)
                last_detection_time = time.time()

        if door_unlocked and time.time() - unlock_time > DOOR_LOCK_DELAY:
            DoorControl.lock()
            door_unlocked = False

        cv2.imshow("Face Recognition", frame_face)
        if image_cam != face_cam:
            cv2.imshow("Door Camera", frame_capture)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_cam.release()
    if image_cam != face_cam:
        image_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()