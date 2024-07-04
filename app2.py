import cv2
import telepot
import time
import numpy as np

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

def find_available_camera():
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found an available webcam at index {i}")
            cap.release()
            return i
        cap.release()
    print("No available webcam found.")
    return None

def main():
    doorUnlock = False
    last_pic_time = 0.0
    prevTime = 0
    recognized = False
    
    cam_index = find_available_camera()
    
    if cam_index is None:
        print("Error: No webcams found.")
        return
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            print("Error: Failed to capture image.")
            break

        if frame is None or frame.size == 0:
            print("Error: Captured frame is empty.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_detected = len(faces) > 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if face_detected:
            print("Face Detected")
            if start and (time.time() - last_pic_time > 30):
                send_picture(frame)
                last_pic_time = time.time()
                print("Sending Picture")

        cv2.imshow("Security Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
