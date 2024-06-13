import cv2
import telepot
import time
from time import sleep
import datetime
from subprocess import call 
import face_recognition
import pickle

import RPi.GPIO as GPIO #for relay
#lcd library
from Bluetin_Echo import Echo #ultrasonic sensor library


start = False
first = False
doorUnlock = False
speed_of_sound = 340  


RELAY_PIN = 4 #change if different
ECHO_PIN = ...
TRIG_PIN = ...
echo = Echo(TRIG_PIN, ECHO_PIN, speed_of_sound)

GPIO.setmode(GPIO.BOARD)

# relay pins setup
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)

 #ultrasonic sensor setup
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
GPIO.output(TRIG_PIN, GPIO.LOW)

def unlock_door():
    GPIO.output(RELAY_PIN, GPIO.LOW)
    bot.sendMessage(chat_id, 'Door Unlocked')
    doorUnlock = True

def lock_door():
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    bot.sendMessage(chat_id, 'Door Locked')
    doorUnlock = False

def get_distance():
    distance_samples = 10
    dist = echo.read('cm', distance_samples)
    return dist


def send_picture(img):
    global chat_id
    filename = "image.jpg"
    cv2.imwrite(filename, img)
    bot.sendPhoto(chat_id, open(filename, 'rb'))
    bot.sendMessage(chat_id, 'Person at the door')


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
            bot.sendMessage(chat_id, 'LCD Display: Access Denied')
            #LCD SHOULD DISPLAY Access Denied
            lock_door()

        elif telegramText == '/allow':
            bot.sendMessage(chat_id, 'Allowing Access')
            bot.sendMessage(chat_id, 'LCD Display: Access Granted')
            #LCD SHOULD DISPLAY Access Granted
            unlock_door()

        elif telegramText == '/help':
            bot.sendMessage(chat_id, '\nCommands:\n/start\n/stop\n/allow\n/decline\n/open\n/close\n/help')
        elif telegramText == '/stop':
            bot.sendMessage(chat_id, 'Security camera is deactivated')
            start = False


bot = telepot.Bot('7484485509:AAFdv9PzCpLHAna7xXW7B9K6vBxW6TS3uqY')
bot.message_loop(handle)  


def main():
    sent_pic = False
    last_pic_time = 0.0
    prevTime = 0

    currentname = "unknown"
    encodingsP = "encodings.pickle"
    cascade = "haarcascade_frontalface_default.xml"
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open(encodingsP, "rb").read())
    detector = cv2.CascadeClassifier(cascade)
    print("[INFO] starting video stream...")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (500, 500))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        distance = get_distance() #get distance from sensor in centinmeters
        if distance < 2000: #if less than two meters perfrom detection

            rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                minNeighbors=5, minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"],
                    encoding)
                name = "Unknown"

                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    unlock_door()
                    prevTime = time.time()
                    print("Face Recognized, Door unlock")

                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    name = max(counts, key=counts.get)

                    if currentname != name:
                        currentname = name
                        print(currentname)
                
                else:
                    if start and time.time() - last_pic_time > 30:
                        send_picture(frame)

                names.append(name)

            if doorUnlock == True and time.time() - prevTime > 5:
                lock_door()
                print("Door locked back")

            
            for ((top, right, bottom, left), name) in zip(boxes, names):
                cv2.rectangle(frame, (left, top), (right, bottom),
                    (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (255, 0, 0), 2)
            
        cv2.imshow("Security Camera", frame)


        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    echo.stop()