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

start = False
first = False

RELAY_PIN = 21 #change if different

GPIO.setmode(GPIO.BCM)

# # relay pins setup
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.HIGH)



def unlock_door():
    GPIO.output(RELAY_PIN, GPIO.LOW)
    print("Door Unlocked")
    

def lock_door():
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    print("Door Locked")


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
            time.sleep(5)
            lock_door()

        elif telegramText == '/help':
            bot.sendMessage(chat_id, '\nCommands:\n/start\n/stop\n/allow\n/decline\n/open\n/close\n/help')
        elif telegramText == '/stop':
            bot.sendMessage(chat_id, 'Security camera is deactivated')
            start = False


bot = telepot.Bot('7484485509:AAFdv9PzCpLHAna7xXW7B9K6vBxW6TS3uqY')
bot.message_loop(handle)  


def main():
    doorUnlock = False
    last_pic_time = 0.0
    prevTime = 0
    recognized = False

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

        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
        
        recognized = False

        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        matches = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                recognized = True

                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

                if currentname != name:
                    currentname = name
                    print(currentname)
                

            names.append(name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                .8, (255, 0, 0), 2)

        if recognized:
            print("Face Recognized, Door unlock")
            unlock_door()
            doorUnlock = True
            prevTime = time.time()

        else:
            if start and (time.time() - last_pic_time > 30) and len(matches) > 0:
                send_picture(frame)
                last_pic_time = time.time()


        if doorUnlock == True and time.time() - prevTime > 5:
            lock_door()
            doorUnlock = False
            print("Door locked back")
            #LCD should display door locked
            
        cv2.imshow("Security Camera", frame)


        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()