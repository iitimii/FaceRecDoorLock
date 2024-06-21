import cv2
import telepot
import time
import RPi.GPIO as GPIO #for relay
from RPLCD.i2c import CharLCD
from deepface import DeepFace


lcd = CharLCD(i2c_expander='PCF8574', address=0x3F, port=1, cols=16, rows=2, dotsize=8, backlight_enabled=True)
start = False
first = False

RELAY_PIN = 21 #change if different

GPIO.setmode(GPIO.BCM)

# # relay pins setup
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)

lcd.clear()
lcd.write_string("Security Door Lock")



def unlock_door():
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    print("Door Unlocked")
    

def lock_door():
    GPIO.output(RELAY_PIN, GPIO.LOW)
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
            lcd.clear()
            lcd.write_string('Opening')

        elif telegramText == '/close':
            bot.sendMessage(chat_id, 'Locking Door')
            lock_door()
            lcd.clear()
            lcd.write_string('Closing')

        elif telegramText == '/decline':
            bot.sendMessage(chat_id, 'Declining Access')
            lcd.clear()
            lcd.write_string('Declining Access')
            lock_door()

        elif telegramText == '/allow':
            bot.sendMessage(chat_id, 'Allowing Access')
            lcd.clear()
            lcd.write_string('Allowing Access')
            unlock_door()
            time.sleep(10)
            lock_door()
            lcd.clear()
            lcd.write_string('Door Locked')

        elif telegramText == '/help':
            bot.sendMessage(chat_id, '\nCommands:\n/start\n/stop\n/allow\n/decline\n/open\n/close\n/help')
        elif telegramText == '/stop':
            bot.sendMessage(chat_id, 'Security camera is deactivated')
            start = False


bot = telepot.Bot('7484485509:AAFdv9PzCpLHAna7xXW7B9K6vBxW6TS3uqY')
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


def main(model_name="Dlib", detector_backend='mediapipe'):
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        if frame is None or frame.size == 0:
            print("Error: Captured frame is empty.")
            continue

        recognized = False
        face_detected = False

        try: 
            faces = DeepFace.extract_faces(frame, detector_backend=detector_backend, align=True, enforce_detection=True)
            face_detected = True
        except Exception as e:
            faces = []
            face_detected = False
        
        try: 
            dfs = DeepFace.find(frame, db_path='dataset', model_name=model_name, detector_backend=detector_backend, align=True, enforce_detection=True, silent=True)

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
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0))
                        cv2.putText(frame, name, (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
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
                cv2.circle(frame, (x+w//2, y+h//2), 5, (0, 0, 255), -1)

        

        if recognized == True:
            print("Face Recognized, Door unlock")
            unlock_door()
            doorUnlock = True
            lcd.clear()
            lcd.write_string('Face Recognised\n Opening')
            prevTime = time.time()

        else:
            if start and (time.time() - last_pic_time > 30) and face_detected:
                send_picture(frame)
                last_pic_time = time.time()


        if doorUnlock == True and time.time() - prevTime > 10:
            lock_door()
            doorUnlock = False
            print("Door locked back")
            lcd.clear()
            lcd.write_string('Door Locked')
            
        cv2.imshow("Security Camera", frame)


        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()



main()