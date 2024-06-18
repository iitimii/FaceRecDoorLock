import os
import cv2
from train_model import train_model
from RPLCD.i2c import CharLCD


name = 'ChangeName' 
output_directory = f"dataset/{name}" 
os.makedirs(output_directory, exist_ok=True)

lcd = CharLCD(i2c_expander='PCF8574', address=0x3F, port=1, cols=16, rows=2, dotsize=8, backlight_enabled=True)
lcd.clear()
lcd.write_string("press space to take a photo")

cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("press space to take a photo", 500, 300)

img_counter = 0


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("press space to take a photo", frame)
    lcd.clear()
    lcd.write_string("press space to take a photo")

    k = cv2.waitKey(1)
    if k%256 == 27:
        lcd.clear()
        lcd.write_string("Escape hit, closing...")
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        lcd.clear()
        lcd.write_string("saving image")
        img_name = "dataset/"+ name +"/image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cv2.destroyAllWindows()

train_model()