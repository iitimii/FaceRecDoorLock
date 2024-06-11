import cv2
import os  
import time


face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create a directory to store detected faces
output_directory = "detected_faces"
os.makedirs(output_directory, exist_ok=True)

cap = cv2.VideoCapture(0)
while True:
    ret, im = cap.read()
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))

        # Generate a unique filename using timestamp for every saved image
        timestamp = int(time.time())
        filename = os.path.join(output_directory, f"face_{timestamp}.jpg")
        cv2.imwrite(filename, im[y:y+h, x:x+w])  # Save only the detected face portion

    cv2.imshow("Camera", im)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()