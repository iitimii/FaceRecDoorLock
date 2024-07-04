import cv2
from deepface import DeepFace

def main(model_name="Dlib", detector_backend='opencv'):
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
        frame = cv2.flip(frame, 1)
        
        if not ret:
            print("Error: Failed to capture image.")
            break

        if frame is None or frame.size == 0:
            print("Error: Captured frame is empty.")
            continue

        try:
            faces = DeepFace.detectFace(frame, detector_backend=detector_backend)
            
            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        except Exception as e:
            print('Error detecting faces:', str(e))
        
        cv2.imshow("Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

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

if __name__ == "__main__":
    main()
