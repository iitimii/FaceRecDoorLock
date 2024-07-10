from deepface import DeepFace

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
       
        prevTime = time.time(


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    cv2.destroyAllWindows()



main()