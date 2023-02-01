import pickle
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


mp_drawing = mp.solutions.drawing_utils
mp_hol = mp.solutions.holistic
preprocessor = StandardScaler()

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

with mp_hol.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_hol.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hol.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hol.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_hol.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[coords.x, coords.y, coords.z, coords.visibility] for coords in pose]).flatten())

            face = results.face_landmarks.landmark
            face_row = list(np.array([[coords.x, coords.y, coords.z, coords.visibility] for coords in face]).flatten())

            # detect sign with existing model
            ligne = pose_row + face_row
            X = pd.DataFrame([ligne])
            body_lang_class = model.predict(X)[0]
            body_lang_prob = model.predict_proba(X)[0]
            # print(body_lang_class, body_lang_prob)

            # Coordinates of left ear
            coords = tuple(np.multiply(np.array(
                (results.pose_landmarks.landmark[mp_hol.PoseLandmark.LEFT_EAR].x,
                 results.pose_landmarks.landmark[mp_hol.PoseLandmark.LEFT_EAR].y)),
                [640, 480]).astype(int))  # those are the dimensions of webcam

            # creates rectangle and fixes it's position dynamically
            cv2.rectangle(image, (coords[0], coords[1] + 3),
                          (coords[0] + len(body_lang_class) * 20, coords[1] - 30),
                          (245, 200, 69), -1)
            # adding the prediction text on the image
            cv2.putText(image, body_lang_class, coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            # Get status box
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_lang_class.split(' ')[0]
                        , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_lang_prob[np.argmax(body_lang_prob)], 2))
                        , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



        except:

            pass

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()