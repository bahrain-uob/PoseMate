import cv2
import pickle as pkl
import time
import xgboost as xgb
import math
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

width = 640
height = 480

hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.75)

model = pkl.load(open('./models/xgboost-model-alphabets-24', 'rb'))

labels = {
    "0" : "A", 
    "1" : "B", 
    "2" : "C", 
    "3" : "D",
    "4" : "E",
    "5" : "F",
    "6" : "G",
    "7" : "H",
    "8" : "I",
    "9" : "K",
    "10" : "L",
    "11" : "M",
    "12" : "N",
    "13" : "O",
    "14" : "P",
    "15" : "Q",
    "16" : "R",
    "17" : "S",
    "18" : "T",
    "19" : "U",
    "20" : "V",
    "21" : "W",
    "22" : "X",
    "23" : "Y",
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

rightHandFirst = False

finalLabel = ''
finalProb = 0

connections = [
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (0, 17)
]

null_vector = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

def generatePointVectors(points):
    vectors = []
    for num, connection in enumerate(connections):
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x_final = x1 - x0
        y_final = y1 - y0
        mag = math.sqrt((x_final)**2+(y_final)**2)
        x = round((x_final/mag),5)
        y = round((y_final/mag),5)
        vectors.append(x)
        vectors.append(y)
    vectors.extend(null_vector)    
    return vectors

def classify(vectors):
    arr = np.array(vectors)
    inputData = xgb.DMatrix(arr.data)

    prob_list = model.predict(inputData)[0]
    max_prob = np.amax(prob_list)
    out_label = labels["{}".format(np.argmax(prob_list, axis=0))]
    return out_label, max_prob

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    results = hands.process(image)
    if(results.multi_handedness):
        # results.multi_handedness[0] is first detected hand
        if(results.multi_handedness[0].classification[0].index == 0):  # Index 0 is Left, 1 is Right
            rightHandFirst = False
        else:
            rightHandFirst = True

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:

        rightHandPoints = []
        leftHandPoints = []

        for hand, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if(rightHandFirst):                       # First hand (0) is Right, Second hand (1) is Left
                if(hand == 0): 
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        rightHandPoints.append((landmark.x, landmark.y))
            else:                                     # First hand (0) is Left, Second hand (1) is Right
                if(hand == 1):
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        rightHandPoints.append((landmark.x, landmark.y))
            
            if(len(rightHandPoints) != 0):
                finalVectors = generatePointVectors(rightHandPoints)
                finalLabel, finalProb = classify(finalVectors)
                
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(image, finalLabel, (width - 200, height - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, 1)
    cv2.putText(image, str(finalProb), (10, height - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, 1)
    
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()