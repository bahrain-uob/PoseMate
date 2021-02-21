import cv2
import time
import pandas as pd
import xgboost as xgb
import math
import numpy as np
import mediapipe as mp
import time
import csv

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

width = 640
height = 480

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

rightHandFirst = False
isMultiHand = False
addFrame = True # For first frame

keyFrames = []
keyCheckPoints = []

file_name = "hello.csv"
lock_frame = 96
collecting = False
dataRows = []
frameCounter = 0

nullVectors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
null_24 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

d_threshold = 0.1

connections = [
    (1, 4), (5, 8), (9, 12), (13, 16), (17, 20)
]

r_c_x = 0
r_c_y = 0
l_c_x = 0
l_c_y = 0
r_p_x = 0
r_p_y = 0
l_p_x = 0
l_p_y = 0

def generatePointVectors(rightPoints, leftPoints, previousFrames):
    rightVectors = []
    leftVectors = []

    r_prev_origin_x = 0
    r_prev_origin_y = 0
    l_prev_origin_x = 0
    l_prev_origin_y = 0

    r_dx = 0
    r_dy = 0
    l_dx = 0
    l_dy = 0

    if(len(previousFrames) == 0):
        r_prev_origin_x = 0
        r_prev_origin_y = 0
        l_prev_origin_x = 0
        l_prev_origin_y = 0
    else:
        r_prev_origin_x = previousFrames[0]
        r_prev_origin_y = previousFrames[1]
        l_prev_origin_x = previousFrames[12]
        l_prev_origin_y = previousFrames[13]

    if(len(rightPoints) != 0):
        r_origin_x, r_origin_y = rightPoints[0]

        r_origin_x_rounded = round((r_origin_x), 5)
        r_origin_y_rounded = round((r_origin_y), 5)

        r_dx = r_origin_x_rounded - r_prev_origin_x
        r_dy = r_origin_y_rounded - r_prev_origin_y

        rightVectors.append(r_origin_x_rounded)
        rightVectors.append(r_origin_y_rounded)

    if(len(leftPoints) != 0):
        l_origin_x, l_origin_y = leftPoints[0]

        l_origin_x_rounded = round((l_origin_x), 5)
        l_origin_y_rounded = round((l_origin_y), 5)

        l_dx = l_origin_x_rounded - l_prev_origin_x
        l_dy = l_origin_y_rounded - l_prev_origin_y

        leftVectors.append(l_origin_x_rounded)
        leftVectors.append(l_origin_y_rounded)

    for num, connection in enumerate(connections):

        if(len(rightPoints) != 0):
            r_x0, r_y0 = rightPoints[connection[0]]
            r_x1, r_y1 = rightPoints[connection[1]]
            r_x_final = r_x1 - r_x0
            r_y_final = r_y1 - r_y0
            r_mag = math.sqrt((r_x_final)**2+(r_y_final)**2)

            r_x_vector = round((r_x_final/r_mag) + r_dx,5)
            r_y_vector = round((r_y_final/r_mag) + r_dy,5)

            rightVectors.append(r_x_vector)
            rightVectors.append(r_y_vector)

        if(len(leftPoints) != 0):
            l_x0, l_y0 = leftPoints[connection[0]]
            l_x1, l_y1 = leftPoints[connection[1]]
            l_x_final = l_x1 - l_x0
            l_y_final = l_y1 - l_y0
            l_mag = math.sqrt((l_x_final)**2+(l_y_final)**2)

            l_x_vector = round((l_x_final/l_mag) + l_dx,5)
            l_y_vector = round((l_y_final/l_mag) + l_dy,5)

            leftVectors.append(l_x_vector)
            leftVectors.append(l_y_vector)
            
    finalVectors = []
    if(len(rightVectors) != 0 and len(leftVectors) != 0):
        finalVectors.extend(rightVectors)
        finalVectors.extend(leftVectors)
    if(len(rightVectors) == 0):
        finalVectors.extend(nullVectors)
        finalVectors.extend(leftVectors)
    if(len(leftVectors) == 0):
        finalVectors.extend(rightVectors)
        finalVectors.extend(nullVectors)

    return finalVectors

def checkPreviousFrame(currCheckPoints, prevCheckPoints):

    r_current_dx = currCheckPoints[0]
    r_current_dy = currCheckPoints[1]
    l_current_dx = currCheckPoints[2]
    l_current_dy = currCheckPoints[3]

    r_prev_dx = prevCheckPoints[0]
    r_prev_dy = prevCheckPoints[1]
    l_prev_dx = prevCheckPoints[2]
    l_prev_dy = prevCheckPoints[3]

    r_dx = round(abs(r_current_dx - r_prev_dx), 5)
    r_dy = round(abs(r_current_dy - r_prev_dy), 5)
    l_dx = round(abs(l_current_dx - l_prev_dx), 5)
    l_dy = round(abs(l_current_dy - l_prev_dy), 5)

    if(r_dx >= d_threshold or r_dy >= d_threshold or l_dx >= d_threshold or l_dx >= d_threshold):
        print("Thresold crossed.")

        return True, r_current_dx, r_current_dy, l_current_dx, l_current_dy, r_prev_dx, r_prev_dy, l_prev_dx, l_prev_dy
    else:
        return False, r_current_dx, r_current_dy, l_current_dx, l_current_dy, r_prev_dx, r_prev_dy, l_prev_dx, l_prev_dy

def generateCheckPoints(rightPoints, leftPoints):
    checkPoints = []
    if(len(rightPoints) != 0 and len(leftPoints) != 0):
        r_palm_x, r_palm_y = rightPoints[0]
        r_thumb_x, r_thumb_y = rightPoints[4]
        r_index_x, r_index_y = rightPoints[8]
        r_pinky_x, r_pinky_y = rightPoints[20]

        r_mean_x = round((r_palm_x + r_thumb_x + r_index_x + r_pinky_x)/4, 5)
        r_mean_y = round((r_palm_y + r_thumb_y + r_index_y + r_pinky_y)/4, 5)

        l_palm_x, l_palm_y = leftPoints[0]
        l_thumb_x, l_thumb_y = leftPoints[4]
        l_index_x, l_index_y = leftPoints[8]
        l_pinky_x, l_pinky_y = leftPoints[20]

        l_mean_x = round((l_palm_x + l_thumb_x + l_index_x + l_pinky_x)/4, 5)
        l_mean_y = round((l_palm_y + l_thumb_y + l_index_y + l_pinky_y)/4, 5)

        checkPoints.append(r_mean_x)
        checkPoints.append(r_mean_y)
        checkPoints.append(l_mean_x)
        checkPoints.append(l_mean_y)

    elif(len(rightPoints) != 0 and len(leftPoints) == 0):
        r_palm_x, r_palm_y = rightPoints[0]
        r_thumb_x, r_thumb_y = rightPoints[4]
        r_index_x, r_index_y = rightPoints[8]
        r_pinky_x, r_pinky_y = rightPoints[20]

        r_mean_x = round((r_palm_x + r_thumb_x + r_index_x + r_pinky_x)/4, 5)
        r_mean_y = round((r_palm_y + r_thumb_y + r_index_y + r_pinky_y)/4, 5)

        checkPoints.append(r_mean_x)
        checkPoints.append(r_mean_y)
        checkPoints.append(0)
        checkPoints.append(0)
    elif(len(leftPoints) != 0 and len(rightPoints) == 0):
        l_palm_x, l_palm_y = leftPoints[0]
        l_thumb_x, l_thumb_y = leftPoints[4]
        l_index_x, l_index_y = leftPoints[8]
        l_pinky_x, l_pinky_y = leftPoints[20]

        l_mean_x = round((l_palm_x + l_thumb_x + l_index_x + l_pinky_x)/4, 5)
        l_mean_y = round((l_palm_y + l_thumb_y + l_index_y + l_pinky_y)/4, 5)

        checkPoints.append(0)
        checkPoints.append(0)
        checkPoints.append(l_mean_x)
        checkPoints.append(l_mean_y)

    return checkPoints

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
        if(len(results.multi_handedness) == 1):
            isMultiHand = False
        else:
            isMultiHand = True
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
                else:
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        leftHandPoints.append((landmark.x, landmark.y))
            else:                                     # First hand (0) is Left, Second hand (1) is Right
                if(hand == 0):
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        leftHandPoints.append((landmark.x, landmark.y))
                else: 
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        rightHandPoints.append((landmark.x, landmark.y))

            if(isMultiHand):
                if(hand == 1):
                    finalVectors = generatePointVectors(rightHandPoints, leftHandPoints, keyFrames)
                    checkPoints = generateCheckPoints(rightHandPoints, leftHandPoints)

                    key = cv2.waitKey(1)
                    if(key == ord('s')):    # Press 'S' to start/stop data collection
                        if collecting == False:
                            collecting = True
                        else:
                            collecting = False

                    if(collecting):
                        if(addFrame == True):
                            keyFrames.extend(finalVectors)
                            keyCheckPoints.extend(checkPoints)
                            print("Frame Added. Length of Keyframes: ", len(keyFrames))
                            addFrame = False
                        else:
                            addFrame, r_c_x, r_c_y, l_c_x, l_c_y, r_p_x, r_p_y, l_p_x, l_p_y = checkPreviousFrame(checkPoints, keyCheckPoints)
                            if(addFrame == True):
                                keyCheckPoints = []
                            if(len(keyFrames) == lock_frame):
                                remainingFrames = (96 - lock_frame) / 24
                                i = 0
                                while(i < remainingFrames):
                                    keyFrames.extend(null_24)
                                    i = i + 1
                                dataRows.append(keyFrames)
                                keyFrames = []
                                collecting = False

                        cv2.circle(image, (int(r_c_x * width), int(r_c_y * height)), 3, (255, 0, 0), 2)
                        cv2.circle(image, (int(l_c_x * width), int(l_c_y * height)), 3, (255, 255, 0), 2)

                        cv2.circle(image, (int(r_p_x * width), int(r_p_y * height)), 3, (0, 255, 0), 2)
                        cv2.circle(image, (int(l_p_x * width), int(l_p_y * height)), 3, (0, 255, 255), 2)
                    
                    else:
                        if(len(keyFrames) != 0):
                            if(len(keyFrames) == 24):
                                keyFrames.extend(null_24)
                                keyFrames.extend(null_24)
                                keyFrames.extend(null_24)
                            elif(len(keyFrames) == 48):
                                keyFrames.extend(null_24)
                                keyFrames.extend(null_24)
                            elif(len(keyFrames) == 72):
                                keyFrames.extend(null_24)
                            dataRows.append(keyFrames)
                            frameCounter = frameCounter + 1
                            print("Frame Count ", frameCounter)
                            keyFrames = []
                        
            else:
                finalVectors = generatePointVectors(rightHandPoints, leftHandPoints, keyFrames)
                checkPoints = generateCheckPoints(rightHandPoints, leftHandPoints)

                key = cv2.waitKey(1)
                if(key == ord('s')):    # Press 'S' to start/stop data collection
                    if collecting == False:
                        collecting = True
                    else:
                        collecting = False

                if(collecting):
                    if(addFrame == True):
                        keyFrames.extend(finalVectors)
                        keyCheckPoints.extend(checkPoints)
                        print("Frame Added. Length of Keyframes: ", len(keyFrames))
                        addFrame = False
                    else:
                        addFrame, r_c_x, r_c_y, l_c_x, l_c_y, r_p_x, r_p_y, l_p_x, l_p_y = checkPreviousFrame(checkPoints, keyCheckPoints)
                        if(addFrame == True):
                            keyCheckPoints = []
                        if(len(keyFrames) == lock_frame):
                            remainingFrames = (96 - lock_frame) / 24
                            i = 0
                            while(i < remainingFrames):
                                keyFrames.extend(null_24)
                                i = i + 1
                            dataRows.append(keyFrames)
                            frameCounter = frameCounter + 1
                            print("Frame Count ", frameCounter)
                            keyFrames = []
                            collecting = False

                    cv2.circle(image, (int(r_c_x * width), int(r_c_y * height)), 3, (255, 0, 0), 2)
                    cv2.circle(image, (int(l_c_x * width), int(l_c_y * height)), 3, (255, 255, 0), 2)

                    cv2.circle(image, (int(r_p_x * width), int(r_p_y * height)), 3, (0, 255, 0), 2)
                    cv2.circle(image, (int(l_p_x * width), int(l_p_y * height)), 3, (0, 255, 255), 2)
                
                else:
                    if(len(keyFrames) != 0):
                        if(len(keyFrames) == 24):
                            keyFrames.extend(null_24)
                            keyFrames.extend(null_24)
                            keyFrames.extend(null_24)
                        elif(len(keyFrames) == 48):
                            keyFrames.extend(null_24)
                            keyFrames.extend(null_24)
                        elif(len(keyFrames) == 72):
                            keyFrames.extend(null_24)
                        dataRows.append(keyFrames)
                        frameCounter = frameCounter + 1
                        print("Frame Count ", frameCounter)
                        keyFrames = []

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
    # Decrease FPS
    # time.sleep(1/frameRate)
    # # Calculate FPS
    # counter+=1
    # if (time.time() - start_time) > x :
    #     print("FPS: ", counter / (time.time() - start_time))
    #     counter = 0
    #     start_time = time.time()

with open(file_name, 'a+', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for row in dataRows:
        writer.writerow(row)

hands.close()
cap.release()