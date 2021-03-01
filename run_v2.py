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

modelRight = pkl.load(open('./models/xgboost-model-dynamic-words-16-right-tuned', 'rb'))
modelLeft = pkl.load(open('./models/xgboost-model-dynamic-words-16-left-tuned', 'rb'))

labels = {
    "0" : "me", 
    "1" : "you", 
    "2" : "hello", 
    "3" : "from",
    "4" : "good",
    "5" : "how",
    "6" : "university",
    "7" : "welcome",
    "8" : "hope",
    "9" : "like",
    "10" : "new",
    "11" : "people",
    "12" : "technology",
    "13" : "use",
    "14" : "voice",
    "15" : "create"
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

rightHandFirst = False
isMultiHand = False

# Initialised to True for first frame
addRightFrame = True
addLeftFrame = True

rightKeyFrames = []
leftKeyFrames = []

rightKeyCheckPoints = []
leftKeyCheckPoints = []

rightLabel = ''
rightProb = 0
leftLabel = ''
leftProb = 0

null_12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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

def generatePointVectors(points, previousFrames):
    vectors = []

    prev_origin_x = 0
    prev_origin_y = 0

    dx = 0
    dy = 0

    if(len(previousFrames) == 0):
        prev_origin_x = 0
        prev_origin_y = 0
    else:
        prev_origin_x = previousFrames[0]
        prev_origin_y = previousFrames[1]
    
    origin_x, origin_y = points[0]

    origin_x_rounded = round((origin_x), 5)
    origin_y_rounded = round((origin_y), 5)

    dx = origin_x_rounded - prev_origin_x
    dy = origin_y_rounded - prev_origin_y

    vectors.append(origin_x_rounded)
    vectors.append(origin_y_rounded)

    for num, connection in enumerate(connections):

        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x_final = x1 - x0
        y_final = y1 - y0
        mag = math.sqrt((x_final)**2+(y_final)**2)

        x_vector = round((x_final/mag) + dx,5)
        y_vector = round((y_final/mag) + dy,5)

        vectors.append(x_vector)
        vectors.append(y_vector)

    return vectors

def generateCheckPoints(points):
    checkPoints = []

    palm_x, palm_y = points[0]
    thumb_x, thumb_y = points[4]
    index_x, index_y = points[8]
    pinky_x, pinky_y = points[20]

    mean_x = round((palm_x + thumb_x + index_x + pinky_x)/4, 5)
    mean_y = round((palm_y + thumb_y + index_y + pinky_y)/4, 5)

    checkPoints.append(mean_x)
    checkPoints.append(mean_y)

    return checkPoints

def checkPreviousFrame(currCheckPoints, prevCheckPoints):
    current_dx = currCheckPoints[0]
    current_dy = currCheckPoints[1]

    prev_dx = prevCheckPoints[0]
    prev_dy = prevCheckPoints[1]

    dx = round(abs(current_dx - prev_dx), 5)
    dy = round(abs(current_dy - prev_dy), 5)

    if(dx >= d_threshold or dy >= d_threshold):
        print("Thresold crossed.")
        return True, current_dx, current_dy, prev_dx, prev_dy
    else:
        return False, current_dx, current_dy, prev_dx, prev_dy

def recalculateFrames(frames):
    
    cycledFrames = []
    cycledFrames.extend(frames)
    # Current Origin
    if(len(frames) > 12):
        base_x = cycledFrames[0]
        base_y = cycledFrames[1]

        secondFrame_dx = cycledFrames[12] - base_x
        secondFrame_dy = cycledFrames[13] - base_y

        # New Origin
        new_base_x = cycledFrames[12]
        new_base_y = cycledFrames[13]

        if(len(frames) > 24):
            thirdFrame_dx = cycledFrames[24] - base_x
            thirdFrame_dy = cycledFrames[25] - base_y

            # New second frame
            new_secondFrame_dx = cycledFrames[24] - new_base_x
            new_secondFrame_dy = cycledFrames[25] - new_base_y

            if(len(frames) > 36):
                fourthFrame_dx = cycledFrames[36] - base_x
                fourthFrame_dy = cycledFrames[37] - base_y

                # New third frame
                new_thirdFrame_dx = cycledFrames[36] - new_base_x
                new_thirdFrame_dy = cycledFrames[37] - new_base_y     
    
        i = 12
        while(i < 48):

            # This
            if(i >= 14 and i < 24 and len(frames) > 12):
                cycledFrames[i] = round((cycledFrames[i] - secondFrame_dx), 5)
                cycledFrames[i + 1] = round((cycledFrames[i + 1] - secondFrame_dy), 5)
            # This
            elif(i >= 26 and i < 36 and len(frames) > 24):
                original_keyframe_dx = cycledFrames[i] - thirdFrame_dx
                original_keyframe_dy = cycledFrames[i + 1] - thirdFrame_dy
                
                cycledFrames[i] = round(original_keyframe_dx + new_secondFrame_dx, 5)
                cycledFrames[i + 1] = round(original_keyframe_dy + new_secondFrame_dy, 5)
            # This
            elif(i >= 38 and i < 48 and len(frames) > 36):
                original_keyframe_dx = cycledFrames[i] - fourthFrame_dx
                original_keyframe_dy = cycledFrames[i + 1] - fourthFrame_dy
                
                cycledFrames[i] = round(original_keyframe_dx + new_thirdFrame_dx, 5)
                cycledFrames[i + 1] = round(original_keyframe_dy + new_thirdFrame_dy, 5)
            i = i + 2
    # 0 - 11
    # 12 - 23
    # 24 - 35
    # 36 - 47
    # Cycle out
    cycledFrames = cycledFrames[12:]
    return cycledFrames

def preprocessData(frames):
    
    dataToProcess = []
    dataToProcess.extend(frames)
    if(len(dataToProcess) != 48):
        if(len(dataToProcess) == 12):
            dataToProcess.extend(null_12)
            dataToProcess.extend(null_12)
            dataToProcess.extend(null_12)
        elif(len(dataToProcess) == 24):
            dataToProcess.extend(null_12)
            dataToProcess.extend(null_12)
        elif(len(dataToProcess) == 36):
            dataToProcess.extend(null_12)
        else:
            print("Error in preprocessData. Length of dataToProcess: ", len(dataToProcess))

    group_0 = []
    group_0.extend(dataToProcess[:12])
    group_0.extend(null_12)
    group_0.extend(null_12)
    group_0.extend(null_12)

    group_1 = []
    group_1.extend(dataToProcess[:24])
    group_1.extend(null_12)
    group_1.extend(null_12)

    group_2 = []
    group_2.extend(dataToProcess[:36])
    group_2.extend(null_12)

    group_3 = []
    group_3.extend(dataToProcess[:48])

    arr_0 = np.array(group_0)
    arr_1 = np.array(group_1)
    arr_2 = np.array(group_2)
    arr_3 = np.array(group_3)

    inputData_0 = xgb.DMatrix(arr_0.data)
    inputData_1 = xgb.DMatrix(arr_1.data)
    inputData_2 = xgb.DMatrix(arr_2.data)
    inputData_3 = xgb.DMatrix(arr_3.data)
    # Convert values to DMatrix format
    return xgb.DMatrix(arr_0.data), xgb.DMatrix(arr_1.data), xgb.DMatrix(arr_2.data), xgb.DMatrix(arr_3.data)

def classification(inputData_0, inputData_1, inputData_2, inputData_3, model):
    prob_list_0 = model.predict(inputData_0)[0]
    prob_list_1 = model.predict(inputData_1)[0]
    prob_list_2 = model.predict(inputData_2)[0]
    prob_list_3 = model.predict(inputData_3)[0]

    max_prob_0 = np.amax(prob_list_0)
    max_prob_1 = np.amax(prob_list_1)
    max_prob_2 = np.amax(prob_list_2)
    max_prob_3 = np.amax(prob_list_3)

    out_label_0 = labels["{}".format(np.argmax(prob_list_0, axis=0))]
    out_label_1 = labels["{}".format(np.argmax(prob_list_1, axis=0))]
    out_label_2 = labels["{}".format(np.argmax(prob_list_2, axis=0))]
    out_label_3 = labels["{}".format(np.argmax(prob_list_3, axis=0))]

    label = out_label_0
    prob = max_prob_0

    if(prob < max_prob_1 and max_prob_1 > max_prob_2 and max_prob_1 > max_prob_3):
        prob = max_prob_1
        label = out_label_1
    elif(prob < max_prob_2 and max_prob_2 > max_prob_3 and max_prob_2 > max_prob_1):
        prob = max_prob_2
        label = out_label_2
    elif(prob < max_prob_3 and max_prob_3 > max_prob_1 and max_prob_3 > max_prob_2):
        prob = max_prob_3
        label = out_label_3

    return label, prob

def cleanUp(frames, model):
    temp_frames = []
    temp_frames.extend(frames)

    temp_label = ''
    temp_prob = 0

    if(model is None):
        temp_frames = []
        return temp_frames, temp_label, temp_prob

    while(len(temp_frames) != 0):

        temp_frames = recalculateFrames(temp_frames)
        if(len(temp_frames) != 0):
            # Preprocess
            set0, set1, set2, set3 = preprocessData(temp_frames)
            # Classify
            temp_label, temp_prob = classification(set0, set1, set2, set3, model)
    temp_frames = []
    return temp_frames, temp_label, temp_prob

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

        rightVectors = []
        leftVectors = []

        rightCheckPoints = []
        leftCheckPoints = []

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
                    
                    if(len(rightHandPoints) != 0 and len(leftHandPoints) != 0):
                        rightVectors = generatePointVectors(rightHandPoints, rightKeyFrames)
                        rightCheckPoints = generateCheckPoints(rightHandPoints)
                        
                        leftVectors = generatePointVectors(leftHandPoints, leftKeyFrames)
                        leftCheckPoints = generateCheckPoints(leftHandPoints)

                        if(len(rightKeyFrames) == 48):
                            rightKeyFrames = recalculateFrames(rightKeyFrames)
                            print("Right Frame Cycled:", len(rightKeyFrames))
                        if(len(leftKeyFrames) == 48):
                            leftKeyFrames = recalculateFrames(leftKeyFrames)
                            print("Left Frame Cycled:", len(leftKeyFrames))

                        if(addRightFrame == True or addLeftFrame == True):
                            rightKeyFrames.extend(rightVectors)
                            rightKeyCheckPoints.extend(rightCheckPoints)

                            leftKeyFrames.extend(leftVectors)
                            leftKeyCheckPoints.extend(leftCheckPoints)
                            
                            print("Right Added: ", len(rightKeyFrames), "Left Added: ", len(leftKeyFrames))
                            # Preprocess
                            r_set0, r_set1, r_set2, r_set3 = preprocessData(rightKeyFrames)
                            l_set0, l_set1, l_set2, l_set3 = preprocessData(leftKeyFrames)
                            # Classify
                            rightLabel, rightProb = classification(r_set0, r_set1, r_set2, r_set3, modelRight)
                            leftLabel, leftProb = classification(l_set0, l_set1, l_set2, l_set3, modelLeft)

                            addRightFrame = False
                            addLeftFrame = False
                        else:
                            if(len(rightKeyCheckPoints) == 0):
                                rightKeyCheckPoints.extend(rightCheckPoints)
                            else:
                                addRightFrame, r_c_x, r_c_y, r_p_x, r_p_y = checkPreviousFrame(rightCheckPoints, rightKeyCheckPoints)

                            if(len(leftKeyCheckPoints) == 0):
                                leftKeyCheckPoints.extend(leftCheckPoints)
                            else:
                                addLeftFrame, l_c_x, l_c_y, l_p_x, l_p_y = checkPreviousFrame(leftCheckPoints, leftKeyCheckPoints)

                            if(addRightFrame == True or addLeftFrame == True):
                                rightKeyCheckPoints = []
                                leftKeyCheckPoints = []
                                if(len(rightKeyFrames) == 48):
                                    rightKeyFrames = recalculateFrames(rightKeyFrames)
                                    print("Right Frame Cycled:", len(rightKeyFrames))
                                if(len(leftKeyFrames) == 48):
                                    leftKeyFrames = recalculateFrames(leftKeyFrames)
                                    print("Left Frame Cycled:", len(leftKeyFrames))
                                     
                    cv2.circle(image, (int(r_c_x * width), int(r_c_y * height)), 3, (255, 0, 0), 2)
                    cv2.circle(image, (int(l_c_x * width), int(l_c_y * height)), 3, (255, 255, 0), 2)

                    cv2.circle(image, (int(r_p_x * width), int(r_p_y * height)), 3, (0, 255, 0), 2)
                    cv2.circle(image, (int(l_p_x * width), int(l_p_y * height)), 3, (0, 255, 255), 2)

            else:

                if(len(rightHandPoints) != 0):
                    rightVectors = generatePointVectors(rightHandPoints, rightKeyFrames)
                    rightCheckPoints = generateCheckPoints(rightHandPoints)

                    if(addRightFrame == True):
                        rightKeyFrames.extend(rightVectors)
                        rightKeyCheckPoints.extend(rightCheckPoints)
                        
                        print("Right Frame Added: ", len(rightKeyFrames))
                        # Preprocess
                        r_set0, r_set1, r_set2, r_set3 = preprocessData(rightKeyFrames)
                        # Classify
                        rightLabel, rightProb = classification(r_set0, r_set1, r_set2, r_set3, modelRight)

                        leftLabel = ''
                        leftProb = 0

                        addRightFrame = False
                    else:
                        if(len(rightKeyCheckPoints) == 0):
                            rightKeyCheckPoints.extend(rightCheckPoints)
                        else:
                            addRightFrame, r_c_x, r_c_y, r_p_x, r_p_y = checkPreviousFrame(rightCheckPoints, rightKeyCheckPoints)

                        if(addRightFrame == True):
                            rightKeyCheckPoints = []
                            if(len(rightKeyFrames) == 48):
                                rightKeyFrames = recalculateFrames(rightKeyFrames)
                                print("Right Frame Cycled:", len(rightKeyFrames))

                    if(len(leftKeyFrames) != 0):                                
                        leftKeyFrames, leftLabel, leftProb = cleanUp(leftKeyFrames, None)

                if(len(leftHandPoints) != 0):
                    leftVectors = generatePointVectors(leftHandPoints, leftKeyFrames)
                    leftCheckPoints = generateCheckPoints(leftHandPoints)

                    if(addLeftFrame == True):
                        leftKeyFrames.extend(leftVectors)
                        leftKeyCheckPoints.extend(leftCheckPoints)

                        print("Left Frame Added: ", len(leftKeyFrames))
                        # Preprocess
                        l_set0, l_set1, l_set2, l_set3 = preprocessData(leftKeyFrames)
                        # Classify
                        leftLabel, leftProb = classification(l_set0, l_set1, l_set2, l_set3, modelLeft)

                        rightLabel = ''
                        rightProb = 0

                        addLeftFrame = False
                    else:
                        if(len(leftKeyCheckPoints) == 0):
                            leftKeyCheckPoints.extend(leftCheckPoints)
                        else:
                            addLeftFrame, l_c_x, l_c_y, l_p_x, l_p_y = checkPreviousFrame(leftCheckPoints, leftKeyCheckPoints)                        

                        if(addLeftFrame == True):
                            leftKeyCheckPoints = []
                            if(len(leftKeyFrames) == 48):
                                leftKeyFrames = recalculateFrames(leftKeyFrames)
                                print("Left Frame Cycled:", len(leftKeyFrames))
                    
                    if(len(rightKeyFrames) != 0):                                
                        rightKeyFrames, rightLabel, rightProb = cleanUp(rightKeyFrames, None)
                            
                cv2.circle(image, (int(r_c_x * width), int(r_c_y * height)), 3, (255, 0, 0), 2)
                cv2.circle(image, (int(l_c_x * width), int(l_c_y * height)), 3, (255, 255, 0), 2)

                cv2.circle(image, (int(r_p_x * width), int(r_p_y * height)), 3, (0, 255, 0), 2)
                cv2.circle(image, (int(l_p_x * width), int(l_p_y * height)), 3, (0, 255, 255), 2)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    else:
        if(len(rightKeyFrames) != 0):
            rightKeyFrames, rightLabel, rightProb = cleanUp(rightKeyFrames, modelRight)
        if(len(leftKeyFrames) != 0):
            leftKeyFrames, leftLabel, leftProb = cleanUp(leftKeyFrames, modelLeft)

    cv2.putText(image, rightLabel, (width - 200, height - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, 1)
    cv2.putText(image, str(rightProb), (10, height - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, 1)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()