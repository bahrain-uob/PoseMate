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

frameRate = 1
# #Display FPS
# start_time = time.time()
# x = 1 # displays the frame rate every 1 second
# counter = 0

hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.75)

model = pkl.load(open('./models/xgboost-model-dynamic-words-8-tuned', 'rb'))

# labels = {
#     "0" : "me", 
#     "1" : "you", 
#     "2" : "hello", 
#     "3" : "from",
#     "4" : "good",
#     "5" : "how",
#     "6" : "university",
#     "7" : "welcome",
#     "8" : "hope",
#     "9" : "like",
#     "10" : "new",
#     "11" : "people",
#     "12" : "technology",
#     "13" : "use",
#     "14" : "voice",
#     "15" : "create"
# }
labels = {
    "0" : "me", 
    "1" : "you", 
    "2" : "hello", 
    "3" : "good",
    "4" : "how",
    "5" : "university",
    "6" : "welcome",
    "7" : "people"
}
# labels = {
#     "0" : "you", 
#     "1" : "hello", 
#     "2" : "good",
#     "3" : "how"
# }

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fourcc =cv2.VideoWriter_fourcc('M','J','P','G')
videoWriter = cv2.VideoWriter('hello.mp4', fourcc, 30, (width,height))

rightHandFirst = False
isMultiHand = False
addFrame = True # For first frame

finalLabel = ''
finalProb = 0

keyFrames = []
keyCheckPoints = []

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

def recalculateFrames(frames):
    
    cycledFrames = []
    cycledFrames.extend(frames)
    # Current Origin

    #right hand origins
    r_base_x = cycledFrames[0]
    r_base_y = cycledFrames[1]

    r_24_dx = cycledFrames[24] - r_base_x
    r_25_dy = cycledFrames[25] - r_base_y

    r_48_dx = cycledFrames[48] - r_base_x
    r_49_dy = cycledFrames[49] - r_base_y

    r_72_dx = cycledFrames[72] - r_base_x
    r_73_dy = cycledFrames[73] - r_base_y

    #left hand origins
    l_base_x = cycledFrames[12]
    l_base_y = cycledFrames[13]

    l_36_dx = cycledFrames[36] - l_base_x
    l_37_dy = cycledFrames[37] - l_base_y

    l_60_dx = cycledFrames[60] - l_base_x
    l_61_dy = cycledFrames[61] - l_base_y

    l_84_dx = cycledFrames[84] - l_base_x
    l_85_dy = cycledFrames[85] - l_base_y

    # New Origin
    new_r_base_x = cycledFrames[24]
    new_r_base_y = cycledFrames[25]

    new_r_48_x = cycledFrames[48] - new_r_base_x
    new_r_49_y = cycledFrames[49] - new_r_base_y
    
    new_r_72_x = cycledFrames[72] - new_r_base_x
    new_r_73_y = cycledFrames[73] - new_r_base_y

    new_l_base_x = cycledFrames[36]
    new_l_base_y = cycledFrames[37]

    new_l_60_x = cycledFrames[60] - new_l_base_x
    new_l_61_y = cycledFrames[61] - new_l_base_y
    
    new_l_84_x = cycledFrames[84] - new_l_base_x
    new_l_85_y = cycledFrames[85] - new_l_base_y

    i = 24
    while(i < 96):
        if(i >= 26 and i < 36):
            cycledFrames[i] = round((cycledFrames[i] - r_24_dx), 5)
            cycledFrames[i + 1] = round((cycledFrames[i + 1] - r_25_dy), 5)
        elif(i >= 38 and i < 48):
            cycledFrames[i] = round(cycledFrames[i] - l_36_dx , 5)
            cycledFrames[i + 1] = round(cycledFrames[i + 1] - l_37_dy , 5)
            
        elif(i >= 50 and i < 60):
            r_orignial_keyframe_x = cycledFrames[i] - r_48_dx
            r_orignial_keyframe_y = cycledFrames[i + 1] - r_49_dy
            
            cycledFrames[i] = round(r_orignial_keyframe_x + new_r_48_x, 5)
            cycledFrames[i + 1] = round(r_orignial_keyframe_y + new_r_49_y, 5)
        elif(i >= 62 and i < 72):
            l_orignial_keyframe_x = cycledFrames[i] - l_60_dx
            l_orignial_keyframe_y = cycledFrames[i + 1] - l_61_dy
            
            cycledFrames[i] = round(l_orignial_keyframe_x + new_l_60_x, 5)
            cycledFrames[i + 1] = round(l_orignial_keyframe_y + new_l_61_y, 5)
        elif(i >= 74 and i < 84):
            r_orignial_keyframe_x = cycledFrames[i] - r_72_dx
            r_orignial_keyframe_y = cycledFrames[i + 1] - r_73_dy
            
            cycledFrames[i] = round(r_orignial_keyframe_x + new_r_72_x, 5)
            cycledFrames[i + 1] = round(r_orignial_keyframe_y + new_r_73_y, 5)
        elif(i >= 86 and i < 96):
            l_orignial_keyframe_x = cycledFrames[i] - l_84_dx
            l_orignial_keyframe_y = cycledFrames[i + 1] - l_85_dy
            
            cycledFrames[i] = round(l_orignial_keyframe_x + new_l_84_x, 5)
            cycledFrames[i + 1] = round(l_orignial_keyframe_y + new_l_85_y, 5)
        i = i + 2
    # 0 - 23
    # 24 - 47
    # 48 - 71
    # 72 - 95
    # Cycle out
    cycledFrames = cycledFrames[24:]
    return cycledFrames

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

    if(r_dx >= d_threshold or r_dy >= d_threshold or l_dx >= d_threshold or l_dy >= d_threshold):
        # print("Thresold crossed.")

        return True, r_current_dx, r_current_dy, l_current_dx, l_current_dy, r_prev_dx, r_prev_dy, l_prev_dx, l_prev_dy
    else:
        return False, r_current_dx, r_current_dy, l_current_dx, l_current_dy, r_prev_dx, r_prev_dy, l_prev_dx, l_prev_dy

def preprocessData(frames):
    
    dataToProcess = []
    dataToProcess.extend(frames)
    if(len(dataToProcess) != 96):
        if(len(dataToProcess) == 24):
            dataToProcess.extend(null_24)
            dataToProcess.extend(null_24)
            dataToProcess.extend(null_24)
        elif(len(dataToProcess) == 48):
            dataToProcess.extend(null_24)
            dataToProcess.extend(null_24)
        elif(len(dataToProcess) == 72):
            dataToProcess.extend(null_24)
        else:
            print("Error in preprocessData. Length of dataToProcess: ", len(dataToProcess))

    group_0 = []
    group_0.extend(dataToProcess[:24])
    group_0.extend(null_24)
    group_0.extend(null_24)
    group_0.extend(null_24)

    group_1 = []
    group_1.extend(dataToProcess[:48])
    group_1.extend(null_24)
    group_1.extend(null_24)

    group_2 = []
    group_2.extend(dataToProcess[:72])
    group_2.extend(null_24)

    group_3 = []
    group_3.extend(dataToProcess[:96])

    # df = pd.DataFrame(finalVectors)
    # df_T = df.T
    # df_T.to_csv('output.csv', index=False, header=None)
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

def classification(inputData_0, inputData_1, inputData_2, inputData_3):
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

    # print(out_label_0, max_prob_0, out_label_1, max_prob_1, out_label_2, max_prob_2, out_label_3, max_prob_3)
    print(out_label_0, out_label_1, out_label_2, out_label_3)
    
    label = ''
    prob = 0
    if(max_prob_0 > 0.95):
        label = out_label_0
        prob = max_prob_0

    if(max_prob_1 > 0.90):
        label = out_label_1
        prob = max_prob_1
    
    if(max_prob_2 > 0.85):
        label = out_label_2
        prob = max_prob_2
    
    if(max_prob_3 > 0.80):
        label = out_label_3
        prob = max_prob_3

    return label, prob

def cleanUp(frames):
    keyFrames2 = []
    keyFrames2.extend(frames)
    # 4 Frames
    if(len(keyFrames2) == 96):
        cycledFrames = []
        cycledFrames.extend(keyFrames2)
        #right hand origins
        r_base_x = cycledFrames[0]
        r_base_y = cycledFrames[1]

        r_24_dx = cycledFrames[24] - r_base_x
        r_25_dy = cycledFrames[25] - r_base_y

        r_48_dx = cycledFrames[48] - r_base_x
        r_49_dy = cycledFrames[49] - r_base_y

        r_72_dx = cycledFrames[72] - r_base_x
        r_73_dy = cycledFrames[73] - r_base_y

        #left hand origins
        l_base_x = cycledFrames[12]
        l_base_y = cycledFrames[13]

        l_36_dx = cycledFrames[36] - l_base_x
        l_37_dy = cycledFrames[37] - l_base_y

        l_60_dx = cycledFrames[60] - l_base_x
        l_61_dy = cycledFrames[61] - l_base_y

        l_84_dx = cycledFrames[84] - l_base_x
        l_85_dy = cycledFrames[85] - l_base_y

        # New Origin
        new_r_base_x = cycledFrames[24]
        new_r_base_y = cycledFrames[25]

        new_r_48_x = cycledFrames[48] - new_r_base_x
        new_r_49_y = cycledFrames[49] - new_r_base_y
        
        new_r_72_x = cycledFrames[72] - new_r_base_x
        new_r_73_y = cycledFrames[73] - new_r_base_y

        new_l_base_x = cycledFrames[36]
        new_l_base_y = cycledFrames[37]

        new_l_60_x = cycledFrames[60] - new_l_base_x
        new_l_61_y = cycledFrames[61] - new_l_base_y
        
        new_l_84_x = cycledFrames[84] - new_l_base_x
        new_l_85_y = cycledFrames[85] - new_l_base_y

        i = 24
        while(i < 96):
            if(i >= 26 and i < 36):
                cycledFrames[i] = round((cycledFrames[i] - r_24_dx), 5)
                cycledFrames[i + 1] = round((cycledFrames[i + 1] - r_25_dy), 5)
            elif(i >= 38 and i < 48):
                cycledFrames[i] = round(cycledFrames[i] - l_36_dx , 5)
                cycledFrames[i + 1] = round(cycledFrames[i + 1] - l_37_dy , 5)
                
            elif(i >= 50 and i < 60):
                r_orignial_keyframe_x = cycledFrames[i] - r_48_dx
                r_orignial_keyframe_y = cycledFrames[i + 1] - r_49_dy
                
                cycledFrames[i] = round(r_orignial_keyframe_x + new_r_48_x, 5)
                cycledFrames[i + 1] = round(r_orignial_keyframe_y + new_r_49_y, 5)
            elif(i >= 62 and i < 72):
                l_orignial_keyframe_x = cycledFrames[i] - l_60_dx
                l_orignial_keyframe_y = cycledFrames[i + 1] - l_61_dy
                
                cycledFrames[i] = round(l_orignial_keyframe_x + new_l_60_x, 5)
                cycledFrames[i + 1] = round(l_orignial_keyframe_y + new_l_61_y, 5)
            elif(i >= 74 and i < 84):
                r_orignial_keyframe_x = cycledFrames[i] - r_72_dx
                r_orignial_keyframe_y = cycledFrames[i + 1] - r_73_dy
                
                cycledFrames[i] = round(r_orignial_keyframe_x + new_r_72_x, 5)
                cycledFrames[i + 1] = round(r_orignial_keyframe_y + new_r_73_y, 5)
            elif(i >= 86 and i < 96):
                l_orignial_keyframe_x = cycledFrames[i] - l_84_dx
                l_orignial_keyframe_y = cycledFrames[i + 1] - l_85_dy
                
                cycledFrames[i] = round(l_orignial_keyframe_x + new_l_84_x, 5)
                cycledFrames[i + 1] = round(l_orignial_keyframe_y + new_l_85_y, 5)
            i = i + 2
        # Cycle out
        cycledFrames = cycledFrames[24:]
        keyFrames = cycledFrames

        set_0, set_1, set_2, set_3 = preprocessData(keyFrames)
        classification(set_0, set_1, set_2, set_3)

        cycledFrames = []
        cycledFrames.extend(keyFrames)

        #right hand origins
        r_base_x = cycledFrames[0]
        r_base_y = cycledFrames[1]

        r_24_dx = cycledFrames[24] - r_base_x
        r_25_dy = cycledFrames[25] - r_base_y

        r_48_dx = cycledFrames[48] - r_base_x
        r_49_dy = cycledFrames[49] - r_base_y

        #left hand origins
        l_base_x = cycledFrames[12]
        l_base_y = cycledFrames[13]

        l_36_dx = cycledFrames[36] - l_base_x
        l_37_dy = cycledFrames[37] - l_base_y

        l_60_dx = cycledFrames[60] - l_base_x
        l_61_dy = cycledFrames[61] - l_base_y

        # New Origin
        new_r_base_x = cycledFrames[24]
        new_r_base_y = cycledFrames[25]

        new_r_48_x = cycledFrames[48] - new_r_base_x
        new_r_49_y = cycledFrames[49] - new_r_base_y

        new_l_base_x = cycledFrames[36]
        new_l_base_y = cycledFrames[37]

        new_l_60_x = cycledFrames[60] - new_l_base_x
        new_l_61_y = cycledFrames[61] - new_l_base_y

        i = 24
        while(i < 72):
            if(i >= 26 and i < 36):
                cycledFrames[i] = round((cycledFrames[i] - r_24_dx), 5)
                cycledFrames[i + 1] = round((cycledFrames[i + 1] - r_25_dy), 5)
            elif(i >= 38 and i < 48):
                cycledFrames[i] = round(cycledFrames[i] - l_36_dx , 5)
                cycledFrames[i + 1] = round(cycledFrames[i + 1] - l_37_dy , 5)
                
            elif(i >= 50 and i < 60):
                r_orignial_keyframe_x = cycledFrames[i] - r_48_dx
                r_orignial_keyframe_y = cycledFrames[i + 1] - r_49_dy
                
                cycledFrames[i] = round(r_orignial_keyframe_x + new_r_48_x, 5)
                cycledFrames[i + 1] = round(r_orignial_keyframe_y + new_r_49_y, 5)
            elif(i >= 62 and i < 72):
                l_orignial_keyframe_x = cycledFrames[i] - l_60_dx
                l_orignial_keyframe_y = cycledFrames[i + 1] - l_61_dy
                
                cycledFrames[i] = round(l_orignial_keyframe_x + new_l_60_x, 5)
                cycledFrames[i + 1] = round(l_orignial_keyframe_y + new_l_61_y, 5)
            i = i + 2

        # Cycle out
        cycledFrames = cycledFrames[24:]
        keyFrames = cycledFrames

        set_0, set_1, set_2, set_3 = preprocessData(keyFrames)
        classification(set_0, set_1, set_2, set_3)

        cycledFrames = []
        cycledFrames.extend(keyFrames)

        #right hand origins
        r_base_x = cycledFrames[0]
        r_base_y = cycledFrames[1]

        r_24_dx = cycledFrames[24] - r_base_x
        r_25_dy = cycledFrames[25] - r_base_y

        #left hand origins
        l_base_x = cycledFrames[12]
        l_base_y = cycledFrames[13]

        l_36_dx = cycledFrames[36] - l_base_x
        l_37_dy = cycledFrames[37] - l_base_y

        # New Origin
        new_r_base_x = cycledFrames[24]
        new_r_base_y = cycledFrames[25]

        new_l_base_x = cycledFrames[36]
        new_l_base_y = cycledFrames[37]

        i = 24
        while(i < 48):
            if(i >= 26 and i < 36):
                cycledFrames[i] = round((cycledFrames[i] - r_24_dx), 5)
                cycledFrames[i + 1] = round((cycledFrames[i + 1] - r_25_dy), 5)
            elif(i >= 38 and i < 48):
                cycledFrames[i] = round(cycledFrames[i] - l_36_dx , 5)
                cycledFrames[i + 1] = round(cycledFrames[i + 1] - l_37_dy , 5)
            i = i + 2

        # Cycle out
        cycledFrames = cycledFrames[24:]
        keyFrames = cycledFrames

        set_0, set_1, set_2, set_3 = preprocessData(keyFrames)
        classification(set_0, set_1, set_2, set_3)

    # 3 Frames
    elif(len(keyFrames2) == 72):
        cycledFrames = []
        cycledFrames.extend(keyFrames2)

        #right hand origins
        r_base_x = cycledFrames[0]
        r_base_y = cycledFrames[1]

        r_24_dx = cycledFrames[24] - r_base_x
        r_25_dy = cycledFrames[25] - r_base_y

        r_48_dx = cycledFrames[48] - r_base_x
        r_49_dy = cycledFrames[49] - r_base_y

        #left hand origins
        l_base_x = cycledFrames[12]
        l_base_y = cycledFrames[13]

        l_36_dx = cycledFrames[36] - l_base_x
        l_37_dy = cycledFrames[37] - l_base_y

        l_60_dx = cycledFrames[60] - l_base_x
        l_61_dy = cycledFrames[61] - l_base_y

        # New Origin
        new_r_base_x = cycledFrames[24]
        new_r_base_y = cycledFrames[25]

        new_r_48_x = cycledFrames[48] - new_r_base_x
        new_r_49_y = cycledFrames[49] - new_r_base_y

        new_l_base_x = cycledFrames[36]
        new_l_base_y = cycledFrames[37]

        new_l_60_x = cycledFrames[60] - new_l_base_x
        new_l_61_y = cycledFrames[61] - new_l_base_y

        i = 24
        while(i < 72):
            if(i >= 26 and i < 36):
                cycledFrames[i] = round((cycledFrames[i] - r_24_dx), 5)
                cycledFrames[i + 1] = round((cycledFrames[i + 1] - r_25_dy), 5)
            elif(i >= 38 and i < 48):
                cycledFrames[i] = round(cycledFrames[i] - l_36_dx , 5)
                cycledFrames[i + 1] = round(cycledFrames[i + 1] - l_37_dy , 5)
                
            elif(i >= 50 and i < 60):
                r_orignial_keyframe_x = cycledFrames[i] - r_48_dx
                r_orignial_keyframe_y = cycledFrames[i + 1] - r_49_dy
                
                cycledFrames[i] = round(r_orignial_keyframe_x + new_r_48_x, 5)
                cycledFrames[i + 1] = round(r_orignial_keyframe_y + new_r_49_y, 5)
            elif(i >= 62 and i < 72):
                l_orignial_keyframe_x = cycledFrames[i] - l_60_dx
                l_orignial_keyframe_y = cycledFrames[i + 1] - l_61_dy
                
                cycledFrames[i] = round(l_orignial_keyframe_x + new_l_60_x, 5)
                cycledFrames[i + 1] = round(l_orignial_keyframe_y + new_l_61_y, 5)
            i = i + 2

        # Cycle out
        cycledFrames = cycledFrames[24:]
        keyFrames = cycledFrames

        set_0, set_1, set_2, set_3 = preprocessData(keyFrames)
        classification(set_0, set_1, set_2, set_3)

        cycledFrames = []
        cycledFrames.extend(keyFrames)

        #right hand origins
        r_base_x = cycledFrames[0]
        r_base_y = cycledFrames[1]

        r_24_dx = cycledFrames[24] - r_base_x
        r_25_dy = cycledFrames[25] - r_base_y

        #left hand origins
        l_base_x = cycledFrames[12]
        l_base_y = cycledFrames[13]

        l_36_dx = cycledFrames[36] - l_base_x
        l_37_dy = cycledFrames[37] - l_base_y

        # New Origin
        new_r_base_x = cycledFrames[24]
        new_r_base_y = cycledFrames[25]

        new_l_base_x = cycledFrames[36]
        new_l_base_y = cycledFrames[37]

        i = 24
        while(i < 48):
            if(i >= 26 and i < 36):
                cycledFrames[i] = round((cycledFrames[i] - r_24_dx), 5)
                cycledFrames[i + 1] = round((cycledFrames[i + 1] - r_25_dy), 5)
            elif(i >= 38 and i < 48):
                cycledFrames[i] = round(cycledFrames[i] - l_36_dx , 5)
                cycledFrames[i + 1] = round(cycledFrames[i + 1] - l_37_dy , 5)
            i = i + 2

        # Cycle out
        cycledFrames = cycledFrames[24:]
        keyFrames = cycledFrames

        set_0, set_1, set_2, set_3 = preprocessData(keyFrames)
        classification(set_0, set_1, set_2, set_3)

    # 2 Frames
    elif(len(keyFrames2) == 48):
        cycledFrames = []
        cycledFrames.extend(keyFrames2)

        #right hand origins
        r_base_x = cycledFrames[0]
        r_base_y = cycledFrames[1]

        r_24_dx = cycledFrames[24] - r_base_x
        r_25_dy = cycledFrames[25] - r_base_y

        #left hand origins
        l_base_x = cycledFrames[12]
        l_base_y = cycledFrames[13]

        l_36_dx = cycledFrames[36] - l_base_x
        l_37_dy = cycledFrames[37] - l_base_y

        # New Origin
        new_r_base_x = cycledFrames[24]
        new_r_base_y = cycledFrames[25]

        new_l_base_x = cycledFrames[36]
        new_l_base_y = cycledFrames[37]

        i = 24
        while(i < 48):
            if(i >= 26 and i < 36):
                cycledFrames[i] = round((cycledFrames[i] - r_24_dx), 5)
                cycledFrames[i + 1] = round((cycledFrames[i + 1] - r_25_dy), 5)
            elif(i >= 38 and i < 48):
                cycledFrames[i] = round(cycledFrames[i] - l_36_dx , 5)
                cycledFrames[i + 1] = round(cycledFrames[i + 1] - l_37_dy , 5)
            i = i + 2

        # Cycle out
        cycledFrames = cycledFrames[24:]
        keyFrames = cycledFrames

        set_0, set_1, set_2, set_3 = preprocessData(keyFrames)
        classification(set_0, set_1, set_2, set_3)

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

                    if(addFrame == True):
                        keyFrames.extend(finalVectors)
                        keyCheckPoints.extend(checkPoints)
                        # print("Frame Added. Length of Keyframes: ", len(keyFrames))
                        set_0, set_1, set_2, set_3 = preprocessData(keyFrames)
                        finalLabel, finalProb = classification(set_0, set_1, set_2, set_3)
                        # if temp_label:
                        #     finalLabel = temp_label
                        # if temp_prob != 0:
                        #     finalProb = temp_prob
                        addFrame = False
                    else:
                        addFrame, r_c_x, r_c_y, l_c_x, l_c_y, r_p_x, r_p_y, l_p_x, l_p_y = checkPreviousFrame(checkPoints, keyCheckPoints)
                        if(addFrame == True):
                            keyCheckPoints = []
                            if(len(keyFrames) == 96):
                                keyFrames = recalculateFrames(keyFrames)
                                # print("Frame Cycled. Length of Keyframes: ", len(keyFrames))
                                
                    cv2.circle(image, (int(r_c_x * width), int(r_c_y * height)), 3, (255, 0, 0), 2)
                    cv2.circle(image, (int(l_c_x * width), int(l_c_y * height)), 3, (255, 255, 0), 2)

                    cv2.circle(image, (int(r_p_x * width), int(r_p_y * height)), 3, (0, 255, 0), 2)
                    cv2.circle(image, (int(l_p_x * width), int(l_p_y * height)), 3, (0, 255, 255), 2)
                        
            else:
                finalVectors = generatePointVectors(rightHandPoints, leftHandPoints, keyFrames)
                checkPoints = generateCheckPoints(rightHandPoints, leftHandPoints)

                if(addFrame == True):
                    keyFrames.extend(finalVectors)
                    keyCheckPoints.extend(checkPoints)
                    # print("Frame Added. Length of Keyframes: ", len(keyFrames))
                    set_0, set_1, set_2, set_3 = preprocessData(keyFrames)
                    finalLabel, finalProb = classification(set_0, set_1, set_2, set_3)
                    # if temp_label:
                    #     finalLabel = temp_label
                    # if temp_prob != 0:
                    #     finalProb = temp_prob
                    addFrame = False
                else:
                    addFrame, r_c_x, r_c_y, l_c_x, l_c_y, r_p_x, r_p_y, l_p_x, l_p_y = checkPreviousFrame(checkPoints, keyCheckPoints)
                    if(addFrame == True):
                        keyCheckPoints = []
                        if(len(keyFrames) == 96):
                            keyFrames = recalculateFrames(keyFrames)
                            # print("Frame Cycled. Length of Keyframes: ", len(keyFrames))
                
                cv2.circle(image, (int(r_c_x * width), int(r_c_y * height)), 3, (255, 0, 0), 2)
                cv2.circle(image, (int(l_c_x * width), int(l_c_y * height)), 3, (255, 255, 0), 2)

                cv2.circle(image, (int(r_p_x * width), int(r_p_y * height)), 3, (0, 255, 0), 2)
                cv2.circle(image, (int(l_p_x * width), int(l_p_y * height)), 3, (0, 255, 255), 2)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    else:
        cleanUp(keyFrames)
        keyFrames = []

    cv2.putText(image, finalLabel, (width - 200, height - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, 1)
    cv2.putText(image, str(finalProb), (10, height - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, 1)
    
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
    videoWriter.write(image)
    # Decrease FPS
    # time.sleep(1/frameRate)
    # # Calculate FPS
    # counter+=1
    # if (time.time() - start_time) > x :
    #     print("FPS: ", counter / (time.time() - start_time))
    #     counter = 0
    #     start_time = time.time()

hands.close()
videoWriter.release()
cap.release()