import cv2
import mediapipe as mp
import numpy as np
import math

# function to calculate angle
def calculate_angle(a,b,c):
    a = np.array(a) # first
    b = np.array(b) # second
    c = np.array(c) # third

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# function to calculate distance
def calcualte_distance(a,b):
    # euclidean distance between two points
    dist = math.dist(a,b)
    return dist

# function to calculate mid point
def calculate_mid_point(a,b):
    x = (a[0] + b[0])/2.0
    y = (a[1] + b[1])/2.0
    c = [x,y]
    return c

# function to calculate quadilateral area
def calculate_quad_area(a,b,c,d):
    first = a[0]*b[1] + b[0]*c[1] + c[0]*d[1] + d[0]*a[1]
    second = a[1]*b[0] + b[1]*c[0] + c[1]*d[0] + d[1]*a[0]
    area = 0.5 * abs(first - second)
    return abs(area)

# functions for cases of pose detection
def case1(angle_right,angle_left,tol_angle):
    global case1_res
    case1_res = ''
    # Case1: checking if left and right angle is almost similar
    tolerance_angle = tol_angle
    if abs(int(angle_right) - int(angle_left)) <= tolerance_angle:
        cv2.putText(image, str('Case1: Posture is correct by angle'),(20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
        case1_res = 1
    else:
        # check how much left or right side angle differs
        diff = angle_left - angle_right
        if diff > tol_angle:
            cv2.putText(image, ("Tilt your head: " + str(diff.astype(int)) + " degrees to right"), (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 216, 173), 2, cv2.LINE_AA) # change camera feed from phone web-cam for future use: change this --> {[640,480]}
            case1_res = diff.astype(int)
        else:
            cv2.putText(image, ("Tilt your head: " + str(abs(diff.astype(int))) + " degrees to left"), (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 216, 173), 2, cv2.LINE_AA) # change camera feed from phone web-cam for future use: change this --> {[640,480]}
            case1_res = diff.astype(int)
        cv2.putText(image, str('Case1: Please correct your posture by angle'),(20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
    #print('case1 results: ',case1_res)
    # return case1_res

def case2(dist_right,dist_left, tol_dist):
    # Case2: checking if distance between RS,RE is equal to LS,LE
    global case2_res
    case2_res = ''
    if abs(dist_right - dist_left) <= tol_dist:
        cv2.putText(image, str('Case2: Posture is correct by distance'),(400,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
        case2_res = 1
    else:
        diff_dist = dist_right - dist_left
        if dist_right > dist_left:
            cv2.putText(image, ("Turn your head: " + str(round(abs(diff_dist),2)) + " mm to right"), (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 216, 173), 2, cv2.LINE_AA)
            case2_res = round(abs(diff_dist),2)
        else:
            cv2.putText(image, ("Turn your head: " + str(round(abs(diff_dist),2)) + " mm to left"), (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 216, 173), 2, cv2.LINE_AA)
            case2_res = round(abs(diff_dist),2)
        cv2.putText(image, str('Case2: Please correct your posture by distance'),(400,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

def case3(dist_nose_RS,dist_nose_LS,tol_nose_dist):
    # Case3: Distance between nose to RS,LS
    global case3_res
    case3_res = ''
    if abs(dist_nose_RS - dist_nose_LS) <= tol_nose_dist:
        cv2.putText(image, str('Case3: Posture correct by nose and shoulder distance'),(800,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
        case3_res = 1
    else:
        diff_nose_dist = dist_nose_RS - dist_nose_LS
        if dist_nose_RS > dist_nose_LS:
            cv2.putText(image, ("Raise your head: " + str(round(abs(diff_nose_dist),2)) + " mm to right"), (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 216, 173), 2, cv2.LINE_AA)
            case3_res = round(abs(diff_nose_dist), 2)
        else:
            cv2.putText(image, ("Move your head: " + str(round(abs(diff_nose_dist),2)) + " mm to left"), (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 216, 173), 2, cv2.LINE_AA)
            case3_res = round(abs(diff_nose_dist), 2)
        cv2.putText(image, str('Case3: Please correct your posture by nose and shoulder distance'),(800,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

def case4(area_right, area_left,tol_diff_area):
    # Case4: Area under Nose, RE,RS and center of RS,LS and other side
    global case4_res
    case4_res = ''
    if math.isclose(area_right, area_left, rel_tol=tol_diff_area):
        cv2.putText(image, str('Case4: Posture is correct by quad area'),(20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
        case4_res = 1
    else:
        diff_area = area_right - area_left
        if area_right > area_left:
            cv2.putText(image, ("Straighten your head: " + str(round(abs(diff_area),2)) + " mm to right"), (40,350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 216, 173), 2, cv2.LINE_AA)
            case4_res = round(abs(diff_area), 2)
        else:
            cv2.putText(image, ("Straighten your head: " + str(round(abs(diff_area),2)) + " mm to right"), (40,350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 216, 173), 2, cv2.LINE_AA)
            case4_res = round(abs(diff_area), 2)
        cv2.putText(image, str('Case4: Please correct your posture by quad area'),(20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

def caseChecker():
    '''
    Check for cases to be met like minimum 3 cases to be correct mainly 1,3,4
    '''
    case1(angle_right,angle_left,tol_angle=4.0)
    case2(dist_right,dist_left,tol_dist=0.03)
    case3(dist_nose_RS,dist_nose_LS,tol_nose_dist=0.05)
    case4(area_right, alrea_left,tol_diff_area=0.2)
    if case1_res == case3_res and case4_res == case3_res and case1_res == case4_res:
        print('Cases passed')
    else:
        print('Case needs to be corrected')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# for web cam input
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        # recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # make detection
        results = pose.process(image)

        # drawing pose annotation on image and recolor to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            # extracting coordinates
            LS = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y] # left shoulder cordinates
            LE = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y] # left ear cordinates
            RS = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y] # right shoulder cordinates
            RE = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y] # right ear cordinates
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y] # nose cordinates

            # calculate angle
            angle_right = calculate_angle(RS,RE,LE)
            angle_left = calculate_angle(LS,LE,RE)
            # calculate distance
            dist_right = calcualte_distance(RS,RE)
            dist_left = calcualte_distance(LS,LE)
            # mid distance of RS and LS
            mid_RS_LS = calculate_mid_point(RS,LS)
            dist_nose_RS = calcualte_distance(nose,RS)
            dist_nose_LS = calcualte_distance(nose,LS)
            # area on left and right
            area_right= calculate_quad_area(RE,RS,nose,mid_RS_LS)
            alrea_left = calculate_quad_area(LE,LS,nose,mid_RS_LS)

            # adding different checks
            caseChecker()
        except:
            pass

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())
        # glip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Pose', cv2.flip(image,1))
        cv2.imshow('MediaPipe Neck Pose',image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()