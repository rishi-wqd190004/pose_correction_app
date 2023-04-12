import cv2
import mediapipe as mp
import time
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
            dist_nose_mid_pt = calcualte_distance(nose,mid_RS_LS)
            # area on left and right
            area_right= calculate_quad_area(RE,RS,nose,mid_RS_LS)
            alrea_left = calculate_quad_area(LE,LS,nose,mid_RS_LS)
            print(area_right, alrea_left)

            # visualize for angle
            #cv2.putText(image, str(angle_right.astype(int)), tuple(np.multiply(RE, [640,480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA) # change camera feed from phone web-cam for future use: change this --> {[640,480]}
            #cv2.putText(image, str(angle_left.astype(int)), tuple(np.multiply(LE, [640,480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
            # visualize for distance
            #cv2.putText(image, str(round(dist_right,2)), tuple(np.multiply(RE, [640,480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)

            # adding different checks
            # Case1: checking if left and right angle is almost similar
            tolerance_angle = 4.0 
            if abs(int(angle_right) - int(angle_left)) <= tolerance_angle:
                cv2.putText(image, str('Posture is correct by angle'),(200,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, str('Please correct your posture by angle'),(200,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

            # Case2: checking if distance between RS,RE is equal to LS,LE
            tolerance_dist = 0.01
            if abs(dist_right - dist_left) <= tolerance_dist:
                cv2.putText(image, str('Posture is correct by distance'),(400,400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, str('Please correct your posture by distance'),(400,400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

            # Case3: Nose to center of RS,LS
            if mid_RS_LS == 0: # check the way this works don't hardcode the tolerance
                tolerance_nose_mid = 0.25
                if abs(dist_nose_mid_pt) >= tolerance_nose_mid:
                    print(dist_nose_mid_pt)
                    cv2.putText(image, str('Please correct your posture by nose distance'),(600,600), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
                    cv2.line(image, tuple(nose), tuple(mid_RS_LS), (0,255,0), thickness=2)
            # Case4: Area under Nose, RE,RS and center of RS,LS and other side
            if area_right == alrea_left:
                cv2.putText(image, str('Posture is correct by quad angle'),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, str('Please correct your posture by quad angle'),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
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