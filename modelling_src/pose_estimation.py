import cv2
import mediapipe as mp
import time
import numpy as np

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
def calcualte_distance():
    pass

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import shutup; shutup.please()

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

            # calculate angle
            angle_right = calculate_angle(RS,RE,LE)
            angle_left = calculate_angle(LS,LE,RE)
            print(angle_right, angle_left)

            # visualize
            cv2.putText(image, str(angle_right.astype(int)), tuple(np.multiply(RE, [640,480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA) # change camera feed from phone web-cam for future use: change this --> {[640,480]}
            print(angle_right, angle_left)
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