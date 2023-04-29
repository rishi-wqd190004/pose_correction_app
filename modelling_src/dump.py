# visualize for angle
#cv2.putText(image, str(angle_right.astype(int)), tuple(np.multiply(RE, [640,480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA) # change camera feed from phone web-cam for future use: change this --> {[640,480]}
#cv2.putText(image, str(angle_left.astype(int)), tuple(np.multiply(LE, [640,480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
# visualize for distance
#cv2.putText(image, str(round(dist_right,2)), tuple(np.multiply(RE, [640,480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)