import cv2

cap = cv2.VideoCapture(0)
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters_create()
size = 2.65 #inches

while True: 
    #obtain camera image
    ret0, img = cap.read()                                                                                                                          

    #detect the markers in the image
    markerCorners, markerIds, rejecteCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
    img = cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)

    cv2.imshow('marker', img)
    if cv2.waitKey(20) == ord('q'):
        break

    # if markerIds is not None:
    #     for corners, id in zip(markerCorners, markerIds.flatten()):
    #         # estimate pose for each marker
    #         # r, t, _ = cv2.aruco.estimatePoseSingleMarkers(corners, size, camera_matrix, dist_coeffs)
