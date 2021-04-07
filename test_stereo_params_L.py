import cv2
import gtsam
import numpy as np

# load images to test with
img_l = cv2.imread("imagesL/tag_L.jpg")
img_c = cv2.imread("imagesL/tag_C.jpg")
img_r = cv2.imread("imagesL/tag_R.jpg")

# set up camera and import intrinsic parameters
paramsl = np.load('params/left_params.npz')
paramsr = np.load('params/right_params.npz')
left   = ["L", img_l, paramsl['mtx_l'], paramsl['dist_l'], gtsam.Pose3(gtsam.Rot3(paramsl['R_L']), paramsl['T_L'])]
center = ["C", img_c, paramsl['mtx_c'], paramsl['dist_c'], gtsam.Pose3()]
right  = ["R", img_r, paramsr['mtx_r'], paramsr['dist_r'], gtsam.Pose3(gtsam.Rot3(paramsr['R_R']), paramsr['T_R']).inverse()]
all_cameras = [left, center, right]

# setup aruco finder
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters_create()
size = 2.6 #inches

poses = dict()
for name, img, mtx, dist, transform in all_cameras:
    #detect the markers in the image, and add them to the graph
    markerCorners, markerIds, rejecteCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=aruco_params)
    img = cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)

    if markerIds is not None:
        for corners, id in zip(markerCorners, markerIds.flatten()):
            if id not in poses:
                poses[id] = dict()

            # estimate pose for each marker
            R, T, _ = cv2.aruco.estimatePoseSingleMarkers(corners, size, mtx, dist)
            R, _ = cv2.Rodrigues(R)
            T = T.squeeze()

            poses[id][name] = gtsam.Pose3(gtsam.Rot3(R), T)

test = poses[id]['C'].compose( poses[id]['L'].inverse() )
print("From tags\n", test)
print("From calibration\n", left[-1])

# test = left[-1].compose( poses[id]['L'] )
# print("From center camera\n", poses[id]['C'])
# print("From calibration\n", test)

np.savez('params/tag_left.npz', T=test.translation(), R=test.rotation())