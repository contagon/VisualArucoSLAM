import cv2
import gtsam
import numpy as np

# load images to test with
img_l = cv2.imread("imagesR/tag_L.jpg")
img_c = cv2.imread("imagesR/tag_C.jpg")
img_r = cv2.imread("imagesR/tag_R.jpg")

# set up camera and import intrinsic parameters
paramsl = np.load('params/left_params.npz')
paramsr = np.load('params/right_params.npz')
left   = ["L", img_l, paramsl['mtx_l'], paramsl['dist_l'], gtsam.Pose3(gtsam.Rot3(paramsl['R_L']), paramsl['T_L'])]
center = ["C", img_c, paramsl['mtx_c'], paramsl['dist_c'], gtsam.Pose3()]
right  = ["R", img_r, paramsr['mtx_r'], paramsr['dist_r'], gtsam.Pose3(gtsam.Rot3(paramsr['R_R']), paramsr['T_R'])]
all_cameras = [left, center, right]

# setup aruco finder
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters_create()
size = 2.6 #inches
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 5
aruco_params.cornerRefinementMaxIterations = 30
aruco_params.cornerRefinementMinAccuracy = 1e-5

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

# print(poses)
test = poses[id]['R'].compose( poses[id]['C'].inverse() )
print("From tags\n", test)
print("From calibration\n", right[-1])

np.savez('params/tag_right.npz', T_R=test.translation(), R_R=test.rotation().matrix())