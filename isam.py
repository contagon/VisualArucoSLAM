import gtsam
from gtsam.symbol_shorthand import L, X
import numpy as np
import cv2
from plotting import *

np.set_printoptions(suppress=True)

idx_l = 6
idx_c = 4
idx_r = 2

# set up camera and import intrinsic parameters
paramsl = np.load('params/left_params.npz')
paramsr = np.load('params/right_params.npz')
left   = ['L', paramsl['mtx_l'], paramsl['dist_l'], cv2.VideoCapture(idx_l), gtsam.Pose3(gtsam.Rot3(paramsl['R_L']), paramsl['T_L'])]
center = ['C', paramsl['mtx_c'], paramsl['dist_c'], cv2.VideoCapture(idx_c), gtsam.Pose3()]
right  = ['R', paramsr['mtx_r'], paramsr['dist_r'], cv2.VideoCapture(idx_r), gtsam.Pose3(gtsam.Rot3(paramsr['R_R']), paramsr['T_R']).inverse()]
all_cameras = [left, center, right]

# setup aruco finder
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters_create()
size = 2.65 #inches

# setup iSAM
parameters = gtsam.ISAM2Params()
parameters.setRelinearizeThreshold(0.01)
parameters.setRelinearizeSkip(1)
isam = gtsam.ISAM2(parameters)
model = gtsam.noiseModel.Diagonal.Sigmas(np.array([.1, .1, .1, 5, 5, 5]))
i = 0
seen = set()

# Create a Factor Graph and Values to hold the new data
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

# setup plot
fig = plt.figure(figsize=plt.figaspect(1/2))
ax_2d = fig.add_subplot(1, 2, 1)
ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
plt.ion()

while(True):
    images = []
    for name, mtx, dist, cap, transform in all_cameras:
        #obtain camera image
        ret0, img = cap.read()
        images.append(img)

        #detect the markers in the image, and add them to the graph
        markerCorners, markerIds, rejecteCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=aruco_params)
        img = cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)

        if markerIds is not None:
            for corners, id in zip(markerCorners, markerIds.flatten()):
                # estimate pose for each marker
                R, T, _ = cv2.aruco.estimatePoseSingleMarkers(corners, size, mtx, dist)
                R, _ = cv2.Rodrigues(R)
                T = T.squeeze()

                # add factor to graph
                l_pose = transform.compose( gtsam.Pose3(gtsam.Rot3(R), T) )
                graph.add(gtsam.BetweenFactorPose3(X(i), L(id), l_pose, model))

                # add an estimate if we haven't seen it before
                if id not in seen:
                    seen.add(id)
                    initial_estimate.insert(L(id), gtsam.Pose3()) 


    if i == 0:
        # add in origin for first pose estimate
        initial_estimate.insert(X(0), gtsam.Pose3())
        initial_estimate.insert(X(1), gtsam.Pose3())

        # Add a prior on pose x0
        graph.push_back(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(), model))

    else:
        # Guess new location if close to old one :)
        if i != 1:
            initial_estimate.insert(X(i), estimate.atPose3(X(i-1)))

        # update isam
        isam.update(graph, initial_estimate)
        estimate = isam.calculateEstimate()
        
        # plot
        plot_3d(estimate, ax_3d, seen)
        plot_2d(estimate, isam, ax_2d, seen)
        plt.pause(.00001)

        # clear everything out for next run
        graph.resize(0)
        initial_estimate.clear()

    i += 1
    cv2.imshow('image', cv2.hconcat(images))
    if cv2.waitKey(1) == ord('q'):
        break
    # print(i)