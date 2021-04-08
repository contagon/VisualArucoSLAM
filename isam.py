import gtsam
from gtsam.symbol_shorthand import L, X
import numpy as np
import cv2
from plotting import *
import subprocess
import pickle

np.set_printoptions(suppress=True)

idx_l = 6
idx_c = 4
idx_r = 2
update_every = 5
framerate = 20

# noise for each of them
noise_c = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.pi/8, np.pi/8, np.pi/8, 5, 5, 5]))
noise_l = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.pi/4, np.pi/4, np.pi/4, 10, 10, 10]))
noise_r = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.pi/2, np.pi/2, np.pi/2, 15, 15, 15]))

# set up camera and import intrinsic parameters
paramsl = np.load('params/left_params.npz')
paramsr = np.load('params/right_params.npz')
tagr = np.load('params/tag_right.npz')
tagl = np.load('params/tag_left.npz')
left   = [idx_l, 'L', paramsl['mtx_l'], paramsl['dist_l'], cv2.VideoCapture(idx_l), noise_l, gtsam.Pose3(gtsam.Rot3(tagl['R_L']), tagl['T_L'])]
center = [idx_c, 'C', paramsl['mtx_c'], paramsl['dist_c'], cv2.VideoCapture(idx_c), noise_c, gtsam.Pose3()]
right  = [idx_r, 'R', paramsr['mtx_r'], paramsr['dist_r'], cv2.VideoCapture(idx_r), noise_r, gtsam.Pose3(gtsam.Rot3(paramsr['R_R']), paramsr['T_R']).inverse()]
all_cameras = [left, center, right]

# set all exposures
for idx, name, mtx, dist, cap, noise, transform in all_cameras:
    subprocess.check_call(f"v4l2-ctl -d /dev/video{idx} -c exposure_auto=1 -c exposure_auto_priority=0", shell=True)
    cap.set(cv2.CAP_PROP_EXPOSURE, 100)
    print(f"Camera {name} exposure is set to {cap.get(cv2.CAP_PROP_EXPOSURE)}")

# setup aruco finder
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters_create()
size = 2.6 #inches
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 10
aruco_params.cornerRefinementMaxIterations = 50
aruco_params.cornerRefinementMinAccuracy = 1e-5

# setup iSAM
parameters = gtsam.ISAM2Params()
parameters.setRelinearizeThreshold(0.001)
parameters.setRelinearizeSkip(1)
# parameters.setFactorization("QR")
isam = gtsam.ISAM2(parameters)
model = gtsam.noiseModel.Diagonal.Sigmas(np.array([.1, .1, .1, 5, 5, 5]))
i = 0
seen = set()
seen_iter = set()

# Create a Factor Graph and Values to hold the new data
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

# setup plot
fig = plt.figure(figsize=(10,4))
ax_2d = fig.add_subplot(1, 1, 1)
# ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
plt.ion()
while(True):
    images = []
    aruco_pose = dict()
    pose = dict()
    for idx, name, mtx, dist, cap, noise, transform in all_cameras:
        #obtain camera image
        ret0, img = cap.read()
        images.append(img)
        cap.get(cv2.CAP_PROP_EXPOSURE)

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
                graph.add(gtsam.BetweenFactorPose3(X(i), L(id), l_pose, noise))

                # add an estimate if we haven't seen it before
                if id not in seen.union(seen_iter):
                    initial_estimate.insert(L(id), l_pose) 
                seen_iter.add(id)

                # if id == 1:
                #     aruco_pose[name] = gtsam.Pose3(gtsam.Rot3(R), T)
                #     pose[name] = l_pose
    # print(pose)
    # print("Needs to be to work", aruco_pose['R'].compose( aruco_pose['C'].inverse() ))
    # print()

    #  first iteration
    if i == 0:
        # add in origin for first pose estimate
        initial_estimate.insert(X(0), gtsam.Pose3())
        initial_estimate.insert(X(1), gtsam.Pose3())

        # Add a prior on pose x0
        graph.push_back(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(), model))
        last_i = 1

    # after things are running
    else:
        print(i, seen_iter)
        # if we didn't see anything, pretend it didn't happen
        if len(seen_iter) == 0:
            i -= 1
        # if all new landmarks, add a prior
        elif seen_iter.intersection(seen) == set():
            graph.push_back(gtsam.PriorFactorPose3(X(i), estimate.atPose3(X(last_i)), model))
            initial_estimate.insert(X(i), estimate.atPose3(X(last_i)))
        # Guess new location close to old one :)
        elif i != 1:
            initial_estimate.insert(X(i), estimate.atPose3(X(last_i)))

    # reset what we saw this iteration
    seen = seen.union(seen_iter)
    seen_iter = set()

    # graph.saveGraph(f"graphs/test_out_{i}.dot")
    
    # update isam every update_every iterations
    cv2.imshow('image', cv2.hconcat(images))
    if (i+update_every-1) % update_every == 0:
        last_i = i

        # update isam
        isam.update(graph, initial_estimate)
        estimate = isam.calculateEstimate()


        # clear everything out for next iteration
        graph.resize(0)
        initial_estimate.clear()

        # plot
        # plot_3d(estimate, ax_3d, seen)
        plot_2d(estimate, isam, ax_2d, seen)
        plt.pause(.00001)

        key = cv2.waitKey(1)
    else:
        key = cv2.waitKey(1000//framerate)

    if key == ord('q'):
        break

    i += 1

# save estimate
with open('results.pkl', 'wb') as handle:
    print("Saving...")
    pickle.dump(estimate, handle)

# close cameras
for idx, name, mtx, dist, cap, noise, transform in all_cameras:
    cap.release()