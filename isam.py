import gtsam
from gtsam.symbol_shorthand import L, X

def plot_3d(estimate):
    pass

def plot(estimate):
    pass

i = 0
num_l = 10
seen = set()

# TODO set up camera and import intrinsic parameters

# setup iSAM
parameters = gtsam.ISAM2Params()
parameters.setRelinearizeThreshold(0.01)
parameters.setRelinearizeSkip(1)
isam = gtsam.ISAM2(parameters)
model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))

# Create a Factor Graph and Values to hold the new data
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

while(True):
    #obtain camera image
    ret0, img = cap.read()                                                                                                                          

    #detect the markers in the image, and add them to the graph
    markerCorners, markerIds, rejecteCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
    img = cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)

    if markerIds is not None:
        for corners, id in zip(markerCorners, markerIds.flatten()):
            # estimate pose for each marker
            R, T, _ = cv2.aruco.estimatePoseSingleMarkers(corners, size, mtx, dist)

            # add factor to graph
            l_pose = gtsam.Pose3(R, T)
            # TODO Verify I've got X and L in the right order here
            graph.add(gtsam.BetweenFactorPose3(X(i), L(id), l_pose, model))

            # add an estimate if we haven't seen it before
            if id not in seen:
                seen.add(id)
                # TODO Come up with better inital estimate for landmarks
                initial_estimate.insert(L(id), gtsam.Pose3()) 


    if i == 0:
        # add in origin for first pose estimate
        initial_estimate.insert(X(0), gtsam.Pose3())

        # Add a prior on pose x0
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))  # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        graph.push_back(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(), pose_noise))

    else:
        # Guess new location is close to old one :)
        initial_estimate.insert(X(i), estimate.atPose3(X(i-1)))

        # update isam
        isam.update(graph, initial_estimate)
        estimate = isam.calculateEstimate()

        # clear everything out for next run
        graph.resize(0)
        initial_estimate.clear()