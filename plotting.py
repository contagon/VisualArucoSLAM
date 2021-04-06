import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611
import gtsam.utils.plot as gtsam_plot
from matplotlib import patches
from gtsam.symbol_shorthand import L, X

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Args:
      fignum (int): An integer representing the figure number for Matplotlib.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def plot_3d(result, ax, seen):
    ax.cla()

    # Plot cameras
    i = 0
    while result.exists(X(i)):
        pose_i = result.atPose3(X(i))
        gtsam_plot.plot_pose3_on_axes(ax, pose_i, 10)
        i += 1

    # plot landmarks
    for i in seen:
        pose_i = result.atPose3(L(i))
        gtsam_plot.plot_pose3_on_axes(ax, pose_i, 5)

    # draw
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)
    ax.view_init(-55, -85)

def plot_2d(result, isam, ax, seen):
    idx = np.ix_((0,-1), (0,-1))
    ax.cla()

    # plot cameras
    i = 0
    cameras = []
    while result.exists(X(i)):
        # get all data from pose
        pose = result.atPose3(X(i))
        # R = pose.rotation().matrix()
        # cov = isam.marginalCovariance(X(i))[3:6,3:6]
        # cov = ( R@cov@R.T )[idx]
        t = [result.atPose3(X(i)).x(), result.atPose3(X(i)).z()]

        # do we want pose covariances?
        # ax.add_patch(cov_patch(t, cov, 'b'))
        cameras.append(t)

        i += 1

    # plot landmarks
    landmarks = []
    for i in seen:
        # get all data from pose
        pose = result.atPose3(L(i))
        R = pose.rotation().matrix()
        cov = isam.marginalCovariance(L(i))[3:6,3:6]
        cov = ( R@cov@R.T )[idx]
        t = [result.atPose3(L(i)).x(), result.atPose3(L(i)).z()]

        ax.add_patch(cov_patch(t, cov, 'r'))
        landmarks.append(t)

    cameras = np.array(cameras)
    landmarks = np.array(landmarks)

    ax.plot(cameras[:,0], cameras[:,1], label="Camera Poses", c='b', marker='o')
    ax.scatter(landmarks[:,0], landmarks[:,1], label="Landmarks", c='r')

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect('equal')

def cov_patch(origin, cov, color):
    k = 3
    w, v = np.linalg.eig(cov)
    angle = np.arctan2(v[1, 0], v[0, 0])
    return patches.Ellipse(origin, np.sqrt(w[0]*k), np.sqrt(w[1]*k),
                             np.rad2deg(angle), fill=False, color=color, alpha=0.8)