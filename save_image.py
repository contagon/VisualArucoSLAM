import gtsam
from gtsam.symbol_shorthand import L, X
import numpy as np
import cv2
from plotting import *

np.set_printoptions(suppress=True)

idx_l = 6
idx_c = 2
idx_r = 4

# set up camera and import intrinsic parameters
paramsl = np.load('params/left_params.npz')
paramsr = np.load('params/right_params.npz')
left   = ['L', cv2.VideoCapture(idx_l)]
center = ['C', cv2.VideoCapture(idx_c)]
right  = ['R', cv2.VideoCapture(idx_r)]
all_cameras = [left, center, right]

for name, cap in all_cameras:
    ret, img = cap.read()

    cv2.imwrite(f"imagesL/tag_{name}.jpg", img)